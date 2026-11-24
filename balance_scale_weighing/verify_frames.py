"""Frame-level check of rendered clips against the ground truth.

    python verify_frames.py --dataset-dir <dir with annotations.jsonl + videos/>

For every clip it decodes the MP4 and checks:
  * timer: at every event frame a question refers to, the timer patch in
    the decoded frame is closer to the expected reading than to the
    readings 0.1 s before/after (template match against freshly drawn
    timer patches);
  * weights: every 6th frame, each weight resting on the scale shows at
    least 60% as many pixels of its base colour as its sprite has, and
    their centroid is within 12 px of where the sprite's base-colour
    centroid should be per render.layout; a weight that is not in the
    frame leaves no blob (>150 px) of its colour;
  * counts: in the last frame, the number of weights found this way
    equals the number of weights on the scale.
"""
from __future__ import annotations
import argparse
import json
import math
import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))
import render  # noqa: E402
import sim  # noqa: E402

TOL_PX = 12


def _timer_patch(f):
    img = Image.new("RGB", (render.W * render.SS, render.H * render.SS), (0, 0, 0))
    render._draw_timer(ImageDraw.Draw(img), f)
    img = img.resize((render.W, render.H), Image.LANCZOS)
    x0, y0, x1, y1 = render.TIMER_BOX
    return np.asarray(img, dtype=np.float32)[y0 + 4:y1 - 4, x0 + 4:x1 - 4]


def _frame_for_tenths(t):
    return next(f for f in range(0, 400) if (f * 10) // 24 == t)


def _mask(rgb, colour, floor_y):
    refs = np.array([render.RGB[colour]], dtype=np.float32)
    px = rgb[:floor_y].astype(np.float32)
    d = np.min(np.linalg.norm(px[:, :, None, :] - refs[None, None], axis=3), axis=2)
    return d < 40


def _expected(o, st, scene):
    """Centroid and pixel count of the weight's base-colour pixels, from its sprite placed per the layout."""
    th = scene["theme"]
    bg = render._lum(render._mix(th["bg_top"], th["bg_bot"], 0.5))
    outline = (24, 24, 28, 255) if bg > 110 else (240, 240, 240, 255)
    sp = render._sprite(o["colour"], o["shape"], o["pattern"], 0 if o["mystery"] else o["mass"],
                        round(st["size"], 1), outline)
    sp = sp.resize((sp.width // render.SS, sp.height // render.SS), Image.LANCZOS)
    arr = np.asarray(sp).astype(np.float32)
    m = (np.linalg.norm(arr[:, :, :3] - np.array(render.RGB[o["colour"]], np.float32), axis=2) < 40) & (arr[:, :, 3] > 250)
    ys, xs = np.nonzero(m)
    cx, cy = xs.mean() - sp.width / 2, ys.mean() - sp.height / 2
    r = math.radians(st["rot"])
    return st["x"] + cx * math.cos(r) - cy * math.sin(r), st["y"] + cx * math.sin(r) + cy * math.cos(r), m.sum()


def _cover(rgb, o, st, scene, fy):
    ex, ey, area = _expected(o, st, scene)
    R = int(st["size"]) + 2
    ya, yb, xa, xb = max(0, int(ey) - R), min(fy, int(ey) + R), max(0, int(ex) - R), int(ex) + R
    yy, xx = np.mgrid[ya:yb, xa:xb]
    m = _mask(rgb[ya:yb, xa:xb], o["colour"], yb - ya) & ((xx - ex) ** 2 + (yy - ey) ** 2 < (0.85 * st["size"]) ** 2)
    return round(float(m.sum() / area), 2)


def check_clip(s, root):
    scene = s["params"]
    errs = []
    cap = cv2.VideoCapture(str(root / s["media_path"]))
    frames = []
    while True:
        ok, fr = cap.read()
        if not ok:
            break
        frames.append(cv2.cvtColor(fr, cv2.COLOR_BGR2RGB))
    cap.release()
    if len(frames) != scene["n_frames"]:
        return [f"{len(frames)} frames decoded, expected {scene['n_frames']}"], 0
    n_checks = 0
    # timer at every question-relevant event frame
    x0, y0, x1, y1 = render.TIMER_BOX
    for ev in scene["events"]:
        f = sim.moment(ev)
        got = frames[f].astype(np.float32)[y0 + 4:y1 - 4, x0 + 4:x1 - 4]
        t = (f * 10) // 24
        cands = {d: _timer_patch(_frame_for_tenths(t + d)) for d in (-1, 0, 1) if t + d >= 0}
        err = {d: float(np.abs(got - p).mean()) for d, p in cands.items()}
        if min(err, key=err.get) != 0:
            errs.append(f"timer at frame {f} doesn't read {sim.display(f)}")
        n_checks += 1
    objs = {o["id"]: o for o in scene["objects"]}
    fy = scene["theme"]["floor_y"] - 2
    # event moments: the weight is at its resting spot on the event frame side and away from it 3 frames on the
    # other side (landing: there at f, not at f-3; lift-off: there at f-1, gone from it at f+3)
    for ev in scene["events"]:
        f = sim.moment(ev)
        o = objs[ev["obj"]]
        at, away = (f - 1, f + 3) if ev["kind"] == "lift" else (f, f - 3)
        st = render.layout(scene, at)["objs"][o["id"]]
        cov = [_cover(frames[g], o, st, scene, fy) for g in (at, away)]
        if cov[0] < 0.6 or cov[1] > 0.3:
            errs.append(f"{ev['kind']} of {o['colour']} {o['shape']} not seen at frame {f} (cover {cov})")
        n_checks += 1
    yy, xx = np.mgrid[0:fy, 0:render.W]
    last_f = scene["n_frames"] - 1
    for f in list(range(0, scene["n_frames"], 6)) + [last_f]:
        lay = render.layout(scene, f)
        moving = [st for st in lay["objs"].values() if st["state"] == "moving"]
        n_seen = 0
        for oid, st in lay["objs"].items():
            o = objs[oid]
            if st["state"] == "rest":
                ex, ey, area = _expected(o, st, scene)
                if any(math.hypot(m["x"] - ex, m["y"] - ey) < 2 * st["size"] for m in moving):
                    continue                              # a weight in flight may pass in front
                R = int(st["size"]) + 2
                ya, yb, xa, xb = max(0, int(ey) - R), min(fy, int(ey) + R), max(0, int(ex) - R), int(ex) + R
                sub = frames[f][ya:yb, xa:xb]
                m = _mask(sub, o["colour"], sub.shape[0])
                m &= (xx[ya:yb, xa:xb] - ex) ** 2 + (yy[ya:yb, xa:xb] - ey) ** 2 < (0.85 * st["size"]) ** 2
                cover = m.sum() / area
                if cover < 0.6:
                    errs.append(f"frame {f}: {o['colour']} {o['shape']} not found at ({ex:.0f},{ey:.0f}), "
                                f"cover {cover:.2f}")
                else:
                    n_seen += 1
                    off = math.hypot(xx[ya:yb, xa:xb][m].mean() - ex, yy[ya:yb, xa:xb][m].mean() - ey)
                    if off > TOL_PX:
                        errs.append(f"frame {f}: {o['colour']} {o['shape']} centre off by {off:.1f}px")
                n_checks += 1
            elif st["state"] == "hidden":
                m = _mask(frames[f], o["colour"], fy).astype(np.uint8)
                n, _, stats, _ = cv2.connectedComponentsWithStats(m, 8)
                if any(stats[i, cv2.CC_STAT_AREA] > 150 for i in range(1, n)):
                    errs.append(f"frame {f}: {o['colour']} visible but should be hidden")
                n_checks += 1
        if f == last_f:
            on = sum(1 for st in lay["objs"].values() if st["state"] == "rest")
            if n_seen != on:
                errs.append(f"last frame shows {n_seen} weights, expected {on}")
    return errs, n_checks


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset-dir", required=True)
    args = ap.parse_args()
    root = Path(args.dataset_dir)
    total, bad = 0, 0
    for line in open(root / "annotations.jsonl", encoding="utf8"):
        s = json.loads(line)
        errs, n = check_clip(s, root)
        total += n
        bad += len(errs)
        for e in errs[:5]:
            print(f"{s['sample_id']}: {e}")
    print(f"frame checks: {total}, failures: {bad}")
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
