"""Frame-level check of rendered clips (decoded MP4 pixels vs the scene data).

    python frame_check.py --dataset-dir <dir>

For every clip it decodes the MP4 and checks:
  * frame count = scene n_frames;
  * timer pixels: every 12th frame and every frame a question relies on
    (stops, restarts, marker crossings, timer readings named in a question,
    the frame of every time-valued answer) match a reference rendering of
    the expected timer text much better than the neighbouring readings;
  * markers: on every 2nd frame each marker's colour blob is found within
    3 px of where the scene says its corner is (skipped while a nearer
    marker covers it);
  * crossings read from pixels: a marker's blob crossing the axis line on
    the near half of its orbit (blob larger than its median size), on the same frames (+-1) as the
    ground-truth crossings, so counts and first-crossing order match;
  * stops read from pixels: per solid, the frames on which the solid's
    pixels stop changing / start changing again match the scene's pauses
    (+-1 frame), so the stop count and times match;
  * side answers: the blob is left / right of the axis line as stated;
  * edges: at mid-clip every projected edge has ink along it, so the drawn
    edge count equals the solid's edge count;
  * nothing else is drawn: pixels outside the solids' bounding circles and
    the timer box equal the background (no stray text, labels or objects).
Prints a summary and exits 1 on any failure.
"""
from __future__ import annotations
import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))
import answers  # noqa: E402
import render  # noqa: E402
import sim  # noqa: E402
from common.video5 import font  # noqa: E402


def frames(path: Path) -> list[np.ndarray]:
    import cv2
    cap = cv2.VideoCapture(str(path))
    out = []
    while True:
        ok, fr = cap.read()
        if not ok:
            break
        out.append(fr[:, :, ::-1].astype(np.int16))
    cap.release()
    return out


def background(scene) -> np.ndarray:
    return np.asarray(render._background(render.bg_key(scene)).resize((sim.W, sim.H), Image.LANCZOS), dtype=np.int16)


def timer_ref(scene, bgimg, text):
    img = Image.fromarray(bgimg.astype(np.uint8))
    x, y, anchor = render.timer_xy(scene)
    ImageDraw.Draw(img).text((x, y), text, fill=tuple(scene["timer"]["ink"]), font=font(scene["timer"]["size"]),
                             anchor=anchor)
    x0, y0, x1, y1 = sim.timer_box(scene)
    return np.asarray(img, dtype=np.int16)[y0:y1, x0:x1]


def timer_ok(scene, bgimg, fr, f) -> bool:
    x0, y0, x1, y1 = sim.timer_box(scene)
    crop = fr[y0:y1, x0:x1]
    tn = sim.tenths(f)
    d_true = np.abs(crop - timer_ref(scene, bgimg, f"{sim.fmt_t(tn)} s")).mean()
    d_other = min(np.abs(crop - timer_ref(scene, bgimg, f"{sim.fmt_t(t)} s")).mean() for t in (tn - 1, tn + 1) if t >= 0)
    return d_true < 4 and d_true < 0.6 * d_other


def blob(fr, rgb, x, y, r):
    R = int(r * 2 + 8)
    x0, y0 = max(0, int(x) - R), max(0, int(y) - R)
    win = fr[y0:int(y) + R + 1, x0:int(x) + R + 1]
    mask = np.abs(win - np.array(rgb)).sum(axis=2) < 90
    if mask.sum() < 20:
        return None
    ys, xs = np.nonzero(mask)
    return xs.mean() + x0, ys.mean() + y0, int(mask.sum())


def covered(ms, i):
    """Marker i is (partly) hidden by a nearer marker."""
    x, y, r, z = ms[i]
    return any(j != i and ms[j][3] > z and math.hypot(ms[j][0] - x, ms[j][1] - y) < r + ms[j][2] + 8
               for j in range(len(ms)))


def check_clip(s: dict, root: Path) -> tuple[list[str], dict]:
    scene, sid = s["params"], s["sample_id"]
    errs, stats = [], {"timer": 0, "blobs": 0, "edges": 0, "crossings": 0, "stops": 0, "sides": 0}
    frs = frames(root / s["media_path"])
    n = scene["n_frames"]
    if len(frs) != n:
        return [f"{sid}: {len(frs)} frames decoded, expected {n}"], stats
    bgimg = background(scene)
    nm = len(scene["marks"])
    # ---- markers on every 2nd frame
    tracks = {i: {} for i in range(nm)}
    for f in range(0, n, 2):
        ms = [render.mark_screen(scene, i, f) for i in range(nm)]
        for i in range(nm):
            if covered(ms, i):
                continue
            x, y, r, _ = ms[i]
            bb = blob(frs[f], scene["marks"][i]["rgb"], x, y, r)
            stats["blobs"] += 1
            if bb is None or math.hypot(bb[0] - x, bb[1] - y) > 3:
                errs.append(f"{sid}: frame {f} marker {scene['marks'][i]['colour']} not at its corner")
            else:
                tracks[i][f] = bb
    # ---- crossings read from pixels (upright-axis bodies)
    for i, mk in enumerate(scene["marks"]):
        body = scene["bodies"][mk["body"]]
        if not sim.y_only(body):
            continue
        full = {}
        for f in range(n):
            ms = [render.mark_screen(scene, j, f) for j in range(nm)]
            if covered(ms, i):
                continue
            x, y, r, _ = ms[i]
            bb = blob(frs[f], mk["rgb"], x, y, r)
            if bb is not None:
                full[f] = bb
        if len(full) < n * 0.6:
            continue
        amid = np.median([v[2] for v in full.values()])   # nearer markers are drawn larger
        seen = []
        fs = sorted(full)
        for a, b in zip(fs, fs[1:]):
            if b != a + 1:
                continue
            xa, xb = full[a][0] - body["cx"], full[b][0] - body["cx"]
            if (xa > 0) != (xb > 0) and (full[a][2] + full[b][2]) / 2 > amid:
                seen.append(b)
        truth = answers.crossings(scene, i)
        gaps = [f for f in range(n) if f not in full]
        ok = len(seen) == len(truth) and all(abs(p - q) <= 1 for p, q in zip(seen, truth))
        if not ok and not any(abs(g - t) <= 2 for g in gaps for t in truth):
            errs.append(f"{sid}: marker {mk['colour']} pixel crossings {seen} != {truth}")
        stats["crossings"] += len(truth)
    # ---- stops read from pixels
    for b, body in enumerate(scene["bodies"]):
        R = sim.body_radius(scene, b)
        x0, x1 = int(max(0, body["cx"] - R)), int(min(sim.W, body["cx"] + R))
        y0, y1 = int(max(0, body["cy"] - R)), int(min(sim.H, body["cy"] + R))
        diff = [int((np.abs(frs[f + 1][y0:y1, x0:x1] - frs[f][y0:y1, x0:x1]).max(axis=2) > 40).sum())
                for f in range(n - 1)]
        still = [d < 25 for d in diff]          # a turning solid changes hundreds of pixels per frame
        stops = [f for f in range(1, n - 1) if still[f] and not still[f - 1]]
        res = [f + 1 for f in range(1, n - 1) if not still[f] and still[f - 1]]
        t_stop, t_res = sim.stop_frames(body), sim.resume_frames(body)
        if len(stops) != len(t_stop) or any(abs(p - q) > 1 for p, q in zip(stops, t_stop)) or \
                len(res) != len(t_res) or any(abs(p - q) > 1 for p, q in zip(res, t_res)):
            errs.append(f"{sid}: body {b} pixel stops {stops}/{res} != {t_stop}/{t_res}")
        stats["stops"] += len(t_stop)
    # ---- edges at mid clip (an edge fully under a marker is looked for on a later frame instead)
    for b, body in enumerate(scene["bodies"]):
        _, edges, _ = sim.geometry(body["solid"])
        hit = 0
        for a, c in edges:
            ok = None
            for f in [n // 2 + 5 * k for k in range(8)]:
                f = f % n
                sx, sy, z = render.project(scene, b, f)
                ms = [render.mark_screen(scene, i, f) for i in range(nm)]
                tried = False
                for u in np.linspace(0.2, 0.8, 13):
                    x, y = sx[a] + (sx[c] - sx[a]) * u, sy[a] + (sy[c] - sy[a]) * u
                    if any(math.hypot(x - m[0], y - m[1]) < m[2] + 6 for m in ms):
                        continue
                    tried = True
                    px = frs[f][int(round(y)) - 1:int(round(y)) + 2, int(round(x)) - 1:int(round(x)) + 2]
                    if np.abs(px - bgimg[int(round(y)), int(round(x))]).sum(axis=2).max() > 60:
                        ok = True
                        break
                if tried:
                    ok = bool(ok)
                    break
            hit += bool(ok)
        stats["edges"] += hit
        if hit != len(edges):
            errs.append(f"{sid}: body {b} shows ink on {hit} of {len(edges)} edges")
    # ---- nothing outside the solids and the timer
    yy, xx = np.mgrid[0:sim.H, 0:sim.W]
    outside = np.ones((sim.H, sim.W), bool)
    for b, body in enumerate(scene["bodies"]):
        outside &= np.hypot(xx - body["cx"], yy - body["cy"]) > sim.body_radius(scene, b)
    tx0, ty0, tx1, ty1 = sim.timer_box(scene)
    outside[ty0 - 4:ty1 + 4, tx0 - 4:tx1 + 4] = False
    for f in (0, n // 3, 2 * n // 3, n - 1):
        dev = np.abs(frs[f] - bgimg).sum(axis=2)[outside]
        if (dev > 40).mean() > 0.0005:
            errs.append(f"{sid}: frame {f} has something drawn outside the solids and the timer")
    # ---- timer
    tf = set(range(0, n, 12))
    for b, body in enumerate(scene["bodies"]):
        tf |= set(sim.stop_frames(body)) | set(sim.resume_frames(body))
    for i in range(nm):
        tf |= set(answers.crossings(scene, i)) if sim.y_only(scene["bodies"][scene["marks"][i]["body"]]) else set()
    for q in s["questions"]:
        p = q["params"]
        if "t" in p:
            tf |= {f for f in range(n) if sim.tenths(f) == p["t"]}
        if q["answer_text"].endswith(" s") and q["derivation_key"] != "period":
            fs = [f for f in range(n) if sim.timer_text(f) == q["answer_text"]]
            tf |= set(fs)
            ev = {"stop_time": lambda: sim.stop_frames(scene["bodies"][p["b"]]),
                  "resume_time": lambda: sim.resume_frames(scene["bodies"][p["b"]]),
                  "first_pass_time": lambda: answers.crossings(scene, p["m"]),
                  "pass_time": lambda: answers.crossings(scene, p["m"])}[q["derivation_key"]]()
            if not set(ev) & set(fs):
                errs.append(f"{sid}/{q['qid']}: answer {q['answer_text']} is not on the timer at any event frame")
        if q["derivation_key"] in ("mark_side", "side_at_time", "mark_at_stop"):
            m = p["m"]
            if q["derivation_key"] == "mark_side":
                fq = 0 if p["when"] == "start" else n - 1
            elif q["derivation_key"] == "side_at_time":
                fq = min(f for f in range(n) if sim.tenths(f) == p["t"])
            else:
                fq = sim.stop_frames(scene["bodies"][scene["marks"][m]["body"]])[0]
            x, y, r, _ = render.mark_screen(scene, m, fq)
            bb = blob(frs[fq], scene["marks"][m]["rgb"], x, y, r)
            cx = scene["bodies"][scene["marks"][m]["body"]]["cx"]
            if bb is None or ("left" if bb[0] < cx else "right") not in q["answer_text"].split(" of ")[0]:
                errs.append(f"{sid}/{q['qid']}: marker pixels are not on the stated side")
            stats["sides"] += 1
    for f in sorted(tf):
        stats["timer"] += 1
        if not timer_ok(scene, bgimg, frs[f], f):
            errs.append(f"{sid}: timer pixels at frame {f} don't read {sim.timer_text(f)}")
    return errs, stats


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset-dir", required=True)
    args = ap.parse_args()
    root = Path(args.dataset_dir)
    errs, n, tot = [], 0, {}
    with open(root / "annotations.jsonl", encoding="utf8") as fh:
        for line in fh:
            s = json.loads(line)
            n += 1
            e, st = check_clip(s, root)
            errs += e
            for k, v in st.items():
                tot[k] = tot.get(k, 0) + v
    print(f"frame check: {n} clips, {len(errs)} failures; checked {tot}")
    for e in errs[:40]:
        print("  -", e)
    sys.exit(1 if errs else 0)


if __name__ == "__main__":
    main()
