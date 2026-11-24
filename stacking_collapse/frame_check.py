"""Frame-level check of rendered clips against the scene data.

    python frame_check.py --dataset-dir <dir> [--report frame_check.txt] [--jobs 4]

For every clip it decodes the MP4 (OpenCV) and checks:
  * clock: template-matches the timer panel against every reading 0.0-9.9 s
    on every 6th frame, on every event frame and on every frame a question
    names by its clock reading;
  * no other text: the clip's label list is empty;
  * blocks: on every 3rd frame and every event frame, each block that is
    wholly in the frame is found where the track puts it (enough pixels of
    its colour inside its outline, their centroid within 3 px of where the
    drawn block puts it);
  * landing: on its landing frame a block sits on its rest spot (centroid
    within 2.5 px); the first frame it is seen sitting still there equals
    the event frame (so one frame earlier it is still falling), and so the
    order of landing is the one the questions use;
  * pulls: on the first frame of a pull the block has moved sideways by
    the slide speed; collapses: four frames after a collapse starts, the
    top block of the falling part has moved towards the side it tips to;
  * grid: the spacing of the grid lines / dots measured in the first frame
    equals the unit; widths read off the frames in grid squares equal the
    answers of the width questions;
  * answers read off the frames: clock answers on the event frame; the
    number of blocks at rest on the frames showing a named clock reading
    and at the end; which block is on top at the end.
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
import render  # noqa: E402
import sim  # noqa: E402
import questions  # noqa: E402
from common.video5 import font  # noqa: E402

COL_TOL = 60          # RGB distance for "this block's colour"


def frames_of(path: Path) -> list:
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


_TIMER = {}


def _timer_templates():
    if not _TIMER:
        tb = sim.TIMER_BOX
        for k in range(100):
            img = Image.new("RGB", (sim.W, sim.H), (0, 0, 0))
            d = ImageDraw.Draw(img)
            d.rounded_rectangle(tb, radius=10, fill=render.TIMER_BG, outline=render.TIMER_EDGE, width=2)
            d.text(((tb[0] + tb[2]) / 2, (tb[1] + tb[3]) / 2), sim.clock_text(k), fill=render.TIMER_FG,
                   font=font(sim.TIMER_FONT), anchor="mm")
            _TIMER[k] = np.asarray(img, dtype=np.int16)[tb[1] + 4:tb[3] - 4, tb[0] + 4:tb[2] - 4]
    return _TIMER


def read_timer(fr) -> int:
    tb = sim.TIMER_BOX
    crop = fr[tb[1] + 4:tb[3] - 4, tb[0] + 4:tb[2] - 4]
    tm = _timer_templates()
    return min(tm, key=lambda k: np.abs(crop - tm[k]).mean())


def _rect_of(poly):
    q = np.asarray(poly, dtype=float)
    c = q.mean(axis=0)
    ang = math.atan2(q[1, 1] - q[0, 1], q[1, 0] - q[0, 0])
    return c[0], c[1], ang, math.hypot(*(q[1] - q[0])), math.hypot(*(q[3] - q[0]))


def inside(fr, poly, rgb, shrink):
    """(pixels of colour rgb inside the outline shrunk by `shrink` px, mask size, centroid)."""
    cx, cy, ang, w, h = _rect_of(poly)
    p = sim.rect(cx, cy, ang, max(2.0, w - 2 * shrink), max(2.0, h - 2 * shrink))
    xs = [v[0] for v in p]
    ys = [v[1] for v in p]
    x0, y0 = max(0, int(math.floor(min(xs)))), max(0, int(math.floor(min(ys))))
    x1, y1 = min(sim.W, int(math.ceil(max(xs))) + 1), min(sim.H, int(math.ceil(max(ys))) + 1)
    if x1 <= x0 or y1 <= y0:
        return 0, 0, None
    m = Image.new("L", (x1 - x0, y1 - y0), 0)
    ImageDraw.Draw(m).polygon([(x - x0, y - y0) for x, y in p], fill=255)
    m = np.asarray(m) > 0
    crop = fr[y0:y1, x0:x1]
    col = (np.abs(crop - np.array(rgb, dtype=np.int16)).sum(axis=2) < COL_TOL) & m
    ys_, xs_ = np.nonzero(col)
    if len(xs_) == 0:
        return 0, int(m.sum()), None
    return len(xs_), int(m.sum()), (float(xs_.mean()) + x0, float(ys_.mean()) + y0)


_EXP = {}


def expected(scene, j):
    """(colour fraction inside the shrunk outline, centroid offset from the centre) of block j drawn alone."""
    st, b, S = scene["style"], scene["blocks"][j], scene["S"]
    key = (b["colour"], b["pattern"], b["w"] * S, b["h"] * S, st["outline"], st["stripe"], render.edge(st))
    if key not in _EXP:
        img = Image.new("RGB", (sim.W * render.SS, sim.H * render.SS), (128, 128, 128))
        sp = render.sprite(*key)
        render._paste(img, sp, 640, 360, 0.0)
        fr = np.asarray(img.reduce(render.SS), dtype=np.int16)
        poly = sim.rect(640, 360, 0, b["w"] * S, b["h"] * S)
        cnt, area, c = inside(fr, poly, sim.COLOURS[b["colour"]], st["outline"] + 2)
        _EXP[key] = (cnt / area, (c[0] - 640, c[1] - 360))
    return _EXP[key]


def shrink_of(scene):
    return scene["style"]["outline"] + 2


def found(fr, scene, j, poly, tol=3.0, frac=0.75):
    """Block j where `poly` puts it: enough of its colour, centroid where the drawn block puts it."""
    fx, (ox, oy) = expected(scene, j)
    cnt, area, c = inside(fr, poly, sim.COLOURS[scene["blocks"][j]["colour"]], shrink_of(scene))
    if area == 0 or c is None or cnt < frac * fx * area:
        return False, cnt, c
    cx, cy, ang, _, _ = _rect_of(poly)
    ex = cx + ox * math.cos(ang) - oy * math.sin(ang)
    ey = cy + ox * math.sin(ang) + oy * math.cos(ang)
    return math.hypot(c[0] - ex, c[1] - ey) <= tol, cnt, c


def in_frame(poly, m=2):
    q = np.asarray(poly)
    return q[:, 0].min() >= m and q[:, 1].min() >= m and q[:, 0].max() <= sim.W - m and q[:, 1].max() <= sim.H - m


def rest_polys(tr, j):
    """Distinct outlines block j rests in (upright, unchanged over 2+ frames)."""
    P = tr["poly"]
    out = []
    for f in range(1, tr["n_frames"]):
        a, b = P[f, j], P[f - 1, j]
        if np.isnan(a[0, 0]) or np.isnan(b[0, 0]) or not np.allclose(a, b) or abs(a[1, 1] - a[0, 1]) > 1e-9:
            continue
        if not any(np.allclose(a, o) for o in out):
            out.append(a.copy())
    return out


def at_rest(frames, scene, j, polys, f):
    """Block j sits still in one of its rest outlines over frames f-1, f, f+1 (as seen in the video)."""
    for p in polys:
        if all(found(frames[g], scene, j, p, tol=2.5, frac=0.85)[0]
               for g in (f - 1, f, f + 1) if 0 <= g < len(frames)):
            return True
    return False


def grid_spacing(fr, scene):
    """Spacing (px) of the grid lines / dots along a background row near the top of the first frame:
    the lag with the strongest autocorrelation."""
    S, oy = scene["S"], scene["oy"]
    half = S / 2 if scene["style"]["grid"] == "lines" else 0      # between the lines, or through the dots
    ys = [oy - S * k - half for k in range(-40, 40) if 80 <= oy - S * k - half <= 150]
    if not ys:
        return None
    y = int(round(ys[0]))
    row = fr[max(0, y - 1):y + 2, 230:sim.W - 10].mean(axis=0)
    dev = np.abs(row - np.median(row, axis=0)).sum(axis=1)
    tr = render.track(scene)
    polys = [tr["poly"][0, j] for j in range(len(scene["blocks"]))] + ([tr["plank"][0]] if tr["plank"] is not None else [])
    polys += [np.array([[x0, yt], [x1, yt], [x1, sim.H], [x0, sim.H]]) for x0, yt, x1 in tr["pillars"]]
    for q in polys:                                   # leave out columns where something covers the row
        if not np.isnan(q[0, 0]) and q[:, 1].min() - 3 <= y <= q[:, 1].max() + 3:
            a, b = int(q[:, 0].min()) - 233, int(q[:, 0].max()) - 227
            dev[max(0, a):max(0, b)] = 0
    dev = dev - dev.mean()
    if not dev.any():
        return None
    ac = [float((dev[:-k] * dev[k:]).sum() / (len(dev) - k)) for k in range(24, 61)]
    return float(24 + int(np.argmax(ac)))


def check_clip(scene, qs, frames) -> list[str]:
    errs = []
    n = scene["n_frames"]
    if len(frames) != n:
        return [f"decoded {len(frames)} frames, expected {n}"]
    if scene["labels"]:
        errs.append("labels present")
    F = questions.Facts(scene)
    tr = render.track(scene)
    P = tr["poly"]
    nb = len(scene["blocks"])
    ev = set()
    for e in F.lands:
        ev |= {e["f"] - 1, e["f"]}
    for e in F.pulls:
        ev |= {e["f"] - 1, e["f"]} | ({e["settle"]} if e["settle"] else set())
    for e in F.falls:
        ev |= {e["f"] - 1, e["f"], e["f"] + 4}
    ev = {f for f in ev if 0 <= f < n} | {0, n - 1}
    named = set()
    for q in qs:
        p = q["params"]
        if q["derivation_key"] in ("count_at", "between"):
            for kk in ("k", "k1", "k2"):
                if kk in p:
                    lo, hi = sim.frames_showing(p[kk])
                    named |= set(range(lo, hi + 1))
    # clock
    for f in sorted(set(range(0, n, 6)) | ev | named):
        got = read_timer(frames[f])
        if got != sim.tenths(f):
            errs.append(f"frame {f}: clock reads {sim.clock_text(got)}, should be {sim.clock_text(sim.tenths(f))}")
    # blocks where the track puts them
    for f in sorted(set(range(0, n, 3)) | ev):
        for j in range(nb):
            p = P[f, j]
            if np.isnan(p[0, 0]) or not in_frame(p):
                continue
            ok, cnt, c = found(frames[f], scene, j, p, tol=3.0 if abs(p[1, 1] - p[0, 1]) < 1e-9 else 4.0)
            if not ok:
                errs.append(f"frame {f}: {F.nm[j]} not found where drawn ({cnt} px, {c})")
    # landings, and the landing order read off the frames
    for e in F.lands:
        j, f = e["block"], e["f"]
        rp = P[f, j]
        if not found(frames[f], scene, j, rp, tol=2.5, frac=0.85)[0]:
            errs.append(f"{F.nm[j]} not at rest on its landing frame {f}")
        f0 = next((g for g in range(1, n - 1) if found(frames[g], scene, j, rp, tol=2.5, frac=0.85)[0]
                   and found(frames[g + 1], scene, j, rp, tol=2.5, frac=0.85)[0]), None)
        if f0 != f:
            errs.append(f"{F.nm[j]} seen landing on frame {f0}, event frame {f}")
    # pulls and collapses
    for e in F.pulls:
        j, f = e["block"], e["f"]
        a, b = P[f - 1, j], P[f, j]
        ok0, _, c0 = found(frames[f - 1], scene, j, a)
        ok1, _, c1 = found(frames[f], scene, j, b) if in_frame(b) else (True, 0, None)
        if not ok0 or (c1 is not None and abs((c1[0] - c0[0]) - e["dir"] * scene["pull_v"]) > 2.5):
            errs.append(f"{F.nm[j]} pull at frame {f} not seen ({c0} -> {c1})")
    for e in F.falls:
        j = e["blocks"][-1]
        f = e["f"] + 3
        a, b = P[e["tip"], j], P[f, j]
        if np.isnan(b[0, 0]) or not in_frame(b):
            continue
        _, _, c0 = found(frames[e["tip"]], scene, j, a)
        ok, _, c1 = found(frames[f], scene, j, b, tol=4.0)
        if not ok or c0 is None or (c1[0] - c0[0]) * (1 if e["side"] == "right" else -1) <= 0.5:
            errs.append(f"collapse at frame {e['f']}: {F.nm[j]} not seen moving {e['side']} ({c0} -> {c1})")
    # grid spacing in the frame = the unit
    sp = grid_spacing(frames[0], scene)
    if sp is None or abs(sp - scene["S"]) > 0.6:
        errs.append(f"grid spacing read off the frame {sp}, unit {scene['S']}")
    # answers read off the frames
    rests = {j: rest_polys(tr, j) for j in range(nb)}

    def standing(f, base=None):
        out = []
        for j in range(nb):
            if base is not None:
                px = [p.mean(axis=0)[0] for p in rests[j]]
                cen = [(x0 + x1) / 2 for x0, _, x1 in sim.bases_px(scene)]
                if not px or min(range(len(cen)), key=lambda b: abs(cen[b] - px[0])) != base:
                    continue
            if at_rest(frames, scene, j, rests[j], f):
                out.append(j)
        return out
    for q in qs:
        key, p, gt = q["derivation_key"], q["params"], q["groundtruth"]
        if key in ("land_time", "pull_time"):
            j = F.nm.index(p["x"])
            fe = [e["f"] for e in (F.lands if key == "land_time" else F.pulls) if e["block"] == j][0]
            if sim.clock_text(read_timer(frames[fe])) != gt:
                errs.append(f"{q['qid']} {key}: clock on frame {fe} reads {sim.clock_text(read_timer(frames[fe]))}")
        if key == "count_at":
            lo, hi = sim.frames_showing(p["k"])
            for f in (lo, (lo + hi) // 2, hi):
                got = len(standing(f, p["base"]))
                if got != gt:
                    errs.append(f"{q['qid']} count_at: {got} at rest on frame {f}, answer {gt}")
        if key == "end_count":
            got = len(standing(n - 2))
            if got != gt:
                errs.append(f"{q['qid']} end_count: {got} at rest at the end, answer {gt}")
        if key == "top_end":
            st = standing(n - 2, p["base"])
            top = min(st, key=lambda j: P[n - 2, j][:, 1].min()) if st else None
            if top is None or F.nm[top] != gt:
                errs.append(f"{q['qid']} top_end: top at the end read as {top}")
        if key == "width":
            j = F.nm.index(p["x"])
            rp = rests[j][0]
            cx, cy, _, w, h = _rect_of(rp)
            x0, x1 = int(cx - w / 2 - 4), int(cx + w / 2 + 5)
            y = int(round(cy))
            fr = frames[[f for f in range(n) if np.allclose(P[f, j], rp)][0]]
            col = np.abs(fr[y - 1:y + 2, x0:x1] - np.array(sim.COLOURS[scene["blocks"][j]["colour"]])).sum(axis=2)
            cols = np.nonzero((col < COL_TOL).any(axis=0))[0]
            wpx = cols.max() - cols.min() + 1 + 2 * scene["style"]["outline"] if len(cols) else 0
            if sp is None or int(round(wpx / sp)) != gt:
                errs.append(f"{q['qid']} width: read {wpx} px / {sp} = {wpx / sp if sp else None}, answer {gt}")
    return errs


def _one(arg):
    root, line = arg
    s = json.loads(line)
    frames = frames_of(Path(root) / s["media_path"])
    try:
        errs = check_clip(s["params"], s["questions"], frames)
    except Exception as e:  # report, don't crash the pool
        errs = [f"checker error {e!r}"]
    return s["sample_id"], s["params"]["scenario"], errs


def main():
    from multiprocessing import Pool
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset-dir", required=True)
    ap.add_argument("--report", default="frame_check.txt")
    ap.add_argument("--jobs", type=int, default=4)
    args = ap.parse_args()
    root = Path(args.dataset_dir)
    rows = open(root / "annotations.jsonl", encoding="utf8").read().splitlines()
    lines, total = [], 0
    with Pool(args.jobs) as pool:
        for sid, sc, errs in pool.imap(_one, [(str(root), r) for r in rows]):
            total += len(errs)
            lines += [f"{sid}: {e}" for e in errs]
            print(f"{sid} ({sc}): {len(errs)} failures", flush=True)
    lines.append(f"RESULT: {len(rows)} clips, {total} frame-check failures")
    (root / args.report).write_text("\n".join(lines) + "\n")
    print(lines[-1])
    for e in lines[:-1][:20]:
        print("  -", e)
    sys.exit(1 if total else 0)


if __name__ == "__main__":
    main()
