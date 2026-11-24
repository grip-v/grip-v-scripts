"""Frame-level check of rendered clips against the scene data.

    python frame_check.py --dataset-dir <dir> [--report frame_check.txt]

For every clip it decodes the MP4 (OpenCV) and, on the decoded frames:
  * reads the clock by template matching (all readings 0.0-9.9 s) on every
    event frame, every frame a question names by its clock reading and every
    12th frame; the best match must be the renderer's text for that frame;
  * identifies every stop marker by shape (mask IoU against all six shapes
    drawn at the stop's centre); the best match must be the stop's shape;
  * measures, on every frame, how far along its path each route is drawn
    (pixels of the route's own colour sampled along the path, away from the
    markers and the ring) and compares it with the renderer's track; fits
    the start / finish / erase / gone frames from the measured growth and
    requires them within 1 frame of the scene's;
  * finds the black ring (traveller) on every frame and requires its centre
    within 2 px of the track;
  * rebuilds the answers.py timeline from those MEASURED events and ring
    positions and re-derives every question; non-clock answers must equal
    the stored answer, clock answers must be within 0.1 s (options are 0.5 s
    apart);
  * repeats sim.check's layout tests (nothing reaches the clock or leaves
    the frame; no other text is drawn).
Prints a summary and exits 1 on any failure.
"""
from __future__ import annotations
import argparse
import copy
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

COL_TOL = 55.0
CLOCK_KEYS = ("finish_clock", "erase_clock", "arrive_clock")


def decode(path: Path) -> list:
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


# ---------------------------------------------------------------- clock

_TIMER = {}


def _timer_templates(scene):
    key = (tuple(scene["timer"]["box"]), scene["timer"]["size"])
    if key not in _TIMER:
        tb = scene["timer"]["box"]
        out = {}
        for k in range(100):
            img = Image.new("RGB", (sim.W, sim.H), (0, 0, 0))
            d = ImageDraw.Draw(img)
            d.rounded_rectangle(tb, radius=10, fill=render.TIMER_BG, outline=render.TIMER_EDGE, width=2)
            d.text(((tb[0] + tb[2]) / 2, (tb[1] + tb[3]) / 2), sim.clock_text(k), fill=render.TIMER_FG,
                   font=font(scene["timer"]["size"]), anchor="mm")
            out[sim.clock_text(k)] = np.asarray(img, dtype=np.int16)[tb[1] + 4:tb[3] - 4, tb[0] + 4:tb[2] - 4]
        _TIMER[key] = out
    return _TIMER[key]


def read_timer(scene, fr) -> str:
    tb = scene["timer"]["box"]
    crop = fr[tb[1] + 4:tb[3] - 4, tb[0] + 4:tb[2] - 4]
    tm = _timer_templates(scene)
    return min(tm, key=lambda k: np.abs(crop - tm[k]).mean())


# ---------------------------------------------------------------- stops

def read_shape(scene, fr, m) -> tuple:
    st = scene["style"]
    half = int(st["mh"] * 1.3 + 3)
    cx, cy = int(round(m["x"])), int(round(m["y"]))
    crop = fr[cy - half:cy + half + 1, cx - half:cx + half + 1]
    got = np.sqrt(((crop - np.array(st["ink"])) ** 2).sum(axis=2)) < 60
    best, bi = None, -1.0
    for shp in sim.SHAPES:
        img = Image.new("L", (2 * half + 1, 2 * half + 1), 0)
        render._fill_shape(ImageDraw.Draw(img), shp, m["x"] - cx + half, m["y"] - cy + half, st["mh"], 255)
        ref = np.asarray(img) > 127
        iou = (ref & got).sum() / max(1, (ref | got).sum())
        if iou > bi:
            best, bi = shp, iou
    return best, bi


# ---------------------------------------------------------------- routes

def samples(scene, r):
    """Sample points (x, y, arc length from the growth anchor) along the drawn path,
    skipping the stop markers."""
    R, cum = render.draw_path(scene, r)
    L = cum[-1]
    st = scene["style"]
    stops = [(m["x"], m["y"]) for m in render.stop_marks(scene)]
    out = []
    s = 0.0
    while s <= L:
        x, y = render.cut(R, cum, max(0.0, s - 0.01), max(s, 0.01))[-1]
        if all(math.hypot(x - a, y - b) > st["mh"] * 1.3 + 4 for a, b in stops):
            out.append((x, y, s if r["grow"] == "a" else L - s))
        s += 3.0
    return np.array(out), L


def pattern_on(scene, r, pts, L):
    """(which samples lie on ink of the line pattern once drawn, the drawn length each
    one needs: its own position, or for a dot the position of the dot's centre)."""
    sa = pts[:, 2] if r["grow"] == "a" else L - pts[:, 2]      # arc length from the a end
    st = scene["style"]
    if r["style"] == "dashed":
        on, off = st["dash"]
        return np.mod(sa, on + off) < on, pts[:, 2]
    if r["style"] == "dotted":
        period, rad = st["dot"]
        centre = np.floor(sa / period) * period + period / 2
        need = centre if r["grow"] == "a" else L - centre
        return np.abs(sa - centre) <= rad, need
    return np.ones(len(pts), bool), pts[:, 2]


_OFF = [(dx, dy) for dx in (-1, 0, 1) for dy in (-1, 0, 1)]


def inked(fr, pts, rgb):
    xs = np.clip(np.round(pts[:, 0]).astype(int), 1, sim.W - 2)
    ys = np.clip(np.round(pts[:, 1]).astype(int), 1, sim.H - 2)
    c = np.array(rgb, dtype=float)
    best = np.full(len(xs), 1e9)
    for dx, dy in _OFF:
        d = np.sqrt(((fr[ys + dy, xs + dx] - c) ** 2).sum(axis=1))
        best = np.minimum(best, d)
    return best < COL_TOL


def tol_px(scene, r):
    """(how far the measured end may run past the drawn length: round cap / dot,
    how far it may fall short: a dash gap / dot spacing)."""
    st = scene["style"]
    hi = st["w"] / 2 + 6
    if r["style"] == "dashed":
        return hi, st["dash"][1] + 6
    if r["style"] == "dotted":
        return st["dot"][1] + 6, st["dot"][0] + 6
    return hi, 6.0


def fit_events(est, L, lo_v, hi_v):
    """Start / finish (and erase start / gone) frames fitted to the measured growth;
    lo_v / hi_v: the visible part of the path (arc length from the anchor)."""
    n = len(est)
    span = hi_v - lo_v
    full = [i for i in range(n) if est[i] >= hi_v - 0.05 * span]
    if not full:
        return None
    i_full, j_full = full[0], full[-1]
    mid = lambda v: lo_v + 0.1 * span < v < hi_v - 0.1 * span  # noqa: E731
    out = []
    grow = [i for i in range(i_full) if mid(est[i])]
    if len(grow) < 2:
        return None
    a, b = np.polyfit(grow, [est[i] for i in grow], 1)
    out += [-b / a, (L - b) / a]
    shrink = [i for i in range(j_full + 1, n) if mid(est[i])]
    if len(shrink) >= 2:
        a, b = np.polyfit(shrink, [est[i] for i in shrink], 1)
        out += [(L - b) / a, -b / a]
    else:
        out += [None, None]
    return out


# ---------------------------------------------------------------- ring

def find_ring(scene, fr):
    tb = scene["timer"]["box"]
    m = np.sqrt(((fr - np.array(sim.TOKEN_RGB)) ** 2).sum(axis=2)) < 50
    m[:tb[3] + 10, :tb[2] + 10] = False
    ys, xs = np.nonzero(m)
    if len(xs) < 50:
        return None
    return float(xs.mean()), float(ys.mean())


# ---------------------------------------------------------------- per clip

def check_clip(scene, qs, frames) -> list:
    errs = []
    nf = scene["n_frames"]
    if len(frames) != nf:
        return [f"decoded {len(frames)} frames, expected {nf}"]
    # clock
    probe = set(range(0, nf, 12)) | {f + d for f in sim.event_frames(scene) for d in (-1, 0, 1)}
    for q in qs:
        if "t" in q["params"]:
            lo, hi = sim.frames_showing(q["params"]["t"])
            probe |= set(range(lo, hi + 1))
    for i in sorted(probe):
        if 0 <= i < nf and read_timer(scene, frames[i]) != render.timer_text(i):
            errs.append(f"frame {i}: clock misread")
    # stops
    for m in render.stop_marks(scene):
        shp, iou = read_shape(scene, frames[nf // 2], m)
        if shp != m["shape"] or iou < 0.6:
            errs.append(f"stop {m['shape']} read as {shp} (IoU {iou:.2f})")
    # ring
    ring = [None] * nf
    if scene["token"]:
        for i in range(nf):
            ring[i] = find_ring(scene, frames[i])
            want = render.token_xy(scene, i)
            if ring[i] is None or math.hypot(ring[i][0] - want[0], ring[i][1] - want[1]) > 2.0:
                errs.append(f"frame {i}: ring at {ring[i]}, expected {want}")
                break
    # routes
    tl = copy.deepcopy(answers.extract(scene))
    R_px = scene["style"]["token_r"] + 6
    for r in scene["routes"]:
        pts, L = samples(scene, r)
        tol_hi, tol_lo = tol_px(scene, r)
        est, bad = [], None
        t_first = float(pts[:, 2].min())
        on, need = pattern_on(scene, r, pts, L)
        for i in range(nf):
            free = np.ones(len(pts), bool)
            if ring[i] is not None:
                free = np.hypot(pts[:, 0] - ring[i][0], pts[:, 1] - ring[i][1]) > R_px
            ok = inked(frames[i], pts, r["rgb"]) & free
            e = float(pts[ok, 2].max()) if ok.any() else 0.0
            est.append(e)
            true = render.progress(r, i) * L
            vis = pts[(need <= true + 1e-6) & free & on, 2]
            want = float(vis.max()) if len(vis) else 0.0       # the farthest sample the line has reached
            if bad is None and true == 0 and ok.any():
                bad = f"{r['color']}: ink on frame {i} before it is drawn / after it is gone"
            short = e < want - tol_lo and want - t_first > 10   # the first px past the marker rim can blur away
            if bad is None and (e > true + tol_hi or short):
                bad = f"{r['color']}: frame {i} drawn to {e:.0f}px, expected {want:.0f}-{true:.0f}px"
        if bad:
            errs.append(bad)
        cover = inked(frames[r["draw"][1] + 1], pts, r["rgb"]).mean()
        need = {"solid": 0.9, "dashed": 0.5, "dotted": 0.3}[r["style"]]
        if cover < need:
            errs.append(f"{r['color']}: only {cover:.0%} of the {r['style']} line visible when complete")
        fe = fit_events(est, L, float(pts[:, 2].min()), float(pts[:, 2].max()))
        if fe is None:
            errs.append(f"{r['color']}: growth not measurable")
            continue
        want = [r["draw"][0], r["draw"][1]] + (list(r["erase"]) if r["erase"] else [None, None])
        got = [None if v is None else int(round(v)) for v in fe]
        for name, g, w in zip(("start", "finish", "erase", "gone"), got, want):
            if (g is None) != (w is None) or (g is not None and abs(g - w) > 1):
                errs.append(f"{r['color']}: measured {name} frame {g}, scene {w}")
        tr = tl["routes"][r["color"]]
        tr["s"], tr["e"], tr["g0"], tr["g1"] = got
    if scene["token"]:
        if all(p is not None for p in ring):
            tl["token"] = answers.token_rides(ring, tl["stops"], tl["routes"], 1.5)
        else:
            errs.append("ring lost on some frame")
    # every question, re-derived from the measured timeline
    for q in qs:
        k = q["derivation_key"]
        try:
            v = answers.derive(k, q["params"], tl)
        except Exception as ex:  # noqa: BLE001
            errs.append(f"{q['qid']} ({k}): derive failed on measured timeline: {ex!r}")
            continue
        if k in CLOCK_KEYS:
            if v is None or abs(float(v.split()[0]) - float(str(q["groundtruth"]).split()[0])) > 0.1 + 1e-9:
                errs.append(f"{q['qid']} ({k}): measured {v!r}, stored {q['groundtruth']!r}")
        elif str(v) != str(q["groundtruth"]):
            errs.append(f"{q['qid']} ({k}): measured {v!r}, stored {q['groundtruth']!r}")
    errs += [e for e in sim.check(scene) if e.startswith("layout")]
    return errs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset-dir", required=True)
    ap.add_argument("--report", default="frame_check.txt")
    args = ap.parse_args()
    root = Path(args.dataset_dir)
    lines, total, n = [], 0, 0
    for line in open(root / "annotations.jsonl", encoding="utf8"):
        s = json.loads(line)
        n += 1
        errs = check_clip(s["params"], s["questions"], decode(root / s["media_path"]))
        total += len(errs)
        lines += [f"{s['sample_id']}: {e}" for e in errs]
        print(f"{s['sample_id']}: {len(errs)} failures", flush=True)
    lines.append(f"RESULT: {n} clips, {total} frame-check failures")
    (root / args.report).write_text("\n".join(lines) + "\n")
    print(lines[-1])
    for e in lines[:-1][:20]:
        print("  -", e)
    sys.exit(1 if total else 0)


if __name__ == "__main__":
    main()
