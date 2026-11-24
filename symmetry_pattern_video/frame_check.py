"""Frame-level check of rendered clips against the scene data.

    python frame_check.py --dataset-dir <dir> [--report frame_check.txt] [--jobs 4]

For every clip it decodes the MP4 (OpenCV) and checks:
  * clock: template-matches the timer panel against all readings 0.0-9.9 s
    on every 6th frame, on every event frame and on every frame a question
    names by its clock reading;
  * shapes: on every 6th frame where nothing is half-drawn, fading or
    overlapping a like-coloured shape, the blobs of each colour (strokes
    joined by a small dilation) are counted and their centres matched to the
    track's shapes (within 0.45 of the shape size);
  * events: a wave / a popping shape is absent on the frame before it starts
    and present on its first frame; a gliding group / the disc is still from
    its settle / move-end frame on and was moving into it;
  * tracing direction: halfway through each wave every shape's decoded
    stroke pixels overlap the half-outline drawn the stored way round more
    than the half drawn the other way round;
  * turns: the notch is found where the track puts it;
  * clock answers: the clock read on the decoded event frame equals the answer;
  * layout: the timer panel is outside the pattern / disc circle.
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
from common.video5 import font  # noqa: E402


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


def colour_mask(fr, rgb):
    return np.abs(fr - np.array(rgb)).sum(axis=2) < 70


def classified(fr, scene, rgb):
    """Pixels whose nearest scene colour (shape colours, background, disc,
    rim, notch, timer) is `rgb`, within 60."""
    st = scene["style"]
    pal = [tuple(rgb)] + [sim.COLORS[c] for c in sorted({s["color"] for s in scene["shapes"]}) if sim.COLORS[c] != tuple(rgb)]
    pal += [tuple(c) for c in st["bg"]] + [tuple(st["disc"]), tuple(st["rim"]), tuple(st["notch"])]
    cand = np.array(pal, dtype=np.int16)
    d = np.abs(fr[:, :, None, :] - cand[None, None]).sum(axis=3)
    return (d.argmin(axis=2) == 0) & (d.min(axis=2) < 60)


def blobs(fr, rgb, min_px=30, scene=None):
    import cv2
    raw = (classified(fr, scene, rgb) if scene is not None else colour_mask(fr, rgb)).astype(np.uint8)
    m = cv2.dilate(raw, np.ones((11, 11), np.uint8))
    k, lab, stats, cent = cv2.connectedComponentsWithStats(m, connectivity=8)
    ink = np.bincount(lab[raw > 0], minlength=k)          # undilated pixels per component
    comps = [[stats[j, 0], stats[j, 1], stats[j, 0] + stats[j, 2], stats[j, 1] + stats[j, 3], ink[j]]
             for j in range(1, k) if ink[j] >= min_px]
    # one shape can be several strokes (a double outline, a centre dot, dashes): merge
    # components whose boxes overlap
    merged = True
    while merged:
        merged = False
        for a in range(len(comps)):
            for b in range(a + 1, len(comps)):
                A, B = comps[a], comps[b]
                if A[0] <= B[2] and B[0] <= A[2] and A[1] <= B[3] and B[1] <= A[3]:
                    comps[a] = [min(A[0], B[0]), min(A[1], B[1]), max(A[2], B[2]), max(A[3], B[3]), A[4] + B[4]]
                    comps.pop(b)
                    merged = True
                    break
            if merged:
                break
    return [((c[0] + c[2]) / 2, (c[1] + c[3]) / 2) for c in comps]


def _cent(pts):
    return (sum(p[0] for p in pts) / len(pts), sum(p[1] for p in pts) / len(pts))


def _box(pts, pad=6):
    xs, ys = [p[0] for p in pts], [p[1] for p in pts]
    return (max(0, int(min(xs)) - pad), max(0, int(min(ys)) - pad), min(sim.W, int(max(xs)) + pad + 1),
            min(sim.H, int(max(ys)) + pad + 1))


def _ink(fr, it):
    x0, y0, x1, y1 = _box(it["pts"])
    return int(colour_mask(fr[y0:y1, x0:x1], sim.COLORS[it["color"]]).sum())


def _path_mask(pts, frac, width):
    img = Image.new("L", (sim.W, sim.H), 0)
    line = render._path(pts, frac)
    if len(line) >= 2:
        ImageDraw.Draw(img).line(line, fill=255, width=width)
    return np.asarray(img) > 0


def check_clip(scene, qs, frames) -> list:
    errs = []
    nf = scene["n_frames"]
    if len(frames) != nf:
        return [f"decoded {len(frames)} frames, expected {nf}"]
    tr = render.track(scene)
    sc = scene["scenario"]
    cx, cy = scene["center"]
    reach = scene["disc_r"] + 8 if sc == "turns" else scene["rp"]
    tb = sim.TIMER_BOX
    nx, ny = min(max(cx, tb[0]), tb[2]), min(max(cy, tb[1]), tb[3])
    if math.hypot(cx - nx, cy - ny) <= reach:
        errs.append("timer panel inside the pattern circle")
    ev = sim.event_frames(scene)
    # clock
    want = set(range(0, nf, 6)) | set(ev)
    for q in qs:
        if "t" in q["params"]:
            lo, hi = sim.frames_showing(q["params"]["t"])
            want |= set(range(lo, hi + 1))
    for f in sorted(want):
        if read_timer(frames[f]) != sim.tenths(f):
            errs.append(f"frame {f}: clock misread")
    # blobs per colour on calm frames
    for f in range(0, nf, 6):
        items = [it for it in tr[f]["shapes"] if it is not None]
        if any(it["tip"] is not None or 0 < it["fill_a"] < 1 for it in items):
            continue
        if sc == "reveal" and any(0 <= f - a < 3 for a in scene["timeline"]["appear"]):
            continue
        by = {}
        for it in items:
            by.setdefault(it["color"], []).append(it)
        for col, its in by.items():
            cs = [((_box(it["pts"], 0)[0] + _box(it["pts"], 0)[2]) / 2, (_box(it["pts"], 0)[1] + _box(it["pts"], 0)[3]) / 2)
                  for it in its]
            sizes = [max(math.dist(c, p) for p in it["pts"]) for c, it in zip(cs, its)]
            if any(math.dist(cs[a], cs[b]) < sizes[a] + sizes[b] + 14 for a in range(len(cs))
                   for b in range(a + 1, len(cs))):
                continue                                   # touching / overlapping: not separable
            others = [_box(it["pts"]) for it in items if it["color"] != col]
            if any(o[0] < b[2] and b[0] < o[2] and o[1] < b[3] and b[1] < o[3]
                   for o in others for b in (_box(it["pts"]) for it in its)):
                continue                                   # another shape drawn over one of these
            got = blobs(frames[f], sim.COLORS[col], scene=scene)
            # ignore stray blobs far from every shape (anti-aliased edges of other colours)
            got = [g for g in got if min(math.dist(g, c) for c in cs) < 2 * max(sizes)]
            if len(got) != len(cs):
                errs.append(f"frame {f}: {len(got)} {col} blobs, track has {len(cs)}")
                continue
            for c, r in zip(cs, sizes):
                if min(math.dist(c, g) for g in got) > 0.45 * r + 3:
                    errs.append(f"frame {f}: no {col} blob at {c}")
    # events
    if sc == "waves":
        for o, (f0, f1) in enumerate(sim.wave_frames(scene)):
            idx = [i for i, s in enumerate(scene["shapes"]) if s["orbit"] == o]
            ref = tr[f1 + 4]["shapes"]
            before = sum(_ink(frames[f0 - 1], ref[i]) for i in idx)
            at = sum(_ink(frames[f0], ref[i]) for i in idx)
            if before > 20 or at < 10:
                errs.append(f"wave {o + 1}: ink {before} -> {at} at its start frame {f0}")
            # tracing direction, halfway through the wave
            fm = (f0 + f1) // 2
            for i in idx:
                it = tr[fm]["shapes"][i]
                got = colour_mask(frames[fm], sim.COLORS[it["color"]])
                fwd = _path_mask(it["pts"], it["frac"], 8)
                rev_pts = [it["pts"][0]] + it["pts"][1:][::-1]
                rev = _path_mask(rev_pts, it["frac"], 8)
                if (got & fwd).sum() <= (got & rev).sum():
                    errs.append(f"frame {fm}: shape {i} not traced the stored way round")
    if sc == "reveal":
        for i, a in enumerate(scene["timeline"]["appear"]):
            it = tr[a + 3]["shapes"][i]
            if _ink(frames[a - 1], it) > 20 or _ink(frames[a], it) < 10:
                errs.append(f"shape {i} not appearing on frame {a}")
    if sc == "assemble":
        for e in sim.settle_frames(scene):
            idx = [i for i, s in enumerate(scene["shapes"]) if sim.settle_frames(scene)[s["orbit"]] == e]
            for i in idx:
                x0, y0, x1, y1 = _box(tr[e]["shapes"][i]["pts"])
                still = np.abs(frames[e + 2][y0:y1, x0:x1] - frames[e][y0:y1, x0:x1]).mean()
                moved = np.abs(frames[e][y0:y1, x0:x1] - frames[e - 3][y0:y1, x0:x1]).mean()
                if still > 1.0 and not _crossed(tr, e, e + 2, (x0, y0, x1, y1), i):
                    errs.append(f"shape {i} still moving after its settle frame {e}")
                if moved < 0.5:
                    errs.append(f"shape {i} not moving into its settle frame {e}")
    if sc == "turns":
        r = int(scene["disc_r"]) + 10
        box = (int(cx) - r, int(cy) - r, int(cx) + r, int(cy) + r)
        for m in scene["timeline"]["moves"]:
            for f, moving in ((m["f0"], True), (m["f1"], False)):
                a = frames[f - 1 if moving else f][box[1]:box[3], box[0]:box[2]]
                b = frames[f + 2][box[1]:box[3], box[0]:box[2]]
                d = np.abs(a - b).mean()
                if (moving and d < 0.15) or (not moving and d > 0.4):
                    errs.append(f"disc {'not moving' if moving else 'still moving'} at frame {f} ({d:.2f})")
        for f in range(0, nf, 6):
            M = sim.pattern_matrix(scene, f)
            if abs(M[0][0] * M[1][1] - M[0][1] * M[1][0]) < 0.5:
                continue                                    # disc edge-on in the middle of a flip
            x, y = tr[f]["notch"]
            vx, vy = x - cx, y - cy
            ln = math.hypot(vx, vy) or 1
            px, py = x - vx / ln * 9, y - vy / ln * 9          # inside the notch triangle
            got = colour_mask(frames[f][int(py) - 3:int(py) + 4, int(px) - 3:int(px) + 4], scene["style"]["notch"])
            if got.mean() < 0.5:
                errs.append(f"frame {f}: notch not where the track puts it")
    # clock answers on decoded event frames
    for q in qs:
        k, p = q["derivation_key"], q["params"]
        f = None
        if k == "wave_time":
            f = min(sim.wave_frames(scene)[s["orbit"]][0] for s in scene["shapes"]
                    if [s["color"], s["type"]] == p["g"])
        elif k == "done_time":
            f = {"waves": lambda: sim.wave_frames(scene)[-1][1], "reveal": lambda: max(scene["timeline"]["appear"]),
                 "assemble": lambda: max(sim.settle_frames(scene)),
                 "turns": lambda: scene["timeline"]["moves"][-1]["f1"]}[sc]()
        if f is not None and sim.clock_text(read_timer(frames[f])) != q["groundtruth"]:
            errs.append(f"{q['qid']}: clock on the event frame {f} != {q['groundtruth']}")
    return errs


def _crossed(tr, f0, f1, box, i):
    """Another shape passes through the box between the two frames."""
    x0, y0, x1, y1 = box
    for f in (f0, f1):
        for j, it in enumerate(tr[f]["shapes"]):
            if j == i or it is None:
                continue
            bx0, by0, bx1, by1 = _box(it["pts"])
            if bx0 < x1 and x0 < bx1 and by0 < y1 and y0 < by1:
                return True
    return False


def _one(args):
    root, line = args
    s = json.loads(line)
    errs = check_clip(s["params"], s["questions"], frames_of(Path(root) / s["media_path"]))
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
