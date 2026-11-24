"""Frame-level check of rendered clips against the scene data.

    python frame_check.py --dataset-dir <dir> [--report frame_check.txt]

For every clip it decodes the MP4 (OpenCV) and, at the event frames:
  * at the middle of every rest (the start, between rounds, the end) finds
    each piece as a blob of its own colour, turns the blob centre into a
    square through the board geometry and compares it with the replayed
    position; a captured piece must have no blob at all;
  * at every frame a question names by its clock reading (and at every
    landing a clock answer refers to) template-matches the clock panel
    against all readings 0.0-9.9 s and requires the best match to be the
    expected text (and, for square/moving questions, re-reads the piece
    squares / counts the pieces away from their squares);
  * template-matches every coordinate label (A-H, 1-8) at its stored
    position against all 16 glyphs; best match must be its own text;
  * checks that each hole centre shows the hole colour;
  * checks label boxes never overlap each other, the board, the timer or
    any box a piece sweeps (sim.check geometry, repeated here).
Prints a summary and exits 1 on any failure.
"""
from __future__ import annotations
import argparse
import json
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

GLYPHS = list(sim.FILES) + [str(k) for k in range(1, 9)]


def _frames(path: Path) -> list:
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
            out[k] = np.asarray(img, dtype=np.int16)[tb[1] + 4:tb[3] - 4, tb[0] + 4:tb[2] - 4]
        _TIMER[key] = out
    return _TIMER[key]


def read_timer(scene, fr) -> int:
    tb = scene["timer"]["box"]
    crop = fr[tb[1] + 4:tb[3] - 4, tb[0] + 4:tb[2] - 4]
    tm = _timer_templates(scene)
    return min(tm, key=lambda k: np.abs(crop - tm[k]).mean())


def read_label(scene, fr, lb) -> str:
    st = scene["style"]
    x, y = int(round(lb["x"])), int(round(lb["y"]))
    box = (x - 20, y - 20, x + 20, y + 20)
    crop = fr[box[1]:box[3], box[0]:box[2]]
    best, bd = None, 1e9
    for g in GLYPHS:
        img = Image.new("RGB", (40, 40), tuple(st["bg"]))
        ImageDraw.Draw(img).text((20 + lb["x"] - x, 20 + lb["y"] - y), g, fill=tuple(st["ink"]),
                                 font=font(lb["size"]), anchor="mm")
        ref = np.asarray(img, dtype=np.int16)
        # compare as ink masks so faint background patterns do not matter
        inkd = np.abs(ref - np.array(st["bg"])).sum(axis=2) > 120
        got = np.abs(crop - np.array(st["bg"])).sum(axis=2) > 120
        d = (inkd ^ got).mean()
        if d < bd:
            best, bd = g, d
    return best


def find_pieces(scene, fr, centres=False) -> dict:
    """Square under each piece's colour blob (None when no blob); with
    centres=True the blob centre in cells instead."""
    st = scene["style"]
    N, c, x0, y0 = scene["N"], st["cell"], st["x0"], st["y0"]
    reg = fr[y0:y0 + N * c, x0:x0 + N * c]
    out = {}
    min_px = (0.04 if st["piece_style"] == "figure" else 0.12) * (st["piece_scale"] * c) ** 2
    for i, p in enumerate(scene["pieces"]):
        m = np.abs(reg - np.array(p["rgb"])).sum(axis=2) < 70
        ys, xs = np.nonzero(m)
        if len(xs) < min_px:
            out[i] = None
            continue
        if centres:
            out[i] = (float(np.mean(xs)) / c, float(np.mean(ys)) / c)
            continue
        f = int(np.median(xs) // c)
        r = N - 1 - int(np.median(ys) // c)
        out[i] = (f, r)
    return out


def check_clip(scene, qs, frames) -> list:
    errs = []
    R, nf = scene["rounds"], scene["n_frames"]
    if len(frames) != nf:
        errs.append(f"decoded {len(frames)} frames, expected {nf}")
        return errs
    P = sim.positions(scene)
    # rest frames: middle of the intro, of every pause, of the hold
    rest_frames = [(0, sim.round_frames(scene, 1)[0] // 2)]
    for r in range(1, R + 1):
        land = sim.round_frames(scene, r)[1]
        nxt = sim.round_frames(scene, r + 1)[0] if r < R else nf - 1
        rest_frames.append((r, (land + nxt) // 2))
    for r, i in rest_frames:
        got = find_pieces(scene, frames[i])
        want = {k: P[r][k] for k in range(scene["n"])}
        if got != want:
            errs.append(f"frame {i}: pieces {got} != {want}")
        if read_timer(scene, frames[i]) != sim.timer_tenths(i):
            errs.append(f"frame {i}: clock misread")
    # moments named by questions / clock answers
    for q in qs:
        k, p = q["derivation_key"], q["params"]
        if k in ("square_at_time", "n_moving_at"):
            lo, hi = sim.frames_showing(p["t"])
            for i in range(lo, hi + 1):
                if read_timer(scene, frames[i]) != p["t"]:
                    errs.append(f"{q['qid']}: frame {i} clock != {p['t']}")
                if k == "square_at_time":
                    sqr = find_pieces(scene, frames[i])[p["x"]]
                    if sqr is None or sim.name(sqr) != q["groundtruth"]:
                        errs.append(f"{q['qid']}: frame {i} shows piece on {sqr}, answer {q['groundtruth']}")
                else:
                    # a piece is moving if its blob has left where it sat when the glide began
                    s0 = max(r0 for r0 in (sim.round_frames(scene, r)[0] for r in range(1, R + 1)) if r0 < i)
                    cs, c0 = find_pieces(scene, frames[i], centres=True), find_pieces(scene, frames[s0], centres=True)
                    moving = sum(1 for k2, v in cs.items() if v is not None and c0[k2] is not None and
                                 max(abs(v[0] - c0[k2][0]), abs(v[1] - c0[k2][1])) > 0.12)
                    if moving != q["groundtruth"]:
                        errs.append(f"{q['qid']}: {moving} pieces off-square at frame {i}")
        if k in ("land_time", "illegal_time"):
            tenths = int(round(float(q["groundtruth"].split()[0]) * 10))
            lo, hi = sim.frames_showing(tenths)
            land = next((i for i in range(lo, hi + 1) if _is_landing(scene, i)), None)
            if land is None or read_timer(scene, frames[land]) != tenths:
                errs.append(f"{q['qid']}: no landing on a frame reading {q['groundtruth']}")
    # labels: glyph read-back and layout
    mid = frames[nf // 2]
    for lb in scene["labels"]:
        if read_label(scene, mid, lb) != lb["text"]:
            errs.append(f"label {lb['text']} misread")
    for h in scene["holes"]:
        x, y = render.square_centre(scene, *sim.sq(h))
        px = mid[int(y) - 2:int(y) + 3, int(x) - 2:int(x) + 3].reshape(-1, 3).mean(axis=0)
        if min(np.abs(px - np.array(render.HOLE)).sum(), np.abs(px - np.array(render.HOLE_HATCH)).sum()) > 60:
            errs.append(f"hole {h} not drawn")
    errs += [e for e in sim.check(scene) if "label" in e or "timer" in e]
    return errs


def _is_landing(scene, i) -> bool:
    return any(sim.round_frames(scene, r)[1] == i for r in range(1, scene["rounds"] + 1))


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
        errs = check_clip(s["params"], s["questions"], _frames(root / s["media_path"]))
        total += len(errs)
        lines += [f"{s['sample_id']}: {e}" for e in errs]
    lines.append(f"RESULT: {n} clips, {total} frame-check failures")
    (root / args.report).write_text("\n".join(lines) + "\n")
    print(lines[-1])
    for e in lines[:-1][:20]:
        print("  -", e)
    sys.exit(1 if total else 0)


if __name__ == "__main__":
    main()
