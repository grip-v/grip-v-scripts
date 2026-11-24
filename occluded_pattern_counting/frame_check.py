"""Frame-level checks for rendered occluded_pattern_counting clips (decodes the MP4s).

    python frame_check.py --dataset-dir ../output_videos/occluded_pattern_counting_preview

Per clip:
- clock: sampled frames' CLOCK_BOX is template-matched against render._clock drawings for 0.0-10.0 s and must read
  render.clock_tenths(f);
- visibility: on sampled frames inside every hold, each object with sim.states == 0 has a 5x5 median at its centre
  within COLOR_TOL (max channel) of sim.COLORS[colour]; each object with state == 2 (hidden under an opaque occluder)
  has a centre that is NOT that close;
- blobs: at the middle frame of every hold, the number of connected blobs of object-coloured pixels (clock box
  excluded) equals the number of visible objects (state 0) -- the counting check;
- answers: first_move_time / last_stop_time / first_hold_len are re-derived from the decoded clock on the relevant
  frames (from scene["holds"]) and must equal the stored groundtruth; render.clock_text(frame) must equal it too;
- margins: every slide start/end is >= 12 frames from both clip ends.
"""
from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE), str(HERE.parent)]
import sim  # noqa: E402
import render  # noqa: E402

COLOR_TOL = 40      # max-channel tolerance for "centre shows the pure colour" (h264 chroma blur)
STEP = 3            # sampled frame step inside holds and for the clock
MIN_BLOB = 40       # px; smaller blobs are anti-aliasing / codec noise
CHECKS = ("clock", "clear", "hidden", "blobs", "answers", "margins")


def decode(path):
    cap = cv2.VideoCapture(str(path))
    out = []
    while True:
        ok, fr = cap.read()
        if not ok:
            break
        out.append(cv2.cvtColor(fr, cv2.COLOR_BGR2RGB))
    cap.release()
    return out


def _crop(img):
    x0, y0, x1, y1 = render.CLOCK_BOX
    return img[y0 + 4:y1 - 3, x0 + 4:x1 - 3]


_T = None


def templates():
    global _T
    if _T is None:
        rows = []
        for k in range(101):
            img = Image.new("RGB", (sim.W, sim.H), (0, 0, 0))
            render._clock(ImageDraw.Draw(img), (k * 24 + 9) // 10)
            rows.append(_crop(np.asarray(img)).astype(np.float32).ravel())
        T = np.stack(rows)
        _T = (T, (T * T).sum(1))
    return _T


def read_clocks(frames):
    """Decoded tenths for each given image."""
    T, tt = templates()
    C = np.stack([_crop(f).astype(np.float32).ravel() for f in frames])
    d = (C * C).sum(1)[:, None] - 2 * C @ T.T + tt[None, :]
    return [int(k) for k in d.argmin(1)]


def centre(img, x, y):
    xi, yi = int(round(x)), int(round(y))
    return np.median(img[yi - 2:yi + 3, xi - 2:xi + 3].reshape(-1, 3), axis=0)


def count_blobs(img, colors):
    mask = np.zeros(img.shape[:2], bool)
    im = img.astype(np.int16)
    for c in colors:
        mask |= np.abs(im - np.array(c, np.int16)).max(axis=2) <= COLOR_TOL
    x0, y0, x1, y1 = render.CLOCK_BOX
    mask[max(0, y0 - 2):y1 + 2, max(0, x0 - 2):x1 + 2] = False
    n, _, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), connectivity=8)
    return sum(1 for k in range(1, n) if stats[k, cv2.CC_STAT_AREA] >= MIN_BLOB)


def check_clip(s, root, rep, errs):
    sc, sid = s["params"], s["sample_id"]
    frames = decode(root / s["media_path"])
    F = sc["frames"]
    if len(frames) != F:
        errs.append(f"{sid}: decoded {len(frames)} frames, expected {F}")
        rep["clock"][1] += F
        return
    st, hs = sim.states(sc), sim.holds(sc)
    objs = sc["objects"]

    # a. clock on sampled frames (every STEP-th, plus the first/last frame of each hold and the frames around slides)
    fs_clock = sorted({f for f in range(0, F, STEP)} | {g for a, b in hs for g in (a, b, b + 1, a - 1) if 0 <= g < F}
                      | {F - 1})
    clocks = dict(zip(fs_clock, read_clocks([frames[f] for f in fs_clock])))
    bad = [f for f in fs_clock if clocks[f] != render.clock_tenths(f)]
    rep["clock"][0] += len(fs_clock) - len(bad)
    rep["clock"][1] += len(fs_clock)
    if bad:
        errs.append(f"{sid}: clock misread on {len(bad)}/{len(fs_clock)} frames, first f={bad[0]} read "
                    f"{clocks[bad[0]]} want {render.clock_tenths(bad[0])}")

    # b. centre colour vs cover state inside holds
    fs = sorted({f for a, b in hs for f in list(range(a, b + 1, STEP)) + [(a + b) // 2]})
    for f in fs:
        for i, o in enumerate(objs):
            if st[f][i] not in (0, 2):
                continue
            px = centre(frames[f], o["x"], o["y"])
            diff = int(np.abs(px - np.array(sim.COLORS[o["color"]])).max())
            if st[f][i] == 0:
                rep["clear"][1] += 1
                if diff <= COLOR_TOL:
                    rep["clear"][0] += 1
                else:
                    errs.append(f"{sid}: f={f} object {i} ({o['color']}) clear but centre {px.astype(int).tolist()} "
                                f"differs from colour by {diff}")
            else:
                rep["hidden"][1] += 1
                if diff > COLOR_TOL:
                    rep["hidden"][0] += 1
                else:
                    errs.append(f"{sid}: f={f} object {i} ({o['color']}) hidden but centre {px.astype(int).tolist()} "
                                f"is within {diff} of its colour")

    # c. blob count at the middle of every hold
    cols = [sim.COLORS[c] for c in sorted({o["color"] for o in objs})]
    for k, (a, b) in enumerate(hs):
        m = (a + b) // 2
        want = sum(1 for v in st[m] if v == 0)
        got = count_blobs(frames[m], cols)
        rep["blobs"][1] += 1
        if got == want:
            rep["blobs"][0] += 1
        else:
            errs.append(f"{sid}: hold {k} (f={m}) blob count {got}, visible objects {want}")

    # d. clock answers from the decoded frames
    f_move, f_stop = hs[0][1] + 1, hs[-1][0]
    for q in s["questions"]:
        k = q["derivation_key"]
        if k not in ("first_move_time", "last_stop_time", "first_hold_len"):
            continue
        rep["answers"][1] += 1
        try:
            if k == "first_move_time":
                got, want_text = f"{clocks[f_move] / 10:.1f} s", render.clock_text(f_move)
            elif k == "last_stop_time":
                got, want_text = f"{clocks[f_stop] / 10:.1f} s", render.clock_text(f_stop)
            else:
                got = f"{(clocks[f_move] - clocks[0]) / 10:.1f} s"
                want_text = f"{(render.clock_tenths(f_move) - render.clock_tenths(0)) / 10:.1f} s"
        except (KeyError, TypeError, IndexError) as e:
            got, want_text = f"<undecodable: {e!r}>", None
        if got == str(q["groundtruth"]) and want_text == str(q["groundtruth"]):
            rep["answers"][0] += 1
        else:
            errs.append(f"{sid}/{q['qid']} {k}: decoded {got!r}, render text {want_text!r}, "
                        f"stored {q['groundtruth']!r}")

    # e. slide margins
    for (a, b), (c, d) in zip(hs, hs[1:]):
        for name, f in (("slide start", b), ("slide end", c)):
            rep["margins"][1] += 1
            if sim.EDGE_F <= f <= F - 1 - sim.EDGE_F:
                rep["margins"][0] += 1
            else:
                errs.append(f"{sid}: {name} at f={f} within {sim.EDGE_F} frames of a clip end ({F} frames)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset-dir", default=str(HERE.parent / "output_videos" / "occluded_pattern_counting_preview"))
    ap.add_argument("--report", default="frame_check_report.txt")
    args = ap.parse_args()
    root = Path(args.dataset_dir).resolve()
    rep = {c: [0, 0] for c in CHECKS}
    errs, n = [], 0
    for line in open(root / "annotations.jsonl", encoding="utf8"):
        n += 1
        check_clip(json.loads(line), root, rep, errs)
    r = rep
    summary = (f"frame_check: {n} clips, clock frames ok {r['clock'][0]}/{r['clock'][1]}, "
               f"clear centres ok {r['clear'][0]}/{r['clear'][1]}, hidden centres ok {r['hidden'][0]}/{r['hidden'][1]}, "
               f"hold blob counts ok {r['blobs'][0]}/{r['blobs'][1]}, "
               f"clock answers ok {r['answers'][0]}/{r['answers'][1]}, "
               f"slide margins ok {r['margins'][0]}/{r['margins'][1]}")
    lines = [summary, f"errors: {len(errs)}"] + [f"  - {e}" for e in errs]
    (root / args.report).write_text("\n".join(lines) + "\n", encoding="utf8")
    print("\n".join(lines[:2 + 40]))
    sys.exit(1 if errs else 0)


if __name__ == "__main__":
    main()
