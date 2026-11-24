"""Frame-level checks for rendered translucent_occlusion clips (decodes the MP4s).

    python frame_check.py --dataset-dir ../output_videos/translucent_occlusion_preview

Per clip:
- clock: every decoded frame's CLOCK_BOX is template-matched against render._clock drawings for 0.0-10.0 s
  and must read render.clock_tenths(f);
- visibility: on every 3rd frame (and on every hide/reveal event frame and the frame before it) each actor with
  sim.states == 0 (clear of the panels) has a 5x5 median at its centre within COLOR_TOL (max channel) of
  sim.COLORS[colour]; each actor with state == 2 (completely hidden) has a centre pixel that is NOT that close
  (the panel tint dominates);
- answers: hide_time / reveal_time / hidden_for are re-derived from the decoded clock on the event frames
  (sim.events) and must equal the stored groundtruth; render.clock_text(event frame) must equal it too;
- margins: every hide/reveal event is >= 12 frames from both clip ends.
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
STEP = 3
CHECKS = ("clock", "clear", "hidden", "answers", "margins")


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
    T, tt = templates()
    C = np.stack([_crop(f).astype(np.float32).ravel() for f in frames])
    d = (C * C).sum(1)[:, None] - 2 * C @ T.T + tt[None, :]
    return [int(k) for k in d.argmin(1)]


def centre(img, x, y):
    xi, yi = int(round(x)), int(round(y))
    return np.median(img[yi - 2:yi + 3, xi - 2:xi + 3].reshape(-1, 3), axis=0)


def check_clip(s, root, rep, errs):
    sc, sid = s["params"], s["sample_id"]
    frames = decode(root / s["media_path"])
    F = sc["frames"]
    if len(frames) != F:
        errs.append(f"{sid}: decoded {len(frames)} frames, expected {F}")
        rep["clock"][1] += F
        return
    tr, st, ev = sim.track(sc), sim.states(sc), sim.events(sc)

    # a. clock on every frame
    clocks = read_clocks(frames)
    bad = [f for f in range(F) if clocks[f] != render.clock_tenths(f)]
    rep["clock"][0] += F - len(bad)
    rep["clock"][1] += F
    if bad:
        errs.append(f"{sid}: clock misread on {len(bad)}/{F} frames, first f={bad[0]} read {clocks[bad[0]]} "
                    f"want {render.clock_tenths(bad[0])}")

    # b. centre colour vs cover state
    fs = sorted(set(range(0, F, STEP)) | {f + d for f, _, _ in ev for d in (-1, 0) if 0 <= f + d < F})
    for f in fs:
        for i, a in enumerate(tr[f]):
            if a is None or st[f][i] not in (0, 2):
                continue
            px = centre(frames[f], a[0], a[1])
            diff = int(np.abs(px - np.array(sim.COLORS[a[3]])).max())
            if st[f][i] == 0:
                rep["clear"][1] += 1
                if diff <= COLOR_TOL:
                    rep["clear"][0] += 1
                else:
                    errs.append(f"{sid}: f={f} actor {i} ({a[3]} {a[4]}) clear but centre {px.astype(int).tolist()} "
                                f"differs from colour by {diff}")
            else:
                rep["hidden"][1] += 1
                if diff > COLOR_TOL:
                    rep["hidden"][0] += 1
                else:
                    errs.append(f"{sid}: f={f} actor {i} ({a[3]} {a[4]}) hidden but centre {px.astype(int).tolist()} "
                                f"is within {diff} of its pure colour")

    # d. clock answers from the decoded event frames
    first = {}
    for f, i, k in ev:
        first.setdefault((i, k), f)
    for q in s["questions"]:
        k, p = q["derivation_key"], q["params"]
        if k not in ("hide_time", "reveal_time", "hidden_for"):
            continue
        rep["answers"][1] += 1
        try:
            fh, fr = first.get((p["x"], "hide")), first.get((p["x"], "reveal"))
            if k == "hide_time":
                got = f"{clocks[fh] / 10:.1f} s"
                want_text = render.clock_text(fh)
            elif k == "reveal_time":
                got = f"{clocks[fr] / 10:.1f} s"
                want_text = render.clock_text(fr)
            else:
                got = f"{(clocks[fr] - clocks[fh]) / 10:.1f} s"
                want_text = f"{(render.clock_tenths(fr) - render.clock_tenths(fh)) / 10:.1f} s"
        except (KeyError, TypeError, IndexError) as e:
            got, want_text = f"<undecodable: {e!r}>", None
        if got == str(q["groundtruth"]) and want_text == str(q["groundtruth"]):
            rep["answers"][0] += 1
        else:
            errs.append(f"{sid}/{q['qid']} {k} {p}: decoded {got!r}, render text {want_text!r}, "
                        f"stored {q['groundtruth']!r}")

    # e. event margins
    for f, i, k in ev:
        rep["margins"][1] += 1
        if sim.EDGE_F <= f <= F - 1 - sim.EDGE_F:
            rep["margins"][0] += 1
        else:
            errs.append(f"{sid}: {k} of actor {i} at f={f} within {sim.EDGE_F} frames of a clip end ({F} frames)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset-dir", default=str(HERE.parent / "output_videos" / "translucent_occlusion_preview"))
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
               f"clock answers ok {r['answers'][0]}/{r['answers'][1]}, "
               f"event margins ok {r['margins'][0]}/{r['margins'][1]}")
    lines = [summary, f"errors: {len(errs)}"] + [f"  - {e}" for e in errs]
    (root / args.report).write_text("\n".join(lines) + "\n", encoding="utf8")
    print("\n".join(lines[:2 + 40]))
    sys.exit(1 if errs else 0)


if __name__ == "__main__":
    main()
