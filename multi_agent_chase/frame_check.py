"""Frame-level checks for rendered multi_agent_chase clips (decodes the MP4s).

    python frame_check.py --dataset-dir ../output_videos/multi_agent_chase_preview

Per clip:
- clock: every decoded frame's CLOCK_BOX is template-matched against render._clock drawings for 0.0-10.0 s
  and must read render.clock_tenths(f);
- visibility: around every tracked ball centre (0.7 r) at least half the pixels are the ball colour or its
  pale variant on every frame the ball is alive (frames where another ball overlaps that disc are skipped);
  after a catch fewer than 15 pixels of the ball's full colour remain in the arena. The frame where the colour
  disappears is the decoded catch frame and must equal the frame where the track's alive flag flips;
- size/position: for isolated alive balls (every 4th frame) the blob of ball-coloured pixels around the tracked
  centre has its centroid within 3 px of the track and equivalent radius within 2.5 px of the tracked r;
- answers: catch_time / first_catch_time / left_at / caught_between are re-derived from the decoded clock and
  decoded catch frames and must equal the stored groundtruth;
- text: outside the arena (+12 px) and the clock box, frame 0 and the last frame equal the plain backdrop.
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
sys.path[:0] = [str(HERE), str(HERE.parent)]
import sim  # noqa: E402
import render  # noqa: E402

TOL = 28           # per-channel tolerance for colour matching (h264 chroma blur)
TIGHT = 14         # per-channel tolerance for "full colour is gone"
VIS_MIN = 0.5
GONE_MAX = 15
CENTRE_TOL = 3.0
SIZE_TOL = 2.5
BG_TOL = 10
CHECKS = ("clock", "visibility", "visibility_drawn", "catch", "size", "answers", "margins")


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
            img = Image.new("RGB", render.SIZE, (0, 0, 0))
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


def near(img, rgb, tol=TOL):
    lo = np.clip(np.array(rgb) - tol, 0, 255).astype(np.uint8)
    hi = np.clip(np.array(rgb) + tol, 0, 255).astype(np.uint8)
    return cv2.inRange(img, lo, hi) > 0


def near_blend(img, a, b, tol=TOL):
    """Pixels within tol (max channel) of the straight RGB segment a-b: the colour, its pale variant and the
    codec-blurred mixtures of the two along thin pattern lines (grid, stripes, dots)."""
    a, b = np.array(a, np.float32), np.array(b, np.float32)
    px = img.astype(np.float32)
    t = np.clip(((px - a) @ (b - a)) / float(((b - a) ** 2).sum()), 0, 1)
    proj = a + t[..., None] * (b - a)
    return np.abs(px - proj).max(-1) <= tol


def palette(name):
    col = sim.COLORS[name]
    return col, render._mix(col, (255, 255, 255), 0.55), render._mix(col, (0, 0, 0), 0.45)


def check_clip(s, root, rep, errs):
    sc = s["params"]
    sid = s["sample_id"]
    frames = decode(root / s["media_path"])
    F = sc["frames"]
    if len(frames) != F:
        errs.append(f"{sid}: decoded {len(frames)} frames, expected {F}")
        rep["clock"][1] += F
        return
    tr = sim.track(sc)
    n = len(sc["balls"])
    x0, y0, x1, y1 = sc["arena"]
    H, W = render.SIZE[1], render.SIZE[0]
    yy, xx = np.mgrid[0:H, 0:W]

    # ---- a. clock
    clocks = read_clocks(frames)
    bad = [f for f in range(F) if clocks[f] != render.clock_tenths(f)]
    rep["clock"][0] += F - len(bad)
    rep["clock"][1] += F
    if bad:
        errs.append(f"{sid}: clock misread on {len(bad)}/{F} frames, first f={bad[0]} read {clocks[bad[0]]} "
                    f"want {render.clock_tenths(bad[0])}")

    # ---- b. visibility + decoded catch frames
    # state[f, i]: True drawn, False absent, None unknown (the ball's disc is mostly hidden behind another ball)
    state = [[None] * n for _ in range(F)]
    last = [tuple(tr[0][i][:3]) for i in range(n)]
    for f in range(F):
        img = frames[f]
        arena = img[y0:y1, x0:x1]
        alive_discs = np.zeros((y1 - y0, x1 - x0), dtype=np.uint8)
        for j in range(n):
            if tr[f][j][3]:
                cv2.circle(alive_discs, (int(round(tr[f][j][0] - x0)), int(round(tr[f][j][1] - y0))),
                           int(math.ceil(tr[f][j][2])) + 3, 1, -1)
        for i in range(n):
            col, pale, _ = palette(sc["balls"][i]["color"])
            x, y, r, alive = tr[f][i]
            if alive:
                last[i] = (x, y, r)
            else:
                # spec: almost none of the ball's full colour left in the arena (pixels on other balls' edges,
                # where antialiasing blends their colours, are not counted)
                cnt = int((near(arena, col, TIGHT) & (alive_discs == 0)).sum())
                rep["visibility"][1] += 1
                if cnt < GONE_MAX:
                    rep["visibility"][0] += 1
                else:
                    errs.append(f"{sid}: f={f} ball {i} gone in track but {cnt} px of its colour remain")
            lx, ly, lr = last[i]
            ys, xs = slice(max(0, int(ly - lr)), int(ly + lr) + 2), slice(max(0, int(lx - lr)), int(lx + lr) + 2)
            disc = (xx[ys, xs] - lx) ** 2 + (yy[ys, xs] - ly) ** 2 <= (0.7 * lr) ** 2
            for j in range(n):
                if j != i and tr[f][j][3]:
                    disc &= (xx[ys, xs] - tr[f][j][0]) ** 2 + (yy[ys, xs] - tr[f][j][1]) ** 2 > (tr[f][j][2] + 1) ** 2
            full = (xx[ys, xs] - lx) ** 2 + (yy[ys, xs] - ly) ** 2 <= (0.7 * lr) ** 2
            if disc.sum() < 0.3 * full.sum():
                continue
            sub = img[ys, xs]
            frac = float((near_blend(sub, col, pale) & disc).sum()) / disc.sum()
            state[f][i] = frac >= VIS_MIN
            if alive:
                rep["visibility_drawn"][1] += 1
                if frac >= VIS_MIN:
                    rep["visibility_drawn"][0] += 1
                else:
                    errs.append(f"{sid}: f={f} ball {i} ({sc['balls'][i]['color']}) fraction {frac:.2f} < {VIS_MIN}")
    dec_catch = {}
    for i in range(n):
        flip = next((f for f in range(F) if not tr[f][i][3]), None)
        known = [(f, state[f][i]) for f in range(F) if state[f][i] is not None]
        gone = next((f for f, st in known if not st), None)
        if flip is None and gone is None:
            continue
        rep["catch"][1] += 1
        after_ok = gone is not None and all(not st for f, st in known if f >= gone)
        if flip is not None and gone == flip and after_ok:
            rep["catch"][0] += 1
        else:
            errs.append(f"{sid}: ball {i} decoded gone at {gone}, track flips at {flip}")
        if gone is not None:
            dec_catch[i] = gone
    present = np.array([[not (i in dec_catch and dec_catch[i] <= f) for i in range(n)] for f in range(F)])

    # ---- c. size / position
    for f in range(0, F, 4):
        img = frames[f]
        for i in range(n):
            x, y, r, alive = tr[f][i]
            if not alive or any(tr[f][j][3] and j != i and math.dist((x, y), tr[f][j][:2]) < r + tr[f][j][2] + 10
                                for j in range(n)):
                continue
            col, pale, outl = palette(sc["balls"][i]["color"])
            h = int(math.ceil(1.5 * r))
            ya, yb = max(y0, int(y) - h), min(y1, int(y) + h + 1)
            xa, xb = max(x0, int(x) - h), min(x1, int(x) + h + 1)
            sub = img[ya:yb, xa:xb]
            m = (near(sub, col) | near(sub, pale) | near(sub, outl)).astype(np.uint8)
            m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
            nl, lab, st, ce = cv2.connectedComponentsWithStats(m, connectivity=8)
            rep["size"][1] += 1
            if nl < 2:
                errs.append(f"{sid}: f={f} ball {i} no blob found")
                continue
            j = 1 + int(np.argmax(st[1:, cv2.CC_STAT_AREA]))
            filled = np.zeros_like(m)
            cs, _ = cv2.findContours((lab == j).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            cv2.drawContours(filled, cs, -1, 1, cv2.FILLED)
            ys_, xs_ = np.nonzero(filled)
            cx, cy = xs_.mean() + xa, ys_.mean() + ya
            req = math.sqrt(len(xs_) / math.pi)
            dc = math.hypot(cx - x, cy - y)
            if dc <= CENTRE_TOL and abs(req - r) <= SIZE_TOL:
                rep["size"][0] += 1
            else:
                errs.append(f"{sid}: f={f} ball {i} centre off {dc:.2f}px (track {x:.1f},{y:.1f}; blob "
                            f"{cx:.1f},{cy:.1f}), radius {req:.2f} vs track {r:.2f}")

    # ---- d. answers from the decoded clock / catch frames
    def first_showing(t):
        return next((f for f in range(F) if clocks[f] == t), None)

    alive_at = lambda f: int(present[f].sum())  # noqa: E731
    for q in s["questions"]:
        k, p = q["derivation_key"], q["params"]
        if k not in ("catch_time", "first_catch_time", "left_at", "caught_between"):
            continue
        rep["answers"][1] += 1
        try:
            if k == "catch_time":
                got = f"{clocks[dec_catch[p['x']]] / 10:.1f} s"
            elif k == "first_catch_time":
                got = f"{clocks[min(dec_catch.values())] / 10:.1f} s"
            elif k == "left_at":
                got = alive_at(first_showing(p["t"]))
            else:
                fa, fb = first_showing(p["t1"]), first_showing(p["t2"])
                got = sum(fa < f <= fb for f in dec_catch.values())
        except (KeyError, TypeError, ValueError) as e:
            got = f"<undecodable: {e!r}>"
        if str(got) == str(q["groundtruth"]):
            rep["answers"][0] += 1
        else:
            errs.append(f"{sid}/{q['qid']} {k} {p}: decoded {got!r} vs stored {q['groundtruth']!r}")

    # ---- e. nothing but the clock outside the arena
    bg = np.array(sim.STYLES[sc["style"]["palette"]][0])
    out_m = np.ones((H, W), dtype=bool)
    out_m[max(0, y0 - 12):y1 + 13, max(0, x0 - 12):x1 + 13] = False
    cx0, cy0, cx1, cy1 = render.CLOCK_BOX
    out_m[cy0 - 3:cy1 + 4, cx0 - 3:cx1 + 4] = False      # codec ringing around the clock box
    for f in (0, F - 1):
        diff = np.abs(frames[f].astype(np.int16) - bg).max(-1)
        nbad = int(((diff > BG_TOL) & out_m).sum())
        rep["margins"][1] += 1
        if nbad == 0:
            rep["margins"][0] += 1
        else:
            ys_, xs_ = np.nonzero((diff > BG_TOL) & out_m)
            errs.append(f"{sid}: f={f} {nbad} non-background px outside arena, e.g. ({xs_[0]},{ys_[0]}) "
                        f"max diff {int(diff[ys_[0], xs_[0]])}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset-dir", default=str(HERE.parent / "output_videos" / "multi_agent_chase_preview"))
    ap.add_argument("--report", default="frame_check_report.txt")
    args = ap.parse_args()
    root = Path(args.dataset_dir).resolve()
    rep = {c: [0, 0] for c in CHECKS}
    errs, n = [], 0
    for line in open(root / "annotations.jsonl", encoding="utf8"):
        s = json.loads(line)
        n += 1
        check_clip(s, root, rep, errs)
    r = rep
    summary = (f"frame_check: {n} clips, clock frames ok {r['clock'][0]}/{r['clock'][1]}, "
               f"visibility ok {r['visibility_drawn'][0]}/{r['visibility_drawn'][1]} drawn + {r['visibility'][0]}/{r['visibility'][1]} gone, "
               f"catch frames ok {r['catch'][0]}/{r['catch'][1]}, "
               f"centre/size ok {r['size'][0]}/{r['size'][1]}, "
               f"clock answers ok {r['answers'][0]}/{r['answers'][1]}, "
               f"margins ok {r['margins'][0]}/{r['margins'][1]}")
    lines = [summary, f"errors: {len(errs)}"] + [f"  - {e}" for e in errs]
    (root / args.report).write_text("\n".join(lines) + "\n", encoding="utf8")
    print("\n".join(lines[:2 + 40]))
    sys.exit(1 if errs else 0)


if __name__ == "__main__":
    main()
