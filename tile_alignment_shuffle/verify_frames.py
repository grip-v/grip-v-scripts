"""Frame-level check of rendered tile_alignment_shuffle clips against the ground truth.

    python verify_frames.py --dataset-dir <dir with annotations.jsonl + videos/>

It decodes every MP4 and reads the answers back off the pixels. No answer
comes from sim.py or answers.py; render.py is used only to draw reference
patches (tile sprites, clock and number templates) and for the layout boxes.

  * resting boards: a frame is "at rest" when the board area differs from
    the previous frame by < REST_DIFF grey levels; each run of rest frames
    is one resting board. Every cell of it is classified by matching its
    centre patch against freshly drawn sprites of the clip's designs
    (best match must beat the runner-up by a clear margin);
  * moves: consecutive resting boards differ in a few cells. For swaps /
    cycles / double swaps the movers all look different, so the only
    design-consistent rearrangement of the changed cells is the move; for
    slides every row/column shift by one cell is tried and exactly one may
    fit. Tiles are then followed through the moves by identity;
  * every question is re-answered from these boards, moves and paths
    (counts, cells, orders, lines, what-if replays) and compared with the
    stored answer; clock questions read the clock patch at the first rest
    frame of the board in question;
  * clock: at the first rest frame after every move, the clock patch is
    template-matched against all readings and must equal that frame's time;
  * numbers: each row/column number is template-matched at its scene
    position, its ink must be >= 24 px tall, and its patch must stay the
    same in every frame (nothing ever moves over it); the clock box too;
  * geometry: in every frame, no drawn tile (incl. lifted ones) overlaps a
    number or the clock box (render.state boxes).
"""
from __future__ import annotations
import argparse
import itertools
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
from common.video5 import font  # noqa: E402

REST_DIFF = 1.2
SHAPES = {"circle", "square", "triangle", "star", "heart", "hexagon", "diamond", "cross", "crescent", "pentagon"}


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


def clock(f):
    t = f * 10 // 24
    return f"{t // 10}.{t % 10} s"


def ordinal(k):
    suf = "th" if 10 <= k % 100 <= 20 else {1: "st", 2: "nd", 3: "rd"}.get(k % 10, "th")
    return f"{k}{suf}"


def cell(rc):
    return f"row {rc[0] + 1}, column {rc[1] + 1}"


def dname(d):
    return f"{d['color']} {d['kind']}" + ("" if d["kind"] in SHAPES else " symbol")


# --------------------------------------------------------------------------- reading the frames


class Reader:
    def __init__(self, scene):
        self.s = scene
        self.n = scene["n"]
        lay = scene["layout"]
        self.cell = lay["span"] / self.n
        self.x0, self.y0 = lay["x0"], lay["y0"]
        self.size = render.tile_size(scene)
        self.half = int(self.size * 0.32)                 # centre patch: inside the card, off the border
        st = scene["style"]
        self.refs = []
        for d in scene["designs"]:
            spr = np.asarray(render.sprite(d["kind"], d["color"], self.size, scene["card"], st["radius"],
                                           st["border"], st["scale"]).convert("RGB")).astype(float)
            c = self.size // 2
            self.refs.append(spr[c - self.half:c + self.half, c - self.half:c + self.half])
        p = lay["pad"]
        self.board_box = (lay["x0"] - p, lay["y0"] - p, lay["x0"] + lay["span"] + p, lay["y0"] + lay["span"] + p)

    def patch(self, fr, r, c):
        cx = int(round(self.x0 + self.cell * (c + 0.5)))
        cy = int(round(self.y0 + self.cell * (r + 0.5)))
        return fr[cy - self.half:cy + self.half, cx - self.half:cx + self.half].astype(float)

    def board(self, fr, errs, tag):
        g = [[None] * self.n for _ in range(self.n)]
        for r in range(self.n):
            for c in range(self.n):
                p = self.patch(fr, r, c)
                ds = sorted((np.abs(p - ref).mean(), i) for i, ref in enumerate(self.refs))
                if ds[0][0] > 35 or ds[0][0] > 0.6 * ds[1][0]:     # H.264 halves colour resolution: judge by margin
                    errs.append(f"{tag}: cell {r},{c} unclear (best {ds[0][0]:.1f}, next {ds[1][0]:.1f})")
                g[r][c] = ds[0][1]
        return g

    def rest_runs(self, frames):
        x0, y0, x1, y1 = self.board_box
        crop = [f[y0:y1, x0:x1].astype(np.int16) for f in frames]
        still = [False] + [np.abs(crop[i] - crop[i - 1]).mean() < REST_DIFF for i in range(1, len(frames))]
        runs, i = [], 0
        while i < len(frames):
            if still[i] or i == 0:
                j = i
                while j + 1 < len(frames) and still[j + 1]:
                    j += 1
                if j - i >= 2:
                    runs.append((i, j))
                i = j + 1
            else:
                i += 1
        return runs


def solve_move(before, after, scenario, n):
    """Cell rearrangement {src: dst} that turns design board `before` into `after`."""
    changed = [(r, c) for r in range(n) for c in range(n) if before[r][c] != after[r][c]]
    if scenario == "slide":
        fits = []
        for axis, i, d in itertools.product(("row", "col"), range(n), (1, -1)):
            cells = [(i, c) for c in range(n)] if axis == "row" else [(r, i) for r in range(n)]
            mv = {a: b for a, b in zip(cells, cells[d:] + cells[:d])}
            g = [row[:] for row in before]
            for a, b in mv.items():
                g[b[0]][b[1]] = before[a[0]][a[1]]
            if g == after:
                fits.append(mv)
        return fits
    fits = []
    for perm in itertools.permutations(changed):
        if all(a != b and before[a[0]][a[1]] == after[b[0]][b[1]] for a, b in zip(changed, perm)):
            fits.append(dict(zip(changed, perm)))
    return fits


def lines_on(g, n, look=lambda x: x):
    ls = [[(r, c) for c in range(n)] for r in range(n)] + [[(r, c) for r in range(n)] for c in range(n)]
    ls += [[(i, i) for i in range(n)], [(i, n - 1 - i) for i in range(n)]]
    return [cs for cs in ls if len({look(g[r][c]) for r, c in cs}) == 1]


def line_text(cs):
    if len({r for r, _ in cs}) == 1:
        return f"row {cs[0][0] + 1}"
    if len({c for _, c in cs}) == 1:
        return f"column {cs[0][1] + 1}"
    return "the diagonal from top-left to bottom-right" if (0, 0) in cs else "the diagonal from top-right to bottom-left"


# --------------------------------------------------------------------------- text checks


def text_patch(text, size, fg, bg, box=None):
    w, h = (box[2] - box[0], box[3] - box[1]) if box else (int(size * 1.4), int(size * 1.4))
    img = Image.new("RGB", (w, h), bg)
    ImageDraw.Draw(img).text((w / 2, h / 2), text, fill=fg, font=font(size), anchor="mm")
    return np.asarray(img).astype(float)


class Clock:
    def __init__(self, scene, n_frames):
        b = scene["timer"]["box"]
        self.box = (b[0] + 4, b[1] + 4, b[2] - 4, b[3] - 4)
        self.cands = {}
        for f in range(n_frames):
            t = clock(f)
            if t not in self.cands:
                img = render.background(scene).copy()
                render.draw_timer(img, scene, f)
                x0, y0, x1, y1 = self.box
                self.cands[t] = np.asarray(img)[y0:y1, x0:x1].astype(float)

    def read(self, fr):
        x0, y0, x1, y1 = self.box
        p = fr[y0:y1, x0:x1].astype(float)
        return min(self.cands, key=lambda t: np.abs(p - self.cands[t]).mean())


def check_labels(scene, frames, errs):
    import sim
    th = sim.THEMES[scene["theme"]]
    f0 = frames[0].astype(float)
    for lab in scene["labels"]:
        s = lab["size"]
        x, y, w = int(round(lab["x"])), int(round(lab["y"])), int(s * 0.7)
        box = (x - w, y - w, x + w, y + w)
        p = f0[box[1]:box[3], box[0]:box[2]]
        scores = {str(k): np.abs(p - text_patch(str(k), s, th["ink"], th["bg"], box)).mean() for k in range(1, 10)}
        best = min(scores, key=scores.get)
        if best != lab["text"]:
            errs.append(f"label at ({x},{y}) reads {best}, scene says {lab['text']}")
        ink = np.abs(p - np.array(th["bg"], float)).sum(axis=2) > 120
        rows = np.where(ink.any(axis=1))[0]
        if not len(rows) or rows[-1] - rows[0] + 1 < 24:
            errs.append(f"label {lab['text']} ink only {0 if not len(rows) else rows[-1] - rows[0] + 1} px tall")
        for i in range(0, len(frames), 3):
            if np.abs(frames[i][box[1]:box[3], box[0]:box[2]].astype(float) - p).mean() > 3:
                errs.append(f"label {lab['text']} at ({x},{y}) changes in frame {i}")
                break
    b = scene["timer"]["box"]
    edge = [(b[0] - 3, b[1] - 3, b[2] + 3, b[1] + 1)]          # strip along the clock box top: never covered
    for x0, y0, x1, y1 in edge:
        ref = f0[max(0, y0):y1, x0:x1]
        if any(np.abs(fr[max(0, y0):y1, x0:x1].astype(float) - ref).mean() > 3 for fr in frames[::3]):
            errs.append("something moves over the clock box")


def check_geometry(scene, errs):
    import sim
    boxes = [sim.label_box(lab) for lab in scene["labels"]] + [tuple(scene["timer"]["box"])]
    size = render.tile_size(scene)
    for f in range(scene["n_frames"]):
        st = render.state(scene, f)
        lift = math.sin(math.pi * st["u"])                    # exactly what render.frame draws
        big = round(size * (1 + scene["style"]["lift"] * lift))
        sh = round(3 + 8 * lift)                              # shadow falls down-right only
        for tid, x, y, lifted in st["draw"]:
            h = (big if lifted else size) / 2
            t = (x - h, y - h, x + h + (sh if lifted else 0), y + h + (sh if lifted else 0))
            if st["kind"] == "slide":
                lay = scene["layout"]
                t = (max(t[0], lay["x0"]), max(t[1], lay["y0"]), min(t[2], lay["x0"] + lay["span"]),
                     min(t[3], lay["y0"] + lay["span"]))
            for b in boxes:
                if t[0] < b[2] and b[0] < t[2] and t[1] < b[3] and b[1] < t[3]:
                    errs.append(f"frame {f}: a moving tile overlaps text box {b}")
                    return


# --------------------------------------------------------------------------- answers from pixels


def verify(sample, frames, errs):
    scene = sample["params"]
    n, D, sid = scene["n"], scene["designs"], sample["sample_id"]
    rd = Reader(scene)
    runs = rd.rest_runs(frames)
    boards, first_rest = [], []
    for a, b in runs:
        g = rd.board(frames[(a + b) // 2], errs, f"{sid} rest {a}-{b}")
        if not boards or g != boards[-1]:
            boards.append(g)
            first_rest.append(a)
    moves = []                                  # list of {src: dst}
    for k, (b0, b1) in enumerate(zip(boards, boards[1:])):
        fits = solve_move(b0, b1, scene["scenario"], n)
        if len(fits) != 1:
            errs.append(f"{sid}: move {k + 1} fits {len(fits)} rearrangements")
            return {}
        moves.append(fits[0])
    ids = [[(r, c) for c in range(n)] for r in range(n)]     # tile id = starting cell
    idb = [ids]
    for mv in moves:
        g = [row[:] for row in idb[-1]]
        for a, b in mv.items():
            g[b[0]][b[1]] = idb[-1][a[0]][a[1]]
        idb.append(g)
    pos = [{t: (r, c) for r, row in enumerate(g) for c, t in enumerate(row)} for g in idb]
    tids = list(pos[0])
    path = {t: [p[t] for p in pos] for t in tids}
    look = {t: boards[0][t[0]][t[1]] for t in tids}             # design index of each tile
    nm = {t: sum(1 for k in range(len(moves)) if path[t][k] != path[t][k + 1]) for t in tids}
    first = {t: next((k for k in range(len(moves)) if path[t][k] != path[t][k + 1]), None) for t in tids}
    endg = boards[-1]
    ck = Clock(scene, len(frames))
    # clock at every move's first rest frame
    for k, f in enumerate(first_rest):
        got = ck.read(frames[f])
        if got != clock(f) and got != clock(f - 1) and got != clock(f + 1):
            errs.append(f"{sid}: clock reads {got} at frame {f}, expected {clock(f)}")

    def replay(skip):
        g = [row[:] for row in idb[0]]
        cnt = {}
        for j, mv in enumerate(moves):
            if j == skip:
                continue
            new = [row[:] for row in g]
            for a, b in mv.items():
                new[b[0]][b[1]] = g[a[0]][a[1]]
                cnt[g[a[0]][a[1]]] = cnt.get(g[a[0]][a[1]], 0) + 1
            g = new
        return g, cnt

    def dboard(g):
        return [[look[t] for t in row] for row in g]

    got = {}
    first_line = next((k for k, g in enumerate(boards) if lines_on(g, n)), None)
    for q in sample["questions"]:
        key, p = q["derivation_key"], q["params"]
        at = (lambda x: tuple(x))
        try:
            if key == "count_design":
                v = sum(row.count(p["design"]) for row in boards[0])
            elif key == "n_false":
                flat = [x for row in boards[0] for x in row]
                v = sum(1 for x in flat if flat.count(x) == 1)
            elif key == "n_kinds":
                v = len({x for row in boards[0] for x in row})
            elif key == "n_moves":
                v = len(moves)
            elif key == "tile_at_start":
                v = dname(D[boards[0][p["cell"][0]][p["cell"][1]]])
            elif key == "tile_at_end":
                v = dname(D[endg[p["cell"][0]][p["cell"][1]]])
            elif key == "line_where":
                v = line_text(lines_on(endg, n)[0])
            elif key == "aligned_design":
                cs = lines_on(endg, n)[0]
                v = dname(D[endg[cs[0][0]][cs[0][1]]])
            elif key == "partner":
                t = at(p["cell"])
                k = first[t]
                src = path[t][k]
                v = dname(D[boards[k + 1][src[0]][src[1]]])
            elif key in ("first_mover", "more_moves"):
                a, b = at(p["cells"][0]), at(p["cells"][1])
                A, B = f"the one that starts in {cell(a)}", f"the one that starts in {cell(b)}"
                if key == "first_mover":
                    fa, fb = (first[x] if first[x] is not None else 10 ** 6 for x in (a, b))
                    v = A if fa < fb else B
                else:
                    v = A if nm[a] > nm[b] else B if nm[b] > nm[a] else (
                        "they move the same number of times" if nm[a] else "neither of them ever moves")
            elif key == "moves_of":
                v = nm[at(p["cell"])]
            elif key == "end_cell":
                v = cell(path[at(p["cell"])][-1])
            elif key == "cell_after":
                v = cell(path[at(p["cell"])][p["k"]])
            elif key == "first_rest_time":
                v = ck.read(frames[first_rest[first[at(p["cell"])] + 1]])
            elif key == "line_after":
                v = f"the {ordinal(first_line)} move"
            elif key == "line_time":
                v = ck.read(frames[first_rest[first_line]])
            elif key == "most_moves":
                v = cell(max(tids, key=lambda t: nm[t]))
            elif key == "start_of":
                v = cell(idb[-1][p["cell"][0]][p["cell"][1]])
            elif key == "never_moved":
                v = sum(1 for t in tids if nm[t] == 0)
            elif key == "back_home":
                v = sum(1 for t in tids if nm[t] and path[t][-1] == t)
            elif key == "cells_visited":
                v = len(set(path[at(p["cell"])]))
            elif key == "line_origin":
                cs = lines_on(endg, n)[0]
                v = sum(1 for r, c in cs if idb[-1][r][c] in cs)
            elif key == "partner_origin":
                t = at(p["cell"])
                mv = moves[first[t]]
                src = path[t][first[t]]
                dst = mv[src]
                v = cell(idb[first[t]][dst[0]][dst[1]])
            elif key == "cf_skip":
                g, _ = replay(p["k"])
                t = at(p["cell"])
                v = cell(next((r, c) for r in range(n) for c in range(n) if g[r][c] == t))
            elif key == "cf_skip_moves":
                _, cnt = replay(p["k"])
                v = cnt.get(at(p["cell"]), 0)
            elif key == "cf_skip_lines":
                g, _ = replay(p["k"])
                v = len(lines_on(dboard(g), n))
            elif key == "cf_false_real":
                v = len(lines_on(endg, n, lambda x: p["as"] if x == p["design"] else x))
            else:
                errs.append(f"{sid}/{q['qid']}: no pixel reader for {key}")
                continue
        except Exception as e:  # noqa: BLE001
            errs.append(f"{sid}/{q['qid']} {key}: pixel reading failed ({e!r})")
            continue
        got[q["qid"]] = v
        if str(v) != str(q["groundtruth"]):
            errs.append(f"{sid}/{q['qid']} {key}: pixels say {v!r}, stored {q['groundtruth']!r}")
    return {"rest_boards": len(boards), "moves": len(moves), "answered": len(got)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset-dir", required=True)
    ap.add_argument("--report", default="frame_check.txt")
    args = ap.parse_args()
    root = Path(args.dataset_dir)
    lines_out, total_err, n_q = [], 0, 0
    for line in open(root / "annotations.jsonl", encoding="utf8"):
        s = json.loads(line)
        errs = []
        frames = decode(root / s["media_path"])
        if len(frames) != s["params"]["n_frames"] or frames[0].shape != (720, 1280, 3):
            errs.append(f"{s['sample_id']}: {len(frames)} frames of {frames[0].shape if frames else None}")
        info = verify(s, frames, errs)
        check_labels(s["params"], frames, errs)
        check_geometry(s["params"], errs)
        n_q += info.get("answered", 0)
        total_err += len(errs)
        lines_out.append(f"{s['sample_id']} {s['params']['scenario']}: {info} errors {len(errs)}")
        lines_out += [f"  - {e}" for e in errs]
        print(lines_out[-1 - len(errs)], flush=True)
    lines_out.append(f"RESULT: {sum(1 for x in lines_out if not x.startswith('  '))} clips, {n_q} questions answered from pixels, {total_err} errors")
    (root / args.report).write_text("\n".join(lines_out) + "\n", encoding="utf8")
    print(lines_out[-1])
    sys.exit(1 if total_err else 0)


if __name__ == "__main__":
    main()
