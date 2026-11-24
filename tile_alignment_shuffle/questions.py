"""10-question ladder for tile_alignment_shuffle (common/ladder5.py, LADDER = "ten").

Per clip: 1-3 / 2-3 / 2-3 / 1-3 / 1-3 questions on levels 1-5 (L.ten_counts).
Templates are drawn first (motion templates weighted up), then their params;
every template is used once before any template repeats with new params,
and no derivation_key + params pair repeats within a clip.

Tiles are named by look ("blue star", "red & symbol"); one particular tile
is named by the cell it starts or ends in ("row 2, column 3", read off the
row/column numbers). Moves are named by order ("the 3rd move"). Clock times
are read off the on-screen clock and only asked where the reading is the
same one frame either side of the event. Every answer here comes from
sim.py; answers.py re-derives it from the renderer's per-frame geometry.
"""
from __future__ import annotations
import json
import math
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from common import ladder5 as L  # noqa: E402

import sim  # noqa: E402

LADDER = "ten"
LINE = "complete row, column or diagonal made of one kind of tile"
LINES = "complete rows, columns or diagonals made of one kind of tile"
MOVE_NOTE = "(a move is one round of tiles changing places; tiles that move at the same time are one move)"
KEEP = "every other move still moves whatever tiles sit in the same cells, in the same way"
NEVER = "it never moves"
STATIC = {"count_design", "n_false", "n_kinds", "tile_at_start", "tile_at_end", "line_where", "aligned_design",
          "cf_false_real"}


def mf(frames: float) -> float:
    """min_fps for a window of `frames` frames: one sample must fall inside it."""
    return math.ceil(100 * sim.FPS / max(1, frames)) / 100


def ordinal(k: int) -> str:
    suf = "th" if 10 <= k % 100 <= 20 else {1: "st", 2: "nd", 3: "rd"}.get(k % 10, "th")
    return f"{k}{suf}"


class Ctx:
    def __init__(self, scene):
        self.s = scene
        self.n = scene["n"]
        self.T, self.D = scene["tiles"], scene["designs"]
        self.S = len(scene["moves"])
        self.gs = sim.grids(scene)
        self.start, self.end = self.gs[0], self.gs[-1]
        self.moves = sim.move_counts(scene)
        self.first = sim.first_moves(scene)
        self.movers = [t for t in range(self.n * self.n) if self.moves[t] > 0]
        self.dnames = [sim.design_name(d) for d in self.D]
        tm = scene["timing"]
        self.period = tm["move"] + tm["pause"]
        self.track = mf(self.period)                     # one sample per move
        self.clock = mf(self.period / 2)                 # nearest of options one move apart
        self.whole = mf(scene["n_frames"])
        self.at_start = mf(tm["intro"])
        self.at_end = mf(tm["hold"])
        self.fin = sim.alignments(self.T, self.end)

    def name(self, tid):
        return self.dnames[self.T[tid]]

    def cell0(self, tid):
        return sim.where(self.start, tid)

    def path(self, tid):
        return [sim.where(g, tid) for g in self.gs]

    def skip(self, k):
        return sim.replay(self.s, self.s["moves"][:k] + self.s["moves"][k + 1:])


# --------------------------------------------------------------------------- option helpers


def _q(c, rng, lv, key, text, gt, distractors, params, fps):
    opts, a = L.mc(rng, gt, distractors)
    if len(opts) != 4:
        raise ValueError(f"{key}: only {len(opts)} options")
    return L.question(lv, key, text, gt, opts, a, key not in STATIC, dict(params, min_fps=fps))


def _cells(c, rng, gt, prefer):
    n = c.n
    near = [(gt[0] + dr, gt[1] + dc) for dr, dc in ((0, 1), (0, -1), (1, 0), (-1, 0), (1, 1), (-1, -1))
            if 0 <= gt[0] + dr < n and 0 <= gt[1] + dc < n]
    rng.shuffle(near)
    rest = [(r, col) for r in range(n) for col in range(n)]
    rng.shuffle(rest)
    out = []
    for x in list(prefer) + near + rest:
        x = tuple(x)
        if x != tuple(gt) and x not in out:
            out.append(x)
    return [sim.cell_name(x) for x in out]


def _designs(c, rng, gt_name, prefer=()):
    """Other kinds on the board first; then decoys mixing a shape/symbol and a colour that are both
    on the board but never together (small grids have only 3 kinds)."""
    rest = [d for d in c.dnames if d != gt_name and d not in prefer]
    rng.shuffle(rest)
    have = {(d["kind"], d["color"]) for d in c.D}
    mixes = [sim.design_name({"kind": k, "color": col}) for k in sorted({d["kind"] for d in c.D})
             for col in sorted({d["color"] for d in c.D}) if (k, col) not in have]
    rng.shuffle(mixes)
    return [d for d in prefer if d != gt_name] + rest + mixes


def _ints(gt, lo=0, hi=10 ** 6, extra=()):
    out = [x for x in extra if lo <= x <= hi and x != gt]
    return out + L.near_ints(gt, lo, hi)


# --------------------------------------------------------------------------- candidates


def candidates(c: Ctx, rng) -> dict:
    sc, n = c.s, c.n
    cand = {lv: [] for lv in range(1, 6)}
    cells = [(r, col) for r in range(n) for col in range(n)]
    mov_cells = [c.cell0(t) for t in c.movers]
    line = bool(c.fin)
    # L1 observation
    cand[1] += [("count_design", {"design": d}) for d, x in enumerate(c.D) if not x["false"]]
    if sim.false_tiles(sc) or rng.random() < 0.5:     # keep "0" from being the usual answer
        cand[1].append(("n_false", {}))
    cand[1] += [("n_kinds", {}), ("n_moves", {})]
    cand[1] += [("tile_at_start", {"cell": list(x)}) for x in cells]
    cand[1] += [("tile_at_end", {"cell": list(x)}) for x in cells]
    if line:
        cand[1].append(("line_where", {}))
    # L2 relational
    cand[2] += [("partner", {"cell": list(x)}) for x in mov_cells]
    for _ in range(12):
        a, b = rng.sample(cells, 2)
        ta, tb = c.start[a[0]][a[1]], c.start[b[0]][b[1]]
        fa, fb = c.first.get(ta), c.first.get(tb)
        if fa != fb:
            cand[2].append(("first_mover", {"cells": [list(a), list(b)]}))
        if c.moves[ta] or c.moves[tb]:
            cand[2].append(("more_moves", {"cells": [list(a), list(b)]}))
    if line:
        cand[2].append(("aligned_design", {}))
    # L3 measurement
    cand[3] += [("moves_of", {"cell": list(x)}) for x in cells]
    cand[3] += [("end_cell", {"cell": list(x)}) for x in mov_cells]
    for x in mov_cells:
        t = c.start[x[0]][x[1]]
        p = c.path(t)
        ks = [k for k in range(1, c.S) if p[k] != p[0] and p[k] != p[-1]]
        if ks:
            cand[3].append(("cell_after", {"cell": list(x), "k": rng.choice(ks)}))
        if sim.stable_reading(sim.move_end(sc, c.first[t])):
            cand[3].append(("first_rest_time", {"cell": list(x)}))
    if line:
        cand[3].append(("line_after", {}))
        if sim.stable_reading(sim.move_end(sc, sc["line_move"] - 1)):
            cand[3].append(("line_time", {}))
    # L4 integration
    top = max(c.moves.values())
    if list(c.moves.values()).count(top) == 1:
        cand[4].append(("most_moves", {}))
    cand[4] += [("start_of", {"cell": list(sim.where(c.end, t))}) for t in c.movers]
    cand[4].append(("never_moved", {}))
    if any(c.cell0(t) == sim.where(c.end, t) for t in c.movers) or rng.random() < 0.3:
        cand[4].append(("back_home", {}))
    cand[4] += [("cells_visited", {"cell": list(x)}) for x in mov_cells]
    if line:
        cand[4].append(("line_origin", {}))
    if sc["scenario"] in ("swap", "neighbour", "double"):
        cand[4] += [("partner_origin", {"cell": list(x)}) for x in mov_cells]
    # L5 counterfactual
    for x in mov_cells:
        t = c.start[x[0]][x[1]]
        ks = [k for k in range(c.S) if sim.where(c.skip(k), t) != sim.where(c.end, t)]
        if ks:
            cand[5].append(("cf_skip", {"cell": list(x), "k": rng.choice(ks)}))
        ks = [k for k in range(c.S) if t in sim.movers(c.gs[k], sc["moves"][k])]
        cand[5].append(("cf_skip_moves", {"cell": list(x), "k": rng.choice(ks)}))
    by_ans = {}                                      # pick the answer value first, so 0 and 1 are even
    for k in range(c.S):
        by_ans.setdefault(len(sim.alignments(c.T, c.skip(k))), []).append(k)
    ans = rng.choice(sorted(by_ans))
    if len(by_ans) > 1 or rng.random() < 0.3:       # a clip with no line can only answer 0
        cand[5].append(("cf_skip_lines", {"k": rng.choice(by_ans[ans])}))
    for ft in sim.false_tiles(sc):
        mimic = next(i for i, x in enumerate(c.D) if not x["false"] and x["kind"] == c.D[c.T[ft]]["kind"])
        cand[5].append(("cf_false_real", {"design": c.T[ft], "as": mimic}))
    for lv in cand:                                  # drop repeated (key, params) pairs
        seen, keep = set(), []
        for key, p in cand[lv]:
            sig = (key, json.dumps(p, sort_keys=True))
            if sig not in seen:
                seen.add(sig)
                keep.append((key, p))
        cand[lv] = keep
    return cand


# --------------------------------------------------------------------------- questions


def make(c: Ctx, rng, lv, key, p):
    sc = c.s
    at = lambda x: c.start[x[0]][x[1]]  # noqa: E731
    cn = lambda x: sim.cell_name(x)  # noqa: E731
    if key == "count_design":
        gt = c.T.count(p["design"])
        return _q(c, rng, lv, key, f"How many {c.dnames[p['design']]} tiles are on the board?", gt,
                  _ints(gt, 1), p, c.whole)
    if key == "n_false":
        gt = len(sim.false_tiles(sc))
        return _q(c, rng, lv, key, "How many tiles match no other tile on the board (no other tile has the same "
                  "shape or symbol in the same colour)?", gt, _ints(gt, 0), p, c.whole)
    if key == "n_kinds":
        gt = len(c.D)
        return _q(c, rng, lv, key, "How many different kinds of tile are on the board? (Two tiles are the same "
                  "kind only if they have the same shape or symbol in the same colour.)", gt, _ints(gt, 1), p,
                  c.whole)
    if key == "n_moves":
        return _q(c, rng, lv, key, f"How many moves happen in the clip? {MOVE_NOTE}", c.S, _ints(c.S, 1), p,
                  c.track)
    if key == "tile_at_start":
        gt = c.name(at(p["cell"]))
        return _q(c, rng, lv, key, f"At the very start, what kind of tile is in {cn(p['cell'])}?", gt,
                  _designs(c, rng, gt, [c.name(c.end[p['cell'][0]][p['cell'][1]])]), p, c.at_start)
    if key == "tile_at_end":
        gt = c.name(c.end[p["cell"][0]][p["cell"][1]])
        return _q(c, rng, lv, key, f"At the end, what kind of tile is in {cn(p['cell'])}?", gt,
                  _designs(c, rng, gt, [c.name(at(p['cell']))]), p, c.at_end)
    if key == "line_where":
        kind, cells, _ = c.fin[0]
        gt = sim.line_name(kind, cells)
        others = [sim.line_name(k2, cs) for k2, cs in sim.lines(c.n) if sim.line_name(k2, cs) != gt]
        rng.shuffle(others)
        return _q(c, rng, lv, key, f"At the end, where is the {LINE}?", gt, others, p, c.at_end)
    if key == "aligned_design":
        gt = c.dnames[c.fin[0][2]]
        return _q(c, rng, lv, key, f"At the end, which kind of tile makes up the {LINE}?", gt,
                  _designs(c, rng, gt), p, c.at_end)
    if key == "partner":
        t = at(p["cell"])
        k = c.first[t]
        x = sim.where(c.gs[k], t)
        gt = c.name(c.gs[k + 1][x[0]][x[1]])
        return _q(c, rng, lv, key, f"The first time the tile that starts in {cn(p['cell'])} moves, what kind of "
                  "tile moves into the cell it leaves?", gt, _designs(c, rng, gt, [c.name(t)]), p, c.track)
    if key == "first_mover":
        a, b = (tuple(x) for x in p["cells"])
        fa, fb = c.first.get(at(a), 10 ** 6), c.first.get(at(b), 10 ** 6)
        opts = [f"the one that starts in {cn(a)}", f"the one that starts in {cn(b)}",
                "they first move at the same time", "neither of them ever moves"]
        gt = opts[0] if fa < fb else opts[1]
        return _q(c, rng, lv, key, f"Which tile moves first: the one that starts in {cn(a)} or the one that "
                  f"starts in {cn(b)}?", gt, opts, p, c.track)
    if key == "more_moves":
        a, b = (tuple(x) for x in p["cells"])
        ma, mb = c.moves[at(a)], c.moves[at(b)]
        opts = [f"the one that starts in {cn(a)}", f"the one that starts in {cn(b)}",
                "they move the same number of times", "neither of them ever moves"]
        gt = opts[0] if ma > mb else opts[1] if mb > ma else opts[2]
        return _q(c, rng, lv, key, f"Which tile moves to a new cell more times: the one that starts in {cn(a)} "
                  f"or the one that starts in {cn(b)}?", gt, opts, p, c.track)
    if key == "moves_of":
        gt = c.moves[at(p["cell"])]
        return _q(c, rng, lv, key, f"How many times does the tile that starts in {cn(p['cell'])} move to a new "
                  f"cell? {MOVE_NOTE}", gt, _ints(gt, 0), p, c.track)
    if key == "end_cell":
        pth = c.path(at(p["cell"]))
        return _q(c, rng, lv, key, f"Where does the tile that starts in {cn(p['cell'])} end up?", cn(pth[-1]),
                  _cells(c, rng, pth[-1], pth[::-1]), p, c.track)
    if key == "cell_after":
        pth = c.path(at(p["cell"]))
        k = p["k"]
        return _q(c, rng, lv, key, f"Where is the tile that starts in {cn(p['cell'])} right after the "
                  f"{ordinal(k)} move? {MOVE_NOTE}", cn(pth[k]), _cells(c, rng, pth[k], [pth[-1]] + pth), p,
                  c.track)
    if key == "first_rest_time":
        t = at(p["cell"])
        k = c.first[t]
        gt = sim.timer_text(sim.move_end(sc, k))
        ds = [sim.timer_text(sim.move_end(sc, j)) for j in (k - 1, k + 1, k - 2, k + 2, k + 3, k - 3)
              if 0 <= j < c.S]
        ds += [sim.timer_text(sim.move_start(sc, k)), sim.timer_text(sc["n_frames"] - 1)]
        return _q(c, rng, lv, key, f"What does the clock read when the tile that starts in {cn(p['cell'])} first "
                  "comes to rest in a new cell?", gt, ds, p, c.clock)
    if key == "line_after":
        f = sc["line_move"]
        ds = [c.S] + [k for k in (f - 1, f + 1, f - 2, f + 2, f - 3) if 1 <= k <= c.S]
        return _q(c, rng, lv, key, f"Right after which move is there a {LINE} on the board for the first time? "
                  f"{MOVE_NOTE}", f"the {ordinal(f)} move", [f"the {ordinal(k)} move" for k in ds], p, c.track)
    if key == "line_time":
        f = sc["line_move"] - 1
        gt = sim.timer_text(sim.move_end(sc, f))
        ds = [sim.timer_text(sim.move_end(sc, j)) for j in (c.S - 1, f - 1, f + 1, f - 2, f + 2, f - 3)
              if 0 <= j < c.S]
        ds.append(sim.timer_text(sc["n_frames"] - 1))
        return _q(c, rng, lv, key, f"What does the clock read when a {LINE} first appears on the board?", gt, ds,
                  p, c.clock)
    if key == "most_moves":
        t = max(c.moves, key=c.moves.get)
        rivals = sorted((x for x in c.movers if x != t), key=lambda x: -c.moves[x])
        return _q(c, rng, lv, key, "Which tile moves to a new cell the most times? Name the cell it starts in.",
                  cn(c.cell0(t)), _cells(c, rng, c.cell0(t), [c.cell0(x) for x in rivals]), p, c.track)
    if key == "start_of":
        t = c.end[p["cell"][0]][p["cell"][1]]
        pth = c.path(t)
        return _q(c, rng, lv, key, f"The tile that ends in {cn(p['cell'])}: which cell did it start in?",
                  cn(pth[0]), _cells(c, rng, pth[0], pth[1:]), p, c.track)
    if key == "never_moved":
        gt = sum(1 for v in c.moves.values() if v == 0)
        return _q(c, rng, lv, key, "How many tiles never move during the whole clip?", gt, _ints(gt, 0), p, c.track)
    if key == "back_home":
        gt = sum(1 for t in c.movers if c.cell0(t) == sim.where(c.end, t))
        return _q(c, rng, lv, key, "How many tiles move at least once but end in the same cell they started in?",
                  gt, _ints(gt, 0), p, c.track)
    if key == "cells_visited":
        pth = c.path(at(p["cell"]))
        gt = len(set(pth))
        return _q(c, rng, lv, key, f"How many different cells does the tile that starts in {cn(p['cell'])} sit in "
                  "during the clip, counting its starting cell?", gt, _ints(gt, 1, extra=[c.moves[at(p['cell'])]]),
                  p, c.track)
    if key == "line_origin":
        cells = c.fin[0][1]
        gt = sum(1 for r, col in cells if c.cell0(c.end[r][col]) in cells)
        return _q(c, rng, lv, key, f"The {LINE} at the end: how many of its tiles were already somewhere in that "
                  "same row, column or diagonal at the start?", gt, _ints(gt, 0, c.n), p, c.track)
    if key == "partner_origin":
        t = at(p["cell"])
        k = c.first[t]
        m = sc["moves"][k]
        x = sim.where(c.gs[k], t)
        y = next(tuple(b) for a, b in m["pairs"] if tuple(a) == x)
        o = c.gs[k][y[0]][y[1]]
        gt = c.cell0(o)
        return _q(c, rng, lv, key, f"The first time the tile that starts in {cn(p['cell'])} moves, it trades places "
                  "with another tile. Which cell did that other tile start the clip in?", cn(gt),
                  _cells(c, rng, gt, [y, x, sim.where(c.end, o)]), p, c.track)
    if key == "cf_skip":
        t, k = at(p["cell"]), p["k"]
        gt = sim.where(c.skip(k), t)
        return _q(c, rng, lv, key, f"Suppose the {ordinal(k + 1)} move had not happened, and {KEEP}. Where would "
                  f"the tile that starts in {cn(p['cell'])} end up? {MOVE_NOTE}", cn(gt),
                  _cells(c, rng, gt, [sim.where(c.end, t), c.cell0(t)] + c.path(t)), p, c.track)
    if key == "cf_skip_moves":
        t, k = at(p["cell"]), p["k"]
        g, cnt = [row[:] for row in c.start], 0
        for j, m in enumerate(sc["moves"]):
            if j != k:
                cnt += t in sim.movers(g, m)
                g = sim.apply(g, m)
        return _q(c, rng, lv, key, f"Suppose the {ordinal(k + 1)} move had not happened, and {KEEP}. How many "
                  f"times would the tile that starts in {cn(p['cell'])} move to a new cell? {MOVE_NOTE}", cnt,
                  _ints(cnt, 0, extra=[c.moves[t], c.moves[t] - 1]), p, c.track)
    if key == "cf_skip_lines":
        k = p["k"]
        gt = len(sim.alignments(c.T, c.skip(k)))
        return _q(c, rng, lv, key, f"Suppose the {ordinal(k + 1)} move had not happened, and {KEEP}. How many "
                  f"{LINES} would the final board have? {MOVE_NOTE}", gt, _ints(gt, 0, extra=[len(c.fin)]), p,
                  c.track)
    if key == "cf_false_real":
        tiles = c.T[:]
        ft = c.T.index(p["design"])
        tiles[ft] = p["as"]
        gt = len(sim.alignments(tiles, c.end))
        return _q(c, rng, lv, key, f"If the {c.dnames[p['design']]} tile were a {c.dnames[p['as']]} instead, how "
                  f"many {LINES} would the final board have?", gt, _ints(gt, 0, extra=[len(c.fin)]), p, c.at_end)
    raise KeyError(key)


def _pick(rng, cands, k):
    keys = []
    for key, _ in cands:
        if key not in keys:
            keys.append(key)
    params = {key: [p for k2, p in cands if k2 == key] for key in keys}
    chosen, pool = [], list(keys)
    while pool and len(chosen) < k:
        key = rng.choices(pool, weights=[1 if x in STATIC else 3 for x in pool])[0]
        pool.remove(key)
        chosen.append((key, rng.choice(params[key])))
    if len(chosen) < k:
        rest = [(key, p) for key, p in cands if (key, p) not in chosen]
        rng.shuffle(rest)
        chosen += rest[:k - len(chosen)]
    if len(chosen) < k:
        raise ValueError("not enough question candidates")
    return chosen


def build(scene: dict, rng: random.Random) -> list[dict]:
    c = Ctx(scene)
    counts = L.ten_counts(rng)
    cand = candidates(c, rng)
    qs = []
    for lv, k in zip(range(1, 6), counts):
        for key, p in _pick(rng, cand[lv], k):
            qs.append(make(c, rng, lv, key, p))
    return L.number(qs)
