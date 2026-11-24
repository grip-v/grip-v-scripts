"""10-question ladder for sandpile_cascade (common/ladder5.py, LADDER = "ten").

Cells are named by position ("the second tray from the left", "the cell in
row 2, column 3", "the third cell clockwise from the hole"), dropped grains
by order ("the second grain to be dropped"), moments by the clock. Ground
truth comes from sim.py (events from sim.play, grains at rest from
sim.counts, what-ifs re-simulated with sim.relax_waves); answers.py
re-derives every answer from the drawn track. Every question stores
derivation_key + params (+ min_fps).
"""
from __future__ import annotations
import random
import sys
from fractions import Fraction as Fr
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE))
from common import ladder5 as L  # noqa: E402

import sim  # noqa: E402

LADDER = "ten"
FPS_STEPS = (1, 2, 3, 4, 6, 8, 12, 24)
ORD = sim.ORD
MARGIN = 8          # frames between events a question compares (1/3 s)
WIN = 7             # frames of calm either side of a clock reading a question names
TOP = "topple (pass its grains on)"
# correct-letter weights per option count, so A-D come out even over the dataset
WEIGHTS = {2: (1, 1), 3: (1, 1, 1), 4: (0.176, 0.176, 0.318, 0.330)}


def fps_for(gap_frames) -> int:
    gap = float(gap_frames) / sim.FPS
    for f in FPS_STEPS:
        if f * gap >= 1 - 1e-9:
            return f
    return 24


def clock(f) -> str:
    return sim.clock_text(sim.tenths(int(f)))


def half_s(frames) -> str | None:
    """Seconds to the nearest 0.5 s, only when the value is within 0.1 s of it."""
    s = Fr(frames, sim.FPS)
    h = round(s * 2) / 2
    if abs(s - Fr(h)) > Fr(1, 10) or h <= 0:
        return None
    return f"{h:g} s"


def _place(rng, gt, distractors):
    opts = [gt] + list(distractors)
    if len(set(opts)) != len(opts) or not 2 <= len(opts) <= 4:
        return None
    rest = opts[1:]
    rng.shuffle(rest)
    k = rng.choices(range(len(opts)), weights=WEIGHTS[len(opts)])[0]
    opts = rest[:k] + [gt] + rest[k:]
    return opts, L.LETTERS[k]


def _pick_d(rng, gt, pool, k=3):
    pool = [p for p in dict.fromkeys(pool) if p != gt]
    if len(pool) < k:
        return None
    return rng.sample(pool, k)


def _q(level, key, text, gt, opts, motion, params):
    if opts is None or params["min_fps"] > 4:          # must stay answerable from ~4 fps samples
        return None
    o, letter = opts
    return L.question(level, key, text, gt, [str(x) for x in o], letter, motion, params)


def _ints(rng, gt, lo=0, hi=999):
    pool = [v for v in (gt + 1, gt - 1, gt + 2, gt - 2, gt + 3, gt - 3, gt + 4) if lo <= v <= hi]
    if len(pool) < 3:
        return None
    d = pool[:2] + [rng.choice(pool[2:])]
    return _place(rng, gt, d)


def _clock_opts(rng, f):
    k = sim.tenths(int(f))
    pool = [sim.clock_text(k + dk) for dk in (5, -5, 10, -10, 15) if k + dk >= 0]
    d = _pick_d(rng, sim.clock_text(k), pool[:4] if len(pool) >= 4 else pool)
    return d and _place(rng, sim.clock_text(k), d)


def _half_opts(rng, gt):
    h = float(gt.split()[0])
    pool = [f"{h + dh:g} s" for dh in (0.5, -0.5, 1.0, -1.0, 1.5) if h + dh > 0]
    d = _pick_d(rng, gt, pool[:4] if len(pool) >= 4 else pool)
    return d and _place(rng, gt, d)


def _yn(rng, gt):
    return _place(rng, gt, ["no" if gt == "yes" else "yes"])


def grain_name(i) -> str:
    return f"the {ORD[i]} grain to be dropped"


class Facts:
    def __init__(self, scene):
        self.s = scene
        self.sc = scene["scenario"]
        self.u = sim.UNIT[self.sc]
        self.conv = sim.CONVENTION[self.sc]
        self.n = scene["n_frames"]
        self.T = scene["T"]
        self.kinds, self.out = sim.topology(scene)
        self.live = [c for c, k in enumerate(self.kinds) if k == "cell"]
        self.nm = {c: sim.cell_name(scene, c) for c in self.live}
        self.ds = {c: sim.cell_desc(scene, c) for c in self.live}
        pl = sim.play(scene)
        self.ev = pl["events"]
        self.aval = pl["aval"]
        self.lands = [e for e in self.ev if e["kind"] == "land"]
        self.tops = [e for e in self.ev if e["kind"] == "topple"]
        self.C = sim.counts(scene)
        tm = scene["timing"]
        self.M, self.D, self.F = tm["M"], tm["D"], tm["F"]
        self.P = self.M + self.D
        self.first_top = {}
        self.n_top = {c: 0 for c in self.live}
        for e in self.tops:
            self.first_top.setdefault(e["cell"], e["f"])
            self.n_top[e["cell"]] += 1
        self.lost = sum(a["lost"] for a in self.aval)
        # how long every dropped grain can be seen before anything else happens to it
        tl = sim.timeline(scene)
        P = tl["poses"]
        g0 = sum(scene["init"])
        self.drop_win, self.appear = [], []
        for i, e in enumerate(self.lands):
            vis = [f for f in range(self.n) if P[f, g0 + i, 0] == P[f, g0 + i, 0]]
            first = vis[0]
            self.appear.append(first)
            nxt = [t["f"] for t in self.tops if t["f"] > e["f"]]
            stop = min(nxt[0] if nxt else self.n, self.lands[i + 1]["start"] if i + 1 < len(self.lands) else self.n)
            self.drop_win.append(stop - first)
        self.fps_drops = fps_for(min(self.drop_win))
        self.fps_top = fps_for(self.P)
        self.motion_fps = max(self.fps_drops, self.fps_top)
        self._cov = None

    @property
    def cov(self):
        if self._cov is None:
            self._cov = sim.covered(self.s)
        return self._cov

    def name(self, c):
        return self.nm[c]

    def calm(self, f0, f1, cell=None) -> bool:
        """No change at rest in [f0, f1] (in one cell, or anywhere)."""
        if f0 < 0 or f1 >= self.n:
            return False
        X = self.C[f0:f1 + 1] if cell is None else self.C[f0:f1 + 1, cell]
        return bool((X == X[0]).all())

    def cell_opts(self, rng, gt, prefer=()):
        pool = [c for c in prefer if c != gt] + [c for c in self.live if c != gt]
        pool = list(dict.fromkeys(pool))
        if len(pool) < 3:
            return None
        near = pool[:max(3, len(prefer))]
        d = rng.sample(near, 3) if len(near) >= 3 else rng.sample(pool, 3)
        return _place(rng, self.name(gt), [self.name(c) for c in d])

    def neighbours(self, c):
        nb = set()
        for t in self.out[c]:
            if t >= 0 and self.kinds[t] == "cell":
                nb.add(t)
        for x in self.live:
            if c in self.out[x]:
                nb.add(x)
        return sorted(nb)


def _p(F, c):
    return F.ds[c]


# ------------------------------------------------------------------ L1 observation

def q_n_drops(F, rng):
    gt = len(F.lands)
    return _q(1, "n_drops", "How many grains are dropped onto the board during the video?", gt,
              _ints(rng, gt, 1), True, {"min_fps": F.fps_drops})


def q_n_cells(F, rng):
    gt = len(F.live)
    extra = " (not counting the hole)" if F.sc == "ring" else ""
    return _q(1, "n_cells", f"How many {F.u}s does the board have{extra}?", gt, _ints(rng, gt, 1), False,
              {"min_fps": 1})


def q_start_count(F, rng):
    c = rng.choice(F.live)
    gt = int(F.C[0, c])
    return _q(1, "start_count", f"How many grains are in {F.name(c)} at the very start?{F.conv}", gt,
              _ints(rng, gt, 0), False, {"cell": _p(F, c), "min_fps": 1})


def q_start_total(F, rng):
    gt = int(F.C[0].sum())
    if gt > 24:
        return None
    return _q(1, "start_total", "How many grains are on the board at the very start?", gt, _ints(rng, gt, 0), False,
              {"min_fps": 1})


def q_end_count(F, rng):
    c = rng.choice(F.live)
    gt = int(F.C[-1, c])
    return _q(1, "end_count", f"How many grains are in {F.name(c)} at the end of the video?{F.conv}", gt,
              _ints(rng, gt, 0), False, {"cell": _p(F, c), "min_fps": 1})


def q_first_drop_cell(F, rng):
    c = F.lands[0]["cell"]
    return _q(1, "first_drop_cell", f"Onto which {F.u} does the first dropped grain land?{F.conv}", F.name(c),
              F.cell_opts(rng, c, F.neighbours(c)), True, {"min_fps": fps_for(F.drop_win[0])})


def q_first_topple_cell(F, rng):
    c = F.tops[0]["cell"]
    return _q(1, "first_topple_cell", f"Which {F.u} is the first to {TOP}?{F.conv}", F.name(c),
              F.cell_opts(rng, c, F.neighbours(c) + [e["cell"] for e in F.lands]), True,
              {"min_fps": F.fps_top})


def q_ever_topples(F, rng):
    c = rng.choice(F.live)
    gt = "yes" if F.n_top[c] else "no"
    return _q(1, "ever_topples", f"Does {F.name(c)} {TOP} at any point in the video?{F.conv}", gt, _yn(rng, gt), True,
              {"cell": _p(F, c), "min_fps": F.fps_top})


def q_most_end(F, rng):
    end = F.C[-1]
    c = rng.choice(F.live)
    others = [x for x in F.live if end[x] < end[c]]
    if len(others) < 3:
        return None
    d = rng.sample(others, 3)
    return _q(1, "most_end", f"At the end of the video, which of these {F.u}s holds the most grains?{F.conv}",
              F.name(c), _place(rng, F.name(c), [F.name(x) for x in d]), False,
              {"cells": sorted([_p(F, x) for x in [c] + d]), "min_fps": 1})


# ------------------------------------------------------------------ L2 relational

def q_topple_order(F, rng):
    xs = [c for c in F.live if c in F.first_top]
    if len(xs) < 2:
        return None
    a, b = rng.sample(xs, 2)
    gap = abs(F.first_top[a] - F.first_top[b])
    if gap < MARGIN:
        return None
    gt = F.name(a) if F.first_top[a] < F.first_top[b] else F.name(b)
    return _q(2, "topple_order", f"Which {F.u} is the first to {TOP}: {F.name(a)} or {F.name(b)}?{F.conv}", gt,
              _place(rng, gt, [F.name(b) if gt == F.name(a) else F.name(a)]), True,
              {"a": _p(F, a), "b": _p(F, b), "min_fps": fps_for(gap)})


def q_more_topples(F, rng):
    a, b = rng.sample(F.live, 2)
    if F.n_top[a] == F.n_top[b]:
        return None
    gt = F.name(a) if F.n_top[a] > F.n_top[b] else F.name(b)
    return _q(2, "more_topples", f"Which {F.u} topples more times over the whole video: {F.name(a)} or "
              f"{F.name(b)}?{F.conv}", gt, _place(rng, gt, [F.name(b) if gt == F.name(a) else F.name(a)]), True,
              {"a": _p(F, a), "b": _p(F, b), "min_fps": F.fps_top})


def q_bigger_cascade(F, rng):
    if len(F.aval) < 2:
        return None
    i, j = sorted(rng.sample(range(len(F.aval)), 2))
    ti, tj = F.aval[i]["topples"], F.aval[j]["topples"]
    if ti == tj:
        return None
    gt = grain_name(i) if ti > tj else grain_name(j)
    return _q(2, "bigger_cascade", f"Which dropped grain sets off more topples (counting every time a {F.u} "
              f"passes its grains on until things settle): {grain_name(i)} or {grain_name(j)}?", gt,
              _place(rng, gt, [grain_name(j) if gt == grain_name(i) else grain_name(i)]), True,
              {"i": i, "j": j, "min_fps": F.motion_fps})


def q_trigger(F, rng):
    xs = [c for c in F.live if c in F.first_top]
    c = rng.choice(xs)
    e = next(t for t in F.tops if t["cell"] == c)
    k = e["drop"]
    if len(F.lands) < 3:
        return None
    pool = [grain_name(i) for i in range(len(F.lands)) if i != k]
    d = _pick_d(rng, grain_name(k), pool, min(3, len(pool)))
    return _q(2, "trigger", f"Which dropped grain sets off the cascade in which {F.name(c)} topples for the first "
              f"time?{F.conv}", grain_name(k), d and _place(rng, grain_name(k), d), True,
              {"cell": _p(F, c), "min_fps": F.motion_fps})


def q_start_vs_end(F, rng):
    c = rng.choice(F.live)
    a, b = int(F.C[0, c]), int(F.C[-1, c])
    gt = "more" if b > a else "fewer" if b < a else "the same number"
    return _q(2, "start_vs_end", f"Compared with the very start, does {F.name(c)} hold more grains, fewer grains or "
              f"the same number at the end of the video?{F.conv}", gt,
              _place(rng, gt, [x for x in ("more", "fewer", "the same number") if x != gt]), False,
              {"cell": _p(F, c), "min_fps": 1})


def q_more_end(F, rng):
    a, b = rng.sample(F.live, 2)
    va, vb = F.C[-1, a], F.C[-1, b]
    if va == vb:
        return None
    gt = F.name(a) if va > vb else F.name(b)
    return _q(2, "more_end", f"At the end of the video, which holds more grains: {F.name(a)} or {F.name(b)}?{F.conv}",
              gt, _place(rng, gt, [F.name(b) if gt == F.name(a) else F.name(a)]), False,
              {"a": _p(F, a), "b": _p(F, b), "min_fps": 1})


def q_receives(F, rng):
    xs = [c for c in F.live if c in F.first_top]
    c = rng.choice(xs)
    tg = sorted({t for t in F.out[c] if t >= 0 and F.kinds[t] == "cell"})
    if not tg:
        return None
    gt = rng.choice(tg)
    others = [x for x in F.live if x not in tg and x != c]
    if len(others) < 3:
        return None
    near = sorted(others, key=lambda x: (abs(x - gt), x))[:5]
    d = rng.sample(near, 3)
    return _q(2, "receives", f"When {F.name(c)} topples for the first time, which of these {F.u}s gets one of its "
              f"grains?{F.conv}", F.name(gt), _place(rng, F.name(gt), [F.name(x) for x in d]), True,
              {"cell": _p(F, c), "opts": sorted(_p(F, x) for x in [gt] + d), "min_fps": fps_for(F.M)})


def _first_loss(F):
    for e in F.tops:
        lost = [t for t in e["to"] if t < 0 or F.kinds[t] == "hole"]
        if lost:
            return e, lost
    return None, None


def q_lost_side(F, rng):
    if F.sc not in ("row", "grid"):
        return None
    e, lost = _first_loss(F)
    if e is None:
        return None
    wave = [t for t in F.tops if t["f"] == e["f"]]
    sides = set()
    G = sim.geometry(F.s)
    xs = [c["c"][0] for c in G["cells"]]
    ys = [c["c"][1] for c in G["cells"]]
    for t in wave:
        for v in t["to"]:
            if v < 0:
                x, y = G["virt"][-1 - v]
                sides.add("left" if x < min(xs) else "right" if x > max(xs) else "top" if y < min(ys) else "bottom")
    if len(sides) != 1:
        return None
    gt = sides.pop()
    if F.sc == "row":
        return _q(2, "lost_side", "Off which end of the row does the first grain to leave the board fall: the left "
                  "end or the right end?", f"the {gt} end", _place(rng, f"the {gt} end",
                                                                    ["the right end" if gt == "left" else "the left end"]),
                  True, {"min_fps": fps_for(F.M)})
    opts = [f"the {s} edge" for s in ("top", "bottom", "left", "right")]
    return _q(2, "lost_side", "Over which edge of the board does the first grain to leave the board go?",
              f"the {gt} edge", _place(rng, f"the {gt} edge", [o for o in opts if o != f"the {gt} edge"]), True,
              {"min_fps": fps_for(F.M)})


# ------------------------------------------------------------------ L3 measurement

def q_topple_time(F, rng):
    c = rng.choice(list(F.first_top))
    f = F.first_top[c]
    return _q(3, "topple_time", f"At what time on the clock does {F.name(c)} {TOP} for the first time?{F.conv}",
              clock(f), _clock_opts(rng, f), True, {"cell": _p(F, c), "min_fps": 4})


def q_land_time(F, rng):
    i = rng.randrange(len(F.lands))
    f = F.lands[i]["f"]
    return _q(3, "land_time", f"At what time on the clock does {grain_name(i)} land?", clock(f), _clock_opts(rng, f),
              True, {"i": i, "min_fps": 4})


def q_total_topples(F, rng):
    gt = len(F.tops)
    return _q(3, "total_topples", f"How many times in total does a {F.u} {TOP} over the whole video? (Count every "
              f"topple of every {F.u}.)", gt, _ints(rng, gt, 0), True, {"min_fps": F.fps_top})


def q_cell_topples(F, rng):
    c = rng.choice(list(F.first_top)) if rng.random() < 0.8 else rng.choice(F.live)
    gt = F.n_top[c]
    return _q(3, "cell_topples", f"How many times does {F.name(c)} {TOP} over the whole video?{F.conv}", gt,
              _ints(rng, gt, 0), True, {"cell": _p(F, c), "min_fps": F.fps_top})


def q_count_at(F, rng):
    c = rng.choice(F.live)
    ks = []
    for k in range(sim.tenths(EDGE_F), sim.tenths(F.n - 1 - EDGE_F)):
        a, b = sim.frames_showing(k)
        if F.calm(a - WIN, b + WIN, c) and not F.calm(0, a, c) and not F.cov[a:b + 1, c].any():
            ks.append(k)
    if not ks:
        return None
    k = rng.choice(ks)
    a, b = sim.frames_showing(k)
    gt = int(F.C[a, c])
    return _q(3, "count_at", f"How many grains are in {F.name(c)} when the clock shows {sim.clock_text(k)}?{F.conv}",
              gt, _ints(rng, gt, 0), True, {"cell": _p(F, c), "t": k, "min_fps": fps_for(b - a + 1 + 2 * WIN)})


EDGE_F = sim.EDGE


def q_lost_total(F, rng):
    gt = F.lost
    where = "into the hole" if F.sc == "ring" else "off the board"
    return _q(3, "lost_total", f"How many grains go {where} over the whole video?", gt, _ints(rng, gt, 0), True,
              {"min_fps": fps_for(F.M)})


def q_end_total(F, rng):
    gt = int(F.C[-1].sum())
    if gt > 24:
        return None
    return _q(3, "end_total", "How many grains are on the board at the end of the video?", gt, _ints(rng, gt, 0),
              False, {"min_fps": 1})


def q_cascade_span(F, rng):
    xs = [a for a in F.aval if a["waves"] >= 2]
    if not xs:
        return None
    a = rng.choice(xs)
    last = max(t["f"] for t in F.tops if t["drop"] == a["drop"])
    gt = half_s(last - a["land"])
    if gt is None:
        return None
    return _q(3, "cascade_span", f"How many seconds pass between the moment {grain_name(a['drop'])} lands and the "
              f"start of the last topple in the cascade it sets off?", gt, _half_opts(rng, gt), True,
              {"i": a["drop"], "min_fps": 4})


# ------------------------------------------------------------------ L4 integration

def q_cascade_size(F, rng):
    i = rng.randrange(len(F.aval))
    gt = F.aval[i]["topples"]
    return _q(4, "cascade_size", f"How many topples does {grain_name(i)} set off in total (counting every time a "
              f"{F.u} passes its grains on until things settle)?", gt, _ints(rng, gt, 0), True,
              {"i": i, "min_fps": F.motion_fps})


def q_n_toppled_cells(F, rng):
    gt = len(F.first_top)
    return _q(4, "n_toppled_cells", f"How many different {F.u}s {TOP} at least once during the video?", gt,
              _ints(rng, gt, 0, len(F.live)), True, {"min_fps": F.fps_top})


def q_biggest(F, rng):
    if len(F.aval) < 3:
        return None
    ids = rng.sample(range(len(F.aval)), min(4, len(F.aval)))
    tp = sorted((F.aval[i]["topples"], i) for i in ids)
    if tp[-1][0] == tp[-2][0]:
        return None
    gt = grain_name(tp[-1][1])
    return _q(4, "biggest", "Which of these dropped grains sets off the most topples?", gt,
              _place(rng, gt, [grain_name(i) for _, i in tp[:-1]]), True,
              {"ids": sorted(ids), "min_fps": F.motion_fps})


def q_drops_before(F, rng):
    gt = F.tops[0]["drop"] + 1
    return _q(4, "drops_before", f"How many grains have been dropped (counting the one that sets it off) when the "
              f"first {F.u} topples?", gt, _ints(rng, gt, 1), True, {"min_fps": F.motion_fps})


def q_last_topple_cell(F, rng):
    last = F.tops[-1]["f"]
    wave = [t for t in F.tops if t["f"] == last]
    if len(wave) != 1:
        return None
    c = wave[0]["cell"]
    return _q(4, "last_topple_cell", f"Which {F.u} is the last to {TOP} in the video?{F.conv}", F.name(c),
              F.cell_opts(rng, c, F.neighbours(c) + list(F.first_top)), True, {"min_fps": F.fps_top})


def q_topples_between(F, rng):
    ks = []
    for k in range(sim.tenths(EDGE_F), sim.tenths(F.n - 1 - EDGE_F)):
        a, b = sim.frames_showing(k)
        if all(abs(t["f"] - f) > WIN for t in F.tops for f in (a, b)):
            ks.append(k)
    if len(ks) < 2:
        return None
    k1, k2 = sorted(rng.sample(ks, 2))
    if k2 - k1 < 10:
        return None
    f1, f2 = sim.frames_showing(k1)[1], sim.frames_showing(k2)[0]
    gt = sum(1 for t in F.tops if f1 < t["f"] < f2)
    if gt == 0 and rng.random() < 0.7:
        return None
    return _q(4, "topples_between", f"How many topples happen between {sim.clock_text(k1)} and "
              f"{sim.clock_text(k2)} on the clock (counting every {F.u} that passes its grains on)?", gt,
              _ints(rng, gt, 0), True, {"t1": k1, "t2": k2, "min_fps": F.fps_top})


def q_cascade_lost(F, rng):
    i = rng.randrange(len(F.aval))
    gt = F.aval[i]["lost"]
    where = "into the hole" if F.sc == "ring" else "off the board"
    return _q(4, "cascade_lost", f"How many grains go {where} in the cascade set off by {grain_name(i)}?", gt,
              _ints(rng, gt, 0), True, {"i": i, "min_fps": max(fps_for(F.M), F.fps_drops)})


def q_count_after(F, rng):
    i = rng.randrange(len(F.aval))
    a = F.aval[i]
    if a["topples"] == 0:
        return None
    f = F.appear[i + 1] - 1 if i + 1 < len(F.lands) else F.n - 1          # before the next grain shows up
    c = rng.choice(sorted(set(a["cells"]) | {x for t in a["cells"] for x in F.neighbours(t)}))
    if f - a["end"] < 3:
        return None
    gt = int(F.C[f, c])
    if F.cov[f, c]:
        return None
    return _q(4, "count_after", f"Once the cascade set off by {grain_name(i)} has settled, how many grains are in "
              f"{F.name(c)}?{F.conv}", gt, _ints(rng, gt, 0), True,
              {"i": i, "cell": _p(F, c), "min_fps": max(F.motion_fps, fps_for(f - a["end"] + 1))})


# ------------------------------------------------------------------ L5 counterfactual (re-simulated)

def _cf(F, init, drops):
    """Re-simulate the drops (cell list) from init; returns per-drop (topples, lost) and per-cell topples."""
    h = list(init)
    per, cells = [], {c: 0 for c in F.live}
    for c in drops:
        h[c] += 1
        waves = sim.relax_waves(h, F.kinds, F.out, F.T)
        lost = 0
        for w in waves:
            for x in w:
                cells[x] += 1
                lost += sum(1 for t in F.out[x] if t < 0 or F.kinds[t] == "hole")
        per.append((sum(len(w) for w in waves), lost))
    return per, cells, h


def _rule(F):
    return " " + sim.rule_text(F.s)


def q_cf_skip_total(F, rng):
    i = rng.randrange(len(F.lands))
    drops = [e["cell"] for k, e in enumerate(F.lands) if k != i]
    per, _, _ = _cf(F, F.s["init"], drops)
    gt = sum(p[0] for p in per)
    return _q(5, "cf_skip_total", f"Suppose {grain_name(i)} had never been dropped, and everything else stayed the "
              f"same.{_rule(F)} How many topples would there be in total?", gt, _ints(rng, gt, 0), True,
              {"i": i, "min_fps": F.fps_drops})


def q_cf_skip_cell(F, rng):
    i = rng.randrange(len(F.lands))
    drops = [e["cell"] for k, e in enumerate(F.lands) if k != i]
    _, cells, _ = _cf(F, F.s["init"], drops)
    cand = [c for c in F.live if F.n_top[c]] if rng.random() < 0.7 else F.live
    c = rng.choice(cand)
    gt = "yes" if cells[c] else "no"
    return _q(5, "cf_skip_cell", f"Suppose {grain_name(i)} had never been dropped, and everything else stayed the "
              f"same.{_rule(F)} Would {F.name(c)} topple at any point?{F.conv}", gt, _yn(rng, gt), True,
              {"i": i, "cell": _p(F, c), "min_fps": F.fps_drops})


def q_cf_move(F, rng):
    i = rng.randrange(len(F.lands))
    c0 = F.lands[i]["cell"]
    c = rng.choice([x for x in F.live if x != c0])
    drops = [e["cell"] for e in F.lands]
    drops[i] = c
    per, _, _ = _cf(F, F.s["init"], drops)
    gt = per[i][0]
    return _q(5, "cf_move", f"Suppose {grain_name(i)} had landed on {F.name(c)} instead, and everything before it "
              f"stayed the same.{_rule(F)} How many topples would that grain set off?{F.conv}", gt,
              _ints(rng, gt, 0), True, {"i": i, "cell": _p(F, c), "min_fps": F.fps_drops})


def q_cf_extra_end(F, rng):
    c = rng.choice(F.live)
    h = [int(v) for v in F.C[-1]]
    h[c] += 1
    waves = sim.relax_waves(h, F.kinds, F.out, F.T)
    gt = sum(len(w) for w in waves)
    return _q(5, "cf_extra_end", f"Suppose one more grain were dropped onto {F.name(c)} after the end of the video."
              f"{_rule(F)} How many topples would follow?{F.conv}", gt, _ints(rng, gt, 0), False,
              {"cell": _p(F, c), "min_fps": 1})


def q_cf_init(F, rng):
    d = rng.choice([-1, 1])
    xs = [c for c in F.live if F.s["init"][c] + d >= 0]
    c = rng.choice(xs)
    init = list(F.s["init"])
    init[c] += d
    h = list(init)
    pre = sim.relax_waves(h, F.kinds, F.out, F.T)          # an extra grain may already topple at the start
    per, _, _ = _cf(F, h, [e["cell"] for e in F.lands])
    gt = sum(len(w) for w in pre) + sum(p[0] for p in per)
    how = "one grain fewer" if d < 0 else "one grain more"
    return _q(5, "cf_init", f"Suppose {F.name(c)} had started with {how}, and the same grains were dropped in the "
              f"same places.{_rule(F)} How many topples would there be in total?{F.conv}", gt, _ints(rng, gt, 0), True,
              {"cell": _p(F, c), "d": d, "min_fps": F.fps_drops})


def q_cf_lost(F, rng):
    i = rng.randrange(len(F.lands))
    drops = [e["cell"] for k, e in enumerate(F.lands) if k != i]
    per, _, _ = _cf(F, F.s["init"], drops)
    gt = sum(p[1] for p in per)
    where = "into the hole" if F.sc == "ring" else "off the board"
    return _q(5, "cf_lost", f"Suppose {grain_name(i)} had never been dropped, and everything else stayed the same."
              f"{_rule(F)} How many grains would go {where} in total?", gt, _ints(rng, gt, 0), True,
              {"i": i, "min_fps": F.fps_drops})


TEMPLATES = {
    1: [q_n_drops, q_n_cells, q_start_count, q_start_total, q_end_count, q_first_drop_cell, q_first_topple_cell,
        q_ever_topples, q_most_end],
    2: [q_topple_order, q_more_topples, q_bigger_cascade, q_trigger, q_start_vs_end, q_more_end, q_receives,
        q_lost_side],
    3: [q_topple_time, q_land_time, q_total_topples, q_cell_topples, q_count_at, q_lost_total, q_end_total,
        q_cascade_span],
    4: [q_cascade_size, q_n_toppled_cells, q_biggest, q_drops_before, q_last_topple_cell, q_topples_between,
        q_cascade_lost, q_count_after],
    5: [q_cf_skip_total, q_cf_skip_cell, q_cf_move, q_cf_extra_end, q_cf_init, q_cf_lost],
}


def _sig(q):
    return q["derivation_key"], tuple(sorted((k, str(v)) for k, v in q["params"].items() if k != "min_fps"))


def pools(F, rng) -> dict:
    """Distinct candidate questions per level (several draws per template)."""
    out = {}
    for lv in range(1, 6):
        got = {}
        tpl = TEMPLATES[lv][:]
        rng.shuffle(tpl)
        for _ in range(4):
            for fn in tpl:
                q = fn(F, rng)
                if q and _sig(q) not in got:
                    got[_sig(q)] = q
        out[lv] = list(got.values())
    return out


def _pick(pool, want, rng, motion_first=True, static_ok=False):
    """`want` questions: different templates first. Above L1, motion questions
    come first (one static question may compete with them when `static_ok`)
    and at most one static (single-frame) question unless nothing else is left."""
    chosen, keys = [], {}
    rest = pool[:]
    rng.shuffle(rest)
    while len(chosen) < want:
        static_used = sum(1 for q in chosen if not q["requires_motion"])
        rest.sort(key=lambda q: (keys.get(q["derivation_key"], 0),
                                 motion_first and (not q["requires_motion"]) and static_used >= 1,
                                 motion_first and not static_ok and not q["requires_motion"]))
        q = rest.pop(0)
        chosen.append(q)
        keys[q["derivation_key"]] = keys.get(q["derivation_key"], 0) + 1
    return chosen


def build(scene: dict, rng: random.Random) -> list[dict]:
    F = Facts(scene)
    P = pools(F, rng)
    avail = [len(P[lv]) for lv in range(1, 6)]
    combos = [c for c in L._count_combos() if sum(c) == L.TEN_TOTAL and all(a >= k for a, k in zip(avail, c))]
    if not combos:
        raise RuntimeError(f"not enough questions per level {avail} for {scene['scenario']}")
    counts = rng.choice(combos)
    out = []
    for lv, want in zip(range(1, 6), counts):
        out += _pick(P[lv], want, rng, motion_first=lv > 1, static_ok=rng.random() < 0.3)
    return L.number(out)
