"""10-question ladder for conveyor_belt_sort (common/ladder5.py, LADDER = "ten").

Objects are named by pattern, colour and shape ("the striped red star",
unique in each clip), bins by position ("the upper bin"), moments by the
clock. Ground truth comes from sim.py's exact frames (and sim.simulate for
the what-ifs); answers.py re-derives every answer from the drawn track.
Every question stores derivation_key + params (+ min_fps).
"""
from __future__ import annotations
import math
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
ORD = ["first", "second", "third", "fourth", "fifth", "sixth", "seventh", "eighth", "ninth"]
BIN = {"upper": "the upper bin", "middle": "the middle bin", "lower": "the lower bin"}
STATES = ["still inside the tunnel", "riding on the belt", "sliding toward a bin", "already resting in a bin"]
SHAPE_PL = {"circle": "circles", "square": "squares", "triangle": "triangles", "star": "stars"}
MARGIN = 8          # frames between events a question compares (1/3 s)
WIN = 7             # frames of calm either side of a clock reading a question names
# correct-letter weights per option count, so A-D come out even over the dataset
WEIGHTS = {2: (1, 1), 3: (1, 1, 1), 4: (0.204, 0.204, 0.279, 0.313)}


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
    if opts is None:
        return None
    o, letter = opts
    return L.question(level, key, text, gt, [str(x) for x in o], letter, motion, params)


def _ints(rng, gt, lo=0, hi=99):
    pool = [v for v in (gt + 1, gt - 1, gt + 2, gt - 2, gt + 3) if lo <= v <= hi]
    d = pool[:2] + [rng.choice(pool[2:])] if len(pool) >= 3 else None
    return d and _place(rng, gt, d)


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


def _bin_opts(rng, F, gt):
    return _place(rng, BIN[gt], [BIN[b] for b in F.bins if b != gt])


class Facts:
    def __init__(self, scene):
        self.s = scene
        self.objs = scene["objects"]
        self.n = scene["n_frames"]
        self.v = Fr(scene["v2"], 2)
        self.bins = sim.bin_names(scene)
        self.rule = scene["rule"]
        self.attr = self.rule.get("attr")
        self.e = [o["keys"][1][0] for o in self.objs]
        self.gate = [o["keys"][2][0] for o in self.objs]
        self.land = [o["keys"][4][0] for o in self.objs]
        self.fz = [math.ceil(sim.ext(o) / self.v) for o in self.objs]
        self.done = [i for i, o in enumerate(self.objs) if not o["pending"]]
        self.pend = [i for i, o in enumerate(self.objs) if o["pending"]]
        self.nm = [sim.name(o) for o in self.objs]
        self.sizes = sorted({o["size"] for o in self.objs}, key=lambda z: sim.SIZE_SCALE[z])
        self.turns = self.rule["kind"] == "alternate"
        self.gate_gap = min(b - a for a, b in zip(self.gate, self.gate[1:]))

    def binof(self, i):
        return self.objs[i]["bin"]

    def resting(self, b, f):
        return sum(1 for i in self.done if self.binof(i) == b and self.land[i] <= f)

    def state(self, j, f):
        """(state index, start, end) of object j at frame f, or None in a blurred stretch."""
        e, fz, g, l = self.e[j], self.fz[j], self.gate[j], self.land[j]
        spans = [(0, -10**6, e - fz), (1, e + fz, g - 2), (2, g + 2, l), (3, l, 10**6)]
        for k, a, b in spans:
            if a <= f < b:
                return k, a, b
        return None

    def calm_runs(self, changes):
        """Clock readings k whose frames +- WIN contain none of `changes`
        (frames at which the asked quantity changes): [(k, run length)]."""
        out = []
        ch = sorted(changes)
        for k in range(1, 100):
            lo, hi = sim.frames_showing(k)
            a, b = lo - WIN, hi + WIN
            if a < sim.EDGE or b > self.n - 1 - sim.EDGE:
                continue
            if any(a <= c <= b for c in ch):
                continue
            before = [c for c in ch if c < a]
            after = [c for c in ch if c > b]
            r0 = max(before) if before else 0
            r1 = min(after) if after else self.n
            out.append((k, r1 - r0))
        return out

    def belt_changes(self):
        ch = []
        for i in range(len(self.objs)):
            ch += list(range(self.e[i] - self.fz[i], self.e[i] + self.fz[i] + 1))
            if i in self.done:
                ch += list(range(self.gate[i] - 2, self.gate[i] + 3))
        return ch

    def on_belt(self, f):
        return sum(1 for i in range(len(self.objs)) if self.e[i] <= f < self.gate[i])


def _done_pair(F, rng):
    if len(F.done) < 2:
        return None
    return rng.sample(F.done, 2)


# ------------------------------------------------------------------ L1 observation

def q_n_objects(F, rng):
    n = len(F.objs)
    return _q(1, "n_objects", "How many objects come out of the tunnel during the video?", n,
              _ints(rng, n, 1), False, {"min_fps": 1})


def q_in_bin_end(F, rng):
    b = rng.choice(F.bins)
    ins = [i for i in F.done if F.binof(i) == b]
    outs = [i for i in range(len(F.objs)) if i not in ins]
    i = rng.choice(ins)
    d = _pick_d(rng, F.nm[i], [F.nm[j] for j in outs])
    return d and _q(1, "in_bin_end", f"Which of these objects is resting in {BIN[b]} when the video ends?", F.nm[i],
                    _place(rng, F.nm[i], d), False, {"bin": b, "among": sorted(d + [F.nm[i]]), "min_fps": 1})


def q_count_end(F, rng):
    b = rng.choice(F.bins)
    n = F.resting(b, F.n - 1)
    return _q(1, "count_end", f"How many objects are resting in {BIN[b]} when the video ends?", n,
              _ints(rng, n), False, {"bin": b, "min_fps": 1})


def q_on_belt_end(F, rng):
    p = rng.choice(F.pend)
    d = _pick_d(rng, F.nm[p], [F.nm[j] for j in F.done])
    return d and _q(1, "on_belt_end", "Which of these objects is still on the belt when the video ends?", F.nm[p],
                    _place(rng, F.nm[p], d), False, {"among": sorted(d + [F.nm[p]]), "min_fps": 1})


def q_count_kind(F, rng):
    a = rng.choice(["colour", "shape"])
    v = rng.choice(sorted({o[a] for o in F.objs}))
    n = sum(1 for o in F.objs if o[a] == v)
    word = f"{v} objects" if a == "colour" else SHAPE_PL[v]
    return _q(1, "count_kind", f"How many {word} come out of the tunnel during the video?", n, _ints(rng, n, 1), False,
              {"attr": a, "value": v, "min_fps": 1})


def q_first_out(F, rng):
    d = _pick_d(rng, F.nm[0], [F.nm[j] for j in range(1, len(F.objs))])
    return d and _q(1, "first_out", "Which object comes out of the tunnel first?", F.nm[0],
                    _place(rng, F.nm[0], d), True, {"min_fps": fps_for(F.e[1] - F.e[0])})


# ------------------------------------------------------------------ L2 relational

def q_same_bin(F, rng):
    pr = _done_pair(F, rng)
    if not pr:
        return None
    i, j = pr
    gt = "yes" if F.binof(i) == F.binof(j) else "no"
    return _q(2, "same_bin", f"Do {F.nm[i]} and {F.nm[j]} end up in the same bin?", gt,
              _place(rng, gt, ["no" if gt == "yes" else "yes"]), False, {"x": F.nm[i], "y": F.nm[j], "min_fps": 1})


def q_land_order(F, rng):
    pr = _done_pair(F, rng)
    if not pr:
        return None
    i, j = pr
    gap = abs(F.land[i] - F.land[j])
    if gap < MARGIN:
        return None
    gt = F.nm[i] if F.land[i] < F.land[j] else F.nm[j]
    return _q(2, "land_order", f"Which comes to rest in its bin first: {F.nm[i]} or {F.nm[j]}?", gt,
              _place(rng, gt, [F.nm[j] if gt == F.nm[i] else F.nm[i], "they come to rest at the same moment"]),
              True, {"x": F.nm[i], "y": F.nm[j], "min_fps": fps_for(gap)})


def _next_window(F, i):
    gaps = [F.e[i + 1] - F.e[i]]
    if i + 2 < len(F.objs):
        gaps.append(F.e[i + 2] - F.e[i + 1])
    return min(gaps)


def q_next_out(F, rng):
    i = rng.randrange(len(F.objs) - 1)
    d = _pick_d(rng, F.nm[i + 1], [F.nm[j] for j in range(len(F.objs)) if j not in (i, i + 1)])
    return d and _q(2, "next_out", f"Which object comes out of the tunnel right after {F.nm[i]}?", F.nm[i + 1],
                    _place(rng, F.nm[i + 1], d), True, {"x": F.nm[i], "min_fps": fps_for(_next_window(F, i))})


def q_bin_after(F, rng):
    c = [i for i in F.done if i + 1 in F.done]
    if not c:
        return None
    i = rng.choice(c)
    gt = F.binof(i + 1)
    return _q(2, "bin_after", f"Which bin does the object that comes out of the tunnel right after {F.nm[i]} end up "
              f"in?", BIN[gt], _bin_opts(rng, F, gt), True, {"x": F.nm[i], "min_fps": fps_for(_next_window(F, i))})


def value_word(attr, v) -> str:
    if attr == "size":
        return {"small": "small objects", "medium": "medium-sized objects", "large": "large objects"}[v]
    if attr == "colour":
        return f"{v} objects"
    return SHAPE_PL[v]


def q_rule_value(F, rng):
    if F.turns:
        return None
    v = rng.choice(sorted({F.objs[i][F.attr] for i in F.done}))
    gt = F.rule["map"][v]
    return _q(2, "rule_value", f"Judging by the objects that are already sorted, which bin do {value_word(F.attr, v)} "
              f"go to?", BIN[gt], _bin_opts(rng, F, gt), False, {"attr": F.attr, "value": v, "min_fps": 1})


def q_state_at_land(F, rng):
    i = rng.choice(F.done)
    j = rng.choice([k for k in range(len(F.objs)) if k != i])
    st = F.state(j, F.land[i])
    if st is None:
        return None
    k, a, b = st
    w = min(F.land[i] - a, b - F.land[i])
    if w < 6:
        return None
    gt = STATES[k]
    return _q(2, "state_at_land", f"At the moment {F.nm[i]} comes to rest in its bin, where is {F.nm[j]}?", gt,
              _place(rng, gt, [s for s in STATES if s != gt]), True, {"x": F.nm[i], "y": F.nm[j],
                                                                      "min_fps": fps_for(w)})


def q_first_bin(F, rng):
    order = sorted(F.done, key=lambda i: F.land[i])
    if len(order) < 2 or F.land[order[1]] - F.land[order[0]] < MARGIN:
        return None
    gt = F.binof(order[0])
    return _q(2, "first_bin", "Which bin does the first object to come to rest end up in?", BIN[gt],
              _bin_opts(rng, F, gt), True, {"min_fps": fps_for(F.land[order[1]] - F.land[order[0]])})


# ------------------------------------------------------------------ L3 measurement

def q_land_time(F, rng):
    i = rng.choice(F.done)
    return _q(3, "land_time", f"At what clock time does {F.nm[i]} come to rest in its bin?", clock(F.land[i]),
              _clock_opts(rng, F.land[i]), True, {"x": F.nm[i], "min_fps": 4})


def q_gate_time(F, rng):
    i = rng.choice(F.done)
    return _q(3, "gate_time", f"At what clock time does {F.nm[i]} reach the end of the belt and turn into a chute?",
              clock(F.gate[i]), _clock_opts(rng, F.gate[i]), True, {"x": F.nm[i], "min_fps": 4})


def q_count_at(F, rng):
    b = rng.choice(F.bins)
    runs = [(k, r) for k, r in F.calm_runs([F.land[i] for i in F.done if F.binof(i) == b])
            if 0 < F.resting(b, sim.frames_showing(k)[0]) or rng.random() < 0.2]
    if not runs:
        return None
    k, r = rng.choice(runs)
    n = F.resting(b, sim.frames_showing(k)[0])
    return _q(3, "count_at", f"How many objects are resting in {BIN[b]} when the clock shows {sim.clock_text(k)}?", n,
              _ints(rng, n), True, {"bin": b, "k": k, "min_fps": fps_for(r)})


def q_belt_count_at(F, rng):
    runs = F.calm_runs(F.belt_changes())
    if not runs:
        return None
    k, r = rng.choice(runs)
    n = F.on_belt(sim.frames_showing(k)[0])
    return _q(3, "belt_count_at", f"How many objects are on the belt (out of the tunnel and not yet at the end of the "
              f"belt) when the clock shows {sim.clock_text(k)}?", n, _ints(rng, n), True, {"k": k, "min_fps": fps_for(r)})


def q_belt_time(F, rng):
    gt = half_s(F.s["frames"]["belt"])
    return gt and _q(3, "belt_time", "About how many seconds does an object take to ride from the tunnel mouth to the "
                     "end of the belt?", gt, _half_opts(rng, gt), True, {"min_fps": 4})


def q_entry_gap(F, rng):
    i = rng.randrange(len(F.objs) - 1)
    if abs(F.fz[i] - F.fz[i + 1]) > 2:
        return None
    gt = half_s(F.e[i + 1] - F.e[i])
    return gt and _q(3, "entry_gap", f"About how many seconds pass between {F.nm[i]} and {F.nm[i + 1]} coming out of "
                     f"the tunnel?", gt, _half_opts(rng, gt), True, {"x": F.nm[i], "y": F.nm[i + 1], "min_fps": 4})


def q_ride_time(F, rng):
    i = rng.choice(F.done)
    gt = half_s(F.land[i] - F.gate[i])
    return gt and _q(3, "ride_time", f"About how many seconds does {F.nm[i]} take from the end of the belt to coming "
                     f"to rest in its bin?", gt, _half_opts(rng, gt), True, {"x": F.nm[i], "min_fps": 4})


# ------------------------------------------------------------------ L4 integration

def _rule_fps(F):
    return fps_for(F.gate_gap) if F.turns else 1


def q_pending_bin(F, rng):
    p = rng.choice(F.pend)
    gt = F.binof(p)
    return _q(4, "pending_bin", f"{F.nm[p][0].upper() + F.nm[p][1:]} is still on the belt when the video ends. Which "
              f"bin will it end up in?", BIN[gt], _bin_opts(rng, F, gt), F.turns, {"x": F.nm[p], "min_fps": _rule_fps(F)})


def q_final_count(F, rng):
    b = rng.choice(F.bins)
    n = sum(1 for o in F.objs if o["bin"] == b)
    return _q(4, "final_count", f"Once every object, including those still on the belt, has been sorted, how many "
              f"objects will be in {BIN[b]}?", n, _ints(rng, n), F.turns, {"bin": b, "min_fps": _rule_fps(F)})


def q_pending_land(F, rng):
    p = rng.choice(F.pend)
    return _q(4, "pending_land", f"If the video kept going, at what clock time would {F.nm[p]} come to rest in its "
              f"bin?", clock(F.land[p]), _clock_opts(rng, F.land[p]), True, {"x": F.nm[p], "min_fps": 4})


def q_landed_between(F, rng):
    runs = F.calm_runs([F.land[i] for i in F.done])
    if len(runs) < 2:
        return None
    for _ in range(6):
        (k1, r1), (k2, r2) = sorted(rng.sample(runs, 2))
        f1, f2 = sim.frames_showing(k1)[0], sim.frames_showing(k2)[0]
        n = sum(1 for i in F.done if f1 < F.land[i] <= f2)
        if k2 - k1 >= 15 and n >= 1:
            return _q(4, "landed_between", f"How many objects come to rest in any bin between {sim.clock_text(k1)} and "
                      f"{sim.clock_text(k2)} on the clock?", n, _ints(rng, n), True,
                      {"k1": k1, "k2": k2, "min_fps": fps_for(min(r1, r2))})
    return None


def q_kth_land(F, rng):
    order = sorted(F.done, key=lambda i: F.land[i])
    k = rng.randrange(len(order))
    gaps = [F.land[order[k]] - F.land[order[k - 1]]] if k > 0 else []
    if k + 1 < len(order):
        gaps.append(F.land[order[k + 1]] - F.land[order[k]])
    if not gaps or min(gaps) < MARGIN:
        return None
    i = order[k]
    d = _pick_d(rng, F.nm[i], [F.nm[j] for j in F.done])
    return d and _q(4, "kth_land", f"Which object is the {ORD[k]} to come to rest in a bin?", F.nm[i],
                    _place(rng, F.nm[i], d), True, {"k": k + 1, "min_fps": fps_for(min(gaps))})


def q_last_land_bin(F, rng):
    order = sorted(F.done, key=lambda i: F.land[i])
    gap = F.land[order[-1]] - F.land[order[-2]]
    if gap < MARGIN:
        return None
    gt = F.binof(order[-1])
    return _q(4, "last_land_bin", "Which bin receives the last object that comes to rest during the video?", BIN[gt],
              _bin_opts(rng, F, gt), True, {"min_fps": fps_for(gap)})


# ------------------------------------------------------------------ L5 counterfactual (re-simulated)

def _cf_end_margin(F, res):
    lands = [r[1] for r in res if r is not None]
    if any(x is None for x in lands):
        return None
    return min(abs(x - (F.n - 1)) for x in lands)


def _reversed(F):
    if F.turns:
        return {"kind": "alternate", "start": "lower" if F.rule["start"] == "upper" else "upper"}
    return {"kind": "attr", "attr": F.attr, "map": {k: ("lower" if v == "upper" else "upper") for k, v in F.rule["map"].items()}}


def q_cf_reverse(F, rng):
    if len(F.bins) != 2:
        return None
    res = sim.simulate(F.s, rule=_reversed(F))
    m = _cf_end_margin(F, res)
    if m is None or m < MARGIN:
        return None
    b = rng.choice(F.bins)
    n = sum(1 for g, land, bb, k in res if bb == b and land <= F.n - 1)
    naive = sum(1 for i in F.done if F.binof(i) != b)
    return _q(5, "cf_reverse", f"If the gate had sent every object to the other bin instead, how many objects would be "
              f"resting in {BIN[b]} when the video ends?", n, _ints(rng, n), n != naive,
              {"bin": b, "min_fps": 1 if n == naive else fps_for(m)})


def q_cf_reverse_time(F, rng):
    if len(F.bins) != 2:
        return None
    res = sim.simulate(F.s, rule=_reversed(F))
    x = rng.choice(range(len(F.objs)))
    if res[x][1] is None:
        return None
    f = math.ceil(res[x][1])
    return _q(5, "cf_reverse_time", f"If the gate had sent every object to the other bin instead, at what clock time "
              f"would {F.nm[x]} come to rest?", clock(f), _clock_opts(rng, f), True, {"x": F.nm[x], "min_fps": 4})


def q_cf_fast(F, rng):
    res = sim.simulate(F.s, speed=Fr(2))
    m = _cf_end_margin(F, res)
    if m is None or m < MARGIN:
        return None
    n = sum(1 for g, land, b, k in res if land <= F.n - 1)
    return _q(5, "cf_fast", "If the belt, the chutes and the trays all moved twice as fast (with the objects coming out "
              "of the tunnel at the same moments), how many objects would be resting in the bins when the video ends?",
              n, _ints(rng, n), True, {"min_fps": fps_for(m)})


def q_cf_remove_time(F, rng):
    x = rng.choice(F.done)
    later = [j for j in range(x + 1, len(F.objs)) if F.turns or F.binof(j) == F.binof(x)]
    if not later:
        later = list(range(x + 1, len(F.objs)))
    if not later:
        return None
    y = rng.choice(later)
    res = sim.simulate(F.s, drop=(x,))
    if res[y][1] is None:
        return None
    f = math.ceil(res[y][1])
    return _q(5, "cf_remove_time", f"If {F.nm[x]} had never come onto the belt, at what clock time would {F.nm[y]} "
              f"come to rest in its bin?", clock(f), _clock_opts(rng, f), True,
              {"x": F.nm[x], "y": F.nm[y], "min_fps": 4})


def q_cf_remove_bin(F, rng):
    x = rng.choice(F.done)
    later = list(range(x + 1, len(F.objs)))
    if not later:
        return None
    y = rng.choice(later)
    res = sim.simulate(F.s, drop=(x,))
    gt = res[y][2]
    return _q(5, "cf_remove_bin", f"If {F.nm[x]} had never come onto the belt, which bin would {F.nm[y]} end up in?",
              BIN[gt], _bin_opts(rng, F, gt), F.turns, {"x": F.nm[x], "y": F.nm[y], "min_fps": _rule_fps(F)})


def q_cf_new(F, rng):
    o = {"size": rng.choice(F.sizes), "colour": rng.choice(sorted(sim.COLOURS)), "shape": rng.choice(sim.SHAPES),
         "pattern": rng.choice(sim.PATTERNS)}
    if not F.turns:
        o[F.attr] = rng.choice(sorted({F.objs[i][F.attr] for i in F.done}))
        gt = F.rule["map"][o[F.attr]]
    else:
        last = F.objs[-1]["bin"]
        gt = "lower" if last == "upper" else "upper"
    size = f"{o['size']} " if len(F.sizes) > 1 else ""
    desc = f"a {size}{o['pattern']} {o['colour']} {o['shape']}"
    return _q(5, "cf_new", f"If one more object, {desc}, came out of the tunnel after the last one, which bin would it "
              f"go to?", BIN[gt], _bin_opts(rng, F, gt), F.turns,
              {"size": o["size"] if size else "", "colour": o["colour"], "shape": o["shape"], "pattern": o["pattern"],
               "min_fps": _rule_fps(F)})


def q_cf_delay(F, rng):
    x = rng.choice(F.done)
    dt = rng.choice([12, 24])
    ent = list(F.e)
    ent[x] += dt
    if any(j != x and abs(ent[j] - ent[x]) < F.s["frames"]["slot"] for j in range(len(ent))):
        return None
    res = sim.simulate(F.s, entries=ent)
    if res[x][1] is None:
        return None
    f = math.ceil(res[x][1])
    return _q(5, "cf_delay", f"If {F.nm[x]} had come out of the tunnel {dt / sim.FPS:g} s later, at what clock time "
              f"would it come to rest in its bin?", clock(f), _clock_opts(rng, f), True,
              {"x": F.nm[x], "dt": dt, "min_fps": 4})


def q_cf_swap(F, rng):
    if F.turns:
        return None
    p = rng.choice(F.pend)
    cur = F.objs[p][F.attr]
    others = [v for v in sorted({F.objs[i][F.attr] for i in F.done}) if F.rule["map"][v] != F.rule["map"][cur]]
    if not others:
        return None
    v = rng.choice(others)
    gt = F.rule["map"][v]
    word = {"size": f"{v} instead of {cur}", "colour": f"{v} instead of {cur}", "shape": f"a {v} instead of a {cur}"}
    return _q(5, "cf_swap", f"If {F.nm[p]} (still on the belt at the end) were {word[F.attr]}, which bin would it go "
              f"to?", BIN[gt], _bin_opts(rng, F, gt), False, {"x": F.nm[p], "value": v, "min_fps": 1})


TEMPLATES = {
    1: [q_n_objects, q_in_bin_end, q_count_end, q_on_belt_end, q_first_out, q_count_kind],
    2: [q_same_bin, q_land_order, q_next_out, q_bin_after, q_rule_value, q_state_at_land, q_first_bin],
    3: [q_land_time, q_gate_time, q_count_at, q_belt_count_at, q_belt_time, q_entry_gap, q_ride_time],
    4: [q_pending_bin, q_final_count, q_pending_land, q_landed_between, q_kth_land, q_last_land_bin],
    5: [q_cf_reverse, q_cf_reverse_time, q_cf_fast, q_cf_remove_time, q_cf_remove_bin, q_cf_new, q_cf_delay, q_cf_swap],
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
