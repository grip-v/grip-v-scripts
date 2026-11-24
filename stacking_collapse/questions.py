"""10-question ladder for stacking_collapse (common/ladder5.py, LADDER = "ten").

Blocks are named by pattern and colour ("the striped red block", unique in
each clip), pedestals by side ("the left pedestal"), moments by the clock,
widths in grid squares. Ground truth comes from sim.py (event frames from
sim.timeline, what-ifs re-simulated with sim.play); answers.py re-derives
every answer from the drawn track. Every question stores derivation_key +
params (+ min_fps).
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
ORD = ["first", "second", "third", "fourth", "fifth", "sixth", "seventh", "eighth", "ninth", "tenth"]
SIDES = ["to the left", "to the right"]
MARGIN = 8          # frames between events a question compares (1/3 s)
WIN = 7             # frames of calm either side of a clock reading a question names
# correct-letter weights per option count, so A-D come out even over the dataset
WEIGHTS = {2: (1, 1), 3: (1, 1, 1), 4: (0.165, 0.165, 0.334, 0.336)}


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


def _ints(rng, gt, lo=0, hi=99):
    pool = [v for v in (gt + 1, gt - 1, gt + 2, gt - 2, gt + 3, gt - 3) if lo <= v <= hi]
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


class Facts:
    def __init__(self, scene):
        self.s = scene
        self.sc = scene["scenario"]
        self.twin = self.sc == "twin"
        self.plank = self.sc == "plank"
        self.pull = self.sc == "pull_out"
        self.n = scene["n_frames"]
        self.bl = scene["blocks"]
        self.nm = [sim.name(b) for b in self.bl]
        self.tl = sim.timeline(scene)
        ev = self.tl["events"]
        self.lands = [e for e in ev if e["kind"] == "land"]
        self.pulls = [e for e in ev if e["kind"] == "pull"]
        self.falls = [e for e in ev if e["kind"] == "fall"]
        self.res = sim.play(scene)
        self.init = [j for j, _ in scene["init"]]
        self.start = {j: 0 for j in self.init}
        self.base_of = {j: 0 for j in self.init}
        for e in self.lands:
            self.start[e["block"]] = e["f"]
            self.base_of[e["block"]] = e["base"]
        self.leave = {}
        for e in self.pulls:
            self.leave[e["block"]] = e["f"]
        for e in self.falls:
            for j in e["blocks"]:
                self.leave[j] = e["f"]
        self.fell = {j for e in self.falls for j in e["blocks"]}
        self.steps = [e["f"] for e in ev if e["kind"] in ("land", "pull")]
        blur = []
        for e in self.lands:
            blur.append((e["f"] - 3, e["f"] + 1))
        for e in self.pulls:
            blur.append((e["f"] - 1, (e["settle"] or e["f"]) + 1))
        for e in self.falls:
            blur.append((e["f"] - 1, e["f"] + scene["tip"]["t1"]))
        self.blur = sorted(blur)

    def end(self, j):
        return self.leave.get(j, self.n)

    def count(self, f, base=None):
        return sum(1 for j in self.start if self.start[j] <= f < self.end(j)
                   and (base is None or self.base_of[j] == base))

    def calm(self, f0, f1):
        """Frames of calm around [f0, f1] (distance to the nearest blurred stretch), 0 if too close."""
        lo, hi = -10 ** 6, 10 ** 6
        for a, b in self.blur:
            if b < f0:
                lo = max(lo, b)
            elif a > f1:
                hi = min(hi, a)
            else:
                return 0
        if f0 - lo < WIN or hi - f1 < WIN:
            return 0
        return min(hi, self.n - 1) - max(lo, 0)

    def where(self, base=None):
        if self.plank:
            return "on the plank"
        if self.twin:
            return "on the two pedestals together" if base is None else f"on {sim.base_name(self.s, base)}"
        return "on the pedestal"

    def falls_text(self, k):
        if self.plank:
            return "the plank tips"
        if len(self.falls) == 1:
            return "blocks fall"
        return f"blocks fall for the {ORD[k]} time"

    def seen(self, j):
        """Frames in which block j is wholly inside the frame."""
        S, b = self.s["S"], self.bl[j]
        k = 0
        for f in range(self.n):
            p = self.tl["pose"][f][j]
            if p is not None:
                x0, y0, x1, y1 = sim._bbox(sim.rect(p[0], p[1], p[2], b["w"] * S, b["h"] * S))
                k += x0 >= 0 and y0 >= 0 and x1 <= sim.W and y1 <= sim.H
        return k

    def visible_min(self):
        return min(self.seen(j) for j in self.start)

    def fall_fps(self, e):
        return fps_for(e["gone"] - e["f"])

    def step_gaps(self):
        g = [b - a for a, b in zip(self.steps, self.steps[1:])]
        return min(g) if g else self.n


# ---------------------------------------------------------------- L1 observation

def q_n_dropped(F, rng):
    if F.pull:
        return None
    n = len(F.lands)
    return _q(1, "n_dropped", "How many blocks are dropped in this clip?", n, _ints(rng, n, 1), True,
              {"min_fps": fps_for(F.visible_min())})


def q_n_init(F, rng):
    if not F.pull:
        return None
    n = len(F.init)
    return _q(1, "n_init", "How many blocks does the tower have at the start of the clip?", n, _ints(rng, n, 2),
              False, {"min_fps": 1})


def q_n_pulled(F, rng):
    if not F.pull:
        return None
    n = len(F.pulls)
    return _q(1, "n_pulled", "How many blocks are pulled out of the tower?", n, _ints(rng, n, 1), True,
              {"min_fps": fps_for(F.step_gaps())})


def q_n_falls(F, rng):
    if F.plank:
        return None
    n = len(F.falls)
    gaps = [b["f"] - a["f"] for a, b in zip(F.falls, F.falls[1:])] + [e["gone"] - e["f"] for e in F.falls]
    where = "the pedestals" if F.twin else "the tower"
    return _q(1, "n_falls", f"How many times do blocks fall off {where}?", n, _ints(rng, n, 0, 5), True,
              {"min_fps": fps_for(min(gaps)) if gaps else 1})


def q_plank_tips(F, rng):
    if not F.plank:
        return None
    gt = "yes" if F.falls else "no"
    return _q(1, "plank_tips", "Does the plank tip over?", gt, _yn(rng, gt), True,
              {"min_fps": F.fall_fps(F.falls[0]) if F.falls else 1})


def q_end_count(F, rng):
    gt = F.count(F.n - 1)
    verb = "still lying" if F.plank else "still stacked"
    return _q(1, "end_count", f"How many blocks are {verb} {F.where()} at the end of the clip?", gt,
              _ints(rng, gt, 0), False, {"min_fps": 1})


def q_top_end(F, rng):
    if F.plank:
        return None
    bases = [b for b, st in enumerate(F.res["stacks"]) if st]
    if not bases:
        return None
    b = rng.choice(bases)
    gt = F.nm[F.res["stacks"][b][-1][0]]
    pool = [F.nm[j] for j in F.start if F.base_of[j] == b]
    d = _pick_d(rng, gt, pool)
    if d is None:
        return None
    where = f"on {sim.base_name(F.s, b)}" if F.twin else "of the tower"
    return _q(1, "top_end", f"Which block is on top {where} at the end of the clip?", gt, _place(rng, gt, d), False,
              {"base": b, "min_fps": 1})


def q_first(F, rng):
    if F.pull or len(F.lands) < 4:
        return None
    gt = F.nm[F.lands[0]["block"]]
    d = [F.nm[e["block"]] for e in F.lands[1:4]]
    return _q(1, "first_land", "Which block lands first?", gt, _place(rng, gt, d), True,
              {"min_fps": fps_for(F.lands[1]["f"] - F.lands[0]["f"])})


def q_first_pull(F, rng):
    if not F.pull:
        return None
    gt = F.nm[F.pulls[0]["block"]]
    d = _pick_d(rng, gt, [F.nm[j] for j in F.init])
    if d is None:
        return None
    gap = F.pulls[1]["f"] - F.pulls[0]["f"] if len(F.pulls) > 1 else F.pulls[0]["gone"] - F.pulls[0]["f"]
    return _q(1, "first_pull", "Which block is pulled out first?", gt, _place(rng, gt, d), True,
              {"min_fps": fps_for(min(gap, F.pulls[0]["gone"] - F.pulls[0]["f"]))})


def q_count_look(F, rng):
    pats = sorted({b["pattern"] for b in F.bl})
    if len(pats) < 2:
        return None
    p = rng.choice(pats)
    gt = sum(1 for b in F.bl if b["pattern"] == p)
    return _q(1, "count_look", f"How many {p} blocks appear in the clip?", gt, _ints(rng, gt, 0), not F.pull,
              {"pattern": p, "min_fps": 1 if F.pull else fps_for(F.visible_min())})


# ---------------------------------------------------------------- L2 relational

def q_land_order(F, rng):
    evs = F.pulls if F.pull else F.lands
    pairs = [(a, b) for i, a in enumerate(evs) for b in evs[i + 1:] if b["f"] - a["f"] >= MARGIN]
    if not pairs:
        return None
    a, b = rng.choice(pairs)
    x, y = (a, b) if rng.random() < 0.5 else (b, a)
    gt = F.nm[a["block"]]
    verb = "is pulled out" if F.pull else "lands"
    key = "pull_order" if F.pull else "land_order"
    return _q(2, key, f"Which {verb} first: {F.nm[x['block']]} or {F.nm[y['block']]}?", gt,
              _place(rng, gt, [F.nm[b["block"]]]), True,
              {"x": F.nm[x["block"]], "y": F.nm[y["block"]], "min_fps": fps_for(b["f"] - a["f"])})


def q_on_what(F, rng):
    if F.plank or F.pull:
        return None
    e = rng.choice(F.lands)
    j = e["block"]
    gt = sim.item_name(F.s, e["on"])
    pool = [sim.base_name(F.s, b) for b in range(len(F.s["bases"]))]
    pool += [F.nm[k] for k in F.start if F.base_of[k] == e["base"] and k != j]
    pool += [F.nm[k] for k in F.start if k != j]
    d = _pick_d(rng, gt, pool[:6] if len(set(pool[:6]) - {gt}) >= 3 else pool)
    if d is None:
        return None
    moved = j in F.leave
    return _q(2, "on_what", f"What does {F.nm[j]} land on?", gt, _place(rng, gt, d), moved,
              {"x": F.nm[j], "min_fps": fps_for(F.end(j) - e["f"]) if moved else 1})


def q_settle_on(F, rng):
    cands = [e for e in F.pulls if e["on"] is not None]
    if not cands:
        return None
    e = rng.choice(cands)
    gt = sim.item_name(F.s, e["on"])
    pool = ["the pedestal"] + [F.nm[k] for k in F.init if k != e["block"] and k not in e["above"]] + \
        [F.nm[k] for k in F.init if k != e["block"]]
    d = _pick_d(rng, gt, pool)
    if d is None:
        return None
    nxt = [f for f in F.steps if f > e["f"]] + [x["f"] for x in F.falls if x["f"] > e["settle"]] + [F.n]
    return _q(2, "settle_on", f"After {F.nm[e['block']]} is pulled out, what does the block that was resting on it "
              f"come down onto?", gt, _place(rng, gt, d), True,
              {"x": F.nm[e["block"]], "min_fps": fps_for(min(nxt) - e["settle"])})


def q_fall_side(F, rng):
    if not F.falls:
        return None
    k = rng.randrange(len(F.falls))
    e = F.falls[k]
    gt = SIDES[e["side"] == "right"]
    text = "Which way does the plank tip?" if F.plank else f"When {F.falls_text(k)}, which way do they tip?"
    return _q(2, "fall_side", text, gt, _place(rng, gt, [s for s in SIDES if s != gt]), True,
              {"k": k, "min_fps": F.fall_fps(e)})


def q_cause(F, rng):
    if not F.falls or F.pull:
        return None
    k = rng.randrange(len(F.falls))
    e = F.falls[k]
    c = e["cause"]
    i = [x["block"] for x in F.lands].index(c)
    gt = F.nm[c]
    near = [F.lands[t]["block"] for t in (i - 1, i + 1, i - 2, i + 2, i - 3, i + 3) if 0 <= t < len(F.lands)]
    d = _pick_d(rng, gt, [F.nm[t] for t in near][:3] if len(near) >= 3 else [F.nm[t] for t in near])
    if d is None:
        return None
    gap = F.lands[i]["f"] - F.lands[i - 1]["f"] if i else F.n
    what = "the plank tip" if F.plank else ("blocks fall" if len(F.falls) == 1 else f"blocks fall for the {ORD[k]} "
                                                                                     f"time")
    return _q(2, "cause", f"Which block's landing makes {what}?", gt, _place(rng, gt, d), True,
              {"k": k, "min_fps": fps_for(min(gap, e["gone"] - e["f"]))})


def q_cause_pull(F, rng):
    if not F.falls or not F.pull:
        return None
    e = F.falls[0]
    p = [x for x in F.pulls if x["a"] == e["a"]][0]
    gt = F.nm[p["block"]]
    pool = [F.nm[x["block"]] for x in F.pulls if x is not p] + [F.nm[j] for j in F.init]
    d = _pick_d(rng, gt, pool[:3] if len(set(pool[:3]) - {gt}) >= 3 else pool)
    if d is None:
        return None
    gap = min([abs(x["f"] - p["f"]) for x in F.pulls if x is not p] + [e["gone"] - e["f"]])
    return _q(2, "cause_pull", "Pulling out which block makes blocks fall?", gt, _place(rng, gt, d), True,
              {"min_fps": fps_for(gap)})


def _together(F, x, y):
    return max(F.start[x], F.start[y]) < min(F.end(x), F.end(y))


def q_wider(F, rng):
    pairs = [(x, y) for x in F.start for y in F.start if x < y and abs(F.bl[x]["w"] - F.bl[y]["w"]) >= 1]
    if not pairs:
        return None
    x, y = rng.choice(pairs)
    if rng.random() < 0.5:
        x, y = y, x
    gt = F.nm[x] if F.bl[x]["w"] > F.bl[y]["w"] else F.nm[y]
    other = F.nm[y] if gt == F.nm[x] else F.nm[x]
    return _q(2, "wider", f"Which block is wider: {F.nm[x]} or {F.nm[y]}?", gt, _place(rng, gt, [other]),
              not _together(F, x, y), {"x": F.nm[x], "y": F.nm[y], "min_fps": 1})


def q_twin_tower(F, rng):
    if not F.twin:
        return None
    e = rng.choice(F.lands)
    j = e["block"]
    gt = sim.base_name(F.s, e["base"])
    moved = j in F.leave
    return _q(2, "twin_tower", f"Onto which pedestal does {F.nm[j]} drop?", gt,
              _place(rng, gt, [sim.base_name(F.s, 1 - e["base"])]), moved,
              {"x": F.nm[j], "min_fps": fps_for(F.end(j) - e["f"]) if moved else 1})


def q_twin_first_fall(F, rng):
    if not F.twin:
        return None
    opts = ["the left pedestal", "the right pedestal", "neither"]
    gt = sim.base_name(F.s, F.falls[0]["base"]) if F.falls else "neither"
    return _q(2, "twin_first_fall", "Which pedestal loses blocks first?", gt,
              _place(rng, gt, [o for o in opts if o != gt]), True,
              {"min_fps": F.fall_fps(F.falls[0]) if F.falls else 1})


def q_twin_more(F, rng):
    if not F.twin:
        return None
    a, b = F.count(F.n - 1, 0), F.count(F.n - 1, 1)
    opts = ["the left pedestal", "the right pedestal", "both the same"]
    gt = opts[2] if a == b else opts[0] if a > b else opts[1]
    return _q(2, "twin_more", "At the end of the clip, which pedestal holds more blocks?", gt,
              _place(rng, gt, [o for o in opts if o != gt]), False, {"min_fps": 1})


def q_plank_land_side(F, rng):
    if not F.plank:
        return None
    B = F.s["bases"][0]
    lo, hi = B["x"] - B["w"] / 2, B["x"] + B["w"] / 2
    cands = []
    for a in F.s["actions"]:
        l, r = a["x"] - F.bl[a["block"]]["w"] / 2, a["x"] + F.bl[a["block"]]["w"] / 2
        if (lo - r) * F.s["S"] >= 15 or (l - hi) * F.s["S"] >= 15:          # clearly on one side
            cands.append((a["block"], "left of the support" if r <= lo else "right of the support"))
    if not cands:
        return None
    j, gt = rng.choice(cands)
    other = "right of the support" if gt.startswith("left") else "left of the support"
    moved = j in F.leave
    return _q(2, "plank_land_side", f"On which side of the support does {F.nm[j]} land?", gt,
              _place(rng, gt, [other]), moved,
              {"x": F.nm[j], "min_fps": fps_for(F.end(j) - F.start[j]) if moved else 1})


def q_above(F, rng):
    if F.plank or F.pull:
        return None
    cands = [(e, x) for e in F.lands for x in F.lands if x["on"] == e["block"]
             and sum(1 for y in F.lands if y["on"] == e["block"]) == 1]
    if not cands:
        return None
    e, x = rng.choice(cands)
    gt = F.nm[x["block"]]
    d = _pick_d(rng, gt, [F.nm[k] for k in F.start if k not in (e["block"],)])
    if d is None:
        return None
    return _q(2, "above", f"Which block lands directly on top of {F.nm[e['block']]}?", gt, _place(rng, gt, d), True,
              {"x": F.nm[e["block"]], "min_fps": fps_for(min(x["f"] - e["f"], F.seen(x["block"])))})


# ---------------------------------------------------------------- L3 measurement

def q_land_time(F, rng):
    if F.pull:
        return None
    e = rng.choice(F.lands)
    gt = clock(e["f"])
    return _q(3, "land_time", f"At what clock time does {F.nm[e['block']]} land?", gt, _clock_opts(rng, e["f"]),
              True, {"x": F.nm[e["block"]], "min_fps": 4})


def q_pull_time(F, rng):
    if not F.pull:
        return None
    e = rng.choice(F.pulls)
    gt = clock(e["f"])
    return _q(3, "pull_time", f"At what clock time does {F.nm[e['block']]} start to slide out?", gt,
              _clock_opts(rng, e["f"]), True, {"x": F.nm[e["block"]], "min_fps": 4})


def q_count_at(F, rng):
    base = rng.randrange(2) if F.twin and rng.random() < 0.6 else None
    cands = []
    first = min(F.steps) if F.steps else 0
    for k in range(sim.tenths(max(sim.EDGE, first if not F.pull else 0)), sim.tenths(F.n - 1 - sim.EDGE) + 1):
        f0, f1 = sim.frames_showing(k)
        c = F.calm(f0, f1)
        if c:
            cands.append((k, c))
    if not cands:
        return None
    k, c = rng.choice(cands)
    f0, _ = sim.frames_showing(k)
    gt = F.count(f0, base)
    verb = "lying" if F.plank else "stacked"
    return _q(3, "count_at", f"How many blocks are {verb} {F.where(base)} when the clock reads {sim.clock_text(k)}?",
              gt, _ints(rng, gt, 0), True, {"k": k, "base": base, "min_fps": fps_for(c)})


def q_n_fall(F, rng):
    if not F.falls:
        return None
    k = rng.randrange(len(F.falls))
    e = F.falls[k]
    gt = len(e["blocks"])
    text = ("How many blocks are on the plank when it tips over?" if F.plank else
            f"When {F.falls_text(k)}, how many blocks fall?")
    return _q(3, "n_fall", text, gt, _ints(rng, gt, 1), True, {"k": k, "min_fps": F.fall_fps(e)})


def q_gap(F, rng):
    evs = F.pulls if F.pull else F.lands
    pairs = []
    for i, a in enumerate(evs):
        for b in evs[i + 1:]:
            h = half_s(b["f"] - a["f"])
            if h:
                pairs.append((a, b, h))
    if not pairs:
        return None
    a, b, h = rng.choice(pairs)
    verb = "starting to slide out" if F.pull else "landing"
    return _q(3, "gap", f"How many seconds pass between {F.nm[a['block']]} {verb} and {F.nm[b['block']]} {verb}?",
              h, _half_opts(rng, h), True, {"x": F.nm[a["block"]], "y": F.nm[b["block"]], "min_fps": 4})


def q_width(F, rng):
    j = rng.choice(sorted(F.start))
    gt = F.bl[j]["w"]
    return _q(3, "width", f"How many grid squares wide is {F.nm[j]}?", gt, _ints(rng, gt, 1, 8), False,
              {"x": F.nm[j], "min_fps": 1})


def q_max_count(F, rng):
    if F.plank or F.pull:
        return None
    base = rng.randrange(2) if F.twin else 0
    cs = [F.count(f, base) for f in range(F.n)]
    m = max(cs)
    runs, f = [], 0
    while f < F.n:
        if cs[f] == m:
            g = f
            while g < F.n and cs[g] == m:
                g += 1
            runs.append(g - f)
            f = g
        else:
            f += 1
    if m < 2:
        return None
    where = f"on {sim.base_name(F.s, base)}" if F.twin else "on the pedestal"
    return _q(3, "max_count", f"What is the largest number of blocks stacked {where} at any one moment?", m,
              _ints(rng, m, 1), True, {"base": base, "min_fps": fps_for(max(runs))})


# ---------------------------------------------------------------- L4 integration

def q_total_fallen(F, rng):
    if not F.falls or F.plank:
        return None
    gt = sum(len(e["blocks"]) for e in F.falls)
    return _q(4, "total_fallen", "How many blocks fall in total during the clip?", gt, _ints(rng, gt, 1), True,
              {"min_fps": fps_for(min(e["gone"] - e["f"] for e in F.falls))})


def q_break(F, rng):
    if not F.falls or F.plank:
        return None
    k = rng.randrange(len(F.falls))
    e = F.falls[k]
    gt = sim.item_name(F.s, e["from"])
    pool = [F.nm[e["cause"]]] if e["cause"] not in e["blocks"] else []
    pool += [sim.base_name(F.s, e["base"])] + [F.nm[j] for j in e["blocks"]]
    pool += [F.nm[j] for j in F.start if F.base_of[j] == e["base"]]
    d = _pick_d(rng, gt, pool)
    if d is None:
        return None
    return _q(4, "break_from", f"When {F.falls_text(k)}, what was the falling part resting on just before it "
              f"tipped?", gt, _place(rng, gt, d), True, {"k": k, "min_fps": F.fall_fps(e)})


def q_kth(F, rng):
    evs = F.pulls if F.pull else F.lands
    if len(evs) < 3:
        return None
    k = rng.randrange(1, len(evs))
    gt = F.nm[evs[k]["block"]]
    pool = [F.nm[e["block"]] for i, e in enumerate(evs) if i != k]
    if F.pull:
        pool += [F.nm[j] for j in F.init]
    d = _pick_d(rng, gt, pool)
    if d is None:
        return None
    gaps = [evs[k]["f"] - evs[k - 1]["f"]] + ([evs[k + 1]["f"] - evs[k]["f"]] if k + 1 < len(evs) else [])
    text = f"Which block is the {ORD[k]} to be pulled out?" if F.pull else f"Which block is the {ORD[k]} to land?"
    return _q(4, "kth_pull" if F.pull else "kth_land", text, gt, _place(rng, gt, d), True,
              {"k": k, "min_fps": fps_for(min(gaps))})


def q_count_after(F, rng):
    if F.pull:
        return None
    causes = {e["cause"] for e in F.falls}
    cands = []
    for e in F.lands[1:]:
        if e["block"] in causes:
            continue
        nxt = [a for a, _ in F.blur if a > e["f"] + 1]
        if nxt and min(nxt) - e["f"] < MARGIN:
            continue
        cands.append(e)
    if not cands:
        return None
    e = rng.choice(cands)
    base = e["base"] if F.twin else None
    gt = F.count(e["f"], base)
    verb = "lying" if F.plank else "stacked"
    prev = [b for _, b in F.blur if b < e["f"] - 3]
    nxt = [a for a, _ in F.blur if a > e["f"] + 1] + [F.n]
    gap = min(nxt) - (max(prev) if prev else 0)
    return _q(4, "count_after", f"Right after {F.nm[e['block']]} lands, how many blocks are {verb} "
              f"{F.where(base)} (counting it)?", gt, _ints(rng, gt, 1), True,
              {"x": F.nm[e["block"]], "min_fps": fps_for(gap)})


def q_pull_fall_n(F, rng):
    if not F.pull:
        return None
    p = rng.choice(F.pulls)
    fl = [e for e in F.falls if e["a"] == p["a"]]
    gt = len(fl[0]["blocks"]) if fl else 0
    gap = fl[0]["gone"] - fl[0]["f"] if fl else F.n
    return _q(4, "pull_fall_n", f"How many blocks fall off right after {F.nm[p['block']]} is pulled out?", gt,
              _ints(rng, gt, 0), True, {"x": F.nm[p["block"]], "min_fps": fps_for(gap)})


def q_between(F, rng):
    evs = F.pulls if F.pull else F.lands
    ks = []
    for k in range(sim.tenths(sim.EDGE), sim.tenths(F.n - 1 - sim.EDGE) + 1):
        f0, f1 = sim.frames_showing(k)
        if all(abs(e["f"] - f) > WIN for e in evs for f in (f0, f1)) and not any(f0 <= e["f"] <= f1 for e in evs):
            ks.append(k)
    pairs = [(a, b) for a in ks for b in ks if b - a >= 10]
    if not pairs:
        return None
    a, b = rng.choice(pairs)
    fa, fb = sim.frames_showing(a)[1], sim.frames_showing(b)[0]
    inside = [e for e in evs if fa < e["f"] < fb]
    gt = len(inside)
    gaps = [y["f"] - x["f"] for x, y in zip(inside, inside[1:])] + [WIN * 2]
    verb = "are pulled out" if F.pull else "land"
    return _q(4, "between", f"How many blocks {verb} between {sim.clock_text(a)} and {sim.clock_text(b)} on the "
              f"clock?", gt, _ints(rng, gt, 0), True, {"k1": a, "k2": b, "min_fps": fps_for(min(gaps))})


def q_which_fell(F, rng):
    if not F.falls or F.plank:
        return None
    fallen = sorted(F.fell)
    stay = [j for j in sorted(F.start) if j not in F.fell and j not in F.leave]
    if len(stay) < 3:
        return None
    j = rng.choice(fallen)
    gt = F.nm[j]
    d = [F.nm[k] for k in rng.sample(stay, 3)]
    e = [x for x in F.falls if j in x["blocks"]][0]
    return _q(4, "which_fell", "Which of these blocks falls off at some point?", gt, _place(rng, gt, d), True,
              {"cands": sorted([gt] + d), "min_fps": F.fall_fps(e)})


# ---------------------------------------------------------------- L5 counterfactual (re-simulated)

def _cf(F, actions, bases=None):
    r = sim.play(F.s, actions, bases)
    if not r["ok"] or any(x.get("bad") for x in r["recs"]):
        return None
    return r


def _order_fps(F):
    return fps_for(min(F.step_gaps(), F.visible_min()))


def q_cf_skip(F, rng):
    if F.pull:
        return None
    acts = F.s["actions"]
    order = list(range(len(acts)))
    rng.shuffle(order)
    for i in order:                    # leaving out the last drop always replays cleanly
        r = _cf(F, acts[:i] + acts[i + 1:])
        if r is not None:
            break
    j = acts[i]["block"]
    where = ("each landing on the plank" if F.plank else "each landing on top of whatever is on its pedestal"
             if F.twin else "each landing on top of the stack")
    lead = (f"Suppose {F.nm[j]} had never been dropped, and every other block were dropped in the same order at the "
            f"same left-right spot ({where}).")
    if rng.random() < 0.5:
        gt = "yes" if any(x["fall"] for x in r["recs"]) else "no"
        tail = "Would the plank tip over?" if F.plank else "Would any blocks fall?"
        return _q(5, "cf_skip_any", f"{lead} {tail}", gt, _yn(rng, gt), True,
                  {"x": F.nm[j], "min_fps": _order_fps(F)})
    gt = sim.standing_end(r)
    verb = "lying" if F.plank else "stacked"
    return _q(5, "cf_skip_end", f"{lead} How many blocks would be {verb} {F.where()} at the end?", gt,
              _ints(rng, gt, 0), True, {"x": F.nm[j], "min_fps": _order_fps(F)})


def q_cf_shift(F, rng):
    if F.pull:
        return None
    acts = F.s["actions"]
    i = rng.randrange(len(acts))
    a = acts[i]
    d = rng.choice([-1, 1])
    alt = [dict(x) for x in acts[:i + 1]]
    alt[i]["x"] += d
    r = _cf(F, alt)
    if r is None:
        return None
    fl = r["recs"][-1]["fall"]
    side = "left" if d < 0 else "right"
    lead = (f"Suppose {F.nm[a['block']]} had been dropped one grid square further to the {side}, with everything "
            f"before it the same.")
    e = [x for x in F.lands if x["block"] == a["block"]][0]
    prev = [x["f"] for x in F.lands if x["f"] < e["f"]]
    fps = fps_for(e["f"] - max(prev)) if prev else 1
    if F.plank or rng.random() < 0.5:
        gt = "yes" if fl else "no"
        tail = "Would the plank tip over right after it lands?" if F.plank else \
            "Would any blocks fall right after it lands?"
        return _q(5, "cf_shift", f"{lead} {tail}", gt, _yn(rng, gt), True,
                  {"x": F.nm[a["block"]], "d": d, "min_fps": fps})
    gt = len(fl["blocks"]) if fl else 0
    return _q(5, "cf_shift_n", f"{lead} How many blocks would fall right after it lands?", gt, _ints(rng, gt, 0),
              True, {"x": F.nm[a["block"]], "d": d, "min_fps": fps})


def q_cf_swap(F, rng):
    if F.pull:
        return None
    acts = F.s["actions"]
    cands = [i for i in range(len(acts) - 1) if acts[i].get("base") == acts[i + 1].get("base")]
    if not cands:
        return None
    i = rng.choice(cands)
    alt = acts[:i] + [acts[i + 1], acts[i]] + acts[i + 2:]
    r = _cf(F, alt)
    if r is None:
        return None
    gt = sim.standing_end(r)
    x, y = F.nm[acts[i]["block"]], F.nm[acts[i + 1]["block"]]
    verb = "lying" if F.plank else "stacked"
    return _q(5, "cf_swap", f"Suppose {x} and {y} had been dropped in the opposite order, each at its own left-right "
              f"spot (everything else the same). How many blocks would be {verb} {F.where()} at the end?", gt,
              _ints(rng, gt, 0), True, {"x": x, "y": y, "min_fps": _order_fps(F)})


def q_cf_pull_other(F, rng):
    if not F.pull:
        return None
    acts = F.s["actions"]
    i = rng.randrange(len(acts))
    before = F.res["recs"][i]["before"]
    others = [q for q, _ in before if q != acts[i]["block"]]
    rng.shuffle(others)
    for q in others:
        alt = [dict(x) for x in acts]
        alt[i]["block"] = q
        r = _cf(F, alt)
        if r is None:
            continue
        gt = sim.n_fallen(r)
        x, y = F.nm[acts[i]["block"]], F.nm[q]
        return _q(5, "cf_pull_other", f"Suppose {y} had been pulled out instead of {x} (the other pulls the same). "
                  f"How many blocks would fall in total during the clip?", gt, _ints(rng, gt, 0), True,
                  {"x": x, "y": y, "min_fps": _order_fps(F)})
    return None


def q_cf_no_pull(F, rng):
    if not F.pull:
        return None
    acts = F.s["actions"]
    order = list(range(len(acts)))
    rng.shuffle(order)
    for i in order:                    # leaving out the last pull always replays cleanly
        r = _cf(F, acts[:i] + acts[i + 1:])
        if r is not None:
            break
    gt = sim.standing_end(r)
    x = F.nm[acts[i]["block"]]
    return _q(5, "cf_no_pull", f"Suppose {x} had never been pulled out (the other pulls the same). How many blocks "
              f"would be stacked on the pedestal at the end?", gt, _ints(rng, gt, 0), True,
              {"x": x, "min_fps": _order_fps(F)})


def q_cf_wide_base(F, rng):
    b = rng.randrange(len(F.s["bases"]))
    bases = [dict(B) for B in F.s["bases"]]
    bases[b]["w"] += 2
    r = _cf(F, None, bases)
    if r is None:
        return None
    if F.plank:
        gt = "yes" if any(x["fall"] for x in r["recs"]) else "no"
        text = "Suppose the support were one grid square wider on each side. Would the plank tip over?"
    else:
        gt = "yes" if any(x["fall"] and x["fall"]["base"] == b for x in r["recs"]) else "no"
        name = sim.base_name(F.s, b)
        text = (f"Suppose {name} were one grid square wider on each side (same centre), with every block "
                f"{'pulled' if F.pull else 'dropped'} the same. Would any blocks fall off {name}?")
    return _q(5, "cf_wide_base", text, gt, _yn(rng, gt), True, {"base": b, "min_fps": _order_fps(F)})


def q_cf_mirror(F, rng):
    if not F.plank:
        return None
    acts = F.s["actions"]
    i = rng.randrange(len(acts))
    sx = F.s["bases"][0]["x"]
    alt = [dict(x) for x in acts]
    alt[i]["x"] = 2 * sx - alt[i]["x"]
    if alt[i]["x"] == acts[i]["x"]:
        return None
    r = _cf(F, alt)
    if r is None:
        return None
    gt = "yes" if any(x["fall"] for x in r["recs"]) else "no"
    x = F.nm[acts[i]["block"]]
    return _q(5, "cf_mirror", f"Suppose {x} had landed on the other side of the support, the same distance from its "
              f"centre (everything else the same). Would the plank tip over?", gt, _yn(rng, gt), True,
              {"x": x, "min_fps": _order_fps(F)})


TEMPLATES = {
    1: [q_n_dropped, q_n_init, q_n_pulled, q_n_falls, q_plank_tips, q_end_count, q_top_end, q_first, q_first_pull,
        q_count_look],
    2: [q_land_order, q_on_what, q_settle_on, q_fall_side, q_cause, q_cause_pull, q_wider, q_twin_tower,
        q_twin_first_fall, q_twin_more, q_plank_land_side, q_above],
    3: [q_land_time, q_pull_time, q_count_at, q_n_fall, q_gap, q_width, q_max_count],
    4: [q_total_fallen, q_break, q_kth, q_count_after, q_pull_fall_n, q_between, q_which_fell],
    5: [q_cf_skip, q_cf_shift, q_cf_swap, q_cf_pull_other, q_cf_no_pull, q_cf_wide_base, q_cf_mirror],
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
