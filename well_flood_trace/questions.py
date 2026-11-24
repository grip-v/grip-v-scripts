"""10-question ladder for well_flood_trace (common/ladder5.py, mode "ten").

Sections are named as in the original dataset, by the depth value printed in
them ("the section with depth value 17"); depths are all different, so each
name is unique. The entry is "the entry section (outlined in red)"; a second
entry is "outlined in purple". Times are what the on-screen clock shows. "The
end of this video" means the last frame. A section is "completely full" when
its water turns dark blue (it then passes water on).

Wording follows the original well_flood_trace questions where the original had
that question (reach count, reached-by-end, first section after the entry,
count at a time, when a section begins to rise, the impossible event, which
section filled up first, double pour rate, correcting the impossible event).

Every question has exactly four distinct options and stores params["min_fps"]:
the sampling rate needed to see what it relies on, 1 / (the smallest gap
between the events or the shortest time a needed cue is on screen), rounded up
and clamped to 1..24. Comparisons need >= 0.3 s between the events; clock
readings have options >= 0.6 s apart and use only events whose first frame and
the frame after it show the same reading.

Every template builds its candidate question (or None when the clip can't
support it); the per-level counts are drawn uniformly from the valid splits
the feasible templates can fill, and each level's questions are drawn by weight
without replacement. Scenario-specific templates carry extra weight.
"""
from __future__ import annotations
import itertools
import math
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from common import ladder5 as L  # noqa: E402

import sim  # noqa: E402

LADDER = "ten"               # common/video5.py validates the 10-question ladder
FPS = sim.FPS
MARGIN = 0.3
FULL_MARGIN = 0.9            # a section asked about as "not full" is at most this full
CLOCK_SPACING = 6            # tenths of a second between clock-reading options
SIDE = {"up": "from the section above it", "down": "from the section below it",
        "left": "from the section to its left", "right": "from the section to its right"}
ORDINAL = {2: "second", 3: "third", 4: "fourth", 5: "fifth", 6: "sixth", 7: "seventh"}
RULE = ("Water can only enter a section from an entry pour or from a completely full neighbouring section "
        "with a HIGHER depth value.")
NONE_OPT = "none of them: every fill was physically correct"
KIND_OPTS = {"premature_fill": "a section started filling before any full higher neighbour could feed it",
             "uphill_flow": "water flowed into a section from a neighbour with a LOWER depth value",
             None: "nothing: every fill was physically correct",
             "stall": "a full section stopped passing water on to its lower neighbours"}
CHANGE_WORDS = {"stop": "stopped pouring", "double": "started pouring twice as fast"}
CF_TIMING = ("earlier than shown", "later than shown", "at the same time as shown",
             "not at all before the end of the video")


# ------------------------------------------------------------------ helpers
def cell_name(scene, i) -> str:
    return f"the section with depth value {scene['depth'][i]}"


def entry_name(scene, k) -> str:
    return f"the entry section (outlined in {scene['taps'][k]['colour']})"


def pour_name(scene, k) -> str:
    return "the pour" if len(scene["taps"]) == 1 else f"the pour into {entry_name(scene, k)}"


def secs(tenths: int) -> str:
    return f"{tenths / 10:.1f} s"


def fps_for(gap) -> int:
    if gap is None or gap <= 0:
        return FPS
    return int(min(FPS, max(1, math.ceil(1.0 / gap - 1e-9))))


def side_of(scene, i, j) -> str:
    """Where j sits relative to i."""
    C = scene["C"]
    if j == i - C:
        return "up"
    if j == i + C:
        return "down"
    return "left" if j == i - 1 else "right"


def mc4(rng, correct, distractors, fmt=str):
    seen, opts = {fmt(correct)}, [correct]
    for d in distractors:
        if len(opts) == 4:
            break
        if fmt(d) not in seen:
            opts.append(d)
            seen.add(fmt(d))
    if len(opts) < 4:
        return None
    rng.shuffle(opts)
    texts = [fmt(o) for o in opts]
    return texts, L.LETTERS[texts.index(fmt(correct))]


def spread_values(rng, gt: int, lo: int, hi: int, spacing: int, prefer=()) -> list[int] | None:
    """Three wrong values >= `spacing` apart from each other and from gt, inside [lo, hi],
    with gt's rank among the four drawn uniformly (falling back to other ranks when the
    range is too tight). Values from `prefer` (other real readings) are tried first."""
    rank = rng.randrange(4)
    for r in [rank] + [x for x in range(4) if x != rank]:
        out = []
        for want, sign in ((r, -1), (3 - r, 1)):
            pool = [v for v in range(lo, hi + 1) if (v - gt) * sign >= spacing]
            cand = [v for v in prefer if v in pool]
            rng.shuffle(cand)
            near = sorted(pool, key=lambda v: abs(v - gt))[:max(want * spacing * 2, 1)]
            rng.shuffle(near)
            got = []
            for v in cand + near:
                if len(got) == want:
                    break
                if all(abs(v - u) >= spacing for u in got + out + [gt]):
                    got.append(v)
            if len(got) < want:
                break
            out += got
        else:
            return out
    return None


def count_opts(rng, gt: int, lo: int, hi: int, prefer=()) -> list[int] | None:
    return spread_values(rng, gt, lo, hi, 1, prefer)


class Ctx:
    def __init__(self, scene):
        self.s = scene
        self.R, self.C, self.n = scene["R"], scene["C"], scene["R"] * scene["C"]
        tr = scene["track"]
        self.start, self.full, self.flows, self.first_src = tr["start"], tr["full"], tr["flows"], tr["first_src"]
        self.depth = scene["depth"]
        self.nf = scene["n_frames"]
        self.end = sim.end_time(scene)
        self.wet = [i for i in range(self.n) if self.start[i] is not None and self.start[i] <= self.end]
        self.dry = [i for i in range(self.n) if i not in self.wet]
        self.full_end = [i for i in range(self.n) if self.full[i] is not None and self.full[i] <= self.end]
        self.tap_cells = [t["cell"] for t in scene["taps"]]
        v = scene["violation"]
        self.fed_cells = set(self.tap_cells) | ({v["cell"]} if v and v["kind"] == "premature_fill" else set())

    def name(self, i):
        return cell_name(self.s, i)

    def frac(self, i, t) -> float:
        """How full section i is at time t (0..1), from the stored track."""
        st = self.start[i]
        if st is None or t < st:
            return 0.0
        if self.full[i] is not None and self.full[i] <= t:
            return 1.0
        cur = None
        for sg in self.s["track"]["level"][i]:
            if sg[0] <= t:
                cur = sg
        return min(1.0, (cur[1] + cur[2] * (t - cur[0])) / self.depth[i])

    def full_clear(self, t, cells=None) -> bool:
        """Every listed section that is not full at t is at most 90% full (full ones are
        also dark blue), so full and nearly-full are never confused."""
        cells = range(self.n) if cells is None else cells
        return all(self.frac(i, t) >= 1.0 or self.frac(i, t) <= FULL_MARGIN for i in cells)

    def clock_safe(self, t) -> int | None:
        """Clock reading (tenths) at the first frame showing an event at t, or None when the
        frame after it shows a different reading or the event is outside the clip."""
        if t is None or t > self.end - 0.5:
            return None
        f = sim.frame_of(t)
        if f + 1 >= self.nf or sim.clock_tenths(f) != sim.clock_tenths(f + 1):
            return None
        return sim.clock_tenths(f)

    def inflows(self, j, until=None):
        until = self.end if until is None else until
        return [f for f in self.flows if f[1] == j and f[2] <= until]

    def lasting(self, f) -> float:
        """How long flow f lasts within the clip."""
        return (self.end if f[3] is None else min(f[3], self.end)) - f[2]

    def cf(self, **kw):
        """A what-if world, or None when some section starts filling within 0.3 s of the end."""
        w = sim.run_world(self.s, **kw)
        if w is None or any(t is not None and abs(t - self.end) < MARGIN for t in w["start"]):
            return None
        return w

    def cf_count(self, w) -> int:
        return sum(1 for t in w["start"] if t is not None and t <= self.end)


def Q(key, text, gt, opts, motion, params, fps):
    return {"key": key, "text": text, "gt": gt, "opts": opts, "motion": motion, "params": dict(params, min_fps=fps)}


def near_depth_cells(cx, i, pool):
    """`pool` sorted by how close their depth value is to section i's (plausible distractors)."""
    return sorted((p for p in pool if p != i), key=lambda p: (abs(cx.depth[p] - cx.depth[i]), p))


# ------------------------------------------------------------------ L1 observation
def t_n_wet_end(cx, rng):
    gt = len(cx.wet)
    ds = count_opts(rng, gt, 1, cx.n)
    if ds is None:
        return None
    return Q("n_wet_end", "How many distinct sections had water reach them by the end of this video?",
             gt, mc4(rng, gt, ds), False, {}, 1)


def t_n_full_end(cx, rng):
    if not cx.full_clear(cx.end):
        return None
    gt = len(cx.full_end)
    ds = count_opts(rng, gt, 0, cx.n, prefer=[len(cx.wet)])
    if ds is None:
        return None
    return Q("n_full_end", "How many sections are completely full (dark blue) at the end of this video?",
             gt, mc4(rng, gt, ds), False, {}, 1)


def t_entry_depth(cx, rng):
    k = rng.randrange(len(cx.s["taps"]))
    e = cx.s["taps"][k]["cell"]
    gt = cx.depth[e]
    pool = [cx.depth[j] for j in sim.nbrs(cx.R, cx.C, e)] + [gt + d for d in (-3, -2, -1, 1, 2, 3)]
    pool = [v for v in dict.fromkeys(pool) if 1 <= v <= cx.n and v != gt]
    rng.shuffle(pool)
    return Q("entry_depth", f"What is the depth value of {entry_name(cx.s, k)}?", str(gt),
             mc4(rng, gt, pool), False, {"tap": k}, 1)


def t_reached_one(cx, rng):
    if len(cx.dry) < 3:
        return None
    x = rng.choice(cx.wet)
    edge = [d for d in cx.dry if any(j in cx.wet for j in sim.nbrs(cx.R, cx.C, d))]
    pool = list(dict.fromkeys(edge + near_depth_cells(cx, x, cx.dry)))
    others = pool[:6]
    rng.shuffle(others)
    others = others[:3]
    return Q("reached_one_of_four", "Which of these sections had water reach it by the end of this video?",
             cx.name(x), mc4(rng, x, others, cx.name), False, {"cells": sorted([x] + others)}, 1)


def t_dry_one(cx, rng):
    if not cx.dry or len(cx.wet) < 3:
        return None
    edge = [d for d in cx.dry if any(j in cx.wet for j in sim.nbrs(cx.R, cx.C, d))]
    x = rng.choice(edge or cx.dry)
    others = near_depth_cells(cx, x, cx.wet)[:6]
    rng.shuffle(others)
    others = others[:3]
    return Q("dry_one_of_four", "Which of these sections never had water reach it during this video?",
             cx.name(x), mc4(rng, x, others, cx.name), False, {"cells": sorted([x] + others)}, 1)


# ------------------------------------------------------------------ L2 relational
def _extreme_start(cx, rng, latest: bool):
    xs = cx.wet[:]
    rng.shuffle(xs)
    for x in xs:
        tx = cx.start[x]
        others = [i for i in cx.wet if i != x and ((tx - cx.start[i]) if latest else (cx.start[i] - tx)) >= MARGIN]
        if len(others) < 3:
            continue
        rng.shuffle(others)
        others = others[:3]
        gap = min(abs(cx.start[i] - tx) for i in others)
        word = "LAST" if latest else "FIRST"
        return Q("latest_of_four" if latest else "earliest_of_four",
                 f"Of these sections, which one did water reach {word}?", cx.name(x),
                 mc4(rng, x, others, cx.name), True, {"cells": sorted([x] + others)}, fps_for(gap))
    return None


def t_earliest(cx, rng):
    return _extreme_start(cx, rng, False)


def t_latest(cx, rng):
    return _extreme_start(cx, rng, True)


def t_fed_from(cx, rng):
    xs = [i for i in cx.wet if i not in cx.fed_cells and len(cx.first_src[i]) == 1 and cx.first_src[i][0] >= 0]
    rng.shuffle(xs)
    for x in xs:
        src = cx.first_src[x][0]
        ins = sorted(cx.inflows(x), key=lambda f: f[2])
        first = [f for f in ins if f[0] == src][0]
        win = cx.lasting(first)
        later = [f[2] for f in ins if f[0] != src]
        if later:
            win = min(win, min(later) - cx.start[x])
        if win < MARGIN:
            continue
        gt = SIDE[side_of(cx.s, x, src)]
        return Q("fed_from", f"From which neighbouring section did water first flow into {cx.name(x)}?", gt,
                 mc4(rng, gt, [v for v in SIDE.values() if v != gt]), True, {"cell": x}, fps_for(win))
    return None


def t_full_when_start(cx, rng):
    bs = [b for b in cx.wet if b not in cx.fed_cells]
    rng.shuffle(bs)
    for b in bs:
        tb = cx.start[b]
        done = [i for i in range(cx.n) if i != b and cx.full[i] is not None and cx.full[i] <= tb - MARGIN]
        notyet = [i for i in cx.wet if i != b and (cx.full[i] is None or cx.full[i] >= tb + MARGIN)
                  and cx.frac(i, tb) <= FULL_MARGIN]
        if not done or len(notyet) < 3:
            continue
        x = rng.choice(done)
        rng.shuffle(notyet)
        others = notyet[:3]
        gap = min([tb - cx.full[x]] + [cx.full[i] - tb for i in others if cx.full[i] is not None])
        return Q("full_when_start", f"When water began to rise in {cx.name(b)}, which of these sections was "
                 "already completely full?", cx.name(x), mc4(rng, x, others, cx.name), True,
                 {"cell": b, "cells": sorted([x] + others)}, fps_for(gap))
    return None


def t_first_full(cx, rng):
    """Original L4 'first_complete', asked over four sections (never the entry)."""
    xs = [i for i in cx.full_end if i not in cx.tap_cells]
    rng.shuffle(xs)
    for x in xs:
        tx = cx.full[x]
        others = [i for i in cx.wet if i != x and i not in cx.tap_cells
                  and (cx.full[i] is None or cx.full[i] - tx >= MARGIN)]
        if len(others) < 3:
            continue
        rng.shuffle(others)
        others = others[:3]
        gap = min([cx.full[i] - tx for i in others if cx.full[i] is not None] + [cx.end - tx])
        return Q("first_full_of_four", "Of these sections, which one became completely full (began passing water "
                 "onward) FIRST?", cx.name(x), mc4(rng, x, others, cx.name), True,
                 {"cells": sorted([x] + others)}, fps_for(gap))
    return None


# ------------------------------------------------------------------ L3 measurement
def _all_readings(cx):
    return [sim.clock_tenths(sim.frame_of(t)) for t in cx.start + cx.full if t is not None and t <= cx.end]


def _clock_q(cx, rng, key, times, text_fn):
    xs = [i for i in range(cx.n) if cx.clock_safe(times[i]) is not None]
    if not xs:
        return None
    x = rng.choice(xs)
    gt = cx.clock_safe(times[x])
    ds = spread_values(rng, gt, 3, sim.clock_tenths(cx.nf - 1), CLOCK_SPACING, prefer=_all_readings(cx))
    if ds is None:
        return None
    return Q(key, text_fn(cx.name(x)), secs(gt), mc4(rng, gt, ds, secs), True, {"cell": x},
             fps_for(CLOCK_SPACING / 20))


def t_start_clock(cx, rng):
    return _clock_q(cx, rng, "start_clock", cx.start,
                    lambda nm: f"At about what clock time does water begin to rise in {nm}?")


def t_full_clock(cx, rng):
    return _clock_q(cx, rng, "full_clock", cx.full,
                    lambda nm: f"At about what clock time does {nm} become completely full?")


def _count_at_clock(cx, rng, key, times, text):
    hi = sim.clock_tenths(cx.nf - 1)
    evs = [t for t in times if t is not None]
    good = []
    for T in range(10, hi - 4):
        frs = [f for f in range(cx.nf) if sim.clock_tenths(f) == T]
        if not frs:
            continue
        a, b = frs[0] / FPS, frs[-1] / FPS
        if any(a - MARGIN < t < b + MARGIN for t in evs):
            continue
        if key == "n_full_at_clock" and not cx.full_clear(b):
            continue
        near = min([a - t for t in evs if t <= a] + [t - b for t in evs if t >= b] + [9.9])
        good.append((T, sum(1 for t in evs if t <= a), near))
    if not good:
        return None
    total = sum(1 for t in evs if t <= cx.end)
    mid = [g for g in good if 1 <= g[1] < total]
    T, gt, near = rng.choice(mid or good)
    ds = count_opts(rng, gt, 0, cx.n, prefer=[total])
    if ds is None:
        return None
    return Q(key, text.format(secs(T)), gt, mc4(rng, gt, ds), True, {"tenths": T}, fps_for(near))


def t_count_at_clock(cx, rng):
    return _count_at_clock(cx, rng, "count_at_clock", cx.start,
                           "When the clock shows {}, how many total sections has water reached?")


def t_n_full_at_clock(cx, rng):
    return _count_at_clock(cx, rng, "n_full_at_clock", cx.full,
                           "When the clock shows {}, how many sections are completely full?")


def t_fill_duration(cx, rng):
    xs = [i for i in range(cx.n) if cx.clock_safe(cx.start[i]) is not None and cx.clock_safe(cx.full[i]) is not None
          and cx.clock_safe(cx.full[i]) - cx.clock_safe(cx.start[i]) >= 5]
    if not xs:
        return None
    x = rng.choice(xs)
    gt = cx.clock_safe(cx.full[x]) - cx.clock_safe(cx.start[x])
    ds = spread_values(rng, gt, 2, 99, CLOCK_SPACING)
    if ds is None:
        return None
    return Q("fill_duration", f"About how many seconds does {cx.name(x)} take to fill, from the first water to "
             "completely full?", secs(gt), mc4(rng, gt, ds, secs), True, {"cell": x}, fps_for(CLOCK_SPACING / 20))


def _gap_q(cx, rng, key, times, text):
    safe = [i for i in range(cx.n) if cx.clock_safe(times[i]) is not None]
    pairs = [(a, b) for a in safe for b in safe if times[b] - times[a] >= 0.6]
    if not pairs:
        return None
    a, b = rng.choice(pairs)
    gt = cx.clock_safe(times[b]) - cx.clock_safe(times[a])
    ds = spread_values(rng, gt, 2, 99, CLOCK_SPACING)
    if ds is None:
        return None
    return Q(key, text.format(cx.name(a), cx.name(b)), secs(gt), mc4(rng, gt, ds, secs), True, {"a": a, "b": b},
             fps_for(CLOCK_SPACING / 20))


def t_start_gap(cx, rng):
    return _gap_q(cx, rng, "start_gap", cx.start,
                  "About how many seconds after water reached {} did it reach {}?")


def t_full_gap(cx, rng):
    return _gap_q(cx, rng, "full_gap", cx.full,
                  "About how many seconds after {} became completely full did {} become completely full?")


def t_first_depth(cx, rng):
    """Original L3 'first_depth': depth value of the first section reached after the (red) entry."""
    e = cx.tap_cells[0]
    order = sorted((cx.start[i], i) for i in cx.wet if i not in cx.tap_cells)
    if len(order) < 2 or order[1][0] - order[0][0] < MARGIN or order[0][0] - cx.start[e] < MARGIN:
        return None
    x = order[0][1]
    gt = cx.depth[x]
    pool = [cx.depth[j] for j in sim.nbrs(cx.R, cx.C, e) if j != x] + [cx.depth[i] for _, i in order[1:4]]
    pool += [gt + d for d in (-2, -1, 1, 2)]
    pool = [v for v in dict.fromkeys(pool) if 1 <= v <= cx.n and v != gt]
    return Q("first_depth", "What is the depth value of the section that water reached FIRST after "
             f"{entry_name(cx.s, 0)}?", str(gt), mc4(rng, gt, pool), True, {},
             fps_for(min(order[1][0] - order[0][0], order[0][0] - cx.start[e])))


# ------------------------------------------------------------------ L4 integration
def t_order_three(cx, rng):
    if len(cx.wet) < 3:
        return None
    for _ in range(40):
        cells = rng.sample(cx.wet, 3)
        ts = sorted(cx.start[i] for i in cells)
        gap = min(b - a for a, b in zip(ts, ts[1:]))
        if gap < MARGIN:
            continue
        order = sorted(cells, key=lambda i: cx.start[i])
        perms = [[order[p] for p in pm] for pm in ((0, 2, 1), (1, 0, 2), (1, 2, 0), (2, 0, 1), (2, 1, 0))]
        rng.shuffle(perms)

        def fmt(o):
            return ", then ".join(str(cx.depth[i]) for i in o)
        shown = cells[:]
        rng.shuffle(shown)
        return Q("order_three", "In what order did water reach the sections with depth values "
                 f"{cx.depth[shown[0]]}, {cx.depth[shown[1]]} and {cx.depth[shown[2]]}? (Give the depth values "
                 "in order.)", fmt(order), mc4(rng, order, perms, fmt), True, {"cells": sorted(cells)}, fps_for(gap))
    return None


def t_kth_wet(cx, rng):
    """The k-th section water reached: k-1 started strictly earlier and every other
    start is >= 0.3 s away from it."""
    starts = [(cx.start[i], i) for i in cx.wet]
    ks = []
    for t, i in starts:
        gap = min([abs(t - u) for u, j in starts if j != i] + [9.9])
        k = 1 + sum(1 for u, j in starts if u < t)
        if gap >= MARGIN and 3 <= k <= 7:
            ks.append((k, i, gap))
    if not ks:
        return None
    k, x, gap = rng.choice(ks)
    order = sorted(cx.wet, key=lambda i: (cx.start[i], i))
    pos = order.index(x)
    pool = order[max(0, pos - 1):pos] + order[pos + 1:pos + 2] + order + cx.dry
    pool = [p for p in dict.fromkeys(pool) if p != x][:3]
    got = mc4(rng, x, pool, cx.name)
    if got is None:
        return None
    return Q("kth_wet", f"Counting the entry section as the first, which section did water reach {ORDINAL[k]}?",
             cx.name(x), got, True, {"k": k, "cells": sorted([x] + pool)}, fps_for(gap))


def t_impossible_which(cx, rng):
    """Original L4 'violation_or_confirm': where the impossible event happened, if anywhere."""
    v = cx.s["violation"]

    def fmt(o):
        return o if o == NONE_OPT else cx.name(o)
    if v:
        x = v["dst"] if v["kind"] == "uphill_flow" else v["cell"]
        if v["kind"] == "uphill_flow":
            win = cx.lasting([f for f in cx.flows if f[0] == v["src"] and f[1] == x][0])
        else:
            feeds = [f[2] for f in cx.flows if f[1] == x]
            win = min(feeds + [cx.end]) - cx.start[x]
        pool = ([v["src"]] if v["kind"] == "uphill_flow" else []) + \
            [i for i in sim.nbrs(cx.R, cx.C, x) if i in cx.wet] + cx.wet
        pool = [p for p in dict.fromkeys(pool) if p != x][:2]
        if len(pool) < 2:
            return None
        got = mc4(rng, x, pool + [NONE_OPT], fmt)
        gt, cells = cx.name(x), [x] + pool
    else:
        win = cx.end
        for i in cx.wet:
            if i in cx.tap_cells:
                continue
            ins = sorted(cx.inflows(i), key=lambda f: f[2])
            if ins:
                win = min(win, cx.lasting(ins[0]))
        pool = [i for i in cx.wet if i not in cx.tap_cells]
        rng.shuffle(pool)
        if len(pool) < 3 or win < MARGIN:
            return None
        cells = pool[:3]
        got = mc4(rng, NONE_OPT, cells, fmt)
        gt = NONE_OPT
    return Q("impossible_which", f"{RULE} Which section, if any, received water in a way that breaks this rule?",
             gt, got, True, {"cells": sorted(cells)}, fps_for(win))


def t_violation_kind(cx, rng):
    """Original L4 'violation_or_confirm': what kind of impossible event, if any."""
    v = cx.s["violation"]
    kind = v["kind"] if v else None
    if v and v["kind"] == "premature_fill":
        feeds = [f[2] for f in cx.flows if f[1] == v["cell"]]
        win = min(feeds + [cx.end]) - cx.start[v["cell"]]
    elif v:
        win = cx.lasting([f for f in cx.flows if f[0] == v["src"] and f[1] == v["dst"]][0])
    else:
        win = 1.0
    opts = list(KIND_OPTS.values())
    gt = KIND_OPTS[kind]
    return Q("violation_kind", "One moment in a video like this may show a physically impossible flow event. What, if "
             "anything, went wrong in this video?", gt, mc4(rng, gt, [o for o in opts if o != gt]), True, {},
             fps_for(win))


def t_wet_before_full(cx, rng):
    starts = [t for t in cx.start if t is not None]
    xs = [x for x in cx.full_end if cx.full[x] >= 1.0
          and not any(cx.full[x] - MARGIN <= t < cx.full[x] for t in starts)]
    if not xs:
        return None
    x = rng.choice(xs)
    gt = sum(1 for t in starts if t < cx.full[x])
    ds = count_opts(rng, gt, 1, cx.n, prefer=[len(cx.wet)])
    if ds is None:
        return None
    return Q("wet_before_full", f"How many sections had water in them just before {cx.name(x)} became completely "
             "full?", gt, mc4(rng, gt, ds), True, {"cell": x}, fps_for(MARGIN))


def t_wet_at_entry2(cx, rng):
    if cx.s["scenario"] != "two_entries":
        return None
    t2 = sim.tap_time(cx.s["taps"][1]["segs"][0][0])
    e2 = cx.s["taps"][1]["cell"]
    starts = [(cx.start[i], i) for i in range(cx.n) if cx.start[i] is not None]
    if any(abs(t - t2) < MARGIN and not (i == e2 and abs(t - t2) < 1e-9) for t, i in starts):
        return None
    gt = sum(1 for t, i in starts if t < t2 - 1e-9)
    ds = count_opts(rng, gt, 0, cx.n, prefer=[len(cx.wet)])
    if ds is None:
        return None
    return Q("wet_at_entry2", f"How many sections already had water in them when water started pouring into "
             f"{entry_name(cx.s, 1)}?", gt, mc4(rng, gt, ds), True, {}, fps_for(MARGIN))


def t_new_after_change(cx, rng):
    if cx.s["scenario"] != "pour_change":
        return None
    tp = cx.s["taps"][0]
    tc = sim.tap_time(tp["segs"][0][1])
    starts = [t for t in cx.start if t is not None and t <= cx.end]
    if any(abs(t - tc) < MARGIN for t in starts):
        return None
    gt = sum(1 for t in starts if t > tc)
    ds = count_opts(rng, gt, 0, cx.n, prefer=[len(cx.wet) - gt])
    if ds is None:
        return None
    return Q("new_after_change", f"How many sections first got water after the pour {CHANGE_WORDS[tp['change']]}?",
             gt, mc4(rng, gt, ds), True, {}, fps_for(MARGIN))


def t_walls_blocking(cx, rng):
    if not cx.s["walls"]:
        return None
    highs = [a if cx.depth[a] > cx.depth[b] else b for a, b in cx.s["walls"]]
    if not cx.full_clear(cx.end, highs):
        return None
    gt = sum(1 for h in highs if h in cx.full_end)
    ds = count_opts(rng, gt, 0, len(cx.s["walls"]) + 3, prefer=[len(cx.s["walls"])])
    if ds is None:
        return None
    return Q("walls_blocking", "At the end of this video, how many of the thick dark walls are holding back water, "
             "that is, sit between a completely full section and a neighbour with a lower depth value?", gt,
             mc4(rng, gt, ds), True, {}, 1)


# ------------------------------------------------------------------ L5 counterfactual
def _cf_q(cx, rng, key, text, w, params):
    if w is None:
        return None
    gt = cx.cf_count(w)
    ds = count_opts(rng, gt, 0, cx.n, prefer=[len(cx.wet)])
    if ds is None:
        return None
    return Q(key, text, gt, mc4(rng, gt, ds), True, params, 1)


def _all_pours(cx):
    return "the pour rate" if len(cx.s["taps"]) == 1 else "the pour rate of both entries"


def t_cf_rate(cx, rng):
    m = rng.choice((2.0, 0.5))
    word = "DOUBLE" if m == 2.0 else "HALF"
    return _cf_q(cx, rng, "cf_rate", f"If {_all_pours(cx)} had been {word} what's shown in this video (everything "
                 "else the same), how many distinct sections would water have reached by the end of the video?",
                 cx.cf(rate_mult=m), {"mult": m})


def t_cf_move_entry(cx, rng):
    k = rng.randrange(len(cx.s["taps"]))
    ys = [i for i in range(cx.n) if i not in cx.tap_cells]
    rng.shuffle(ys)
    for y in ys[:6]:
        w = cx.cf(move=(k, y))
        if w is None:
            continue
        return _cf_q(cx, rng, "cf_move_entry", f"If {pour_name(cx.s, k)} had gone into {cx.name(y)} instead (same "
                     "start time and rate), how many distinct sections would water have reached by the end of the "
                     "video?", w, {"tap": k, "cell": y})
    return None


def t_cf_block(cx, rng):
    xs = [i for i in cx.wet if i not in cx.fed_cells]
    rng.shuffle(xs)
    for x in xs[:6]:
        w = cx.cf(blocked=(x,))
        if w is None:
            continue
        return _cf_q(cx, rng, "cf_block", f"If {cx.name(x)} had been sealed so that no water could enter it, how many "
                     "distinct sections would water have reached by the end of the video?", w, {"cell": x})
    return None


def t_cf_x2_which(cx, rng):
    w = cx.cf(rate_mult=2.0)
    if w is None:
        return None
    st = w["start"]
    new = [i for i in cx.dry if st[i] is not None and st[i] <= cx.end - MARGIN]
    stay = [i for i in cx.dry if st[i] is None or st[i] >= cx.end + MARGIN]
    if not new or len(stay) < 3:
        return None
    x = rng.choice(new)
    others = near_depth_cells(cx, x, stay)[:6]
    rng.shuffle(others)
    others = others[:3]
    return Q("cf_x2_which", f"Water never reached these sections in this video. If {_all_pours(cx)} had been DOUBLE, "
             "which one would water have reached by the end of the video?", cx.name(x),
             mc4(rng, x, others, cx.name), True, {"cells": sorted([x] + others)}, 1)


def t_cf_no_violation(cx, rng):
    if not cx.s["violation"]:
        return None
    return _cf_q(cx, rng, "cf_no_violation", "If the one physically impossible event in this video had not happened "
                 "and all the water had behaved correctly, how many distinct sections would water have reached by the "
                 "end of the video?", cx.cf(violation=False), {})


def t_cf_correction(cx, rng):
    """Original L5 'counterfactual_correction': would a section have started filling earlier,
    later, at the same time, or not at all, had the impossible event behaved correctly?"""
    v = cx.s["violation"]
    if not v:
        return None
    w = cx.cf(violation=False)
    if w is None:
        return None
    bad = v["dst"] if v["kind"] == "uphill_flow" else v["cell"]
    ys = [i for i in cx.wet if i not in cx.tap_cells and i != bad]
    rng.shuffle(ys)
    for y in ys:
        shown, true = cx.start[y], w["start"][y]
        if true is None or true > cx.end + MARGIN:
            gt = CF_TIMING[3]
        elif abs(true - shown) < 1e-9:
            gt = CF_TIMING[2]
        elif true <= shown - MARGIN:
            gt = CF_TIMING[0]
        elif true >= shown + MARGIN and true <= cx.end - MARGIN:
            gt = CF_TIMING[1]
        else:
            continue
        return Q("cf_correction", "If the one physically impossible event in this video had instead behaved "
                 f"correctly, would water have started to rise in {cx.name(y)} earlier, later, at the same time, or "
                 "not at all before the end of the video?", gt, mc4(rng, gt, [o for o in CF_TIMING if o != gt]),
                 True, {"cell": y}, 1)
    return None


def t_cf_wall(cx, rng):
    if not cx.s["walls"]:
        return None
    ws = [list(w) for w in cx.s["walls"]]
    rng.shuffle(ws)
    best = None
    for a, b in ws:
        w = cx.cf(walls=[x for x in cx.s["walls"] if list(x) != [a, b]])
        if w is None:
            continue
        if best is None or cx.cf_count(w) != len(cx.wet):
            best = (a, b, w)
            if cx.cf_count(w) != len(cx.wet):
                break
    if best is None:
        return None
    a, b, w = best
    return _cf_q(cx, rng, "cf_wall", f"If the thick wall between {cx.name(a)} and {cx.name(b)} were removed, how many "
                 "distinct sections would water have reached by the end of the video?", w, {"wall": [a, b]})


def t_cf_unchanged(cx, rng):
    if cx.s["scenario"] != "pour_change":
        return None
    return _cf_q(cx, rng, "cf_unchanged", "If the pour had kept going at its starting rate, with no change, how many "
                 "distinct sections would water have reached by the end of the video?", cx.cf(unchanged=True), {})


def t_cf_drop_entry(cx, rng):
    if len(cx.s["taps"]) < 2:
        return None
    k = rng.randrange(2)
    return _cf_q(cx, rng, "cf_drop_entry", f"If no water had ever been poured into {entry_name(cx.s, k)}, how many "
                 "distinct sections would water have reached by the end of the video?", cx.cf(drop_tap=k), {"tap": k})


# (weight, template); scenario-specific templates return None elsewhere
LEVELS = {
    1: [(3, t_n_wet_end), (2, t_n_full_end), (2, t_entry_depth), (2, t_reached_one), (1, t_dry_one)],
    2: [(3, t_earliest), (3, t_latest), (3, t_fed_from), (3, t_full_when_start), (2, t_first_full)],
    3: [(3, t_start_clock), (2, t_full_clock), (3, t_count_at_clock), (2, t_n_full_at_clock), (2, t_fill_duration),
        (2, t_start_gap), (1, t_full_gap), (2, t_first_depth)],
    4: [(3, t_order_three), (2, t_kth_wet), (1, t_impossible_which), (1, t_violation_kind), (2, t_wet_before_full),
        (6, t_wet_at_entry2), (6, t_new_after_change), (4, t_walls_blocking)],
    5: [(2, t_cf_rate), (2, t_cf_move_entry), (2, t_cf_block), (2, t_cf_x2_which), (6, t_cf_no_violation),
        (6, t_cf_correction), (8, t_cf_wall), (8, t_cf_unchanged), (8, t_cf_drop_entry)],
}
# every valid per-level split of the 10 questions (L.TEN_RANGES, total L.TEN_TOTAL)
TEN_SPLITS = [c for c in itertools.product(*(range(lo, hi + 1) for lo, hi in L.TEN_RANGES)) if sum(c) == L.TEN_TOTAL]
BROKEN_BOOST = {t_impossible_which: 6, t_violation_kind: 6}


def build(scene: dict, rng: random.Random) -> list[dict]:
    cx = Ctx(scene)
    pools = {}
    for lv, temps in LEVELS.items():
        pools[lv] = []
        for w, fn in temps:
            q = fn(cx, rng)
            if q is not None and q["opts"] is not None:
                if scene["scenario"] == "broken_flow" and fn in BROKEN_BOOST:
                    w = BROKEN_BOOST[fn]
                pools[lv].append((w, q))
    combos = [c for c in TEN_SPLITS if all(c[lv - 1] <= len(pools[lv]) for lv in range(1, 6))]
    if not combos:
        raise RuntimeError("not enough feasible questions for a 10-question ladder")
    counts = rng.choice(combos)
    picked = []
    for lv in range(1, 6):
        pool = pools[lv][:]
        for _ in range(counts[lv - 1]):
            r = rng.random() * sum(w for w, _ in pool)
            idx = len(pool) - 1
            for k, (w, _) in enumerate(pool):
                r -= w
                if r < 0:
                    idx = k
                    break
            picked.append((lv, pool.pop(idx)[1]))
    qs = []
    for k, (lv, q) in enumerate(picked, start=1):
        opts, letter = q["opts"]
        qs.append(L.question(lv, q["key"], q["text"], q["gt"], opts, letter, q["motion"], q["params"], idx=k))
    return qs
