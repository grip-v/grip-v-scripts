"""10-question ladder for simple_harmonic_motion (common/ladder5.py, mode "ten").

The only text on screen is the clock. Masses are named by colour and shape
("the teal hexagon"; colours are unique per clip). Distances are counted in
tick spacings of the mass's own unnumbered ruler; its rest mark is the long
tick. Times are what the clock shows. Answers come from sim.py's closed
form; answers.py re-derives them from the drawn frames (and re-simulates
the what-ifs numerically). Every question has four distinct options and
params["min_fps"] (see README "Frame sampling").

Templates per level (derivation_key):
  L1 n_objects, shape_of, pattern_of, widest_swing, largest_object, longest_string,
     first_released, n_right_at_t, n_held_at_t
  L2 direction_at_t, farthest_at_t, first_to_far_end, n_faster_than, n_moving_far_at_t,
     released_before_count
  L3 period_of, amplitude_ticks, peak_ticks, crossing_time, crossings_count, far_end_count,
     release_time
  L4 period_difference, order_by_period, total_crossings, period_of_extreme, return_time,
     decay_count
  L5 cf_period, cf_crossings, cf_no_damping, cf_release_later
"""
from __future__ import annotations
import itertools
import json
import math
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import ladder5 as L  # noqa: E402
import sim  # noqa: E402
from answers import (AXIS_WORDS, name, fmt_s, fmt_clock, direction_options, direction_text,  # noqa: E402
                     order_text, clock_tenths, frame_of_clock)

LADDER = "ten"
REDRAWS: list = []        # per build: how many template redraws were needed (diagnostics only)
TEN_SPLITS = [c for c in itertools.product(*(range(lo, hi + 1) for lo, hi in L.TEN_RANGES)) if sum(c) == L.TEN_TOTAL]
FPS = sim.FPS
FPS_STEPS = (1, 2, 3, 4, 6, 8, 12, 24)
EV = 0.3            # s: a question's events stay this far from anything that could change its answer
RULER = ("Distances are counted in tick spacings on each mass's own ruler; the long tick is its rest mark "
         "(where it sits when it is not moving). ")


# ------------------------------------------------------------------ helpers
def fps_for(gaps, floor=1):
    g = min(gaps) if gaps else 99.0
    need = math.ceil(1.0 / g - 1e-9)
    return int(min(24, max(floor, need)))


def fps_sampling(amp, omega, slack):
    """Smallest sampling rate at which the largest sampled offset of a swing of amplitude `amp`
    (px) is within `slack` px of the true peak."""
    for f in FPS_STEPS:
        if amp * (1 - math.cos(omega / (2 * f))) <= slack:
            return f
    return 24


def align(t):
    return 1 if abs(t - round(t)) < 1e-9 else 2


def safe1(v):
    """v (seconds) is >= 0.015 s from a 0.1 s rounding boundary."""
    x = v * 10 - math.floor(v * 10)
    return abs(x - 0.5) >= 0.15


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


def int_opts(rng, gt, lo, hi):
    near = [v for v in (gt + 1, gt - 1, gt + 2, gt - 2, gt + 3, gt - 3, gt + 4) if lo <= v <= hi]
    rng.shuffle(near)
    near.sort(key=lambda v: abs(v - gt))
    return mc4(rng, gt, near)


def secs_opts(rng, gt, cands, spacing):
    """gt and three distractors (from cands) rounded to 0.1 s, all >= spacing apart."""
    g = round(gt, 1)
    cands = [round(c, 1) for c in cands]
    rng.shuffle(cands)
    keep = []
    for c in cands:
        if c >= 0.1 and all(abs(c - u) >= spacing - 1e-9 for u in keep + [g]):
            keep.append(c)
    return mc4(rng, g, keep, fmt_s)


def clock_opts(rng, tenths, extra=()):
    cands = [tenths + d for d in (-15, -10, -5, 5, 10, 15, 20, -20)] + [round(e) for e in extra]
    rng.shuffle(cands)
    keep = []
    for c in cands:
        if c >= 0 and all(abs(c - u) >= 5 for u in keep + [tenths]):
            keep.append(c)
    return mc4(rng, tenths, keep, fmt_clock)


def event_frame(te):
    """(first frame drawn after the event at te seconds, ok): ok when the event sits well inside a frame
    interval and both frames around it show the same clock value."""
    x = te * FPS
    fa = math.ceil(x)
    ok = 0.1 <= x - math.floor(x) <= 0.9 and clock_tenths(fa - 1) == clock_tenths(fa)
    return fa, ok


# ------------------------------------------------------------------ context
class Ctx:
    def __init__(self, scene):
        self.s = scene
        self.objs = scene["objects"]
        self.n = len(self.objs)
        self.cs = [o["colour"] for o in self.objs]
        self.by = {o["colour"]: o for o in self.objs}
        self.dur = sim.duration(scene)
        self.tick = scene["tick"]
        self.orient = scene["orient"]
        self.T = {c: sim.period(o) for c, o in self.by.items()}
        self.cr = {c: sim.crossings(o, 0, self.dur) for c, o in self.by.items()}
        self.tu = {c: sim.turns(o, 0, self.dur) for c, o in self.by.items()}
        self.words = AXIS_WORDS[self.orient]
        self.thing = "pendulum" if self.orient == "pendulum" else "mass"
        self.grid = [k / 2 for k in range(2, int(2 * (self.dur - 1.0)) + 1)]
        self.staggered = scene["scenario"] == "staggered"
        self.damped = scene["scenario"] == "damped"

    def nm(self, c):
        return name(self.s, c)

    def env(self, c, t):
        o = self.by[c]
        return o["A"] * math.exp(-o["zeta"] * o["omega"] * t) if o["mode"] == "damped" else o["A"]

    def visible_cr(self, c):
        """Crossings with a swing of at least half a tick (so they are seen, not guessed)."""
        return [t for t in self.cr[c] if self.env(c, t) >= 0.5 * self.tick]

    def events(self, c):
        """Every time the motion of c changes character: crossings, turns, release."""
        ev = list(self.cr[c]) + [t for t, _ in self.tu[c]]
        if self.by[c]["mode"] == "held":
            ev.append(self.by[c]["t_rel"])
        return sorted(ev)

    def quiet(self, c, t, before=EV, after=EV + 0.1):
        """Distance from the clock window [t, t + 0.1) to the nearest event of c (None if < margin)."""
        d = min([abs(e - t) if e < t else e - t - 0.1 for e in self.events(c)] + [99.0])
        ok = all(not (t - before < e < t + after) for e in self.events(c))
        return d if ok else None

    def held_at(self, c, t):
        o = self.by[c]
        return o["mode"] == "held" and t <= o["t_rel"]

    def u(self, c, t):
        return sim.u_at(self.by[c], t)

    def widest_order(self):
        return sorted(self.cs, key=lambda c: self.by[c]["A"], reverse=True)

    def periods_ok(self, c):
        """At least three visible crossings, so a period can be measured."""
        return len(self.visible_cr(c)) >= 3

    def period_q_fps(self, c):
        return max(4, fps_for([self.T[c] / 2]))


def Q(key, params, text, gt, opts, motion, fps):
    if opts is None:
        return None
    return (key, dict(params, min_fps=fps), text, gt, opts[0], opts[1], motion)


# ------------------------------------------------------------------ L1 observation
def level1(c: Ctx, rng):
    out = []
    s, thing = c.s, c.thing
    plural = "pendulums" if thing == "pendulum" else "masses on springs"
    out.append(Q("n_objects", {}, f"How many {plural} are there in the clip?", c.n,
                 int_opts(rng, c.n, 1, 9), False, 1))
    for col in rng.sample(c.cs, min(2, c.n)):
        o = c.by[col]
        out.append(Q("shape_of", {"c": col}, f"What shape is the {col} {thing}?", o["shape"],
                     mc4(rng, o["shape"], rng.sample([x for x in sim.SHAPES if x != o["shape"]], 4)), False, 1))
        out.append(Q("pattern_of", {"c": col}, f"Which pattern is on the {col} {thing}?", o["pattern"],
                     mc4(rng, o["pattern"], rng.sample([x for x in sim.PATTERNS if x != o["pattern"]], 3)), False, 1))
    if c.n >= 4:
        w = c.widest_order()
        a1, a2 = c.by[w[0]]["A"], c.by[w[1]]["A"]
        if a1 >= 1.1 * a2 and a1 - a2 >= 15:
            o = c.by[w[0]]
            f = 1 if o["mode"] != "free" else fps_sampling(a1, o["omega"], (a1 - a2) / 2)
            out.append(Q("widest_swing", {}, f"Which {thing} gets farthest from its own rest mark at any moment of "
                         "the clip? " + RULER, c.nm(w[0]),
                         mc4(rng, c.nm(w[0]), [c.nm(x) for x in rng.sample(w[1:], 3)]), True, f))
        sz = sorted(c.objs, key=lambda o: o["size"], reverse=True)
        if sz[0]["size"] >= 1.1 * sz[1]["size"]:
            g = sz[0]["colour"]
            out.append(Q("largest_object", {}, f"Which {thing} is drawn the largest?", c.nm(g),
                         mc4(rng, c.nm(g), [c.nm(o["colour"]) for o in rng.sample(sz[1:], 3)]), False, 1))
        if c.orient == "pendulum":
            ls = sorted(c.objs, key=lambda o: o["L"], reverse=True)
            g = ls[0]["colour"]
            out.append(Q("longest_string", {}, "Which pendulum hangs on the longest string?", c.nm(g),
                         mc4(rng, c.nm(g), [c.nm(o["colour"]) for o in rng.sample(ls[1:], 3)]), False, 1))
        if c.staggered:
            rel = sorted(c.objs, key=lambda o: o["t_rel"])
            g = rel[0]["colour"]
            out.append(Q("first_released", {}, "Each mass starts held still by a small stop post. Which mass is "
                         "let go first?", c.nm(g), mc4(rng, c.nm(g), [c.nm(o["colour"]) for o in rng.sample(rel[1:], 3)]),
                         True, fps_for([rel[1]["t_rel"] - rel[0]["t_rel"]])))
    ts = list(c.grid)
    rng.shuffle(ts)
    for t in ts[:12]:
        f0 = frame_of_clock(t)
        fr = [f for f in range(f0, f0 + 4) if clock_tenths(f) == clock_tenths(f0)]
        if any(abs(c.u(x, f / FPS)) < max(15, 0.4 * c.tick) for x in c.cs for f in fr):
            continue
        if any(abs(e - t) < EV or abs(e - t - 0.1) < EV for x in c.cs for e in c.cr[x]):
            continue
        k = sum(1 for x in c.cs if c.u(x, t) > 0)
        out.append(Q("n_right_at_t", {"t": t}, f"At {t:.1f} s on the clock, how many {thing}s are {c.words[2]} their "
                     "own rest mark? " + RULER, k, int_opts(rng, k, 0, c.n), False, align(t)))
        break
    if c.staggered:
        rels = [o["t_rel"] for o in c.objs]
        ts = [t for t in c.grid if t <= max(rels) + 0.5 and all(abs(r - t) >= EV and abs(r - t - 0.1) >= EV for r in rels)]
        for t in rng.sample(ts, min(2, len(ts))):
            k = sum(1 for x in c.cs if c.held_at(x, t))
            gap = min(min(abs(r - t), abs(r - t - 0.1)) for r in rels)
            out.append(Q("n_held_at_t", {"t": t}, f"At {t:.1f} s on the clock, how many masses are still held by "
                         "their stop posts?", k, int_opts(rng, k, 0, c.n), True, max(align(t), fps_for([gap]))))
    return out


# ------------------------------------------------------------------ L2 relational
def level2(c: Ctx, rng):
    out = []
    thing, words = c.thing, c.words
    # direction of one mass at a clock time
    tries = [(x, t) for x in c.cs for t in c.grid]
    rng.shuffle(tries)
    made = 0
    for col, t in tries:
        if made == 3:
            break
        d = c.quiet(col, t)
        if d is None or c.held_at(col, t + 0.1 + EV) or c.held_at(col, t - EV):
            continue
        f0 = frame_of_clock(t)
        us = [c.u(col, (f0 + k) / FPS) for k in (-1, 0, 1)]
        if min(abs(v) for v in us) < 15 or abs(us[2] - us[0]) < 6:
            continue
        gt = direction_text(c.s, sim.v_sign(c.by[col], f0 / FPS), us[1])
        out.append(Q("direction_at_t", {"c": col, "t": t}, f"At {t:.1f} s on the clock, which describes "
                     f"{c.nm(col)}? " + RULER, gt, mc4(rng, gt, direction_options(c.s)), True,
                     max(align(t), fps_for([d]))))
        made += 1
    if c.n >= 4:
        ts = list(c.grid)
        rng.shuffle(ts)
        for t in ts[:12]:
            f0 = frame_of_clock(t)
            fr = [f for f in range(f0, f0 + 4) if clock_tenths(f) == clock_tenths(f0)]
            ranks = [sorted(c.cs, key=lambda x: -abs(c.u(x, f / FPS))) for f in fr]
            if any(r[0] != ranks[0][0] for r in ranks):
                continue
            ok = all(abs(c.u(r[0], f / FPS)) >= 1.1 * abs(c.u(r[1], f / FPS)) and
                     abs(c.u(r[0], f / FPS)) - abs(c.u(r[1], f / FPS)) >= 15 for r, f in zip(ranks, fr))
            if not ok:
                continue
            g = ranks[0][0]
            out.append(Q("farthest_at_t", {"t": t}, f"At {t:.1f} s on the clock, which {thing} is farthest from its "
                         "own rest mark? " + RULER, c.nm(g),
                         mc4(rng, c.nm(g), [c.nm(x) for x in ranks[0][1:4]]), True, align(t)))
            break
        for t in ts[:12]:
            if c.staggered and any(c.held_at(x, t - EV) for x in c.cs):
                continue
            first = {}
            for x in c.cs:
                nxt = [tt for tt, side in c.tu[x] if side > 0 and tt > t - 0.05]
                first[x] = nxt[0] if nxt else math.inf
            order = sorted(c.cs, key=first.get)
            t1, t2 = first[order[0]], first[order[1]]
            if t1 - t - 0.1 < EV or t1 > c.dur - sim.EDGE or t2 - t1 < EV:
                continue
            if c.damped and c.env(order[0], t1) < 0.5 * c.tick:
                continue
            g = order[0]
            out.append(Q("first_to_far_end", {"t": t}, f"After {t:.1f} s on the clock, which {thing} is the first to "
                         f"reach the {words[4]} of its swing (where it stops and turns back)?", c.nm(g),
                         mc4(rng, c.nm(g), [c.nm(x) for x in order[1:4]]), True,
                         max(align(t), fps_for([t1 - t - 0.1, min(t2 - t1, 99)]))))
            break
    # how many oscillate faster
    if all(len(c.visible_cr(x)) >= 2 for x in c.cs):
        tmin = min(c.T.values())
        for col in rng.sample(c.cs, min(2, c.n)):
            k = sum(1 for x in c.cs if x != col and c.T[x] < c.T[col])
            out.append(Q("n_faster_than", {"c": col}, f"How many of the other {thing}s complete a full swing (there and "
                         f"back) in less time than {c.nm(col)}?", k, int_opts(rng, k, 0, c.n), True, fps_for([tmin / 2])))
    ts = list(c.grid)
    rng.shuffle(ts)
    for t in ts[:15]:
        ds = [c.quiet(x, t) for x in c.cs]
        if any(d is None for d in ds):
            continue
        f0 = frame_of_clock(t)
        if any(abs(c.u(x, (f0 + 1) / FPS) - c.u(x, (f0 - 1) / FPS)) < 6 and not c.held_at(x, (f0 + 1) / FPS)
               for x in c.cs):
            continue
        k = sum(1 for x in c.cs if sim.v_sign(c.by[x], f0 / FPS) > 0)
        out.append(Q("n_moving_far_at_t", {"t": t}, f"At {t:.1f} s on the clock, how many {thing}s are moving "
                     f"{words[0]}?", k, int_opts(rng, k, 0, c.n), True, max(align(t), fps_for(ds))))
        break
    if c.staggered:
        rel = sorted(c.objs, key=lambda o: o["t_rel"])
        for col in rng.sample(c.cs, min(2, c.n)):
            k = [o["colour"] for o in rel].index(col)
            mine = c.by[col]["t_rel"]
            gap = min(abs(o["t_rel"] - mine) for o in rel if o["colour"] != col)
            out.append(Q("released_before_count", {"c": col}, f"Each mass starts held by a stop post. How many masses "
                         f"are let go before {c.nm(col)}?", k, int_opts(rng, k, 0, c.n), True, fps_for([gap])))
    return out


# ------------------------------------------------------------------ L3 measurement
def _period_cands(T):
    return [T / 2, 2 * T, 1.5 * T, T / 1.5, T + 0.5, T - 0.5, T + 1.0, T - 1.0, T + 0.4, T * 1.25, T * 0.75]


def level3(c: Ctx, rng):
    out = []
    thing, words = c.thing, c.words
    for col in rng.sample(c.cs, min(3, c.n)):
        T = c.T[col]
        if c.periods_ok(col) and safe1(T):
            out.append(Q("period_of", {"c": col}, f"Read off the clock: how long does {c.nm(col)} take for one full "
                         "swing, there and back?", fmt_s(T), secs_opts(rng, T, _period_cands(T), max(0.3, 0.15 * T)),
                         True, c.period_q_fps(col)))
    for col in rng.sample(c.cs, min(2, c.n)):
        o = c.by[col]
        r = o["A"] / c.tick
        k = round(r)
        if not c.damped:
            slack = (0.5 - abs(r - k)) * c.tick - 2
            f = 1 if o["mode"] == "held" else fps_sampling(o["A"], o["omega"], slack)
            out.append(Q("amplitude_ticks", {"c": col}, f"How many tick spacings from its rest mark does {c.nm(col)} "
                         "get at most, to the nearest whole tick? " + RULER, k,
                         int_opts(rng, k, 1, 12), True, f))
        else:
            good = []
            for tt, side in c.tu[col]:
                v = abs(c.u(col, tt)) / c.tick
                kk = round(v)
                if sim.EDGE <= tt <= c.dur - sim.EDGE and kk >= 1 and abs(v - kk) <= 0.25:
                    good.append((tt, v, kk))
            if good:
                tt, v, kk = rng.choice(good)
                tc = clock_tenths(round(tt * FPS)) / 10
                slack = (0.5 - abs(v - kk)) * c.tick - 2
                f = max(2, fps_sampling(v * c.tick, sim.omega_d(o), slack), fps_for([c.T[col] / 2]))
                out.append(Q("peak_ticks", {"c": col, "t": tc}, f"At about {tc:.1f} s on the clock {c.nm(col)} stops "
                             "for an instant and turns back. How many tick spacings from its rest mark is it then, to "
                             "the nearest whole tick? " + RULER, kk, int_opts(rng, kk, 0, 12), True, f))
    # first crossing after a clock time
    tries = [(x, t) for x in c.cs for t in c.grid]
    rng.shuffle(tries)
    made = 0
    for col, t in tries:
        if made == 2:
            break
        nxt = [e for e in c.visible_cr(col) if e > t + 0.1]
        allc = [e for e in c.cr[col] if e > t - EV]
        if not nxt or allc[0] != nxt[0] or nxt[0] - t - 0.1 < EV or nxt[0] > c.dur - sim.EDGE:
            continue
        fa, ok = event_frame(nxt[0])
        if not ok:
            continue
        tenths = clock_tenths(fa)
        out.append(Q("crossing_time", {"c": col, "t": t}, f"After {t:.1f} s on the clock, when does {c.nm(col)} "
                     "next pass its rest mark? Give the clock reading. " + RULER, fmt_clock(tenths),
                     clock_opts(rng, tenths, [10 * (nxt[0] + c.T[col] / 2)]), True,
                     max(4, fps_for([nxt[0] - t - 0.1, c.T[col] / 2]))))
        made += 1
    # counts in a window
    wins = [(a, b) for a in c.grid for b in c.grid if b - a >= 2.0]
    rng.shuffle(wins)
    for key, word in (("crossings_count", "cross"), ("far_end_count", "end")):
        made = 0
        for (t0, t1), col in zip(wins[:40], itertools.cycle(rng.sample(c.cs, c.n))):
            if made == 1:
                break
            if word == "cross":
                ev = c.cr[col]
                inside = [e for e in ev if t0 <= e <= t1]
            else:
                ev = [tt for tt, _ in c.tu[col]]
                inside = [tt for tt, side in c.tu[col] if side > 0 and t0 <= tt <= t1]
            if any(abs(e - t0) < 0.25 or abs(e - t1) < 0.25 for e in ev):
                continue
            if c.staggered and c.held_at(col, t0):
                continue
            if c.damped and any(c.env(col, e) < 0.5 * c.tick for e in inside):
                continue
            if not inside:
                continue
            k = len(inside)
            margins = [min(abs(e - t0), abs(e - t1)) for e in ev] + [c.T[col] / 2]
            if word == "cross":
                text = (f"Between {t0:.1f} s and {t1:.1f} s on the clock, how many times does {c.nm(col)} pass its "
                        "rest mark? " + RULER)
            else:
                text = (f"Between {t0:.1f} s and {t1:.1f} s on the clock, how many times does {c.nm(col)} reach the "
                        f"{words[4]} of its swing (where it stops and turns back)?")
            out.append(Q(key, {"c": col, "t0": t0, "t1": t1}, text, k, int_opts(rng, k, 0, 30), True,
                         max(align(t0), align(t1), fps_for(margins))))
            made += 1
    if c.staggered:
        for col in rng.sample(c.cs, min(2, c.n)):
            fr = round(c.by[col]["t_rel"] * FPS)
            tenths = clock_tenths(fr)
            out.append(Q("release_time", {"c": col}, f"{c.nm(col).capitalize()} starts held by a stop post. At what "
                         "clock reading is it let go?", fmt_clock(tenths), clock_opts(rng, tenths), True, 4))
    return out


# ------------------------------------------------------------------ L4 integration
EXTREME_WORDS = {"widest": "the {t} that gets farthest from its rest mark",
                 "narrowest": "the {t} that stays closest to its rest mark (smallest swing)",
                 "largest": "the {t} that is drawn the largest", "smallest": "the {t} that is drawn the smallest",
                 "longest": "the pendulum on the longest string", "shortest": "the pendulum on the shortest string",
                 "first_released": "the mass that is let go first", "last_released": "the mass that is let go last"}


def _extreme(c: Ctx, which):
    if which in ("widest", "narrowest"):
        o = sorted(c.objs, key=lambda o: o["A"], reverse=which == "widest")
        ok = o[0]["A"] >= 1.1 * o[1]["A"] and abs(o[0]["A"] - o[1]["A"]) >= 15 if which == "widest" else \
            o[1]["A"] >= 1.1 * o[0]["A"] and abs(o[0]["A"] - o[1]["A"]) >= 15
        return o[0]["colour"] if ok else None
    if which in ("largest", "smallest"):
        o = sorted(c.objs, key=lambda o: o["size"], reverse=which == "largest")
        big, small = (o[0]["size"], o[1]["size"]) if which == "largest" else (o[1]["size"], o[0]["size"])
        return o[0]["colour"] if big >= 1.1 * small else None
    if which in ("longest", "shortest"):
        return (max if which == "longest" else min)(c.objs, key=lambda o: o["L"])["colour"]
    o = sorted(c.objs, key=lambda o: o["t_rel"])
    return (o[0] if which == "first_released" else o[-1])["colour"]


def level4(c: Ctx, rng):
    out = []
    thing = c.thing
    ok = [x for x in c.cs if c.periods_ok(x)]
    pairs = list(itertools.combinations(ok, 2))
    rng.shuffle(pairs)
    made = 0
    for a, b in pairs:
        if made == 2:
            break
        d = abs(c.T[a] - c.T[b])
        if d < 0.4 or not safe1(d):
            continue
        cands = [d + 0.3, d - 0.3, d + 0.6, d - 0.6, d * 2, d / 2, c.T[a], c.T[b], d + 0.9]
        out.append(Q("period_difference", {"a": a, "b": b}, f"How much longer does one full swing (there and back) take "
                     f"for the slower of {c.nm(a)} and {c.nm(b)} than for the faster one?", fmt_s(d),
                     secs_opts(rng, d, cands, 0.3), True, max(4, fps_for([min(c.T[a], c.T[b]) / 2]))))
        made += 1
    if len(ok) >= 3:
        for _ in range(2):
            cs = rng.sample(ok, 3)
            gt = order_text(sorted(cs, key=c.T.get))
            perms = [order_text(p) for p in itertools.permutations(cs) if order_text(p) != gt]
            out.append(Q("order_by_period", {"cs": sorted(cs)}, f"Order {c.nm(cs[0])}, {c.nm(cs[1])} and "
                         f"{c.nm(cs[2])} from the quickest full swing (there and back) to the slowest (colours only).",
                         gt, mc4(rng, gt, rng.sample(perms, len(perms))), True,
                         fps_for([min(c.T[x] for x in cs) / 2])))
    wins = [(a, b) for a in c.grid for b in c.grid if 2.0 <= b - a <= 4.0]
    rng.shuffle(wins)
    for t0, t1 in wins[:40]:
        allev = [e for x in c.cs for e in c.cr[x]]
        if any(abs(e - t0) < 0.25 or abs(e - t1) < 0.25 for e in allev):
            continue
        if c.staggered and any(c.held_at(x, t0) for x in c.cs):
            continue
        if c.damped and any(c.env(x, e) < 0.5 * c.tick for x in c.cs for e in c.cr[x] if t0 <= e <= t1):
            continue
        k = sum(1 for e in allev if t0 <= e <= t1)
        if k == 0:
            continue
        margins = [min(abs(e - t0), abs(e - t1)) for e in allev] + [c.T[x] / 2 for x in c.cs]
        out.append(Q("total_crossings", {"t0": t0, "t1": t1}, f"Between {t0:.1f} s and {t1:.1f} s on the clock, how "
                     f"many times in total do the {thing}s pass their rest marks (count every {thing})? " + RULER,
                     k, int_opts(rng, k, 0, 60), True, max(align(t0), align(t1), fps_for(margins))))
        break
    whichs = ["widest", "narrowest"]
    if c.n >= 3:
        whichs += ["largest", "smallest"]
    if c.orient == "pendulum":
        whichs += ["longest", "shortest"]
    if c.staggered:
        whichs += ["first_released", "last_released"]
    rng.shuffle(whichs)
    made = 0
    for w in whichs:
        if made == 2:
            break
        col = _extreme(c, w)
        if col is None or not c.periods_ok(col) or not safe1(c.T[col]):
            continue
        T = c.T[col]
        out.append(Q("period_of_extreme", {"which": w}, f"How long does {EXTREME_WORDS[w].format(t=thing)} take for "
                     "one full swing, there and back? " + (RULER if w in ("widest", "narrowest") else ""), fmt_s(T),
                     secs_opts(rng, T, _period_cands(T), max(0.3, 0.15 * T)), True, c.period_q_fps(col)))
        made += 1
    if c.staggered:
        for col in rng.sample(c.cs, min(2, c.n)):
            o = c.by[col]
            T = c.T[col]
            if o["t_rel"] + T <= c.dur - sim.EDGE and safe1(T):
                out.append(Q("return_time", {"c": col}, f"{c.nm(col).capitalize()} starts held by a stop post. After it "
                             "is let go, how long does it take to first come back to the spot where it was held?",
                             fmt_s(T), secs_opts(rng, T, _period_cands(T), max(0.3, 0.15 * T)), True,
                             max(4, fps_for([T / 2]))))
    if c.damped:
        for col in rng.sample(c.cs, min(2, c.n)):
            o = c.by[col]
            tu = c.tu[col]
            vals = [abs(c.u(col, t)) / c.tick for t, _ in tu]
            edge = [v for (t, _), v in zip(tu, vals) if t < sim.EDGE or t > c.dur - sim.EDGE]
            # turns in the first / last 0.5 s must be clearly below n, so they never count
            ns = [n for n in range(1, 7) if all(abs(v - n) >= 0.25 for v in vals) and all(v <= n - 0.25 for v in edge)]
            ns = [n for n in ns if 0 < sum(v >= n for v in vals)]
            if not ns:
                continue
            n = rng.choice(ns)
            k = sum(v >= n for v in vals)
            slack = min(abs(v - n) for v in vals) * c.tick - 2
            f = max(fps_sampling(o["A"], sim.omega_d(o), slack), fps_for([c.T[col] / 2]))
            out.append(Q("decay_count", {"c": col, "n": n}, f"Each time {c.nm(col)} stops and turns back, note how far it "
                         f"is from its rest mark. On how many of these turns is it at least {n} tick spacing"
                         f"{'s' if n > 1 else ''} away? " + RULER, k, int_opts(rng, k, 0, 30), True, f))
    return out


# ------------------------------------------------------------------ L5 counterfactual
CHANGE_TEXT = {
    "mass_x4": "were 4 times as heavy (same spring)",
    "mass_d4": "had only a quarter of its mass (same spring)",
    "stiff_x4": "hung on a spring 4 times as stiff (same mass)",
    "stiff_d4": "hung on a spring only a quarter as stiff (same mass)",
    "amp_x2": "were started twice as far from its rest mark (twice the swing)",
    "length_x4": "hung on a string 4 times as long",
    "length_d4": "hung on a string a quarter as long",
    "bob_x4": "had a bob 4 times as heavy (same string)",
}


def changed_osc(c: Ctx, col, change):
    """The oscillator after the change, in sim.py's closed-form parameters."""
    o = dict(c.by[col])
    if "L" in o:
        th0 = sim.theta0(o)
        x0, v0 = th0 * math.cos(o["phi"]), -th0 * o["omega"] * math.sin(o["phi"])
        if change == "length_x4":
            o["L"] *= 4
        elif change == "length_d4":
            o["L"] /= 4
        elif change == "amp_x2":
            x0, v0 = 2 * x0, 2 * v0
        o["omega"] = math.sqrt(c.s["g"] / o["L"])
        th = math.hypot(x0, v0 / o["omega"])
        o["phi"] = math.atan2(-v0 / o["omega"], x0)
        o["A"] = o["L"] * math.sin(th)
        return o
    m, k, b = o["m"], o["k"], o["b"]
    if change == "mass_x4":
        m *= 4
    elif change == "mass_d4":
        m /= 4
    elif change == "stiff_x4":
        k *= 4
    elif change == "stiff_d4":
        k /= 4
    w = math.sqrt(k / m)
    if o["mode"] == "free":
        x0, v0 = o["A"] * math.cos(o["phi"]), -o["A"] * o["omega"] * math.sin(o["phi"])
        if change == "amp_x2":
            x0, v0 = 2 * x0, 2 * v0
        o["A"], o["phi"] = math.hypot(x0, v0 / w), math.atan2(-v0 / w, x0)
    elif change == "amp_x2":
        o["A"] *= 2
    o["omega"] = w
    if o["mode"] == "damped":
        o["zeta"] = b / (2 * math.sqrt(k * m))
    return o


def level5(c: Ctx, rng):
    out = []
    thing = c.thing
    changes = ["length_x4", "length_d4", "amp_x2", "bob_x4"] if c.orient == "pendulum" else \
        ["mass_x4", "mass_d4", "stiff_x4", "stiff_d4", "amp_x2"]
    combos = [(x, ch) for x in c.cs for ch in changes]
    rng.shuffle(combos)
    made = 0
    for col, ch in combos:
        if made == 3:
            break
        if not c.periods_ok(col):
            continue
        o2 = changed_osc(c, col, ch)
        T2 = sim.period(o2)
        if not safe1(T2) or not 0.3 <= T2 <= 7.0:
            continue
        T = c.T[col]
        cands = [T, 2 * T2, T2 / 2, 4 * T, T / 4, T * 2, T / 2, T * math.sqrt(2), T2 + 0.5, T2 - 0.5]
        out.append(Q("cf_period", {"c": col, "change": ch}, f"Suppose {c.nm(col)} {CHANGE_TEXT[ch]}, with everything "
                     "else unchanged. How long would one full swing (there and back) take?", fmt_s(T2),
                     secs_opts(rng, T2, cands, max(0.3, 0.15 * T2)), True, c.period_q_fps(col)))
        made += 1
    rng.shuffle(combos)
    wins = [(a, b) for a in c.grid for b in c.grid if 2.0 <= b - a <= 4.0]
    made = 0
    for col, ch in combos:
        if made == 2:
            break
        if not c.periods_ok(col) or ch == "bob_x4":
            continue
        o2 = changed_osc(c, col, ch)
        T2 = sim.period(o2)
        if T2 / 2 < 0.4:
            continue
        ev = sim.crossings(o2, 0, c.dur + 1)
        for t0, t1 in rng.sample(wins, min(10, len(wins))):
            if c.staggered and t0 < o2["t_rel"] + 0.3:
                continue
            if any(abs(e - t0) < 0.25 or abs(e - t1) < 0.25 for e in ev):
                continue
            inside = [e for e in ev if t0 <= e <= t1]
            if c.damped and any(o2["A"] * math.exp(-o2["zeta"] * o2["omega"] * e) < 0.5 * c.tick for e in inside):
                continue
            k = len(inside)
            out.append(Q("cf_crossings", {"c": col, "change": ch, "t0": t0, "t1": t1}, f"Suppose {c.nm(col)} "
                         f"{CHANGE_TEXT[ch]}, starting from the same place and moving the same way at the start of its "
                         f"motion. How many times would it pass its rest mark between {t0:.1f} s and {t1:.1f} s on "
                         "the clock? " + RULER, k, int_opts(rng, k, 0, 30), True, c.period_q_fps(col)))
            made += 1
            break
    if c.damped:
        for col in rng.sample(c.cs, min(2, c.n)):
            o = c.by[col]
            k = round(o["A"] / c.tick)
            late = [abs(c.u(col, t)) / c.tick for t, _ in c.tu[col]]
            cands = [round(v) for v in late] + [k + 1, k - 1, k + 2]
            cands = [v for v in cands if v >= 0]
            out.append(Q("cf_no_damping", {"c": col}, f"The plunger beside each spring slows the masses down. If "
                         f"{c.nm(col)} had no plunger, how far from its rest mark would it get at most, in tick spacings "
                         "(to the nearest whole tick)? " + RULER, k, mc4(rng, k, list(dict.fromkeys(cands))), True, 1))
    if c.staggered:
        for col in rng.sample(c.cs, min(2, c.n)):
            o = c.by[col]
            for d in rng.sample([0.5, 1.0, 1.5, 2.0], 4):
                te = o["t_rel"] + d + c.T[col] / 4
                fa, ok = event_frame(te)
                if not ok or te > c.dur - EV:
                    continue
                tenths = clock_tenths(fa)
                out.append(Q("cf_release_later", {"c": col, "d": d}, f"{c.nm(col).capitalize()} starts held by a stop "
                             f"post. If it had been let go {d:g} s later, at what clock reading would it first pass its "
                             "rest mark? " + RULER, fmt_clock(tenths),
                             clock_opts(rng, tenths, [tenths - 10 * d, 10 * (te + c.T[col] / 2)]), True,
                             max(4, fps_for([c.T[col] / 4]))))
                break
    return out


# ------------------------------------------------------------------ assembly
def _pick(rng, cands, n):
    by_key = {}
    for cand in cands:
        by_key.setdefault(cand[0], []).append(cand)
    keys = list(by_key)
    rng.shuffle(keys)
    chosen, used = [], set()
    while len(chosen) < n:
        avail = [k for k in keys if by_key[k]]
        if not avail:
            break
        k = ([k for k in avail if k not in used] or avail)[0]
        chosen.append(by_key[k].pop(rng.randrange(len(by_key[k]))))
        used.add(k)
        keys.remove(k)
        keys.append(k)
    return chosen


def build(scene: dict, rng: random.Random) -> list[dict]:
    c = Ctx(scene)
    makers = {1: level1, 2: level2, 3: level3, 4: level4, 5: level5}
    for attempt in range(20):          # a thin pool redraws the template parameters (same rng stream: deterministic)
        made = {}
        for lv in range(1, 6):
            seen, ok = set(), []
            for q in makers[lv](c, random.Random(rng.random())):
                if q is None:
                    continue
                sig = (q[0], json.dumps(q[1], sort_keys=True))
                if sig not in seen:
                    seen.add(sig)
                    ok.append(q)
            made[lv] = ok
        feasible = [cmb for cmb in TEN_SPLITS if all(len(made[lv]) >= cmb[lv - 1] for lv in range(1, 6))]
        if feasible:
            break
    else:
        raise ValueError("not enough question candidates for this scene")
    REDRAWS.append(attempt)
    counts = rng.choice(feasible)
    qs = []
    for lv in range(1, 6):
        for key, params, text, gt, opts, letter, motion in _pick(rng, made[lv], counts[lv - 1]):
            qs.append(L.question(lv, key, text, gt, opts, letter, motion, params, idx=len(qs) + 1))
    return qs
