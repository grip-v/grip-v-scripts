"""10-question ladder for vessels_and_clocks (common/ladder5.py, LADDER = "ten").

Vessels are named by the colour of their rim and plinth. Levels are read on
the shared ruler (0-10 marks, numbered on the even marks, gridlines at every
mark). The amount of water is level x width (all vessels are equally deep,
which the questions say whenever it matters). Times are clock readings.

Margins: level readings sit at least 0.2 mark from a rounding boundary (more
if the level moves during the sampling window), compared levels differ by
at least 0.6 mark, compared amounts of water by at least 15 %, compared
rates by at least 25 %, compared events by at least 0.3 s, and clock-time
options are at least 0.6 s apart. Every question has four distinct options;
what-ifs are re-simulated (closed form here, numerical integration of the
measured flow laws in answers.py).
"""
from __future__ import annotations
import itertools
import json
import math
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from common import ladder5 as L  # noqa: E402

import sim  # noqa: E402

LADDER = "ten"
TEN_SPLITS = [c for c in itertools.product(*(range(lo, hi + 1) for lo, hi in L.TEN_RANGES)) if sum(c) == L.TEN_TOTAL]
FPS = sim.FPS
DEPTH = " (All vessels are equally deep, so a wider vessel holds more water per mark.)"
NONE_FIRST = "none of them"


# ------------------------------------------------------------------ helpers
def fps_for(gaps, floor=1):
    g = min(gaps) if gaps else 99.0
    return int(min(24, max(floor, math.ceil(1.0 / g - 1e-9))))


def safe(x, m=0.2):
    """x sits at least m away from a .5 rounding boundary."""
    return abs((x - math.floor(x)) - 0.5) >= m


def fmt_clock(tenths):
    return f"{tenths / 10:.1f} s"


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


def int_opts(rng, gt, lo, hi, extra=()):
    near = [v for v in (gt + 1, gt - 1, gt + 2, gt - 2, gt + 3, gt - 3) if lo <= v <= hi]
    rng.shuffle(near)
    near.sort(key=lambda v: abs(v - gt))
    ex = [e for e in extra if lo <= e <= hi and e != gt]
    return mc4(rng, gt, ex[:1] + near)


def clock_opts(rng, tenths, extra=(), words=()):
    """gt clock reading + readings at least 6 tenths from it and from each other (+ optional word options)."""
    cands = [round(e) for e in extra] + [tenths + d for d in (-12, -6, 6, 12, 18, -18, 9, -9)]
    keep = []
    for c in cands:
        if c >= 0 and all(abs(c - u) >= 6 for u in keep + [tenths]):
            keep.append(c)
    head, tail = keep[:len(extra)], keep[len(extra):]
    rng.shuffle(tail)
    out = mc4(rng, fmt_clock(tenths), list(words) + [fmt_clock(c) for c in head + tail])
    return out


def word_time_opts(rng, gt_word, extra, words):
    """gt is a phrase ('it would not ... before the clip ends'); distractors are clock readings."""
    keep = []
    for c in [round(e) for e in extra] + [extra[0] + d for d in (6, 12, -6, 18)] if extra else []:
        if c >= 0 and all(abs(c - u) >= 6 for u in keep):
            keep.append(c)
    return mc4(rng, gt_word, list(words) + [fmt_clock(c) for c in keep])


def Q(key, params, text, gt, opts, motion, fps):
    if opts is None:
        return None
    return (key, dict(params, min_fps=fps), text, gt, opts[0], opts[1], motion)


# ------------------------------------------------------------------ context
class Ctx:
    def __init__(self, scene):
        self.s = scene
        self.vs = scene["vessels"]
        self.n = len(self.vs)
        self.tl = sim.t_last(scene)
        self.cols = [v["colour"] for v in self.vs]
        self.evs = sim.events(scene)
        self.solo = {e["i"]: e for e in self.evs if e["kind"] != "valve"}
        self.valve = {e["i"]: e for e in self.evs if e["kind"] == "valve"}
        self.h0 = [v["h0"] for v in self.vs]
        self.hend = [sim.level(scene, i, self.tl) for i in range(self.n)]
        self.vend = [self.hend[i] * self.vs[i]["w"] for i in range(self.n)]
        self.roles = [v["role"] for v in self.vs]

    def nm(self, i):
        return f"the {self.cols[i]} vessel"

    def names(self, first=(), add=()):
        """Vessel option texts: `first` indices, then the rest in order, then extra phrases."""
        order = list(first) + [i for i in range(self.n) if i not in first]
        return [self.nm(i) for i in order] + list(add)

    def ev_id(self, e):
        return ["valve", e["i"]] if e["kind"] == "valve" else ["solo", e["i"]]

    def desc(self, e):
        if e["kind"] == "valve":
            return f"the valve between the {self.cols[e['i']]} and {self.cols[e['j']]} vessels opens"
        return f"{self.nm(e['i'])} {e['kind']}"

    def short(self, e):
        if e["kind"] == "valve":
            return f"{self.cols[e['i']]}-{self.cols[e['j']]} valve opens"
        return f"{self.cols[e['i']]} {'overflows' if e['kind'] == 'overflows' else 'empties'}"

    def clock(self, t):
        return sim.clock_tenths(sim.first_frame(t))

    def what(self, i):
        return "overflow" if self.roles[i] == "tap" else "run empty"

    def device(self, i):
        return "tap" if self.roles[i] == "tap" else "drain"


def _rank_ok(vals, ratio=1.15):
    s = sorted(vals, reverse=True)
    return len(s) < 2 or (s[0] > 0 and s[0] >= ratio * s[1])


# ------------------------------------------------------------------ L1 observation
def level1(c: Ctx, rng):
    out = []
    n = c.n
    if "tap" in c.roles:
        gt = sum(1 for e in c.solo.values() if e["kind"] == "overflows")
        out.append(Q("n_overflow", {}, "How many vessels overflow during the clip?", gt,
                     int_opts(rng, gt, 0, n), True, 1))
    if {"steady", "leaky"} & set(c.roles):
        gt = sum(1 for e in c.solo.values() if e["kind"] == "runs empty")
        out.append(Q("n_empty_end", {}, "How many vessels are empty at the end of the clip?", gt,
                     int_opts(rng, gt, 0, n), True, 1))
    gt = c.roles.count("tap")
    out.append(Q("n_taps", {}, "How many vessels are being filled by a tap?", gt, int_opts(rng, gt, 0, n), False, 1))
    gt = sum(1 for i in range(n) if c.roles[i] == "tap" or (c.roles[i] == "pair" and c.hend[i] > c.h0[i]))
    out.append(Q("n_rising", {}, "How many vessels have a rising water level at some point during the clip?", gt,
                 int_opts(rng, gt, 0, n), True, 1))
    se = sorted(c.solo.values(), key=lambda e: e["t"])
    if se:
        i = se[0]["i"]
        gaps = [se[1]["t"] - se[0]["t"]] if len(se) > 1 else []
        out.append(Q("first_event", {}, "Which vessel is the first to overflow or run empty?", c.nm(i),
                     mc4(rng, c.nm(i), c.names([e["i"] for e in se], [NONE_FIRST])), True, fps_for(gaps)))
    ws = [v["w"] for v in c.vs]
    i = max(range(n), key=lambda k: ws[k])
    out.append(Q("widest", {}, "Which vessel is the widest?", c.nm(i),
                 mc4(rng, c.nm(i), c.names(add=["they are all equally wide"])), False, 1))
    if _rank_ok(c.h0, 1.0) and sorted(c.h0)[-1] - sorted(c.h0)[-2] >= 0.6:
        i = max(range(n), key=lambda k: c.h0[k])
        out.append(Q("highest_start", {}, "Which vessel starts with the highest water level?", c.nm(i),
                     mc4(rng, c.nm(i), c.names(add=["they all start at the same level"])), False, 1))
    return out


# ------------------------------------------------------------------ L2 relational
def level2(c: Ctx, rng):
    out = []
    n = c.n
    for a, b in itertools.permutations(range(n), 2):
        if c.roles[a] == "steady" and c.roles[b] == "leaky":
            x, y = (a, b) if rng.random() < 0.5 else (b, a)
            out.append(Q("steady_vs_leaky", {"x": x, "y": y},
                         f"{c.nm(x).capitalize()} and {c.nm(y)} both drain. Which one's level falls at a steady "
                         "rate, without slowing down as it gets lower?", c.nm(a),
                         mc4(rng, c.nm(a), [c.nm(b), "both fall at a steady rate", "both slow down as they empty"]),
                         True, 1))
    for x, y in itertools.combinations(range(n), 2):
        if rng.random() < 0.5:
            x, y = y, x
        vx, vy = c.vend[x], c.vend[y]
        hx_, hy_ = c.hend[x], c.hend[y]
        if max(hx_, hy_) >= 0.5 and max(vx, vy) >= 1.15 * min(vx, vy):
            g = x if vx > vy else y
            out.append(Q("more_water_end", {"x": x, "y": y},
                         f"At the end of the clip, which holds more water: {c.nm(x)} or {c.nm(y)}?{DEPTH}", c.nm(g),
                         mc4(rng, c.nm(g), [c.nm(x), c.nm(y), "they hold the same amount", "neither holds any water"]),
                         True, 1))
        hx, hy = c.hend[x], c.hend[y]
        if abs(hx - hy) >= 0.6:
            g = x if hx > hy else y
            out.append(Q("higher_end", {"x": x, "y": y},
                         f"At the end of the clip, whose water level is higher: {c.nm(x)} or {c.nm(y)}?", c.nm(g),
                         mc4(rng, c.nm(g), [c.nm(x), c.nm(y), "both are at the same level",
                                            "neither has any water left"]), True, 1))
        if c.roles[x] == c.roles[y] == "tap":
            rx, ry = c.vs[x]["q"] / c.vs[x]["w"], c.vs[y]["q"] / c.vs[y]["w"]
            if max(rx, ry) >= 1.25 * min(rx, ry):
                g = x if rx > ry else y
                out.append(Q("faster_rise", {"x": x, "y": y},
                             f"Before either overflows, whose level rises faster: {c.nm(x)} or {c.nm(y)}?", c.nm(g),
                             mc4(rng, c.nm(g), [c.nm(x), c.nm(y), "both rise at the same speed", "neither rises"]),
                             True, 1))
    for ea, eb in itertools.combinations(c.evs, 2):
        if rng.random() < 0.5:
            ea, eb = eb, ea
        first = ea if ea["t"] < eb["t"] else eb
        out.append(Q("earlier_event", {"a": c.ev_id(ea), "b": c.ev_id(eb)},
                     f"Which happens first: {c.desc(ea)}, or {c.desc(eb)}?", c.desc(first),
                     mc4(rng, c.desc(first), [c.desc(ea), c.desc(eb), "both happen at the same moment",
                                              "neither happens during the clip"]),
                     True, fps_for([abs(ea["t"] - eb["t"])])))
    for a, b in c.s["pairs"]:
        lo = a if c.h0[a] < c.h0[b] else b
        out.append(Q("valve_flow", {"x": a, "y": b},
                     f"When the valve between the {c.cols[a]} and {c.cols[b]} vessels opens, which vessel's level "
                     "rises?", c.nm(lo),
                     mc4(rng, c.nm(lo), [c.nm(a), c.nm(b), "neither level changes", "both levels rise"]), True, 1))
    return out


# ------------------------------------------------------------------ L3 measurement
def level3(c: Ctx, rng):
    out = []
    n = c.n
    for i in range(n):
        h = c.hend[i]
        if 0.3 <= h <= sim.H - 0.3 and safe(h):
            gt = round(h)
            out.append(Q("level_end", {"x": i}, f"At the end of the clip, what is the water level in {c.nm(i)} on "
                         "the ruler (to the nearest mark)?", gt, int_opts(rng, gt, 0, 10), True, 1))
        d = c.h0[i] - h
        if d >= 0.8 and safe(d):
            gt = round(d)
            out.append(Q("drop_marks", {"x": i}, f"By how many marks does the level in {c.nm(i)} fall during the "
                         "clip (to the nearest mark)?", gt, int_opts(rng, gt, 0, 10), True, 1))
    for i, e in c.solo.items():
        g = c.clock(e["t"])
        what = "first overflows" if e["kind"] == "overflows" else "runs empty"
        out.append(Q("event_clock", {"x": i}, f"What does the clock show when {c.nm(i)} {what}?", fmt_clock(g),
                     clock_opts(rng, g), True, 4))
    for a, e in c.valve.items():
        g = c.clock(e["t"])
        out.append(Q("valve_clock", {"x": a}, f"What does the clock show when the valve between the {c.cols[a]} "
                     f"and {c.cols[e['j']]} vessels opens?", fmt_clock(g), clock_opts(rng, g), True, 4))
    secs = list(range(2, int(c.tl)))
    for i in range(n):
        if not secs:
            break
        s = rng.choice(secs)
        f = (s * 10 * FPS + 9) // 10                        # first frame whose clock shows s.0
        t = f / FPS
        h = sim.level(c.s, i, t)
        drift = max(abs(sim.level(c.s, i, t + 0.25) - h), abs(h - sim.level(c.s, i, max(0.0, t - 0.25))))
        if 0.3 <= h <= sim.H - 0.3 and safe(h, 0.2 + drift) and drift > 0.02:
            gt = round(h)
            out.append(Q("level_at_clock", {"x": i, "sec": s}, f"What is the water level in {c.nm(i)} when the "
                         f"clock shows {s}.0 s (to the nearest mark)?", gt, int_opts(rng, gt, 0, 10), True, 4))
    for i in range(n):
        if c.roles[i] != "tap" or (i in c.solo and c.solo[i]["t"] < 2.0):
            continue
        r = c.vs[i]["q"] / c.vs[i]["w"]
        if safe(2 * r) and round(2 * r) >= 1:
            g = round(2 * r) / 2
            fm = lambda x: f"{x:.1f} marks per second"  # noqa: E731
            cands = [g + d for d in (0.5, -0.5, 1.0, -1.0, 1.5) if g + d >= 0.5]
            rng.shuffle(cands)
            cands.sort(key=lambda x: abs(x - g))
            out.append(Q("rise_rate", {"x": i}, f"While the tap fills {c.nm(i)}, how fast does its level rise (to "
                         "the nearest half mark per second)?", fm(g), mc4(rng, g, cands, fm), True, 1))
    return out


# ------------------------------------------------------------------ L4 integration
def _order_opts(rng, seq):
    gt = " → ".join(seq)
    wrong = []
    for k in range(len(seq) - 1):
        alt = list(seq)
        alt[k], alt[k + 1] = alt[k + 1], alt[k]
        wrong.append(" → ".join(alt))
    rng.shuffle(wrong)
    perms = [" → ".join(p) for p in itertools.permutations(seq)]
    rng.shuffle(perms)
    return mc4(rng, gt, wrong + [" → ".join(seq[::-1])] + perms)


def level4(c: Ctx, rng):
    out = []
    n = c.n
    if len(c.evs) >= 3:
        for trio in itertools.combinations(c.evs, 3):
            gaps = [b["t"] - a["t"] for a, b in zip(trio, trio[1:])]
            listed = list(trio)
            rng.shuffle(listed)
            out.append(Q("event_order", {"ev": [c.ev_id(e) for e in listed]},
                         "In what order do these happen: " + "; ".join(c.short(e) for e in listed) + "?",
                         " → ".join(c.short(e) for e in trio), _order_opts(rng, [c.short(e) for e in trio]),
                         True, fps_for(gaps)))
    if _rank_ok(c.vend) and c.hend[max(range(n), key=lambda k: c.vend[k])] >= 0.5:
        i = max(range(n), key=lambda k: c.vend[k])
        out.append(Q("most_water_end", {}, f"At the end of the clip, which vessel holds the most water?{DEPTH}",
                     c.nm(i), mc4(rng, c.nm(i), c.names(sorted(range(n), key=lambda k: -c.hend[k]),
                                                        ["they all hold the same amount"])), True, 1))
    for a, b in c.s["pairs"]:
        sh = sim.shared(c.s, [a, b])
        if safe(sh):
            gt = round(sh)
            naive = round((c.h0[a] + c.h0[b]) / 2)
            out.append(Q("shared_level", {"x": a, "y": b}, f"After the valve between the {c.cols[a]} and "
                         f"{c.cols[b]} vessels opens, the two levels even out. At which mark do they settle (to the "
                         "nearest mark)?", gt, int_opts(rng, gt, 0, 10, [naive]), True, 1))
    for ea, eb in itertools.combinations(c.evs, 2):
        d = c.clock(eb["t"]) - c.clock(ea["t"])
        cands = [d + k for k in (6, -6, 12, -12, 18) if d + k > 0]
        rng.shuffle(cands)
        cands.sort(key=lambda x: abs(x - d))
        keep = []
        for x in cands:
            if all(abs(x - u) >= 6 for u in keep):
                keep.append(x)
        out.append(Q("time_between", {"a": c.ev_id(ea), "b": c.ev_id(eb)}, f"By the clock, how much time passes "
                     f"between the moment {c.desc(ea)} and the moment {c.desc(eb)}?", fmt_clock(d),
                     mc4(rng, fmt_clock(d), [fmt_clock(x) for x in keep]), True, 4))
    losers = [i for i in range(n) if c.h0[i] - c.hend[i] >= 0.5]
    if len(losers) >= 2:
        lost = {i: (c.h0[i] - c.hend[i]) * c.vs[i]["w"] for i in losers}
        if _rank_ok(list(lost.values())):
            i = max(lost, key=lost.get)
            out.append(Q("most_lost", {}, f"Which vessel loses the most water during the clip?{DEPTH}", c.nm(i),
                         mc4(rng, c.nm(i), c.names(sorted(losers, key=lambda k: -(c.h0[k] - c.hend[k])),
                                                   ["they all lose the same amount"])), True, 1))
    return out


# ------------------------------------------------------------------ L5 counterfactual
def _cf_time(c: Ctx, rng, key, params, text, i, t_new):
    """A what-if clock time for vessel i (or 'would not ... before the clip ends')."""
    word = f"it would not {c.what(i)} before the clip ends"
    orig = [c.clock(c.solo[i]["t"])] if i in c.solo else []
    if t_new <= c.tl - 0.3 and t_new >= 0.3 and sim.frac_ok(t_new):
        g = c.clock(t_new)
        return Q(key, params, text, fmt_clock(g), clock_opts(rng, g, orig, [word]), True, 4)
    if t_new >= c.tl + 0.3:
        base = orig or [c.clock(min(c.tl - 1.0, t_new * 0.6))]
        return Q(key, params, text, word, word_time_opts(rng, word, base + [base[0] - 6], []), True, 4)
    return None


def level5(c: Ctx, rng):
    out = []
    n = c.n
    for i in c.solo:
        t2 = 2 * c.solo[i]["t"]
        out.append(_cf_time(c, rng, "cf_wider_time", {"x": i},
                            f"Suppose {c.nm(i)} were twice as wide, with the same starting level and the same "
                            f"{c.device(i)}. What would the clock show when it {'overflows' if c.roles[i] == 'tap' else 'runs empty'}?",
                            i, t2))
    for i in range(n):
        if c.roles[i] == "pair":
            continue
        for dh in rng.sample((1, 2, -1, -2), 4):
            h2 = c.h0[i] + dh
            if not 0.5 <= h2 <= 9.5:
                continue
            t_new = sim.law_time(dict(c.vs[i], h0=h2))
            q = _cf_time(c, rng, "cf_fuller_time", {"x": i, "dh": dh},
                         f"Suppose {c.nm(i)} had started {abs(dh)} mark{'s' if abs(dh) > 1 else ''} "
                         f"{'higher' if dh > 0 else 'lower'}, with the same {c.device(i)}. What would the clock show "
                         f"when it {'overflows' if c.roles[i] == 'tap' else 'runs empty'}?", i, t_new)
            if q:
                out.append(q)
                break
        v2 = dict(c.vs[i], w=2 * c.vs[i]["w"])
        h2 = sim.solo_level(v2, c.tl)
        t2 = sim.law_time(v2)
        edge_ok = abs(t2 - c.tl) >= 0.3
        if edge_ok and (h2 in (0.0, sim.H) or (0.3 <= h2 <= sim.H - 0.3 and safe(h2))):
            gt = round(h2)
            out.append(Q("cf_wider_level", {"x": i}, f"Suppose {c.nm(i)} were twice as wide, with the same starting "
                         f"level and the same {c.device(i)}. What would its level be at the end of the clip (to the "
                         "nearest mark)?", gt, int_opts(rng, gt, 0, 10, [round(c.hend[i])]), True, 1))
    for a, b in c.s["pairs"]:
        for x in rng.sample((a, b), 2):
            done = False
            for e in rng.sample((1, 2, 3, -1, -2), 5):
                hx = c.h0[x] + e
                if not 0.3 <= hx <= 9.7:
                    continue
                alt = dict(c.s, vessels=[dict(v, h0=hx) if k == x else v for k, v in enumerate(c.vs)])
                sh = sim.shared(alt, [a, b])
                if safe(sh):
                    gt = round(sh)
                    out.append(Q("cf_shared_extra", {"x": x, "y": b if x == a else a, "dh": e},
                                 f"Suppose {c.nm(x)} had started {abs(e)} mark{'s' if abs(e) > 1 else ''} "
                                 f"{'higher' if e > 0 else 'lower'}, everything else the same. At which mark would the "
                                 f"{c.cols[a]} and {c.cols[b]} vessels settle after the valve opens (to the nearest "
                                 "mark)?", gt, int_opts(rng, gt, 0, 10, [round(sim.shared(c.s, [a, b]))]), True, 1))
                    done = True
                    break
            if done:
                break
    if c.s["pairs"]:
        paired = {k for p in c.s["pairs"] for k in p}
        vol = [c.h0[i] * c.vs[i]["w"] if i in paired else c.vend[i] for i in range(n)]
        if _rank_ok(vol):
            i = max(range(n), key=lambda k: vol[k])
            what = "the valve had" if len(c.s["pairs"]) == 1 else "neither valve had"
            out.append(Q("cf_valve_never", {}, f"Suppose {what} ever opened (everything else the same). Which vessel "
                         f"would hold the most water at the end of the clip?{DEPTH}", c.nm(i),
                         mc4(rng, c.nm(i), c.names(sorted(range(n), key=lambda k: -c.vend[k]), ["they would all "
                                                                                                 "hold the same"])),
                         True, 1))
    for i in range(n):
        if c.roles[i] != "tap":
            continue
        times = {k: sim.law_time(c.vs[k]) for k in range(n) if c.roles[k] != "pair"}
        times[i] = sim.law_time(dict(c.vs[i], q=2 * c.vs[i]["q"]))
        srt = sorted(times, key=times.get)
        t1 = times[srt[0]]
        t2 = times[srt[1]] if len(srt) > 1 else 99.0
        if EDGE_OK(t1, c.tl) and t2 - t1 >= 0.3:
            g = srt[0]
            out.append(Q("cf_faster_tap", {"x": i}, f"Suppose the tap on {c.nm(i)} poured twice as fast (everything "
                         "else the same). Which vessel would be the first to overflow or run empty?", c.nm(g),
                         mc4(rng, c.nm(g), c.names(srt, [NONE_FIRST])), True, fps_for([t2 - t1])))
    return out


def EDGE_OK(t, tl):
    return 0.3 <= t <= tl - 0.3


# ------------------------------------------------------------------ assembly
def _pick(rng, cands, n):
    """Distinct templates first, then repeats with other params."""
    by_key = {}
    for cand in cands:
        by_key.setdefault(cand[0], []).append(cand)
    keys = sorted(by_key)
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
    if not feasible:
        raise ValueError("not enough question candidates for this scene")
    counts = rng.choice(feasible)
    qs = []
    for lv in range(1, 6):
        for key, params, text, gt, opts, letter, motion in _pick(rng, made[lv], counts[lv - 1]):
            qs.append(L.question(lv, key, text, gt, opts, letter, motion, params, idx=len(qs) + 1))
    return qs
