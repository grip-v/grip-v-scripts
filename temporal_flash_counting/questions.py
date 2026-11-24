"""10-question, 5-rung ladder for temporal_flash_counting (common/ladder5.py, LADDER = "ten").

Nothing in the frame is labelled except the timer, so questions name a shape
by colour + type ("the teal crescent"; "the green lamp" in the signal
scenario; every colour + type is unique in a clip) and a moment by a timer
reading ("when the timer reads 4.5 s"). Every question has 4 options and
stores params["min_fps"]: 24 / k for the largest frame step k at which
sampling every k-th frame still catches every relied-on flash, dark gap,
onset order and timing margin (and still tells short from long flashes when
the question depends on length).

Templates per level (derivation keys):
  L1 flash_count, n_shapes | grid_rows | grid_cols, first_flasher, last_flasher, white_ever, pulse_shape
  L2 relation, more_by, earlier_of_four, longest_flash, lit_during
  L3 long_count, white_count, period, first_time, count_window
  L4 most_frequent, least_frequent, steady, trigger, total_flashes, total_white, busiest_window, sequence
  L5 cf_no_trigger, cf_chain, cf_remove_most, cf_long_only, cf_white_only, cf_no_white_most, cf_stop
"""
from __future__ import annotations
import math
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from common import ladder5 as L  # noqa: E402

import sim  # noqa: E402

LADDER = "ten"
FPS = sim.FPS
MAX_TOTAL = 24


def fps_for(steps, sep=None) -> float:
    """24 / k for the largest frame step k <= every relied-on interval, that
    also keeps short and long flashes apart (sep: [(longest short, shortest long)])."""
    top, k = max(1, int(min(steps))), 1
    while k < top and not any(lm // (k + 1) <= -(-sm // (k + 1)) for sm, lm in (sep or [])):
        k += 1                          # every step up to k must work, not just k itself
    return math.ceil(FPS / k * 10) / 10


def margin_ok(a: int, b: int) -> bool:
    """Counts differ by a clear margin (>= 1 and >= 10 %)."""
    return a - b >= max(1, math.ceil(0.1 * a))


def unique_top(vals: dict):
    order = sorted(vals, key=lambda k: (-vals[k], k))
    if len(order) > 1 and margin_ok(vals[order[0]], vals[order[1]]):
        return order[0], order
    return None, order


def seq_text(pattern) -> str:
    return ", ".join("long" if p else "short" for p in pattern)


def window_text(a: int, b: int | None) -> str:
    return f"from {sim.fmt_reading(a)} to the end" if b is None else f"from {sim.fmt_reading(a)} to {sim.fmt_reading(b)}"


def quarters(scene, m: int) -> list[int] | None:
    """Three timer readings (tenths, multiples of 0.5 s) that cut the clip into
    four stretches: for each quarter point, the nearest reading within 1 s of
    it that no flash onset comes within m frames of (None if there isn't one)."""
    nf = scene["n_frames"]
    onsets = [f["f0"] for f in scene["flashes"]]
    cuts = []
    for k in (1, 2, 3):
        q = sim.tenths(k * nf // 4)
        base = round(q / 5) * 5
        cands = sorted((v for v in range(base - 10, base + 11, 5) if v > (cuts[-1] if cuts else 0) + 5),
                       key=lambda v: (abs(v - q), v))
        ok = [v for v in cands if not any(abs(o - sim.frame_of_reading(v)) < m for o in onsets)]
        if not ok:
            return None
        cuts.append(ok[0])
    return cuts if cuts[-1] < sim.tenths(nf - 1) - 5 else None


class Ctx:
    def __init__(self, scene):
        self.s = scene
        self.n = len(scene["shapes"])
        self.nf = scene["n_frames"]
        self.lab = [sim.label(scene, i) for i in range(self.n)]
        self.noun = "lamp" if scene["scenario"] == "signal" else "shape"
        self.fl = [sim.flashes_of(scene, i) for i in range(self.n)]
        self.cnt = {i: len(self.fl[i]) for i in range(self.n)}
        self.M = sim.margin(scene)
        self.cut = sim.cut_s(scene)
        self.longs = {i: sum(sim.is_long(f, self.cut) for f in self.fl[i]) for i in range(self.n)}
        self.whites = {i: sum(f["white"] for f in self.fl[i]) for i in range(self.n)}
        self.first = {i: self.fl[i][0]["f0"] for i in range(self.n)}
        self.last = {i: self.fl[i][-1]["f0"] for i in range(self.n)}
        self.pulsed = {i: [p for p in scene["pulses"] if p["i"] == i] for i in range(self.n)}
        self.pre = (f" (A flash is a {self.noun} lighting up; growing and shrinking without lighting up is "
                    f"not a flash.)") if scene["pulses"] else ""
        self.all = list(range(self.n))

    def steps(self, shapes, extra=()):
        st = list(extra)
        for i in shapes:
            st += [f["f1"] - f["f0"] for f in self.fl[i]]
            st += [b["f0"] - a["f1"] for a, b in zip(self.fl[i], self.fl[i][1:])]
        return st

    def cut_ok(self, shapes, k) -> bool:
        """Sampling every k frames, a flash lit for L frames shows as ceil(L/k) or
        floor(L/k) lit samples (k frames each); every relied-on flash must stay
        on its side of the short / long cut."""
        cf = self.cut * FPS
        for i in shapes:
            for f in self.fl[i]:
                L_ = f["f1"] - f["f0"]
                if sim.is_long(f, self.cut) and (L_ // k) * k <= cf:
                    return False
                if not sim.is_long(f, self.cut) and -(-L_ // k) * k >= cf:
                    return False
        return True

    def fps(self, shapes, extra=(), length=False):
        top = max(1, int(min(self.steps(shapes, extra))))
        k = 1
        while k < top and (not length or self.cut_ok(shapes, k + 1)):
            k += 1                      # every step up to k must work, not just k itself
        return math.ceil(FPS / k * 10) / 10


def _q(level, key, text, gt, opts, letter, motion, params):
    return L.question(level, key, text, gt, opts, letter, motion, params)


def _num(rng, gt, extra=(), lo=0):
    return L.mc(rng, gt, list(extra) + L.near_ints(gt, lo))


def _shapes(rng, c: Ctx, correct, pool):
    """Options: the correct shape + 3 from `pool` (in that preference order)."""
    others = [k for k in pool if k != correct]
    seen, pick = set(), []
    for k in others:
        if k not in seen:
            seen.add(k)
            pick.append(k)
    pick = pick[:3]
    opts, a = L.mc(rng, c.lab[correct], [c.lab[k] for k in pick])
    return opts, a, [correct] + pick


# ---- L1 observation -------------------------------------------------------------
def level1(c: Ctx):
    s, out = c.s, []
    for x in c.all:
        def mk(rng, x=x):
            o, a = _num(rng, c.cnt[x], lo=1)
            return _q(1, "flash_count", f"How many times does the {c.lab[x]} flash?{c.pre}", c.cnt[x], o, a, True,
                      {"x": x, "min_fps": c.fps([x])})
        out.append(("flash_count", x, mk))
    static = fps_for([c.nf])
    if s.get("rows"):
        for key, val, word in (("grid_rows", s["rows"], "rows"), ("grid_cols", s["cols"], "columns")):
            def mk(rng, key=key, val=val, word=word):
                o, a = _num(rng, val, lo=1)
                return _q(1, key, f"How many {word} of shapes are in the grid?", val, o, a, False, {"min_fps": static})
            out.append((key, None, mk))
    else:
        def mk(rng):
            o, a = _num(rng, c.n, lo=2)
            where = "on the panel" if c.noun == "lamp" else "on the screen"
            return _q(1, "n_shapes", f"How many {c.noun}s are {where}?", c.n, o, a, False, {"min_fps": static})
        out.append(("n_shapes", None, mk))
    for key, when, word in (("first_flasher", c.first, "first"), ("last_flasher", c.last, "last")):
        order = sorted(c.all, key=lambda i: (when[i], i), reverse=(word == "last"))
        lead = abs(when[order[1]] - when[order[0]])
        if lead >= c.M:
            def mk(rng, key=key, order=order, lead=lead, word=word):
                rest = order[1:6]
                rng.shuffle(rest)
                o, a, _ = _shapes(rng, c, order[0], rest + order[6:])
                ix = 0 if word == "first" else -1
                runs = [c.fl[i][ix]["f1"] - c.fl[i][ix]["f0"] for i in c.all]
                text = (f"Which {c.noun} is the first to flash?" if word == "first" else
                        f"Which {c.noun} is the last to start a flash?")
                return _q(1, key, text + c.pre, c.lab[order[0]], o, a, True,
                          {"min_fps": fps_for(runs + [lead])})
            out.append((key, None, mk))
    nonwhite = [k for k in c.all if c.whites[k] == 0]
    for x in c.all:
        if c.whites[x] and len(nonwhite) >= 3:
            def mk(rng, x=x):
                o, a, ids = _shapes(rng, c, x, rng.sample(nonwhite, 3))
                return _q(1, "white_ever", f"Which of these {c.noun}s turns white during at least one of its flashes?",
                          c.lab[x], o, a, True, {"opts": ids, "min_fps": c.fps(ids)})
            out.append(("white_ever", x, mk))
    still = [k for k in c.all if not c.pulsed[k]]
    def grown_frames(x):
        """Longest run of frames on which x is clearly bigger (>= 9 px past its resting edge)."""
        r = s["shapes"][x]["r"]
        return max((sum(1 for k in range(p["f0"], p["f1"])
                        if sim.PULSE_SCALE * math.sin(math.pi * (k - p["f0"] + 0.5) / (p["f1"] - p["f0"])) * r >= 9)
                    for p in c.pulsed[x]), default=0)
    for x in c.all:
        if grown_frames(x) >= 4 and len(still) >= 3:
            def mk(rng, x=x):
                o, a, ids = _shapes(rng, c, x, rng.sample(still, 3))
                big = grown_frames(x)
                return _q(1, "pulse_shape", f"Which of these {c.noun}s grows bigger and then shrinks back at some point?",
                          c.lab[x], o, a, True, {"opts": ids, "min_fps": fps_for([big])})
            out.append(("pulse_shape", x, mk))
    return out


# ---- L2 relational ---------------------------------------------------------------
def level2(c: Ctx):
    s, out = c.s, []
    if s["pair"]:
        a_, b_ = s["pair"]
        gap2, x, y = sim.closest_pair(s, a_, b_)
        gt = sim.relation(x, y, c.M)
        d, ov = abs(x["f0"] - y["f0"]), min(x["f1"], y["f1"]) - max(x["f0"], y["f0"])
        # sampled onsets move by < k frames each, so the onset gap to the runner-up pair moves by < 2k
        ext = [gap2 // 2] + ([d] if d else []) + [abs(ov)]

        def mk(rng):
            o, a = L.mc(rng, gt, list(sim.RELATIONS))
            return _q(2, "relation", f"Look at the moment when the {c.lab[a_]} and the {c.lab[b_]} flash closest "
                      f"together in time. Which describes those two flashes?", gt, o, a, True,
                      {"pair": [a_, b_], "min_fps": c.fps([a_, b_], ext)})
        out.append(("relation", None, mk))
    for i in c.all:
        for j in c.all:
            if c.cnt[i] > c.cnt[j] and i != j:
                def mk(rng, i=i, j=j):
                    v = c.cnt[i] - c.cnt[j]
                    o, a = _num(rng, v, [0])
                    return _q(2, "more_by", f"How many more times does the {c.lab[i]} flash than the {c.lab[j]}?{c.pre}",
                              v, o, a, True, {"pair": [i, j], "min_fps": c.fps([i, j])})
                out.append(("more_by", (i, j), mk))
    glob = min(c.all, key=lambda i: (c.first[i], i))
    for x in c.all:
        later = [k for k in c.all if c.first[k] >= c.first[x] + c.M]
        if x != glob and len(later) >= 3:
            def mk(rng, x=x, later=later):
                o, a, ids = _shapes(rng, c, x, rng.sample(later, 3))
                lead = min(c.first[k] for k in ids[1:]) - c.first[x]
                return _q(2, "earlier_of_four", f"Of these {c.noun}s, which one flashes first?", c.lab[x], o, a, True,
                          {"opts": ids, "min_fps": fps_for([c.fl[k][0]["f1"] - c.fl[k][0]["f0"] for k in ids] + [lead])})
            out.append(("earlier_of_four", x, mk))
    longest = {i: max(f["f1"] - f["f0"] for f in c.fl[i]) for i in c.all}
    for x in c.all:
        weak = [k for k in c.all if k != x and longest[k] <= longest[x] - c.M]
        if len(weak) >= 3:
            def mk(rng, x=x, weak=weak):
                o, a, ids = _shapes(rng, c, x, rng.sample(weak, 3))
                second = max(longest[k] for k in ids[1:])
                return _q(2, "longest_flash", f"Of these {c.noun}s, which one has the single longest flash?",
                          c.lab[x], o, a, True,
                          {"opts": ids, "min_fps": fps_for(c.steps(ids), [(second, longest[x])])})
            out.append(("longest_flash", x, mk))
    for x in c.all:
        fx = sorted(c.fl[x], key=lambda f: f["f0"] - f["f1"])
        big = fx[0]
        if big["f1"] - big["f0"] < 20 or (len(fx) > 1 and (fx[1]["f1"] - fx[1]["f0"]) > (big["f1"] - big["f0"]) - 6):
            continue
        lo, hi = big["f0"], big["f1"]
        inside = [k for k in c.all if k != x and any(lo + c.M <= f["f0"] <= hi - c.M for f in c.fl[k])]
        clear = [k for k in c.all if k != x and not any(f["f0"] < hi + c.M and f["f1"] > lo - c.M for f in c.fl[k])]
        if len(clear) >= 3:
            for y in inside:
                sep = [(fx[1]["f1"] - fx[1]["f0"], big["f1"] - big["f0"])] if len(fx) > 1 else []

                def mk(rng, x=x, y=y, clear=clear, sep=sep):
                    o, a, ids = _shapes(rng, c, y, rng.sample(clear, 3))
                    return _q(2, "lit_during", f"While the {c.lab[x]} is lit during its longest flash, which of these "
                              f"{c.noun}s lights up?", c.lab[y], o, a, True,
                              {"x": x, "opts": ids, "min_fps": fps_for(c.steps([x] + ids, [c.M]), sep)})
                out.append(("lit_during", (x, y), mk))
    return out


# ---- L3 measurement -------------------------------------------------------------
def level3(c: Ctx):
    s, out = c.s, []
    for x in c.all:
        if c.longs[x]:
            def mk(rng, x=x):
                o, a = _num(rng, c.longs[x], [c.cnt[x]])
                return _q(3, "long_count", f"How many of the {c.lab[x]}'s flashes last longer than "
                          f"{c.cut} s?", c.longs[x], o, a, True,
                          {"x": x, "cut": c.cut, "min_fps": c.fps([x], length=True)})
            out.append(("long_count", x, mk))
        if c.whites[x]:
            def mk(rng, x=x):
                o, a = _num(rng, c.whites[x], [c.cnt[x]])
                return _q(3, "white_count", f"How many of the {c.lab[x]}'s flashes turn it white?", c.whites[x],
                          o, a, True, {"x": x, "min_fps": c.fps([x])})
            out.append(("white_count", x, mk))
        if s["shapes"][x]["rhythm"] == "steady":
            on = [f["f0"] for f in c.fl[x]]
            g = (on[-1] - on[0]) / (len(on) - 1) / FPS
            v = round(g * 2) / 2
            if abs(g - v) <= 0.12 and v >= 1.0:
                # sampled onsets move by < k frames, so the mean interval moves by < k / (n - 1)
                kp = int((0.25 - abs(g - v)) * FPS * (len(on) - 1))

                def mk(rng, x=x, v=v, kp=kp):
                    gt = f"{v:.1f} s"
                    o, a = L.mc(rng, gt, [f"{w:.1f} s" for w in (v + 0.5, v - 0.5, v + 1.0, v - 1.0, v + 1.5) if w > 0])
                    return _q(3, "period", f"The {c.lab[x]} flashes at a steady rhythm. About how many seconds "
                              f"pass from the start of one of its flashes to the start of the next (to the nearest "
                              f"half second)?", gt, o, a, True, {"x": x, "min_fps": c.fps([x], [kp])})
                out.append(("period", x, mk))
        tn = sim.tenths(c.first[x])

        def mk(rng, x=x, tn=tn):
            top = sim.tenths(c.nf - 1)
            pool = sorted({sim.tenths(f["f0"]) for k in c.all if k != x for f in c.fl[k]})
            rng.shuffle(pool)
            pool += [tn + d for d in (10, -10, 15, -15, 20, -20, 25, 30)]
            ds = []
            for v in pool:
                if 3 <= v <= top - 3 and abs(v - tn) >= 8 and all(abs(v - w) >= 5 for w in ds) and len(ds) < 3:
                    ds.append(v)
            gt = sim.fmt_reading(tn)
            o, a = L.mc(rng, gt, [sim.fmt_reading(v) for v in ds])
            # a sample k frames apart sees the onset up to k - 1 frames late (+ < 0.1 s of timer rounding);
            # that must stay under half the gap to the nearest wrong reading
            kt = int((min(abs(v - tn) for v in ds) / 20 - 0.1) * FPS)
            return _q(3, "first_time", f"What does the timer read when the {c.lab[x]} first lights up?", gt, o, a,
                      True, {"x": x, "min_fps": c.fps([x], [kt])})
        out.append(("first_time", x, mk))
    onsets = sorted(f["f0"] for f in s["flashes"])
    top = sim.tenths(c.nf - 1)
    for a_ in range(5, top - 15, 5):
        for ln in (20, 25, 30, 40):
            b_ = a_ + ln
            if b_ > top - 5:
                continue
            fa, fb = sim.frame_of_reading(a_), sim.frame_of_reading(b_)
            if any(abs(o - fa) < c.M or abs(o - fb) < c.M for o in onsets):
                continue
            v = sum(fa <= o < fb for o in onsets)
            if 1 <= v <= 15:
                def mk(rng, a_=a_, b_=b_, v=v):
                    o, a = _num(rng, v, lo=0)
                    return _q(3, "count_window", f"Counting every {c.noun}, how many flashes start between "
                              f"{sim.fmt_reading(a_)} and {sim.fmt_reading(b_)} on the timer?{c.pre}", v, o, a, True,
                              {"a": a_, "b": b_, "min_fps": c.fps(c.all, [c.M])})
                out.append(("count_window", (a_, b_), mk))
    return out


# ---- L4 integration ---------------------------------------------------------------
def level4(c: Ctx):
    s, out = c.s, []
    allfps = c.fps(c.all)
    top, order = unique_top(c.cnt)
    if top is not None:
        def mk(rng, top=top, order=order):
            o, a, _ = _shapes(rng, c, top, order[1:4] + order[4:])
            return _q(4, "most_frequent", f"Which {c.noun} flashes the most times over the whole video?{c.pre}",
                      c.lab[top], o, a, True, {"min_fps": allfps})
        out.append(("most_frequent", None, mk))
    neg = {i: -v for i, v in c.cnt.items()}
    low, lorder = unique_top(neg)
    if low is not None and c.cnt[lorder[1]] - c.cnt[low] >= max(1, math.ceil(0.1 * c.cnt[lorder[1]])):
        def mk(rng, low=low, lorder=lorder):
            o, a, _ = _shapes(rng, c, low, lorder[1:4] + lorder[4:])
            return _q(4, "least_frequent", f"Which {c.noun} flashes the fewest times over the whole video?{c.pre}",
                      c.lab[low], o, a, True, {"min_fps": allfps})
        out.append(("least_frequent", None, mk))
    irregular = [i for i in c.all if s["shapes"][i]["rhythm"] == "random" and c.cnt[i] >= 3]
    for x in c.all:
        if s["shapes"][x]["rhythm"] == "steady" and len(irregular) >= 3:
            def mk(rng, x=x):
                o, a, ids = _shapes(rng, c, x, rng.sample(irregular, 3))
                return _q(4, "steady", f"Which of these {c.noun}s flashes at a steady rhythm, with evenly spaced "
                          f"flashes?", c.lab[x], o, a, True,
                          {"opts": ids, "min_fps": c.fps(ids, [steady_step(c, x, ids)])})
            out.append(("steady", x, mk))
    for trig, resp in sim.links(s):
        score = sim.followed(s, trig, resp)
        weak = [k for k in c.all if k not in (trig, resp) and sim.followed(s, k, resp) <= score - 2]
        if len(weak) >= 3:
            def mk(rng, trig=trig, resp=resp, weak=weak):
                o, a, ids = _shapes(rng, c, trig, rng.sample(weak, 3))
                return _q(4, "trigger", f"The {c.lab[resp]} often flashes about one second after one particular "
                          f"{c.noun} flashes. Which {c.noun} is it?", c.lab[trig], o, a, True,
                          {"resp": resp, "opts": ids,
                           "min_fps": c.fps(ids + [resp], [trigger_step(c, resp, trig, ids)])})
            out.append(("trigger", resp, mk))
    tot = len(s["flashes"])
    if tot <= MAX_TOTAL:
        def mk(rng):
            o, a = _num(rng, tot, lo=1)
            return _q(4, "total_flashes", f"Counting every {c.noun}, how many flashes are there in the whole video?"
                      f"{c.pre}", tot, o, a, True, {"min_fps": allfps})
        out.append(("total_flashes", None, mk))
    tw = sum(c.whites.values())
    if 1 <= tw <= 12:
        def mk(rng):
            o, a = _num(rng, tw, lo=0)
            return _q(4, "total_white", f"Counting every {c.noun}, how many flashes turn a {c.noun} white?", tw, o, a,
                      True, {"min_fps": allfps})
        out.append(("total_white", None, mk))
    qb = quarters(c.s, c.M)
    onsets = [f["f0"] for f in s["flashes"]]
    if qb:
        fr = [sim.frame_of_reading(b) for b in qb]
        edges = [0] + fr + [c.nf]
        wc = {k: sum(edges[k] <= o < edges[k + 1] for o in onsets) for k in range(4)}
        best, worder = unique_top(wc)
        if best is not None and wc[worder[0]] - wc[worder[1]] >= 2:
            def mk(rng, best=best):
                texts = [window_text(a, b) for a, b in zip([0] + qb, qb + [None])]
                o, a = L.mc(rng, texts[best], [t for t in texts if t != texts[best]])
                return _q(4, "busiest_window", f"Counting every {c.noun}, in which stretch of the video do the most "
                          f"flashes start?{c.pre}", texts[best], o, a, True,
                          {"cuts": qb, "min_fps": c.fps(c.all, [c.M])})
            out.append(("busiest_window", None, mk))
    for x in c.all:
        pat = tuple(sim.is_long(f, c.cut) for f in c.fl[x])
        if 3 <= len(pat) <= 5 and any(pat) and not all(pat):
            def mk(rng, x=x, pat=pat):
                alts = []
                for k in range(len(pat)):
                    alts.append(tuple(not p if j == k else p for j, p in enumerate(pat)))
                alts.append(tuple(reversed(pat)))
                alts.append(tuple(not p for p in pat))
                rng.shuffle(alts)
                o, a = L.mc(rng, seq_text(pat), [seq_text(p) for p in alts if p != pat])
                return _q(4, "sequence", f"Reading the {c.lab[x]}'s flashes in order, which pattern do they make? "
                          f"(short = lit for less than {c.cut} s, long = lit for more than "
                          f"{c.cut} s)", seq_text(pat), o, a, True,
                          {"x": x, "cut": c.cut, "min_fps": c.fps([x], length=True)})
            out.append(("sequence", x, mk))
    return out


# ---- L5 counterfactual (re-simulated) -------------------------------------------
def steady_step(c: Ctx, x, ids) -> int:
    """Largest k such that, sampling every k' <= k frames at any phase, the
    sampled onsets of x still have the smallest interval spread (CV) of the options."""
    def cv(on):
        g = [b - a for a, b in zip(on, on[1:])]
        m = sum(g) / len(g)
        return (sum((v - m) ** 2 for v in g) / len(g)) ** 0.5 / m
    k = 1
    while k < 24:
        kk = k + 1
        for ph in range(kk):
            sam = {i: [f["f0"] + (ph - f["f0"]) % kk for f in c.fl[i]] for i in ids}
            if min(ids, key=lambda i: cv(sam[i])) != x:
                return k
        k = kk
    return k


def trigger_step(c: Ctx, resp, trig, ids) -> int:
    """Largest k such that, sampling every k' <= k frames at any phase, the
    trigger still has the strictly highest "followed by resp" score of the options."""
    lo, hi = sim.LAG_WIN_F
    k = 1
    while k < 24:
        kk = k + 1
        for ph in range(kk):
            sam = {i: [f["f0"] + (ph - f["f0"]) % kk for f in c.fl[i]] for i in ids + [resp]}
            sc = {i: sum(1 for g in sam[i] if any(lo <= r - g <= hi for r in sam[resp])) for i in ids}
            if any(sc[i] >= sc[trig] for i in ids if i != trig):
                return k
        k = kk
    return k


def _count_in(flashes, i):
    return sum(1 for f in flashes if f["i"] == i)


def level5(c: Ctx):
    s, out = c.s, []
    allfps = c.fps(c.all)
    ch = (s.get("causal") or {}).get("chain") or []
    for trig, resp in sim.links(s):
        v = _count_in(sim.resimulate(s, {trig}), resp)
        if v != c.cnt[resp]:
            def mk(rng, trig=trig, resp=resp, v=v):
                o, a = _num(rng, v, [c.cnt[resp]])
                return _q(5, "cf_no_trigger", f"If the {c.lab[trig]} had never flashed (everything else the same), "
                          f"how many times would the {c.lab[resp]} flash?", v, o, a, True,
                          {"trig": trig, "resp": resp,
                           "min_fps": c.fps([trig, resp], sim.lag_margins(s, trig, resp))})
            out.append(("cf_no_trigger", resp, mk))
    if len(ch) == 3:
        a_, b_, c_ = ch
        v = _count_in(sim.resimulate(s, {a_}), c_)
        if v != c.cnt[c_]:
            def mk(rng, v=v):
                o, a = _num(rng, v, [c.cnt[c_]])
                return _q(5, "cf_chain", f"If the {c.lab[a_]} had never flashed, how many times would the "
                          f"{c.lab[c_]} flash? Assume every {c.noun} that reacts to another {c.noun}'s flashes keeps "
                          f"reacting in the same way.", v, o, a, True,
                          {"trig": a_, "via": b_, "resp": c_, "min_fps": c.fps(ch, sim.lag_margins(s, a_, b_) +
                                                                    sim.lag_margins(s, b_, c_))})
            out.append(("cf_chain", c_, mk))
    top, order = unique_top(c.cnt)
    if top is not None:
        rest = {i: v for i, v in c.cnt.items() if i != top}
        t2, o2 = unique_top(rest)
        if t2 is not None:
            def mk(rng, top=top, t2=t2, o2=o2):
                o, a, _ = _shapes(rng, c, t2, [top] + o2[1:3] + o2[3:])
                return _q(5, "cf_remove_most", f"If the {c.noun} that flashes the most had never flashed, which "
                          f"{c.noun} would have flashed the most?{c.pre}", c.lab[t2], o, a, True, {"min_fps": allfps})
            out.append(("cf_remove_most", None, mk))
    for key, vals, text in (("cf_long_only", c.longs, f"If only flashes lasting longer than {c.cut} s "
                                                       f"counted, which {c.noun} would have the most flashes?"),
                            ("cf_white_only", c.whites, f"If only flashes that turn a {c.noun} white counted, which "
                                                        f"{c.noun} would have the most flashes?")):
        w, wo = unique_top(vals)
        if w is not None and vals[w] >= 2:
            def mk(rng, key=key, w=w, wo=wo, text=text):
                pool = ([top] if top is not None and top != w else []) + wo[1:4] + wo[4:]
                o, a, _ = _shapes(rng, c, w, pool)
                extra = {"cut": c.cut} if key == "cf_long_only" else {}
                return _q(5, key, text, c.lab[w], o, a, True,
                          dict(extra, min_fps=c.fps(c.all, length=(key == "cf_long_only"))))
            out.append((key, None, mk))
    nw = {i: c.cnt[i] - c.whites[i] for i in c.all}
    w, wo = unique_top(nw)
    if w is not None and top is not None and w != top and sum(c.whites.values()):
        def mk(rng, w=w, wo=wo):
            o, a, _ = _shapes(rng, c, w, [top] + wo[1:3] + wo[3:])
            return _q(5, "cf_no_white_most", f"If none of the white flashes had happened, which {c.noun} would have "
                      f"flashed the most?{c.pre}", c.lab[w], o, a, True, {"min_fps": allfps})
        out.append(("cf_no_white_most", None, mk))
    top_t = sim.tenths(c.nf - 1)
    for x in c.all:
        on = [f["f0"] for f in c.fl[x]]
        for tn in range(20, top_t - 14, 5):
            ft = sim.frame_of_reading(tn)
            if any(abs(o - ft) < c.M for o in on):
                continue
            v = sum(o < ft for o in on)
            if 1 <= v < c.cnt[x]:
                def mk(rng, x=x, tn=tn, v=v):
                    o, a = _num(rng, v, [c.cnt[x]])
                    return _q(5, "cf_stop", f"If the video had stopped the moment the timer first read "
                              f"{sim.fmt_reading(tn)}, how many flashes would the {c.lab[x]} have started?", v, o, a,
                              True, {"x": x, "t": tn, "min_fps": c.fps([x], [c.M])})
                out.append(("cf_stop", (x, tn), mk))
    return out


LEVELS = (level1, level2, level3, level4, level5)


def supply(scene) -> list[int]:
    """Distinct candidate questions per level (sim.check needs >= 3 everywhere)."""
    c = Ctx(scene)
    sup = [len(f(c)) for f in LEVELS]
    sup[3] = sum(1 for cd in level4(c) if cd[0] != "trigger")     # the fallback in build() must always work
    return sup


def _pick(rng, cands, n):
    bykey = {}
    for key, ident, mk in cands:
        bykey.setdefault(key, []).append((ident, mk))
    keys = list(bykey)
    rng.shuffle(keys)
    chosen = []
    while len(chosen) < n and any(bykey.values()):
        for key in keys:
            if bykey[key] and len(chosen) < n:
                chosen.append((key,) + bykey[key].pop(rng.randrange(len(bykey[key]))))
    return chosen


def build(scene: dict, rng: random.Random) -> list[dict]:
    """A trigger question (L4) and a counterfactual naming the same responder's
    trigger (L5) never share a clip: the L5 text would give the L4 answer away."""
    counts = L.ten_counts(rng)
    c = Ctx(scene)
    cands = [lvl(c) for lvl in LEVELS]
    picks = [_pick(rng, cands[lv], counts[lv]) for lv in range(4)]
    revealed = {ident for key, ident, _ in picks[3] if key == "trigger"}
    l5 = [cd for cd in cands[4] if not (cd[0] in ("cf_no_trigger", "cf_chain") and cd[1] in revealed)]
    if len(l5) < counts[4]:
        picks[3] = _pick(rng, [cd for cd in cands[3] if cd[0] != "trigger"], counts[3])
        l5 = cands[4]
    picks.append(_pick(rng, l5, counts[4]))
    qs = [mk(rng) for lv in picks for _, _, mk in lv]
    return L.number(qs)
