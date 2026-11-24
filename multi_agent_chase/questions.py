"""10-question ladder for multi_agent_chase (common/ladder5.py, LADDER = "ten").

A ball is named by its colour and pattern ("the blue striped ball"); both are
unique within a clip. Times are clock readings (the clock is the only text in
the frame). Margins: catches are >= 10 frames apart and >= 1 s from the end
(sim.check); size comparisons need a radius ratio >= sim.SIZE_MARGIN; clock
options are >= 0.6 s apart; "when the clock shows X" questions keep every
catch >= 6 frames (0.25 s) outside the frames that show X. What-ifs re-run the
simulation with the change (sim.run(scene, mods)) and are only asked when the
re-run passes the same ambiguity checks as the real clip. Every question has
four distinct options.
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
ORD = ("first", "second", "third", "fourth", "fifth", "sixth", "seventh", "eighth")
ARENA = "the box"
CF_TEXT = {
    "remove": "Suppose {x} had not been in the box at all (everything else the same). ",
    "nogrow": "Suppose catching a ball did NOT make the catcher bigger or slower (everything else the same). ",
    "nozone": "Suppose there were no shaded circle, so no ball were ever safe inside it (everything else the same). ",
    "nopillars": "Suppose the grey pillars were taken away (everything else the same). ",
    "slow": "Suppose {x} moved at half its speed the whole time (everything else the same). ",
    "immune": "Suppose {x} could not be caught (everything else the same). ",
}


# ------------------------------------------------------------------ helpers
def fps_for(gaps, floor=1):
    g = min(gaps) if gaps else 99.0
    return int(min(24, max(floor, math.ceil(1.0 / g - 1e-9))))


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
    near = [v for v in (gt + 1, gt - 1, gt + 2, gt - 2, gt + 3, gt - 3, gt + 4) if lo <= v <= hi]
    rng.shuffle(near)
    near.sort(key=lambda v: abs(v - gt))
    ex = [e for e in extra if lo <= e <= hi and e != gt]
    return mc4(rng, gt, ex[:1] + near)


def clock_opts(rng, tenths, extra=(), step=6):
    cands = [round(e) for e in extra] + [tenths + d for d in (-12, -6, 6, 12, 18, -18, 9, -9)]
    keep = []
    for c in cands:
        if c >= 0 and all(abs(c - u) >= step for u in keep + [tenths]):
            keep.append(c)
    head, tail = keep[:len(extra)], keep[len(extra):]
    rng.shuffle(tail)
    return mc4(rng, fmt_clock(tenths), [fmt_clock(c) for c in head + tail])


def label_opts(rng, c, gt, pool):
    rest = [k for k in pool if k != gt]
    others = [k for k in range(c.n) if k != gt and k not in rest]
    rng.shuffle(others)
    return mc4(rng, c.lbl[gt], [c.lbl[k] for k in rest + others])


def Q(key, params, text, gt, opts, motion, fps):
    if opts is None:
        return None
    return (key, dict(params, min_fps=fps), text, gt, opts[0], opts[1], motion)


class Ctx:
    def __init__(self, scene):
        self.scene = scene
        self.bs = scene["balls"]
        self.n = len(self.bs)
        self.lbl = [sim.label(b) for b in self.bs]
        self.frames = scene["frames"]
        tr, cs = sim.run(scene)
        self.tr, self.cs = tr, cs
        self.cf = [c[0] for c in cs]                       # catch frames
        self.victims = [c[2] for c in cs]
        self.when = {c[2]: c[0] for c in cs}
        self.by = {i: [c[2] for c in cs if c[1] == i] for i in range(self.n)}
        self.catcher_of = {c[2]: c[1] for c in cs}
        self.catchers = [i for i in range(self.n) if self.by[i]]
        self.alive = [i for i in range(self.n) if tr[-1][i][3]]
        self.r0 = [tr[0][i][2] for i in range(self.n)]
        self.rf = [tr[-1][i][2] if tr[-1][i][3] else tr[self.when[i] - 1][i][2] for i in range(self.n)]

    def gaps_around(self, f):
        """Seconds from catch frame f to its neighbouring catches and the clip ends."""
        pts = [0] + self.cf + [self.frames - 1]
        return [abs(f - g) / FPS for g in pts if g != f] or [99.0]

    def all_gaps(self):
        pts = [0] + self.cf + [self.frames - 1]
        return [(b - a) / FPS for a, b in zip(pts, pts[1:])]

    def clock_safe(self, tenths, pad=6):
        """Frames that show this reading, if every catch is >= pad frames outside them."""
        fr = [f for f in range(self.frames) if sim.clock_tenths(f) == tenths]
        if not fr or fr[0] < 12 or fr[-1] > self.frames - 13:
            return None
        if any(fr[0] - pad < g <= fr[-1] + pad for g in self.cf):
            return None
        return fr


def _clear_max(vals: dict):
    order = sorted(vals, key=lambda i: -vals[i])
    if len(order) >= 2 and vals[order[0]] >= sim.SIZE_MARGIN * vals[order[1]]:
        return order[0]
    return None


# ------------------------------------------------------------------ levels
def level1(c: Ctx, rng):
    n, out = c.n, []
    out.append(Q("n_start", {}, f"How many balls are in {ARENA} at the start of the video?", n,
                 int_opts(rng, n, 1, 12), False, 1))
    k = len(c.alive)
    out.append(Q("n_end", {}, f"How many balls are still in {ARENA} at the end of the video?", k,
                 int_opts(rng, k, 0, 12, [n]), True, fps_for(c.all_gaps())))
    k = len(c.cs)
    out.append(Q("n_catches", {}, "How many times does one ball catch another (the caught ball disappears)?", k,
                 int_opts(rng, k, 0, 10), True, fps_for(c.all_gaps())))
    for i in rng.sample(range(n), 2):
        b = c.bs[i]
        pats = [p for p in sim.PATTERNS if p != b["pattern"]]
        rng.shuffle(pats)
        out.append(Q("pattern_of", {"x": i}, f"What pattern does the {b['color']} ball have?", b["pattern"],
                     mc4(rng, b["pattern"], pats), False, 1))
        cols = [col for col in sim.COLORS if col != b["color"]]
        rng.shuffle(cols)
        out.append(Q("color_of", {"x": i}, f"What colour is the {b['pattern']} ball?", b["color"],
                     mc4(rng, b["color"], cols), False, 1))
    g = _clear_max({i: c.r0[i] for i in range(n)})
    if g is not None:
        out.append(Q("biggest_start", {}, "Which ball is the biggest at the start?", c.lbl[g],
                     label_opts(rng, c, g, sorted(range(n), key=lambda i: -c.r0[i])), False, 1))
    if c.scene.get("pillars"):
        k = len(c.scene["pillars"])
        out.append(Q("n_pillars", {}, "How many grey pillars stand in the box?", k, int_opts(rng, k, 0, 6), False, 1))
    z = c.scene.get("zone")
    if z:
        last = c.tr[-1]
        ds = [(math.dist(last[i][:2], (z["x"], z["y"])), last[i][2]) for i in c.alive]
        if all(abs(d - (z["r"] - r)) >= 6 for d, r in ds):
            k = sum(d <= z["r"] - r for d, r in ds)
            out.append(Q("zone_end", {}, "At the end of the video, how many balls are completely inside the shaded "
                         "circle?", k, int_opts(rng, k, 0, 9), True, fps_for([(c.frames - 1 - c.cf[-1]) / FPS])))
    return out


def level2(c: Ctx, rng):
    out = []
    for f, h, p, *_ in c.cs:
        out.append(Q("caught_by", {"x": p}, f"Which ball catches {c.lbl[p]}?", c.lbl[h],
                     label_opts(rng, c, h, [k for k in c.catchers if k != p]), True, fps_for(c.gaps_around(f))))
    for h in c.catchers:
        gt = c.by[h][0]
        out.append(Q("first_victim", {"x": h}, f"Which ball is the first one that {c.lbl[h]} catches?", c.lbl[gt],
                     label_opts(rng, c, gt, c.by[h][1:] + [v for v in c.victims if v != h]), True,
                     fps_for(c.gaps_around(c.when[gt]))))
    pairs = list(itertools.combinations(c.victims, 2))
    rng.shuffle(pairs)
    for x, y in pairs[:3]:
        if rng.random() < 0.5:
            x, y = y, x
        gt = x if c.when[x] < c.when[y] else y
        others = [v for v in c.victims if v not in (x, y)] + c.alive
        out.append(Q("caught_first", {"x": x, "y": y}, f"Which of these balls is caught first: {c.lbl[x]} or "
                     f"{c.lbl[y]}?", c.lbl[gt], label_opts(rng, c, gt, [x, y] + others), True,
                     fps_for([abs(c.when[x] - c.when[y]) / FPS])))
    big = [(x, y) for x, y in itertools.combinations(c.alive, 2)
           if max(c.rf[x], c.rf[y]) >= sim.SIZE_MARGIN * min(c.rf[x], c.rf[y])
           and (c.r0[x] - c.r0[y]) * (c.rf[x] - c.rf[y]) <= 0]          # only pairs whose size order flips
    rng.shuffle(big)
    for x, y in big[:2]:
        gt = x if c.rf[x] > c.rf[y] else y
        out.append(Q("bigger_end", {"x": x, "y": y}, f"At the end of the video, which is bigger: {c.lbl[x]} or "
                     f"{c.lbl[y]}?", c.lbl[gt], label_opts(rng, c, gt, [x, y]), True, fps_for(c.all_gaps())))
    safe = [i for i in c.alive if i not in c.catchers]
    if safe and len(c.victims) >= 3:
        gt = rng.choice(safe)
        out.append(Q("never_caught", {"x": gt}, "Which of these balls is never caught?", c.lbl[gt],
                     mc4(rng, c.lbl[gt], [c.lbl[v] for v in rng.sample(c.victims, len(c.victims))]), True,
                     fps_for(c.all_gaps())))
    idle = [i for i in range(c.n) if not c.by[i]]
    if len(idle) >= 3:
        gt = rng.choice(c.catchers)
        out.append(Q("is_catcher", {"x": gt}, "Which of these balls catches at least one other ball?", c.lbl[gt],
                     mc4(rng, c.lbl[gt], [c.lbl[v] for v in rng.sample(idle, len(idle))]), True,
                     fps_for(c.all_gaps())))
    gt = c.victims[0]
    out.append(Q("first_caught", {}, "Which ball is the first to be caught?", c.lbl[gt],
                 label_opts(rng, c, gt, c.victims[1:] + c.alive), True, fps_for(c.gaps_around(c.cf[0]))))
    return out


def level3(c: Ctx, rng):
    out = []
    for f, h, p, *_ in c.cs:
        tt = sim.clock_tenths(f)
        out.append(Q("catch_time", {"x": p}, f"What does the clock show when {c.lbl[p]} disappears?", fmt_clock(tt),
                     clock_opts(rng, tt), True, fps_for(c.gaps_around(f), 2)))
    for h in sorted(set(c.catchers + c.alive[:2])):
        k = len(c.by[h])
        out.append(Q("n_caught_by", {"x": h}, f"How many balls does {c.lbl[h]} catch?", k, int_opts(rng, k, 0, 8),
                     True, fps_for(c.all_gaps())))
    ticks = [t for t in range(5, sim.clock_tenths(c.frames - 1)) if c.clock_safe(t)]
    for tt in rng.sample(ticks, min(2, len(ticks))):
        fr = c.clock_safe(tt)
        k = sum(c.tr[fr[0]][i][3] for i in range(c.n))
        gaps = [abs(g - fr[0]) / FPS for g in c.cf]
        out.append(Q("left_at", {"t": tt}, f"When the clock shows {fmt_clock(tt)}, how many balls are in {ARENA}?", k,
                     int_opts(rng, k, 0, 12), True, fps_for(gaps, 2)))
    if len(c.cs) >= 2:
        for x, y in rng.sample(list(itertools.combinations(c.victims, 2)), min(2, len(c.cs) * (len(c.cs) - 1) // 2)):
            d = abs(sim.clock_tenths(c.when[x]) - sim.clock_tenths(c.when[y]))
            out.append(Q("time_between", {"x": x, "y": y}, f"How many seconds pass between {c.lbl[x]} disappearing "
                         f"and {c.lbl[y]} disappearing (read the clock)?", fmt_clock(d),
                         clock_opts(rng, d), True, fps_for(c.gaps_around(c.when[x]) + c.gaps_around(c.when[y]), 2)))
    tt = sim.clock_tenths(c.cf[0])
    out.append(Q("first_catch_time", {}, "What does the clock show when the first ball is caught?", fmt_clock(tt),
                 clock_opts(rng, tt), True, fps_for(c.gaps_around(c.cf[0]), 2)))
    return out


def level4(c: Ctx, rng):
    out = []
    for k, (f, h, p, *_) in enumerate(c.cs):
        gt = c.n - (k + 1)
        out.append(Q("left_after", {"x": p}, f"How many balls are in {ARENA} right after {c.lbl[p]} is caught?", gt,
                     int_opts(rng, gt, 0, 12, [gt + 1]), True, fps_for(c.gaps_around(f))))
    if 2 <= len(c.cs) <= 5:
        seq = [c.lbl[p] for p in c.victims]
        wrong = []
        for i in range(len(seq) - 1):
            alt = list(seq)
            alt[i], alt[i + 1] = alt[i + 1], alt[i]
            wrong.append(alt)
        wrong.append(seq[::-1])
        rest = [i for i in c.alive]
        if rest:
            alt = list(seq)
            alt[-1] = c.lbl[rng.choice(rest)]
            wrong.append(alt)
        rng.shuffle(wrong)
        out.append(Q("catch_order", {}, "In what order are the balls caught?", ", then ".join(seq),
                     mc4(rng, ", then ".join(seq), [", then ".join(w) for w in wrong]), True, fps_for(c.all_gaps())))
    counts = {i: len(c.by[i]) for i in range(c.n)}
    srt = sorted(counts.values(), reverse=True)
    if len(c.catchers) >= 2 and srt[0] > srt[1]:
        gt = max(counts, key=counts.get)
        out.append(Q("top_catcher", {}, "Which ball catches the most other balls?", c.lbl[gt],
                     label_opts(rng, c, gt, sorted(c.catchers, key=lambda i: -counts[i])), True,
                     fps_for(c.all_gaps())))
    for k in range(1, len(c.cs)):
        gt = c.victims[k]
        out.append(Q("nth_victim", {"k": k}, f"Which ball is the {ORD[k]} one to be caught?", c.lbl[gt],
                     label_opts(rng, c, gt, c.victims[:k][::-1] + c.victims[k + 1:] + c.alive), True,
                     fps_for(c.gaps_around(c.cf[k]))))
    ticks = [t for t in range(5, sim.clock_tenths(c.frames - 1)) if c.clock_safe(t)]
    if len(ticks) >= 2:
        for _ in range(2):
            a, b = sorted(rng.sample(ticks, 2))
            if b - a < 10:
                continue
            fa, fb = c.clock_safe(a)[0], c.clock_safe(b)[0]
            k = sum(fa < g <= fb for g in c.cf)
            out.append(Q("caught_between", {"t1": a, "t2": b}, f"How many balls are caught between the clock showing "
                         f"{fmt_clock(a)} and the clock showing {fmt_clock(b)}?", k, int_opts(rng, k, 0, 9), True,
                         fps_for(c.all_gaps(), 2)))
    g = _clear_max({i: c.rf[i] for i in c.alive})
    if g is not None and g != _clear_max({i: c.r0[i] for i in range(c.n)}):
        out.append(Q("biggest_end", {}, "Which ball is the biggest at the end of the video?", c.lbl[g],
                     label_opts(rng, c, g, sorted(c.alive, key=lambda i: -c.rf[i])), True, fps_for(c.all_gaps())))
    return out


def cf_mods(c: Ctx):
    """(tag, mods, subject) for every what-if this scene offers."""
    sc = c.scene["scenario"]
    out = [("nogrow", (("nogrow",),), None)]
    for h in c.catchers:
        out.append(("remove", (("remove", h),), h))
        out.append(("slow", (("slow", h),), h))
    for p in c.victims[:2]:
        out.append(("immune", (("immune", p),), p))
        out.append(("remove", (("remove", p),), p))
    if sc == "safe_zone":
        out.append(("nozone", (("nozone",),), None))
    if sc == "pillars":
        out.append(("nopillars", (("nopillars",),), None))
    return out


def level5(c: Ctx, rng):
    out = []
    mods = cf_mods(c)
    rng.shuffle(mods)
    for tag, m, x in mods:
        if len(out) >= 6:
            break
        if sim.run_errors(c.scene, m, need=0):
            continue
        tr, cs = sim.run(c.scene, m)
        lead = CF_TEXT[tag].format(x=c.lbl[x] if x is not None else "")
        params = {"tag": tag, "x": x}
        k = sum(tr[-1][i][3] for i in range(c.n))
        real = len(c.alive)
        cf_gaps = [(b - a) / FPS for a, b in zip([0] + [q[0] for q in cs], [q[0] for q in cs] + [c.frames - 1])]
        out.append(Q("cf_n_end", params, lead + f"How many balls would be in {ARENA} at the end of the video?", k,
                     int_opts(rng, k, 0, 12, [real]), True, fps_for(c.all_gaps() + cf_gaps)))
        if cs:
            first = cs[0][2]
            out.append(Q("cf_first", params, lead + "Which ball would be caught first?", c.lbl[first],
                         label_opts(rng, c, first, [q[2] for q in cs[1:]] + [v for v in c.victims if v != x]), True,
                         fps_for(c.all_gaps() + cf_gaps)))
        if tag == "slow":
            k = sum(q[1] == x for q in cs)
            out.append(Q("cf_n_caught_by", params, lead + f"How many balls would {c.lbl[x]} catch?", k,
                         int_opts(rng, k, 0, 8, [len(c.by[x])]), True, fps_for(c.all_gaps() + cf_gaps)))
    return out


def _pick(rng, cands, n):
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
        cand = by_key[k].pop(rng.randrange(len(by_key[k])))
        if any(cand[2] == ch[2] for ch in chosen):                 # never the same question text twice in a clip
            continue
        chosen.append(cand)
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
    texts = {lv: len({cand[2] for cand in made[lv]}) for lv in range(1, 6)}
    feasible = [cmb for cmb in TEN_SPLITS if all(texts[lv] >= cmb[lv - 1] for lv in range(1, 6))]
    if not feasible:
        raise ValueError("not enough question candidates for this scene")
    counts = rng.choice(feasible)
    qs = []
    for lv in range(1, 6):
        for key, params, text, gt, opts, letter, motion in _pick(rng, made[lv], counts[lv - 1]):
            qs.append(L.question(lv, key, text, gt, opts, letter, motion, params, idx=len(qs) + 1))
    return qs
