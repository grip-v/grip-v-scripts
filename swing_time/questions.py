"""10-question ladder for swing_time (common/ladder5.py, mode "ten").

The video has no labels: posts are named by colour and shape ("the teal
diamond post"; colours are unique per board), the start square is where
the particle begins, the goal is the checkered square and pits are the
hatched squares. Times are what the on-screen clock shows. Every question
has exactly four distinct options and stores params["min_fps"]: 24 / the
smallest gap (in frames) between the events it relies on, rounded up, and
at least 4 for clock readings and duration comparisons.

Templates per level (derivation_key):
  L1 n_swings, n_posts_used, first_pivot, last_pivot, n_posts_total, n_pits
  L2 pivot_of_swing, swing_direction, next_pivot, land_quadrant, earliest_of
  L3 longest_rope, rope_of_post, total_rope, swing_duration, land_time, first_attach_time
  L4 pivot_sequence, pivot_reused, n_clockwise, n_long_ropes, longest_swing_post
  L5 cf_remove_post, cf_remove_n_swings, cf_remove_first_pivot, cf_short_rope, cf_no_pits
Counterfactual answers come from re-solving the changed board (sim.solve).
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
import render  # noqa: E402
from answers import (QUADS, DIRS, NO_ROUTE, NONE_TWICE, post_name, fmt_len, fmt_secs, label_options,  # noqa: E402
                     fmt_seq)

LADDER = "ten"               # common/video5.py validates the 10-question ladder
# every valid per-level split of the 10 questions (L.TEN_RANGES, total L.TEN_TOTAL), in level order
TEN_SPLITS = [c for c in itertools.product(*(range(lo, hi + 1) for lo, hi in L.TEN_RANGES)) if sum(c) == L.TEN_TOTAL]
FPS = sim.FPS
UNIT = "Each grid square is 1 unit wide. "
SHORT_CAPS = (2.0, 2.4, 2.7, 3.2, 3.7, 4.0, 4.4)       # each sits well between two possible rope lengths
LONG_OVER = (1.0, 2.0, 2.4, 2.7, 3.2, 4.0)
ROPES = sim.rope_values()


def min_fps(frames, floor: int = 1) -> int:
    fr = sorted(set(frames))
    gaps = [b - a for a, b in zip(fr, fr[1:])]
    need = math.ceil(FPS / min(gaps) - 1e-9) if gaps else 1
    return int(min(FPS, max(floor, need)))


def safe2(v) -> bool:
    """v is far enough from a 2-decimal rounding boundary (>= 0.0015 units)."""
    x = v * 100 - math.floor(v * 100)
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


def spaced(rng, gt, pool, gap):
    """Shuffle pool and keep values that stay >= gap from gt and from each other."""
    pool = list(pool)
    rng.shuffle(pool)
    out = []
    for v in pool:
        if v > 0 and all(abs(v - u) >= gap - 1e-9 for u in out + [gt]):
            out.append(v)
    return out


def int_wrong(rng, gt, lo=0, hi=60):
    near = [v for v in L.near_ints(gt, lo, hi) + [gt + 2, gt - 2, gt + 3] if lo <= v <= hi and v != gt]
    rng.shuffle(near)
    near.sort(key=lambda v: abs(v - gt))
    return near


class Ctx:
    def __init__(self, scene):
        self.s = scene
        self.sw = scene["swings"]
        self.n = len(self.sw)
        self.ev = sim.events(scene)
        self.N = sim.n_frames(scene)
        self.xy = {p["colour"]: tuple(p["xy"]) for p in scene["posts"]}
        self.colours = [p["colour"] for p in scene["posts"]]
        self.piv = [x["post"] for x in self.sw]
        self.rope, self.turn, here = [], [], tuple(scene["start"])
        for x in self.sw:
            p = self.xy[x["post"]]
            self.rope.append(math.dist(sim.centre(here), p))
            self.turn.append(sim.sweep_of(here, tuple(x["land"]), p, x["dir"]))
            here = tuple(x["land"])
        self.dur = [b - a for a, b in self.ev]
        self.all_ev = [f for e in self.ev for f in e]
        self.once = [c for c in dict.fromkeys(self.piv) if self.piv.count(c) == 1]
        self.rules = ("(Rules: from the centre of its square the particle hooks a post at most "
                      f"{scene['cap']:g} units away and swings the shorter way round (either way on an exact "
                      "half turn) to the centre of another square the same distance from the post"
                      + (" that is not a hatched pit" if scene["pits"] else "") +
                      "; a swing must keep the particle on the board and may not sweep over another post; "
                      "each swing costs its rope length and the particle always takes the cheapest route. "
                      "Squares are 1 unit wide.)")
        self._cf = {}

    def pn(self, c):
        return post_name(self.s, c)

    def idx(self, c):
        return self.piv.index(c)

    def post_wrong(self, rng, gt, prefer=()):
        """Other posts, the ones in `prefer` and those near gt first."""
        gx = self.xy[gt]
        rest = [c for c in self.colours if c != gt]
        rng.shuffle(rest)
        rest.sort(key=lambda c: (c not in prefer, math.dist(self.xy[c], gx) > 3))
        return rest

    def cf(self, **kw):
        k = tuple(sorted(kw.items()))
        if k not in self._cf:
            self._cf[k] = sim.solve(sim.world_of(self.s, **kw))
        return self._cf[k]


def _len_options(rng, gt, base_pool):
    return spaced(rng, gt, base_pool, 0.15)


def _post_q(c, rng, text, gt, prefer, motion, fps):
    o = mc4(rng, gt, c.post_wrong(rng, gt, prefer), c.pn)
    return o and (text, c.pn(gt), *o, motion, fps)


# ------------------------------------------------------------------ L1 observation
def level1(c: Ctx):
    out = []
    n, ev = c.n, c.ev
    out.append(("n_swings", {}, lambda rng: (lambda o: o and (
        "How many swings does the particle make to reach the checkered goal square?", n, *o, True,
        min_fps(c.all_ev)))(mc4(rng, n, int_wrong(rng, n, 1)))))
    k = len(set(c.piv))
    out.append(("n_posts_used", {}, lambda rng: (lambda o: o and (
        "How many different posts does the rope attach to on the way to the goal?", k, *o, True,
        min_fps(c.all_ev)))(mc4(rng, k, int_wrong(rng, k, 1, len(c.colours))))))
    out.append(("first_pivot", {}, lambda rng: _post_q(
        c, rng, "Which post does the rope attach to on the particle's first swing?", c.piv[0], c.piv, True,
        min_fps(ev[0]))))
    out.append(("last_pivot", {}, lambda rng: _post_q(
        c, rng, "Which post does the particle swing around on its last swing, the one that lands on the "
                "checkered goal square?", c.piv[-1], c.piv, True, min_fps(ev[-1]))))
    m = len(c.colours)
    out.append(("n_posts_total", {}, lambda rng: (lambda o: o and (
        "How many posts are standing on the board?", m, *o, False, 1))(mc4(rng, m, int_wrong(rng, m, 2)))))
    if c.s["pits"]:
        k2 = len(c.s["pits"])
        out.append(("n_pits", {}, lambda rng: (lambda o: o and (
            "How many hatched pit squares are on the board?", k2, *o, False, 1))(mc4(rng, k2, int_wrong(rng, k2, 1)))))
    return out


# ------------------------------------------------------------------ L2 relational
def level2(c: Ctx, rng0):
    out = []
    for k in range(2, c.n):                                   # middle swings (first/last have their own L1 templates)
        out.append(("pivot_of_swing", {"swing": k}, lambda rng, k=k: _post_q(
            c, rng, f"Which post does the particle swing around on swing {k} (counting the swings from 1)?",
            c.piv[k - 1], c.piv, True, min_fps(c.all_ev[:2 * k]))))
    for col in c.once:
        i = c.idx(col)
        deg = math.degrees(abs(c.turn[i]))
        if abs(deg - 90) >= 15:
            gt = DIRS[(0 if c.turn[i] < 0 else 2) + (1 if deg > 90 else 0)]
            out.append(("swing_direction", {"post": col}, lambda rng, col=col, gt=gt, i=i: (lambda o: o and (
                f"When the particle swings around {c.pn(col)}, which way does it turn and how far?",
                gt, *o, True, min_fps(c.ev[i])))(mc4(rng, gt, list(DIRS)))))
        land = c.sw[i]["land"]
        dx, dy = land[0] + .5 - c.xy[col][0], land[1] + .5 - c.xy[col][1]
        gtq = QUADS[(0 if dy > 0 else 2) + (1 if dx > 0 else 0)]
        out.append(("land_quadrant", {"post": col}, lambda rng, col=col, gtq=gtq, i=i: (lambda o: o and (
            f"Where does the particle land after its swing around {c.pn(col)}, compared with that post?",
            gtq, *o, True, min_fps(c.ev[i])))(mc4(rng, gtq, list(QUADS)))))
        if i < c.n - 1:
            out.append(("next_pivot", {"post": col}, lambda rng, col=col, i=i: _post_q(
                c, rng, f"Right after the swing around {c.pn(col)}, which post does the rope attach to next?",
                c.piv[i + 1], c.piv, True, min_fps(list(c.ev[i]) + list(c.ev[i + 1])))))
    used = list(dict.fromkeys(c.piv))
    if len(used) >= 2:
        others = [x for x in c.colours if x not in used]
        rng0.shuffle(others)
        pick = sorted(rng0.sample(used, min(4, len(used))) + others[:max(0, 4 - len(used))])
        if len(pick) == 4:
            gt = min(pick, key=lambda x: c.piv.index(x) if x in c.piv else 99)
            frames = [f for x in pick if x in c.piv for f in c.ev[c.idx(x)]]
            out.append(("earliest_of", {"posts": pick}, lambda rng, pick=pick, gt=gt, frames=frames: (lambda o: o and (
                "Of these posts, which one does the rope attach to first?", c.pn(gt), *o, True,
                min_fps(frames)))(mc4(rng, gt, [x for x in pick if x != gt], c.pn))))
    return out


# ------------------------------------------------------------------ L3 measurement
def level3(c: Ctx):
    out = []
    v = max(c.rope)
    if safe2(v):
        out.append(("longest_rope", {}, lambda rng: (lambda o: o and (
            UNIT + "How long is the longest rope the particle uses on any swing (to 2 decimals)?", fmt_len(v),
            *o, True, min_fps(c.all_ev)))(mc4(rng, v, _len_options(rng, v, ROPES[:12]), fmt_len))))
    for col in c.once:
        i = c.idx(col)
        r = c.rope[i]
        if safe2(r):
            out.append(("rope_of_post", {"post": col}, lambda rng, r=r, col=col, i=i: (lambda o: o and (
                UNIT + f"How long is the rope when the particle swings around {c.pn(col)} (to 2 decimals)?",
                fmt_len(r), *o, True, min_fps(c.ev[i])))(mc4(rng, r, _len_options(rng, r, ROPES[:12]), fmt_len))))
        d = c.dur[i] / FPS
        x = d * 10 - math.floor(d * 10)
        if abs(x - 0.5) >= 0.15:
            gt = round(d, 1)
            out.append(("swing_duration", {"post": col}, lambda rng, gt=gt, col=col, i=i: (lambda o: o and (
                f"Reading the clock, about how long does the swing around {c.pn(col)} last, from the moment the "
                "rope appears until the particle lands?", fmt_secs(gt), *o, True, min_fps(c.ev[i], 4)))(
                mc4(rng, gt, spaced(rng, gt, [round(gt + k * 0.1, 1) for k in (-12, -9, -6, 6, 9, 12, 15)], 0.6),
                    fmt_secs))))
    tot = sum(c.rope)
    if safe2(tot):
        pool = [tot - c.rope[i] + r for i in range(c.n) for r in ROPES[:12]] + [tot + r for r in c.rope] + \
               [tot - r for r in c.rope]
        out.append(("total_rope", {}, lambda rng: (lambda o: o and (
            UNIT + "Adding up the rope length of every swing, what is the total (to 2 decimals)?", fmt_len(tot),
            *o, True, min_fps(c.all_ev)))(mc4(rng, tot, spaced(rng, tot, pool, 0.15), fmt_len))))
    hi = round((c.N - 1) / FPS, 1)

    def tq(key, f, text):
        gt = sim.clock_tenths(f) / 10
        others = [sim.clock_tenths(g) / 10 for g in c.all_ev]
        pool = others + [round(gt + k * 0.1, 1) for k in (-15, -11, -8, -6, 6, 8, 11, 15)]
        pool = [x for x in pool if 0 <= x <= hi]
        out.append((key, {}, lambda rng: (lambda o: o and (text, render.clock_text(f), *o, True, min_fps([f], 4)))(
            mc4(rng, gt, spaced(rng, gt, pool, 0.6), fmt_secs))))
    tq("land_time", c.ev[-1][1], "What does the clock show at the moment the particle lands on the checkered goal "
                                 "square?")
    tq("first_attach_time", c.ev[0][0], "What does the clock show at the moment the rope first appears?")
    return out


# ------------------------------------------------------------------ L4 integration
def _seq_wrong(c, rng):
    seq, out = list(c.piv), []
    for _ in range(40):
        alt = list(seq)
        kind = rng.randrange(4)
        if kind == 0 and len(alt) > 1:
            j = rng.randrange(len(alt) - 1)
            alt[j], alt[j + 1] = alt[j + 1], alt[j]
        elif kind == 1:
            j = rng.randrange(len(alt))
            alt[j] = rng.choice([x for x in c.colours if x != alt[j]])
        elif kind == 2:
            alt = alt[::-1]
        else:
            j = rng.randrange(len(alt))
            alt = alt[:j] + alt[j + 1:] if rng.random() < 0.5 else alt[:j] + [rng.choice(c.colours)] + alt[j:]
        if all(a != b for a, b in zip(alt, alt[1:])):
            out.append(alt)
    return out


def level4(c: Ctx):
    out = []
    out.append(("pivot_sequence", {}, lambda rng: (lambda o: o and (
        "In order, what colours are the posts the particle swings around?", fmt_seq(c.piv), *o, True,
        min_fps(c.all_ev)))(mc4(rng, c.piv, _seq_wrong(c, rng), fmt_seq))))
    twice = sorted({x for x in c.piv if c.piv.count(x) > 1})
    if len(twice) <= 1:
        gt = c.pn(twice[0]) if twice else NONE_TWICE
        used = [c.pn(x) for x in dict.fromkeys(c.piv) if x not in twice]

        def reused(rng, gt=gt, used=used):
            pool = list(used)
            rng.shuffle(pool)
            return mc4(rng, gt, ([NONE_TWICE] if twice else []) + pool[:3] +
                       [c.pn(x) for x in c.colours if c.pn(x) not in pool])
        out.append(("pivot_reused", {}, lambda rng: (lambda o: o and (
            "Which post does the rope attach to more than once?", gt, *o, True, min_fps(c.all_ev)))(reused(rng))))
    k = sum(1 for t in c.turn if t < 0)
    out.append(("n_clockwise", {}, lambda rng: (lambda o: o and (
        "How many of the particle's swings go clockwise?", k, *o, True, min_fps(c.all_ev)))(
        mc4(rng, k, int_wrong(rng, k, 0, c.n)))))
    for over in LONG_OVER:
        if over < max(c.rope) and all(abs(r - over) > 0.05 for r in c.rope):
            k2 = sum(1 for r in c.rope if r > over)
            out.append(("n_long_ropes", {"over": over}, lambda rng, k2=k2, over=over: (lambda o: o and (
                UNIT + f"How many swings use a rope longer than {over:g} units?", k2, *o, True,
                min_fps(c.all_ev)))(mc4(rng, k2, int_wrong(rng, k2, 0, c.n)))))
    order = sorted(range(c.n), key=lambda i: -c.dur[i])
    top = order[0]
    if c.n > 1 and c.dur[top] - c.dur[order[1]] >= 12 and c.piv[top] in c.once:
        gap = c.dur[top] - c.dur[order[1]]
        out.append(("longest_swing_post", {}, lambda rng: _post_q(
            c, rng, "Around which post does the particle make its longest-lasting single swing?", c.piv[top],
            c.piv, True, max(min_fps(c.all_ev), math.ceil(2 * FPS / gap)))))
    return out


# ------------------------------------------------------------------ L5 counterfactual
def _cost_options(c, rng, gt):
    base = c.s["cost"]
    if gt is None:
        pool = [base + r for r in ROPES[:10]] + [base]
        return mc4(rng, NO_ROUTE, [fmt_len(x) for x in spaced(rng, base + 50, pool, 0.15)])
    if not safe2(gt):
        return None
    pool = [gt + r for r in ROPES[:10]] + [gt - r for r in ROPES[:6]] + [base + 0.71, base + 1.58]
    keep = [base] if abs(base - gt) >= 0.15 else []
    vals = [x for x in spaced(rng, gt, pool, 0.15) if x > 0.5 and all(abs(x - b) >= 0.15 for b in keep)]
    opts = [fmt_len(b) for b in keep] + ([NO_ROUTE] if rng.random() < 0.5 else [])
    return mc4(rng, fmt_len(gt), opts + [fmt_len(x) for x in vals])


def level5(c: Ctx, rng0):
    out = []
    used = list(dict.fromkeys(c.piv))
    unused = [x for x in c.colours if x not in used]
    cand = used + rng0.sample(unused, min(1, len(unused)))
    for col in cand:
        res = c.cf(drop=col)
        gt = None if res is None else res["cost"]
        out.append(("cf_remove_post", {"post": col}, lambda rng, col=col, gt=gt: (lambda o: o and (
            f"If {c.pn(col)} were pulled out of the board, what would the cheapest total rope length from the start "
            "square to the checkered goal square be (to 2 decimals), or would there be no route at all? " + c.rules,
            fmt_len(gt), *o, False, 1))(_cost_options(c, rng, gt))))
        if res is None or res["second"] - res["cost"] >= sim.MARGIN:
            k = NO_ROUTE if res is None else len(res["path"])
            ints = int_wrong(rng0, c.n if res is None else k, 1, 9)
            out.append(("cf_remove_n_swings", {"post": col}, lambda rng, col=col, k=k, ints=ints: (lambda o: o and (
                f"If {c.pn(col)} were pulled out of the board, how many swings would the new cheapest route take, or "
                "would there be no route at all? " + c.rules, k, *o, False, 1))(
                mc4(rng, k, ([NO_ROUTE] if k != NO_ROUTE else [c.n]) + ints))))
        if res is not None and res["second"] - res["cost"] >= sim.MARGIN:
            by_xy = {tuple(xy): col2 for col2, xy in c.xy.items()}
            first = by_xy[tuple(res["path"][0][0])]
            out.append(("cf_remove_first_pivot", {"post": col}, lambda rng, col=col, first=first: (lambda o: o and (
                f"If {c.pn(col)} were pulled out of the board, which post would the rope attach to on the first "
                "swing of the new cheapest route? " + c.rules, c.pn(first), *o, False, 1))(
                mc4(rng, first, [x for x in c.post_wrong(rng, first, [c.piv[0]] + c.piv) if x != col], c.pn))))
    caps = [x for x in SHORT_CAPS if x < max(c.rope) and x < c.s["cap"]]
    if caps:
        cap = rng0.choice(caps)
        res = c.cf(cap=cap)
        gt = None if res is None else res["cost"]
        out.append(("cf_short_rope", {"cap": cap}, lambda rng, cap=cap, gt=gt: (lambda o: o and (
            f"Suppose no rope could be longer than {cap:g} units (all other rules unchanged). What would the cheapest "
            "total rope length from the start square to the checkered goal square be (to 2 decimals), or would "
            "there be no route at all? " + c.rules, fmt_len(gt), *o, False, 1))(_cost_options(c, rng, gt))))
    if c.s["pits"]:
        res = c.cf(pits=False)
        gt = None if res is None else res["cost"]
        out.append(("cf_no_pits", {}, lambda rng, gt=gt: (lambda o: o and (
            "If the hatched pits were filled in so the particle could land on any square, what would the cheapest "
            "total rope length from the start square to the checkered goal square be (to 2 decimals)? " + c.rules,
            fmt_len(gt), *o, False, 1))(_cost_options(c, rng, gt))))
    return out


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
    pools = {1: level1(c), 2: level2(c, rng), 3: level3(c), 4: level4(c), 5: level5(c, rng)}
    made = {}
    for lv in range(1, 6):
        ok = []
        for key, params, make in pools[lv]:
            r = make(random.Random(rng.random()))
            if r:
                ok.append((key, params, r))
        made[lv] = ok
    feasible = [cmb for cmb in TEN_SPLITS if all(len(made[lv]) >= cmb[lv - 1] for lv in range(1, 6))]
    if not feasible:
        raise ValueError("not enough question candidates for this scene")
    counts = rng.choice(feasible)
    facts = {"piv": c.piv, "rope": c.rope, "turn": c.turn, "frames": c.ev}
    qs = []
    for lv in range(1, 6):
        for key, params, (text, gt, opts, letter, motion, fps) in _pick(rng, made[lv], counts[lv - 1]):
            p = dict(params)
            p["min_fps"] = fps
            q = L.question(lv, key, text, gt, opts, letter, motion, p, idx=len(qs) + 1)
            # the mistake behind each wrong option, from the generator's own route (no rng used)
            q["option_errors"] = label_options(key, q["params"], q["options"], q["answer_text"], facts, scene)
            qs.append(q)
    return qs
