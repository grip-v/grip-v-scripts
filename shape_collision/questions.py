"""10-question ladder for shape_collision (common/ladder5.py, LADDER = "ten").

Shapes are named by colour + outline ("the blue star"; every colour is used
once per clip). Moments are named by the on-screen clock ("when the clock
shows 3.4 s"). Every question has 4 options and stores derivation_key +
params (shape ids, pairs, clock tenths, option lists) + min_fps, which
answers.py uses to re-derive the answer from the rendered track.

Margins: compared events are >= 0.3 s apart (MARGIN); clock moments sit
>= 0.3 s from every event that could change the answer; clock answers are
offered in 0.5 s steps; speeds compared differ by >= 15 %; counts are exact.
"""
from __future__ import annotations
import math
import random
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE))
from common import ladder5 as L  # noqa: E402

import sim  # noqa: E402

LADDER = "ten"
FPS_STEPS = (1, 2, 3, 4, 6, 8, 12, 24)
MARGIN = 0.3
ORD = ["first", "second", "third", "fourth", "fifth", "sixth", "seventh", "eighth", "ninth", "tenth", "eleventh",
       "twelfth"]
RAIL_ORD = ["top", "second", "third", "fourth"]
COLL = "collide with each other (bounces off the walls{posts} don't count)"
NOT_TOUCH = "every other shape moved exactly the same way at the start"


def fps_for(gap_s: float) -> int:
    for f in FPS_STEPS:
        if f * gap_s >= 1 - 1e-9:
            return f
    return 24


def gap_fps(times, lo=0.0, hi=None) -> int:
    """min_fps for a question that relies on events at `times` inside [lo, hi]."""
    pts = sorted([lo] + list(times) + ([hi] if hi is not None else []))
    gaps = [b - a for a, b in zip(pts, pts[1:]) if b - a > 1e-9]
    return fps_for(min(gaps)) if gaps else 1


def clock_fps() -> int:
    return fps_for(0.25)


# ---------------------------------------------------------------- facts (generator side)

class Facts:
    """Everything the templates need, from the exact event-driven solver."""

    def __init__(self, scene):
        self.scene = scene
        bs = scene["bodies"]
        self.n = len(bs)
        self.lbl = [sim.label(b) for b in bs]
        self.tl = sim.t_last(scene)
        self.coll = sim.collisions(scene)
        self.ev = sim.shown(scene)
        self.walls = [[] for _ in bs]
        self.posts = [[] for _ in bs]
        self.wq = [[] for _ in bs]
        for t, kind, i, k, q in self.ev:
            if kind == "w":
                self.walls[i].append(t)
                self.wq[i].append(q)
            elif kind == "p":
                self.posts[i].append(t)
                self.wq[i].append(q)
        self.caps = sim.captures(scene)
        self.fall = {i: t for t, i, _ in self.caps}
        self.hits = [[t for t, a, b in self.coll if x in (a, b)] for x in range(self.n)]
        self.pairs = {}
        for k, (t, i, j) in enumerate(self.coll):
            self.pairs.setdefault((i, j), []).append(k)
        self.start = sim.start_patterns(scene)
        self.hist = [self.start[:]]                     # pattern state after each collision
        for _, i, j in self.coll:
            st = self.hist[-1][:]
            sim.apply_rule(scene, st, i, j)
            self.hist.append(st)
        self.final = self.hist[-1]
        self.first_ev = [min([e[0] for e in self.ev if e[2] == x or (e[1] == "c" and e[3] == x)] or [self.tl])
                         for x in range(self.n)]
        self.speed = [math.hypot(b["vx"], b["vy"]) for b in bs]
        self.posts_txt = " or the posts" if scene["posts"] else ""

    def pair(self, i, j) -> str:
        a, b = sorted([self.lbl[i], self.lbl[j]])
        return f"{a} and {b}"

    def change_times(self, x) -> list:
        """Times at which x's pattern changes."""
        return [t for k, (t, i, j) in enumerate(self.coll) if self.hist[k][x] != self.hist[k + 1][x]]

    def clean_turns(self, x, which) -> bool:
        ts = self.walls[x] if which == "w" else self.posts[x]
        qs = [q for t, kind, i, k, q in self.ev if i == x and kind == which]
        return bool(ts) and all(q >= sim.MIN_WALL for q in qs) and ts[-1] <= self.tl - sim.END_PAD and \
            all(b - a >= MARGIN for a, b in zip(ts, ts[1:]))

    def quiet_tenths(self, times, need_before=True, need_after=True) -> list:
        """Clock readings whose frames are >= MARGIN from every time in `times`."""
        out = []
        for k in range(int(sim.START_PAD * 10) + 1, int((self.tl - sim.START_PAD) * 10)):
            lo, hi = sim.frames_showing(k)
            a, b = lo / sim.FPS, hi / sim.FPS
            if any(a - MARGIN < t < b + MARGIN for t in times):
                continue
            if need_before and not any(t < a for t in times):
                continue
            if need_after and not any(t > b for t in times):
                continue
            out.append(k)
        return out


def _q(lv, key, text, gt, opts, a, motion, params):
    return L.question(lv, key, text, gt, opts, a, motion, params)


def _four(rng, correct, pool):
    """Four options: the correct one + 3 distinct distractors from pool."""
    ds = [d for d in dict.fromkeys(pool) if d != correct]
    if len(ds) < 3:
        return None
    opts, a = L.mc(rng, correct, ds[:3])
    return opts, a


def _ints(rng, gt, lo=0, hi=10 ** 6):
    pool = L.near_ints(gt, lo, hi)
    rng.shuffle(pool)
    return _four(rng, gt, pool)


def _secs(k: int) -> str:
    return sim.clock_text(k)


def _clock_opts(rng, k, hi_tenths):
    """Clock readings in 0.5 s steps around the true reading k (tenths)."""
    cands = [k + d for d in (-15, -10, -5, 5, 10, 15) if 0 <= k + d <= hi_tenths]
    cands.sort(key=lambda c: abs(c - k))
    return _four(rng, _secs(k), [_secs(c) for c in cands[:4]])


# ---------------------------------------------------------------- L1 observation

def q_n_shapes(F, rng):
    gt = F.n
    o = _ints(rng, gt, 2, 12)
    return o and _q(1, "n_shapes", "How many shapes are in the box at the start of the video?", gt, *o, False,
                    {"min_fps": 1})


def q_shape_of(F, rng):
    x = rng.randrange(F.n)
    b = F.scene["bodies"][x]
    pool = [F.scene["bodies"][k]["shape"] for k in rng.sample(range(F.n), F.n)] + rng.sample(sim.SHAPES, len(sim.SHAPES))
    o = _four(rng, b["shape"], pool)
    return o and _q(1, "shape_of", f"What outline does the {b['color']} shape have?", b["shape"], *o, False,
                    {"x": x, "min_fps": 1})


def q_color_of(F, rng):
    shapes = [b["shape"] for b in F.scene["bodies"]]
    xs = [k for k in range(F.n) if shapes.count(shapes[k]) == 1]
    if not xs:
        return None
    x = rng.choice(xs)
    b = F.scene["bodies"][x]
    pool = [F.scene["bodies"][k]["color"] for k in rng.sample(range(F.n), F.n)]
    o = _four(rng, b["color"], pool)
    return o and _q(1, "color_of", f"What colour is the {b['shape']}?", b["color"], *o, False,
                    {"x": x, "min_fps": 1})


def q_n_collisions(F, rng):
    gt = len(F.coll)
    o = _ints(rng, gt, 0)
    return o and _q(1, "n_collisions", f"How many times do two shapes {COLL.format(posts=F.posts_txt)}?", gt, *o,
                    True, {"min_fps": count_fps(F)})


def count_fps(F, xs=None) -> int:
    """min_fps for counting collisions: consecutive collisions of one shape
    must fall in different samples (two different pairs touching at the same
    moment are both visible in one frame), and the first / last ones must be
    seen apart from the start / end."""
    xs = range(F.n) if xs is None else xs
    return max(gap_fps(F.hits[x], 0, F.tl) for x in xs if F.hits[x]) if any(F.hits[x] for x in xs) else 1


def q_first_pair(F, rng):
    if len(F.coll) < 2 or F.coll[1][0] - F.coll[0][0] < MARGIN:
        return None
    _, i, j = F.coll[0]
    pool = [F.pair(a, b) for (a, b) in F.pairs if (a, b) != (i, j)]
    pool += [F.pair(*sorted(rng.sample(range(F.n), 2))) for _ in range(30)]
    o = _four(rng, F.pair(i, j), pool)
    return o and _q(1, "first_pair", "Which two shapes are the first to collide with each other?", F.pair(i, j),
                    *o, True, {"min_fps": gap_fps([F.coll[0][0], F.coll[1][0]])})


def q_start_pattern(F, rng):
    if F.scene["rule"] != "swap":
        return None
    x = rng.randrange(F.n)
    o = _four(rng, F.start[x], rng.sample(F.start, F.n) + rng.sample(sim.PATTERNS, len(sim.PATTERNS)))
    return o and _q(1, "start_pattern", f"What fill pattern does {F.lbl[x]} have at the very start of the video?",
                    F.start[x], *o, False, {"x": x, "min_fps": 1})


def q_end_pattern(F, rng):
    if F.scene["rule"] != "swap":
        return None
    xs = [x for x in range(F.n) if x not in F.fall and F.final[x] != F.start[x]]
    if not xs:
        return None
    x = rng.choice(xs)
    o = _four(rng, F.final[x], [F.start[x]] + rng.sample(F.final, F.n) + rng.sample(sim.PATTERNS, len(sim.PATTERNS)))
    ch = F.change_times(x)
    return o and _q(1, "end_pattern", f"What fill pattern does {F.lbl[x]} have at the end of the video?",
                    F.final[x], *o, False, {"x": x, "min_fps": fps_for(F.tl - ch[-1])})


def q_n_fixtures(F, rng):
    sc = F.scene
    if sc["scenario"] == "posts":
        gt, what = len(sc["posts"]), "round grey posts stand in the box"
    elif sc["scenario"] == "holes":
        gt, what = len(sc["holes"]), "holes are there in the floor of the box"
    elif sc["scenario"] == "rails":
        gt, what = len(sc["lanes"]), "horizontal rails do the shapes ride on"
    else:
        return None
    o = _ints(rng, gt, 1, 9)
    return o and _q(1, "n_fixtures", f"How many {what}?", gt, *o, False, {"min_fps": 1})


def q_tag_start(F, rng):
    if F.scene["rule"] != "infect":
        return None
    x = F.start.index(F.scene["tag"])
    pool = [F.lbl[k] for k in rng.sample(range(F.n), F.n)]
    o = _four(rng, F.lbl[x], pool)
    return o and _q(1, "tag_start", f"Which shape wears the {F.scene['tag']} pattern at the start of the video?",
                    F.lbl[x], *o, False, {"min_fps": 1})


def q_n_tagged_end(F, rng):
    if F.scene["rule"] != "infect":
        return None
    gt = F.final.count(F.scene["tag"])
    tt = [t for k, (t, i, j) in enumerate(F.coll) if F.hist[k] != F.hist[k + 1]]
    o = _ints(rng, gt, 1, F.n)
    return o and _q(1, "n_tagged_end", f"How many shapes wear the {F.scene['tag']} pattern at the end of the video?",
                    gt, *o, False, {"min_fps": fps_for(F.tl - tt[-1])})


def q_n_left_end(F, rng):
    if F.scene["scenario"] != "holes":
        return None
    gt = F.n - len(F.caps)
    o = _ints(rng, gt, 1, F.n)
    return o and _q(1, "n_left_end", "How many shapes are still in the box at the end of the video?", gt, *o, False,
                    {"min_fps": fps_for(F.tl - F.caps[-1][0])})


# ---------------------------------------------------------------- L2 relational

def q_partner_of4(F, rng):
    xs = []
    for x in range(F.n):
        part = {a + b - x for (a, b) in F.pairs if x in (a, b)}
        non = [k for k in range(F.n) if k != x and k not in part]
        if part and len(non) >= 3:
            xs.append((x, sorted(part), non))
    if not xs:
        return None
    x, part, non = rng.choice(xs)
    y = rng.choice(part)
    opts_ids = [y] + rng.sample(non, 3)
    o = _four(rng, F.lbl[y], [F.lbl[k] for k in opts_ids[1:]])
    return o and _q(2, "partner_of4", f"Which of these shapes does {F.lbl[x]} collide with at some point?",
                    F.lbl[y], *o, True,
                    {"x": x, "options": sorted(opts_ids), "min_fps": count_fps(F, [x])})


def q_first_partner(F, rng):
    xs = [x for x in range(F.n) if F.hits[x] and (len(F.hits[x]) < 2 or F.hits[x][1] - F.hits[x][0] >= MARGIN)]
    if not xs:
        return None
    x = rng.choice(xs)
    order = [a + b - x for t, a, b in F.coll if x in (a, b)]
    y = order[0]
    pool = [F.lbl[k] for k in order[1:]] + [F.lbl[k] for k in rng.sample(range(F.n), F.n) if k != x]
    o = _four(rng, F.lbl[y], pool)
    return o and _q(2, "first_partner", f"Which shape is the first one that {F.lbl[x]} collides with?", F.lbl[y],
                    *o, True, {"x": x, "min_fps": gap_fps(F.hits[x][:2])})


def q_more_hits4(F, rng):
    for _ in range(20):
        ids = rng.sample(range(F.n), 4)
        c = sorted((len(F.hits[k]) for k in ids), reverse=True)
        if c[0] > c[1]:
            break
    else:
        return None
    y = max(ids, key=lambda k: len(F.hits[k]))
    o = _four(rng, F.lbl[y], [F.lbl[k] for k in ids if k != y])
    return o and _q(2, "more_hits4", "Which of these shapes collides with other shapes the most times?", F.lbl[y],
                    *o, True, {"options": sorted(ids), "min_fps": count_fps(F, ids)})


def q_earlier_pair(F, rng):
    firsts = sorted((F.coll[ks[0]][0], p) for p, ks in F.pairs.items())
    if len(firsts) < 2:
        return None
    for _ in range(20):
        k = rng.randrange(len(firsts) - 1)
        t0, p0 = firsts[k]
        later = [p for t, p in firsts[k + 1:] if t - t0 >= MARGIN]
        if later:
            break
    else:
        return None
    if k > 0 and t0 - firsts[k - 1][0] < MARGIN:
        return None
    # the correct pair, at least one later colliding pair, the rest pairs that never collide
    never = [(a, b) for a in range(F.n) for b in range(a + 1, F.n) if (a, b) not in F.pairs]
    rng.shuffle(never)
    later = rng.sample(later, min(len(later), rng.randint(1, 3)))
    ds = (later + never)[:3]
    if len(ds) < 3:
        return None
    o = _four(rng, F.pair(*p0), [F.pair(*p) for p in ds])
    tt = [t0] + [F.coll[F.pairs[p][0]][0] for p in ds if p in F.pairs]
    return o and _q(2, "earlier_pair", "Which of these pairs of shapes is the first to collide?", F.pair(*p0), *o,
                    True, {"options": sorted([list(p0)] + [list(p) for p in ds]), "min_fps": gap_fps(sorted(tt)[:2])})


def q_first_wall4(F, rng):
    for _ in range(20):
        ids = rng.sample(range(F.n), 4)
        fw = sorted((F.walls[k][0] if F.walls[k] else 99, k) for k in ids)
        if fw[0][0] < 99 and fw[1][0] - fw[0][0] >= MARGIN and \
                all(q >= sim.MIN_WALL for t, kind, i, kk, q in F.ev if kind == "w" and i == fw[0][1] and t == fw[0][0]):
            break
    else:
        return None
    y = fw[0][1]
    o = _four(rng, F.lbl[y], [F.lbl[k] for k in ids if k != y])
    return o and _q(2, "first_wall4", "Which of these shapes is the first to bounce off a wall of the box?",
                    F.lbl[y], *o, True,
                    {"options": sorted(ids), "min_fps": gap_fps([t for t, _ in fw if t < 99][:2])})


def q_fastest4(F, rng):
    for _ in range(20):
        ids = rng.sample(range(F.n), 4)
        sp = sorted((F.speed[k] for k in ids), reverse=True)
        if sp[0] >= 1.15 * sp[1]:
            break
    else:
        return None
    y = max(ids, key=lambda k: F.speed[k])
    o = _four(rng, F.lbl[y], [F.lbl[k] for k in ids if k != y])
    first = min(F.first_ev[k] for k in ids)
    return o and _q(2, "fastest4", "Which of these shapes is moving fastest at the very start of the video?",
                    F.lbl[y], *o, True, {"options": sorted(ids), "min_fps": fps_for(first / 2)})


def q_tagger_of(F, rng):
    if F.scene["rule"] != "infect":
        return None
    tag = F.scene["tag"]
    got = []
    for k, (t, i, j) in enumerate(F.coll):
        for x, y in ((i, j), (j, i)):
            if F.hist[k][x] != tag and F.hist[k + 1][x] == tag:
                got.append((x, y, t))
    if not got:
        return None
    x, y, t = rng.choice(got)
    pool = [F.lbl[k] for k in rng.sample(range(F.n), F.n) if k not in (x, y)]
    o = _four(rng, F.lbl[y], pool)
    return o and _q(2, "tagger_of", f"Which shape passes the {tag} pattern on to {F.lbl[x]}?", F.lbl[y], *o, True,
                    {"x": x, "min_fps": count_fps(F, [x])})


def q_tagged_first4(F, rng):
    if F.scene["rule"] != "infect":
        return None
    tag = F.scene["tag"]
    when = {}
    for k, (t, i, j) in enumerate(F.coll):
        for x in (i, j):
            if F.hist[k][x] != tag and F.hist[k + 1][x] == tag:
                when[x] = t
    if len(when) < 2:
        return None
    for _ in range(20):
        ids = rng.sample([k for k in range(F.n) if F.start[k] != tag], min(4, F.n - 1))
        ts = sorted((when.get(k, 99), k) for k in ids)
        if len(ids) == 4 and ts[0][0] < 99 and ts[1][0] - ts[0][0] >= MARGIN:
            break
    else:
        return None
    y = ts[0][1]
    o = _four(rng, F.lbl[y], [F.lbl[k] for k in ids if k != y])
    return o and _q(2, "tagged_first4", f"Which of these shapes is the first to get the {tag} pattern?", F.lbl[y],
                    *o, True, {"options": sorted(ids), "min_fps": gap_fps([t for t, _ in ts if t < 99][:2])})


def q_fall_first4(F, rng):
    if F.scene["scenario"] != "holes" or F.n < 4:
        return None
    for _ in range(20):
        ids = rng.sample(range(F.n), 4)
        ts = sorted((F.fall.get(k, 99), k) for k in ids)
        if ts[0][0] < 99 and ts[1][0] - ts[0][0] >= MARGIN:
            break
    else:
        return None
    y = ts[0][1]
    o = _four(rng, F.lbl[y], [F.lbl[k] for k in ids if k != y])
    return o and _q(2, "fall_first4", "Which of these shapes is the first to fall into a hole?", F.lbl[y], *o, True,
                    {"options": sorted(ids), "min_fps": gap_fps([t for t, _ in ts if t < 99][:2])})


def q_post_first4(F, rng):
    if F.scene["scenario"] != "posts":
        return None
    for _ in range(20):
        ids = rng.sample(range(F.n), 4)
        ts = sorted((F.posts[k][0] if F.posts[k] else 99, k) for k in ids)
        if ts[0][0] < 99 and ts[1][0] - ts[0][0] >= MARGIN:
            break
    else:
        return None
    y = ts[0][1]
    o = _four(rng, F.lbl[y], [F.lbl[k] for k in ids if k != y])
    return o and _q(2, "post_first4", "Which of these shapes is the first to bounce off one of the posts?", F.lbl[y],
                    *o, True, {"options": sorted(ids), "min_fps": gap_fps([t for t, _ in ts if t < 99][:2])})


# ---------------------------------------------------------------- L3 measurement

def q_hits_of(F, rng):
    xs = [x for x in range(F.n) if F.hits[x]]
    x = rng.choice(xs)
    gt = len(F.hits[x])
    o = _ints(rng, gt, 0)
    return o and _q(3, "hits_of", f"How many times does {F.lbl[x]} collide with another shape?", gt, *o, True,
                    {"x": x, "min_fps": count_fps(F, [x])})


def q_walls_of(F, rng):
    xs = [x for x in range(F.n) if F.clean_turns(x, "w") and x not in F.fall]
    if not xs:
        return None
    x = rng.choice(xs)
    gt = len(F.walls[x])
    o = _ints(rng, gt, 0)
    return o and _q(3, "walls_of", f"How many times does {F.lbl[x]} bounce off a wall of the box?", gt, *o, True,
                    {"x": x, "min_fps": gap_fps(F.walls[x], 0, F.tl)})


def q_posts_of(F, rng):
    if F.scene["scenario"] != "posts":
        return None
    xs = [x for x in range(F.n) if F.clean_turns(x, "p")]
    if not xs:
        return None
    x = rng.choice(xs)
    gt = len(F.posts[x])
    o = _ints(rng, gt, 0)
    return o and _q(3, "posts_of", f"How many times does {F.lbl[x]} bounce off a post?", gt, *o, True,
                    {"x": x, "min_fps": gap_fps(F.posts[x], 0, F.tl)})


def _frame_ok(t) -> bool:
    """The event sits well inside a frame interval (not on a frame boundary)."""
    u = t * sim.FPS - math.floor(t * sim.FPS)
    return 0.1 < u < 0.9


def q_coll_time(F, rng):
    once = [p for p, ks in F.pairs.items() if len(ks) == 1 and _frame_ok(F.coll[ks[0]][0])]
    if not once:
        return None
    i, j = rng.choice(once)
    t = F.coll[F.pairs[(i, j)][0]][0]
    k = sim.tenths(sim.frame_of(t))
    o = _clock_opts(rng, k, int(F.tl * 10))
    return o and _q(3, "coll_time", f"What does the clock show when {F.pair(i, j)} collide with each other?",
                    _secs(k), *o, True, {"x": i, "y": j, "min_fps": clock_fps()})


def q_fall_time(F, rng):
    xs = [x for x, t in F.fall.items() if _frame_ok(t)]
    if not xs:
        return None
    x = rng.choice(sorted(xs))
    k = sim.tenths(sim.frame_of(F.fall[x]))
    o = _clock_opts(rng, k, int(F.tl * 10))
    return o and _q(3, "fall_time", f"What does the clock show when {F.lbl[x]} falls into a hole?", _secs(k), *o,
                    True, {"x": x, "min_fps": clock_fps()})


def q_span(F, rng):
    if len(F.coll) < 2 or not (_frame_ok(F.coll[0][0]) and _frame_ok(F.coll[-1][0])):
        return None
    a, b = (sim.tenths(sim.frame_of(F.coll[k][0])) for k in (0, -1))
    d = b - a
    if d < 5:
        return None
    o = _clock_opts(rng, d, 100)
    return o and _q(3, "span", "How much time passes between the first and the last collision between two shapes?",
                    _secs(d), *o, True, {"min_fps": clock_fps()})


def q_n_coll_by(F, rng):
    times = [t for t, _, _ in F.coll]
    ks = F.quiet_tenths(times)
    if not ks:
        return None
    k = rng.choice(ks)
    lo, hi = sim.frames_showing(k)
    gt = sum(1 for t in times if t < lo / sim.FPS)
    o = _ints(rng, gt, 0)
    prev = max(t for t in times if t < lo / sim.FPS)
    nxt = min(t for t in times if t > hi / sim.FPS)
    return o and _q(3, "n_coll_by", f"How many collisions between two shapes have happened when the clock shows "
                    f"{_secs(k)}?", gt, *o, True, {"t": k, "min_fps": fps_for(nxt - prev)})


def q_pattern_at(F, rng):
    if F.scene["rule"] != "swap":
        return None
    cands = []
    for x in range(F.n):
        ch = F.change_times(x)
        if ch and x not in F.fall:
            for k in F.quiet_tenths(ch, need_after=False):
                cands.append((x, k))
    if not cands:
        return None
    x, k = rng.choice(cands)
    lo, _ = sim.frames_showing(k)
    st = sim.replay(F.scene, upto=sum(1 for t, _, _ in F.coll if t < lo / sim.FPS))
    gt = st[x]
    ch = F.change_times(x)
    prev = max(t for t in ch if t < lo / sim.FPS)
    nxt = min([t for t in ch if t > lo / sim.FPS] + [F.tl])
    o = _four(rng, gt, [F.start[x], F.final[x]] + [h[x] for h in F.hist] + rng.sample(F.start, F.n))
    return o and _q(3, "pattern_at", f"What fill pattern does {F.lbl[x]} have when the clock shows {_secs(k)}?", gt,
                    *o, True, {"x": x, "t": k, "min_fps": fps_for(nxt - prev)})


def q_n_tagged_at(F, rng):
    if F.scene["rule"] != "infect":
        return None
    tt = [t for k, (t, i, j) in enumerate(F.coll) if F.hist[k] != F.hist[k + 1]]
    ks = F.quiet_tenths(tt)
    if not ks:
        return None
    k = rng.choice(ks)
    lo, hi = sim.frames_showing(k)
    st = sim.replay(F.scene, upto=sum(1 for t, _, _ in F.coll if t < lo / sim.FPS))
    gt = st.count(F.scene["tag"])
    prev = max(t for t in tt if t < lo / sim.FPS)
    nxt = min(t for t in tt if t > hi / sim.FPS)
    o = _ints(rng, gt, 1, F.n)
    return o and _q(3, "n_tagged_at", f"How many shapes wear the {F.scene['tag']} pattern when the clock shows "
                    f"{_secs(k)}?", gt, *o, True, {"t": k, "min_fps": fps_for(nxt - prev)})


def q_n_left_at(F, rng):
    if F.scene["scenario"] != "holes":
        return None
    tt = [t for t, _, _ in F.caps]
    ks = F.quiet_tenths(tt, need_after=False)
    if not ks:
        return None
    k = rng.choice(ks)
    lo, _ = sim.frames_showing(k)
    gt = F.n - sum(1 for t in tt if t < lo / sim.FPS)
    prev = max(t for t in tt if t < lo / sim.FPS)
    nxt = min([t for t in tt if t > lo / sim.FPS] + [F.tl])
    o = _ints(rng, gt, 1, F.n)
    return o and _q(3, "n_left_at", f"How many shapes are still in the box when the clock shows {_secs(k)}?", gt, *o,
                    True, {"t": k, "min_fps": fps_for(nxt - prev)})


# ---------------------------------------------------------------- L4 integration

def q_most_hits(F, rng):
    c = sorted((len(h) for h in F.hits), reverse=True)
    if c[0] == c[1]:
        return None
    y = max(range(F.n), key=lambda k: len(F.hits[k]))
    pool = [F.lbl[k] for k in sorted(range(F.n), key=lambda k: (-len(F.hits[k]), rng.random())) if k != y]
    o = _four(rng, F.lbl[y], pool)
    return o and _q(4, "most_hits", "Which shape collides with other shapes the most times over the whole video?",
                    F.lbl[y], *o, True, {"min_fps": count_fps(F)})


def q_kth_pair(F, rng):
    ks = [k for k in range(1, len(F.coll)) if F.coll[k][0] - F.coll[k - 1][0] >= MARGIN and
          (k + 1 == len(F.coll) or F.coll[k + 1][0] - F.coll[k][0] >= MARGIN)]
    if not ks:
        return None
    k = rng.choice(ks)
    _, i, j = F.coll[k]
    pool = [F.pair(a, b) for _, a, b in F.coll if (a, b) != (i, j)]
    pool += [F.pair(*sorted(rng.sample(range(F.n), 2))) for _ in range(30)]
    o = _four(rng, F.pair(i, j), pool)
    return o and _q(4, "kth_pair", f"Counting collisions between two shapes in order, which two shapes make the "
                    f"{ORD[k]} collision?", F.pair(i, j), *o, True,
                    {"k": k + 1, "min_fps": max(count_fps(F), gap_fps([F.coll[m][0] for m in (k - 1, k, k + 1) if m < len(F.coll)]))})


def q_n_partners(F, rng):
    xs = [x for x in range(F.n) if len(F.hits[x]) >= 2]
    if not xs:
        return None
    x = rng.choice(xs)
    gt = len({a + b - x for (a, b) in F.pairs if x in (a, b)})
    o = _ints(rng, gt, 1, F.n - 1)
    return o and _q(4, "n_partners", f"How many different shapes does {F.lbl[x]} collide with over the whole video?",
                    gt, *o, True, {"x": x, "min_fps": count_fps(F, [x])})


def q_n_worn(F, rng):
    if F.scene["rule"] != "swap":
        return None
    xs = [x for x in range(F.n) if len(F.change_times(x)) >= 2]
    if not xs:
        return None
    x = rng.choice(xs)
    gt = len({h[x] for h in F.hist})
    o = _ints(rng, gt, 1, F.n)
    return o and _q(4, "n_worn", f"How many different fill patterns does {F.lbl[x]} wear during the video, counting "
                    "the one it starts with?", gt, *o, True, {"x": x, "min_fps": count_fps(F, [x])})


def q_visits(F, rng):
    if F.scene["rule"] != "swap":
        return None
    ps = []
    for p in F.start:
        holders = [h.index(p) for h in F.hist]
        if len(set(holders)) >= 3:
            ps.append(p)
    if not ps:
        return None
    p = rng.choice(ps)
    gt = len({h.index(p) for h in F.hist})
    tt = [t for k, (t, i, j) in enumerate(F.coll) if F.hist[k].index(p) != F.hist[k + 1].index(p)]
    o = _ints(rng, gt, 1, F.n)
    return o and _q(4, "visits", f"How many different shapes wear the {p} pattern at some point during the video, "
                    "counting the one that starts with it?", gt, *o, True, {"p": p, "min_fps": gap_fps(tt, 0, F.tl)})


def q_who_has(F, rng):
    if F.scene["rule"] != "swap":
        return None
    xs = [x for x in range(F.n) if F.final.index(F.start[x]) != x and F.final.index(F.start[x]) not in F.fall]
    if not xs:
        return None
    x = rng.choice(xs)
    p = F.start[x]
    y = F.final.index(p)
    path = [h.index(p) for h in F.hist]
    pool = [F.lbl[k] for k in dict.fromkeys(path) if k != y] + [F.lbl[k] for k in rng.sample(range(F.n), F.n)]
    o = _four(rng, F.lbl[y], pool)
    # the first and the last frame are enough, so this one doesn't need the motion
    return o and _q(4, "who_has", f"At the end of the video, which shape wears the pattern that {F.lbl[x]} started "
                    "with?", F.lbl[y], *o, False, {"x": x, "min_fps": 1})


def q_n_kept(F, rng):
    if F.scene["rule"] != "swap":
        return None
    gt = sum(1 for k in range(F.n) if F.final[k] == F.start[k] and k not in F.fall)
    o = _ints(rng, gt, 0, F.n)
    who = "of the shapes still in the box" if F.fall else "shapes"
    return o and _q(4, "n_kept", f"At the end of the video, how many {who} wear the same fill pattern they started "
                    "with?", gt, *o, False, {"min_fps": 1})


def q_n_passed_by(F, rng):
    if F.scene["rule"] != "infect":
        return None
    tag = F.scene["tag"]
    by = {}
    for k, (t, i, j) in enumerate(F.coll):
        for x, y in ((i, j), (j, i)):
            if F.hist[k][x] != tag and F.hist[k + 1][x] == tag:
                by.setdefault(y, []).append(x)
    xs = [x for x in range(F.n) if F.hits[x] and (x in by or F.final[x] == tag)]
    if not xs:
        return None
    x = rng.choice(xs)
    gt = len(by.get(x, []))
    o = _ints(rng, gt, 0, F.n - 1)
    return o and _q(4, "n_passed_by", f"How many shapes does {F.lbl[x]} pass the {tag} pattern on to?", gt, *o, True,
                    {"x": x, "min_fps": count_fps(F, [x])})


def q_kth_tagged(F, rng):
    if F.scene["rule"] != "infect":
        return None
    tag = F.scene["tag"]
    order = []
    for k, (t, i, j) in enumerate(F.coll):
        for x in (i, j):
            if F.hist[k][x] != tag and F.hist[k + 1][x] == tag:
                order.append((t, x))
    ks = [k for k in range(1, len(order)) if order[k][0] - order[k - 1][0] >= MARGIN and
          (k + 1 == len(order) or order[k + 1][0] - order[k][0] >= MARGIN)]
    if not ks:
        return None
    k = rng.choice(ks)
    y = order[k][1]
    pool = [F.lbl[x] for _, x in order if x != y] + [F.lbl[x] for x in rng.sample(range(F.n), F.n)]
    pool = [p for p in pool if p != F.lbl[F.start.index(tag)]]
    o = _four(rng, F.lbl[y], pool)
    return o and _q(4, "kth_tagged", f"Not counting the shape that starts with it, which shape is the {ORD[k]} to get "
                    f"the {tag} pattern?", F.lbl[y], *o, True,
                    {"k": k + 1, "min_fps": max(count_fps(F), gap_fps([order[m][0] for m in (k - 1, k, k + 1) if m < len(order)]))})


def q_last_fall(F, rng):
    if F.scene["scenario"] != "holes" or len(F.caps) < 2 or F.caps[-1][0] - F.caps[-2][0] < MARGIN:
        return None
    y = F.caps[-1][1]
    pool = [F.lbl[x] for _, x, _ in F.caps[:-1]] + [F.lbl[x] for x in rng.sample(range(F.n), F.n)]
    o = _four(rng, F.lbl[y], pool)
    return o and _q(4, "last_fall", "Of the shapes that fall into a hole, which one falls in last?", F.lbl[y], *o,
                    True, {"min_fps": gap_fps([t for t, _, _ in F.caps], 0, F.tl)})


def q_most_posts(F, rng):
    if F.scene["scenario"] != "posts":
        return None
    c = sorted((len(p) for p in F.posts), reverse=True)
    if c[0] == c[1] or not all(F.clean_turns(x, "p") or not F.posts[x] for x in range(F.n)):
        return None
    y = max(range(F.n), key=lambda k: len(F.posts[k]))
    pool = [F.lbl[k] for k in sorted(range(F.n), key=lambda k: (-len(F.posts[k]), rng.random())) if k != y]
    o = _four(rng, F.lbl[y], pool)
    return o and _q(4, "most_posts", "Which shape bounces off the posts the most times?", F.lbl[y], *o, True,
                    {"min_fps": max(gap_fps(p, 0, F.tl) for p in F.posts if p)})


# ---------------------------------------------------------------- L5 counterfactual

def _skip_premise(F, i, j, k):
    many = len(F.pairs[(i, j)]) > 1
    when = f" the {ORD[F.pairs[(i, j)].index(k)]} time" if many else ""
    verb = "traded patterns" if F.scene["rule"] == "swap" else "passed on the pattern"
    return f"Suppose every shape moved exactly the same way, but when {F.pair(i, j)} collided{when} they had not {verb}."


def q_cf_skip_pattern(F, rng):
    if F.scene["rule"] != "swap":
        return None
    cands = []
    for k, (t, i, j) in enumerate(F.coll):
        alt = sim.replay(F.scene, skip=k)
        zs = [z for z in range(F.n) if alt[z] != F.final[z] and z not in F.fall]
        if zs:
            cands.append((k, i, j, alt, zs))
    if not cands:
        return None
    k, i, j, alt, zs = rng.choice(cands)
    z = rng.choice(zs)
    o = _four(rng, alt[z], [F.final[z], F.start[z]] + [h[z] for h in F.hist] + rng.sample(F.start, F.n))
    return o and _q(5, "cf_skip_pattern", f"{_skip_premise(F, i, j, k)} What fill pattern would {F.lbl[z]} have at "
                    "the end?", alt[z], *o, True,
                    {"k": k + 1, "z": z, "min_fps": count_fps(F)})


def q_cf_skip_count(F, rng):
    # under the tag rule only a collision that passed the tag on changes anything
    ks = [k for k in range(len(F.coll)) if F.scene["rule"] == "swap" or F.hist[k] != F.hist[k + 1]]
    k = rng.choice(ks)
    _, i, j = F.coll[k]
    alt = sim.replay(F.scene, skip=k)
    if F.scene["rule"] == "swap":
        gt = sum(1 for z in range(F.n) if alt[z] == F.start[z] and z not in F.fall)
        ask = ("how many of the shapes still in the box at the end would wear the fill pattern they started with?"
               if F.fall else "how many shapes would end with the same fill pattern they started with?")
        real = sum(1 for z in range(F.n) if F.final[z] == F.start[z] and z not in F.fall)
    else:
        gt = alt.count(F.scene["tag"])
        ask = f"how many shapes would wear the {F.scene['tag']} pattern at the end?"
        real = F.final.count(F.scene["tag"])
    pool = [real] + L.near_ints(gt, 0, F.n)
    o = _four(rng, gt, pool)
    return o and _q(5, "cf_skip_count", f"{_skip_premise(F, i, j, k)} Then {ask}", gt, *o, True,
                    {"k": k + 1, "min_fps": count_fps(F)})


def q_cf_stop(F, rng):
    if F.scene["rule"] != "swap" or len(F.coll) < 2 or F.coll[-1][0] - F.coll[-2][0] < MARGIN:
        return None
    _, i, j = F.coll[-1]
    alt = sim.replay(F.scene, upto=len(F.coll) - 1)
    z = rng.choice([i, j])
    o = _four(rng, alt[z], [F.final[z], F.start[z]] + [h[z] for h in F.hist] + rng.sample(F.start, F.n))
    return o and _q(5, "cf_stop", "If the video had stopped just before the last collision between two shapes, what "
                    f"fill pattern would {F.lbl[z]} have at that moment?", alt[z], *o, True,
                    {"z": z, "min_fps": gap_fps([F.coll[-2][0], F.coll[-1][0]])})


def q_cf_zero(F, rng):
    if F.scene["rule"] != "infect":
        return None
    tag = F.scene["tag"]
    z0 = F.start.index(tag)
    real = F.final.count(tag)
    cands = []
    for y in range(F.n):
        if y == z0:
            continue
        st = ["solid"] * F.n
        st[y] = tag
        gt = sim.replay(F.scene, start=st).count(tag)
        cands.append((y, gt))
    rng.shuffle(cands)
    cands.sort(key=lambda c: c[1] == real)        # prefer an answer that differs from what happened
    y, gt = cands[0]
    o = _four(rng, gt, [real] + L.near_ints(gt, 1, F.n))
    return o and _q(5, "cf_zero", f"Suppose every shape moved exactly the same way, but {F.lbl[y]} had started with "
                    f"the {tag} pattern instead of {F.lbl[z0]} (which would start solid). How many shapes would wear "
                    f"the {tag} pattern at the end?", gt, *o, True,
                    {"y": y, "min_fps": count_fps(F)})


def _cf_ok(F, **kw) -> bool:
    """The changed world's events are as clean as a real clip's."""
    _, evs, nears, _ = sim._run(F.scene, **kw)
    if sim.event_errors(F.scene, evs, nears, F.tl):
        return False
    return True


def _remove_fps(F, x):
    """Reading the start motion: two samples before anyone's first event."""
    return fps_for(min(F.first_ev) / 2)


def q_cf_remove_n(F, rng):
    xs = rng.sample(range(F.n), F.n)
    for x in xs[:4]:
        if not _cf_ok(F, drop=x):
            continue
        gt = len(sim.collisions(F.scene, drop=x))
        real_rest = len([c for c in F.coll if x not in c[1:]])
        o = _four(rng, gt, [real_rest, len(F.coll)] + L.near_ints(gt, 0))
        return o and _q(5, "cf_remove_n", f"Suppose {F.lbl[x]} had not been in the box at all, and {NOT_TOUCH}. How "
                        f"many times would two shapes {COLL.format(posts=F.posts_txt)}?", gt, *o, True,
                        {"x": x, "min_fps": _remove_fps(F, x)})
    return None


def q_cf_remove_first(F, rng):
    xs = rng.sample(range(F.n), F.n)
    for x in xs[:4]:
        if not _cf_ok(F, drop=x):
            continue
        cc = sim.collisions(F.scene, drop=x)
        if len(cc) < 2 or cc[1][0] - cc[0][0] < MARGIN:
            continue
        _, i, j = cc[0]
        pool = [F.pair(a, b) for _, a, b in F.coll if x not in (a, b) and (a, b) != (i, j)]
        pool += [F.pair(a, b) for _, a, b in cc[1:] if (a, b) != (i, j)]
        others = [k for k in range(F.n) if k != x]
        pool += [F.pair(*sorted(rng.sample(others, 2))) for _ in range(30)]
        o = _four(rng, F.pair(i, j), pool)
        return o and _q(5, "cf_remove_first", f"Suppose {F.lbl[x]} had not been in the box at all, and {NOT_TOUCH}. "
                        "Which two shapes would be the first to collide with each other?", F.pair(i, j), *o, True,
                        {"x": x, "min_fps": _remove_fps(F, x)})
    return None


def q_cf_no_fixtures(F, rng):
    sc = F.scene["scenario"]
    if sc == "posts":
        kw, what = {"no_posts": True}, "the posts had been taken out of the box"
    elif sc == "holes":
        kw, what = {"no_holes": True}, "the holes had been covered over, so shapes roll across them"
    else:
        return None
    if not _cf_ok(F, **kw):
        return None
    gt = len(sim.collisions(F.scene, **kw))
    o = _four(rng, gt, [len(F.coll)] + L.near_ints(gt, 0))
    return o and _q(5, "cf_no_fixtures", f"Suppose {what}, and every shape moved exactly the same way at the start. How "
                    "many times would two shapes collide with each other (bounces off the walls don't count)?", gt, *o,
                    True, {"min_fps": fps_for(min(F.first_ev) / 2)})


def q_cf_remove_falls(F, rng):
    if F.scene["scenario"] != "holes":
        return None
    for x in rng.sample(range(F.n), F.n)[:4]:
        if not _cf_ok(F, drop=x):
            continue
        gt = len(sim.captures(F.scene, drop=x))
        real = len([c for c in F.caps if c[1] != x])
        o = _four(rng, gt, [real, len(F.caps)] + L.near_ints(gt, 0, F.n - 1))
        return o and _q(5, "cf_remove_falls", f"Suppose {F.lbl[x]} had not been in the box at all, and {NOT_TOUCH}. "
                        "How many shapes would fall into a hole?", gt, *o, True,
                        {"x": x, "min_fps": _remove_fps(F, x)})
    return None


TEMPLATES = {
    1: [q_n_shapes, q_shape_of, q_color_of, q_n_collisions, q_first_pair, q_start_pattern, q_end_pattern,
        q_n_fixtures, q_tag_start, q_n_tagged_end, q_n_left_end],
    2: [q_partner_of4, q_first_partner, q_more_hits4, q_earlier_pair, q_first_wall4, q_fastest4, q_tagger_of,
        q_tagged_first4, q_fall_first4, q_post_first4],
    3: [q_hits_of, q_walls_of, q_posts_of, q_coll_time, q_fall_time, q_span, q_n_coll_by, q_pattern_at,
        q_n_tagged_at, q_n_left_at],
    4: [q_most_hits, q_kth_pair, q_n_partners, q_n_worn, q_visits, q_who_has, q_n_kept, q_n_passed_by, q_kth_tagged,
        q_last_fall, q_most_posts],
    5: [q_cf_skip_pattern, q_cf_skip_count, q_cf_stop, q_cf_zero, q_cf_remove_n, q_cf_remove_first,
        q_cf_no_fixtures, q_cf_remove_falls],
}
# templates that read a single frame; they may fill at most one L1 slot
STATIC = {"n_shapes", "shape_of", "color_of", "start_pattern", "end_pattern", "n_fixtures", "tag_start",
          "n_tagged_end", "n_left_end"}


def _sig(q):
    return q["derivation_key"], tuple(sorted((k, str(v)) for k, v in q["params"].items()))


def build(scene: dict, rng: random.Random) -> list[dict]:
    F = Facts(scene)
    counts = L.ten_counts(rng)
    out, seen = [], set()
    for lv, want in zip(range(1, 6), counts):
        got, keys = [], set()
        tpl = TEMPLATES[lv][:]
        rng.shuffle(tpl)
        for rnd in range(6):
            for fn in tpl:
                if len(got) == want:
                    break
                name = fn.__name__[2:]
                if rnd < 2 and name in keys:
                    continue                    # spread over different templates first
                if rnd < 3 and lv == 1 and name in STATIC and any(q["derivation_key"] in STATIC for q in got):
                    continue
                q = fn(F, rng)
                if not q or _sig(q) in seen:
                    continue
                seen.add(_sig(q))
                keys.add(q["derivation_key"])
                got.append(q)
        if len(got) < want:
            raise RuntimeError(f"level {lv}: only {len(got)} questions for {scene['scenario']}")
        out += got
    return L.number(out)
