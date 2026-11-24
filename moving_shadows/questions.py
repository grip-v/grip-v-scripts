"""10-question ladder for moving_shadows (common/ladder5.py, LADDER = "ten").

Markers are named by their on-screen number ("marker 3"); objects by colour
and shape ("the red star", "the blue post"), colours are unique per clip.
Clock-time questions use the 0.5 s grid; "when" answers are the clock
reading at the first frame that shows the event; durations are rounded to
0.5 s. Every question has four distinct options, stores its derivation_key,
the params answers.py needs, and params["min_fps"].

Margins: compared event times differ by >= 0.3 s (GAP), compared durations
by >= 0.3 s and 10 %, duration answers sit >= 0.15 s from a rounding
boundary, clock answers need the shadow edge >= 1.5 px from the marker
centre on both frames around the event, and clock-time questions need no
relevant event within [t - 0.3, t + 0.4].
"""
from __future__ import annotations
import itertools
import math
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import ladder5 as L  # noqa: E402
import sim  # noqa: E402

LADDER = "ten"
M = sim.GAP
INF = 1e9
NONE_OPT = "no object's shadow"
TWO_OPT = "two objects' shadows"


# ------------------------------------------------------------------ helpers
def fps_for(gaps) -> int:
    gaps = [g for g in gaps if g is not None and g < INF]
    if not gaps:
        return 1
    return max(1, min(24, math.ceil(1 / min(gaps) - 1e-9)))


def tsec(v: float) -> str:
    return f"{v:.1f} s"


def clock_of_frame(f: int) -> str:
    return tsec(sim.clock_tenths(f) / 10)


def first_frame_after(t: float):
    """(first frame shown after t, whether t sits well inside its frame gap)."""
    x = t * sim.FPS
    f = math.floor(x) + 1
    return f, min(x - math.floor(x), f - x) >= 0.15


def window(rng, v, step, lo, hi, fmt):
    """Three distractors: a random run of four values on the step grid that contains v."""
    starts = [k for k in range(-3, 1) if all(lo - 1e-9 <= v + (k + i) * step <= hi + 1e-9 for i in range(4))]
    if not starts:
        return None
    k = rng.choice(starts)
    return [fmt(v + (k + i) * step) for i in range(4) if k + i != 0]


def opts4(rng, correct, distractors):
    opts, a = L.mc(rng, correct, distractors)
    return (opts, a) if len(opts) == 4 else (None, None)


def legs(xs, h):
    """Zigzag legs of a 1-D track with hysteresis h: [(sign, start index, end index)]."""
    out, anchor, ext, d, k_ext, k_anchor = [], xs[0], xs[0], 0, 0, 0
    for k, x in enumerate(xs):
        if d >= 0 and x > ext:
            ext, k_ext = x, k
        if d <= 0 and x < ext:
            ext, k_ext = x, k
        if d == 0 and abs(x - anchor) >= h:
            d = 1 if x > anchor else -1
        elif d > 0 and ext - x >= h:
            out.append((1, k_anchor, k_ext))
            anchor, k_anchor, d, ext = ext, k_ext, -1, x
            k_ext = k
        elif d < 0 and x - ext >= h:
            out.append((-1, k_anchor, k_ext))
            anchor, k_anchor, d, ext = ext, k_ext, 1, x
            k_ext = k
    if d != 0:
        out.append((d, k_anchor, k_ext))
    return out


WAY = {(1,): "to the right", (-1,): "to the left", (-1, 1): "left, then back right", (1, -1): "right, then back left"}


# ------------------------------------------------------------------ context
class Ctx:
    def __init__(self, scene):
        self.sc = scene
        self.T = sim.t_end(scene)
        self.N = sim.n_frames(scene)
        self.cs = scene["casters"]
        self.nc = len(self.cs)
        self.nm = len(scene["markers"])
        self.lbl = [sim.label(o) for o in self.cs]
        self.name = [f"marker {j + 1}" for j in range(self.nm)]
        self.un = sim.union(scene)
        self.per = sim.per_caster(scene)
        self.first = {j: self.un[j][0][0] for j in range(self.nm) if self.un[j]}
        self.dur = {j: sum(b - a for a, b in self.un[j]) for j in range(self.nm)}
        self.ever = [j for j in range(self.nm) if self.un[j]]
        self.never = [j for j in range(self.nm) if not self.un[j]]
        self.light = sim.light_name(scene)
        self.movers = [i for i, o in enumerate(self.cs) if o.get("vx") or o.get("vy")]

    def ilen(self, j):
        """Length of the longest shaded interval of marker j."""
        return max((b - a for a, b in self.un[j]), default=None)

    def edges(self, js):
        return sorted(x for j in js for (i, jj), ivs in self.per.items() if jj == j for a, b in ivs for x in (a, b)
                      if x < self.T - 1e-9)

    def quiet(self, t, js):
        return all(not (t - 0.3 <= x <= t + 0.4) for x in self.edges(js))

    def grid(self):
        return [k / 2 for k in range(2, int(2 * (self.T - 1.0)) + 1)]

    def covering(self, j, t):
        return [i for i in range(self.nc) if any(a < t < b for a, b in self.per[(i, j)])]

    def caster_at_edge(self, j, t):
        for (i, jj), ivs in self.per.items():
            if jj == j and any(abs(x - t) < 1e-7 for a, b in ivs for x in (a, b)):
                return i
        return None

    def clock_safe(self, j, t):
        """Clock answer for an event at marker j, time t: (frame, ok)."""
        f, ok = first_frame_after(t)
        if not ok or not sim.EDGE <= t <= self.T - sim.EDGE:
            return f, False
        i = self.caster_at_edge(j, t)
        if i is None:
            return f, False
        for g in (f - 1, f):
            if abs(sim.edge_pos(self.sc, i, j, t, g / sim.FPS)) < 1.5:
                return f, False
        return f, True

    def caster_opts(self, correct, pool_extra=(NONE_OPT, TWO_OPT)):
        return [x for x in self.lbl + list(pool_extra) if x != correct]


def align(t):
    return 1 if abs(t - round(t)) < 1e-9 else 2


def dur_ok(d):
    v = round(d * 2) / 2
    return v >= 0.5 and abs(d - v) <= 0.25 - 0.15, v


# ------------------------------------------------------------------ templates
# Each template returns a list of candidates (params, make) where make(rng) -> question or None.
def t_n_casters(c: Ctx):
    gt = c.nc

    def make(rng):
        opts, a = opts4(rng, gt, window(rng, gt, 1, 1, 8, int))
        return opts and L.question(1, "n_casters", "How many objects cast a shadow onto the ground in this clip?",
                                   gt, opts, a, False, {"min_fps": 1})
    return [({}, make)]


def t_light_dir(c: Ctx):
    gt = "left to right" if sim.light_dir(c.sc) > 0 else "right to left"

    def make(rng):
        opts, a = opts4(rng, gt, ["right to left" if gt == "left to right" else "left to right",
                                  "it stays in one place", "it goes up and comes back down"])
        return opts and L.question(1, "light_dir", f"Which way does the {c.light} move across the frame?", gt, opts, a,
                                   True, {"min_fps": 1})
    return [({}, make)]


def t_first_marker(c: Ctx):
    sh = sorted((t, j) for j, t in c.first.items())
    if len(sh) < 2 or sh[1][0] - sh[0][0] < M:
        return []
    j = sh[0][1]

    def make(rng):
        opts, a = opts4(rng, c.name[j], [c.name[k] for _, k in sh[1:]] + [c.name[k] for k in c.never])
        return opts and L.question(1, "first_marker", "Which marker is the first to be in shadow?", c.name[j], opts,
                                   a, True, {"min_fps": fps_for([sh[1][0] - sh[0][0], c.un[j][0][1] - c.un[j][0][0]])})
    return [({}, make)]


def t_n_ever(c: Ctx):
    gt = len(c.ever)

    def make(rng):
        opts, a = opts4(rng, gt, window(rng, gt, 1, 0, c.nm, int))
        return opts and L.question(1, "n_ever", "How many of the markers are in shadow at some point in the clip?",
                                   gt, opts, a, True, {"min_fps": fps_for([c.ilen(j) for j in c.ever])})
    return [({}, make)]


def t_never(c: Ctx):
    out = []
    if len(c.ever) < 3:
        return out
    for j in c.never:
        def make(rng, j=j):
            others = rng.sample(c.ever, 3)
            ms = sorted(others + [j])
            opts, a = opts4(rng, c.name[j], [c.name[k] for k in others])
            return opts and L.question(1, "never", "Which of these markers is never in shadow?", c.name[j], opts, a,
                                       True, {"ms": ms, "min_fps": fps_for([c.ilen(k) for k in others])})
        out.append(({"j": j}, make))
    return out


def t_shaded_at_t(c: Ctx):
    out = []
    for t in c.grid():
        if not c.quiet(t, range(c.nm)):
            continue
        on = [j for j in range(c.nm) if sim.shaded_at(c.sc, j, t)]
        off = [j for j in range(c.nm) if j not in on]
        if not on or len(off) < 3:
            continue
        for j in on:
            def make(rng, j=j, t=t, off=off):
                others = rng.sample(off, 3)
                opts, a = opts4(rng, c.name[j], [c.name[k] for k in others])
                return opts and L.question(1, "shaded_at_t", f"When the clock shows {tsec(t)}, which of these markers "
                                           "is in shadow?", c.name[j], opts, a, True,
                                           {"t": t, "ms": sorted(others + [j]), "min_fps": align(t)})
            out.append(({"t": t, "j": j}, make))
    return out


def t_n_shaded_at_t(c: Ctx):
    out = []
    for t in c.grid():
        if not c.quiet(t, range(c.nm)):
            continue
        gt = sum(1 for j in range(c.nm) if sim.shaded_at(c.sc, j, t))

        def make(rng, t=t, gt=gt):
            opts, a = opts4(rng, gt, window(rng, gt, 1, 0, c.nm, int))
            return opts and L.question(1, "n_shaded_at_t", f"How many markers are in shadow when the clock shows "
                                       f"{tsec(t)}?", gt, opts, a, True, {"t": t, "min_fps": align(t)})
        out.append(({"t": t}, make))
    return out


def t_moving_object(c: Ctx):
    if len(c.movers) != 1:
        return []
    i = c.movers[0]

    def make(rng):
        opts, a = opts4(rng, c.lbl[i], c.caster_opts(c.lbl[i], ("none of them", "all of them")))
        return opts and L.question(1, "moving_object", "Which object moves during the clip?", c.lbl[i], opts, a,
                                   True, {"min_fps": 1})
    return [({}, make)]


def t_move_dir(c: Ctx):
    if len(c.movers) != 1:
        return []
    o = c.cs[c.movers[0]]
    gt = ("to the right" if o["vx"] > 0 else "to the left") if o.get("vx") else ("down" if o["vy"] > 0 else "up")

    def make(rng):
        opts, a = opts4(rng, gt, [x for x in ("to the left", "to the right", "up", "down") if x != gt])
        return opts and L.question(1, "move_dir", f"Which way does {c.lbl[c.movers[0]]} move?", gt, opts, a, True,
                                   {"i": c.movers[0], "min_fps": 1})
    return [({}, make)]


# L2 ------------------------------------------------------------------------
def t_caster_of(c: Ctx):
    out = []
    for j in c.ever:
        fs = sorted((ivs[0][0], i) for (i, jj), ivs in c.per.items() if jj == j and ivs)
        if len(fs) > 1 and fs[1][0] - fs[0][0] < M:
            continue
        i = fs[0][1]
        gap = fs[1][0] - fs[0][0] if len(fs) > 1 else INF

        def make(rng, j=j, i=i, gap=gap):
            opts, a = opts4(rng, c.lbl[i], c.caster_opts(c.lbl[i]))
            iv = c.per[(i, j)][0]
            return opts and L.question(2, "caster_of", f"Whose shadow is the first to reach {c.name[j]}?", c.lbl[i],
                                       opts, a, True, {"j": j, "min_fps": fps_for([gap, iv[1] - iv[0]])})
        out.append(({"j": j}, make))
    return out


def t_who_covers_at_t(c: Ctx):
    out = []
    for t in c.grid():
        for j in range(c.nm):
            if not c.quiet(t, [j]):
                continue
            cov = c.covering(j, t)
            if len(cov) > 2:
                continue
            gt = NONE_OPT if not cov else c.lbl[cov[0]] if len(cov) == 1 else TWO_OPT
            if not cov and rng_free_skip(t, j):
                continue

            def make(rng, j=j, t=t, gt=gt):
                opts, a = opts4(rng, gt, c.caster_opts(gt))
                return opts and L.question(2, "who_covers_at_t", f"When the clock shows {tsec(t)}, whose shadow is "
                                           f"over {c.name[j]}?", gt, opts, a, True, {"j": j, "t": t, "min_fps": align(t)})
            out.append(({"j": j, "t": t}, make))
    return out


def rng_free_skip(t, j):
    """Thin out 'no shadow' cases deterministically (keeps most of these questions about a real shadow)."""
    return (int(t * 2) + j) % 3 != 0


def t_shadow_way(c: Ctx):
    out = []
    for i in range(c.nc):
        xs = []
        for f in range(c.N):
            lo, hi = sim.shadow(c.sc, i, f / sim.FPS)
            xs.append((lo + hi) / 2)
        lg, lg2 = legs(xs, 40.0), legs(xs, 15.0)
        if [d for d, _, _ in lg] != [d for d, _, _ in lg2] or tuple(d for d, _, _ in lg) not in WAY:
            continue
        if any(abs(xs[b] - xs[a]) < 60 for _, a, b in lg):
            continue
        gt = WAY[tuple(d for d, _, _ in lg)]
        fps = fps_for([(b - a) / sim.FPS for _, a, b in lg])

        def make(rng, i=i, gt=gt, fps=fps):
            opts, a = opts4(rng, gt, [w for w in WAY.values() if w != gt])
            return opts and L.question(2, "shadow_way", f"Which way does the shadow of {c.lbl[i]} move along the "
                                       "ground?", gt, opts, a, True, {"i": i, "min_fps": fps})
        out.append(({"i": i}, make))
    return out


def t_earliest_of(c: Ctx):
    out = []
    if len(c.ever) < 4:
        return out
    for ms in itertools.combinations(c.ever, 4):
        ts = sorted((c.first[j], j) for j in ms)
        g = ts[1][0] - ts[0][0]
        if g < M:
            continue
        out.append(({"ms": list(ms)}, _earliest_make(c, list(ms), ts[0][1], g)))
    return out


def _earliest_make(c, ms, j, g):
    def make(rng):
        opts, a = opts4(rng, c.name[j], [c.name[k] for k in ms if k != j])
        names = ", ".join(str(k + 1) for k in ms[:-1]) + f" and {ms[-1] + 1}"
        return opts and L.question(2, "earliest_of", f"Of markers {names}, which goes into shadow first?", c.name[j],
                                   opts, a, True, {"ms": ms, "min_fps": fps_for([g])})
    return make


def t_last_marker(c: Ctx):
    sh = sorted((t, j) for j, t in c.first.items())
    if len(sh) < 3 or sh[-1][0] - sh[-2][0] < M:
        return []
    j = sh[-1][1]

    def make(rng):
        opts, a = opts4(rng, c.name[j], [c.name[k] for _, k in sh[:-1]][::-1] + [c.name[k] for k in c.never])
        return opts and L.question(2, "last_marker", "Of the markers that go into shadow, which one is the last to "
                                   "do so for the first time?", c.name[j], opts, a, True,
                                   {"min_fps": fps_for([sh[-1][0] - sh[-2][0], c.un[j][0][1] - c.un[j][0][0]])})
    return [({}, make)]


def _dur_margin(c, j, k):
    a, b = c.dur[j], c.dur[k]
    return a - b >= M and a - b >= 0.1 * a


def t_longest_of(c: Ctx):
    out = []
    if c.nm < 4:
        return out
    for ms in itertools.combinations(range(c.nm), 4):
        ds = sorted(ms, key=lambda j: -c.dur[j])
        if not _dur_margin(c, ds[0], ds[1]) or sum(1 for j in ms if c.un[j]) < 2:
            continue
        m = c.dur[ds[0]] - c.dur[ds[1]]
        fps = math.ceil((len(c.un[ds[0]]) + len(c.un[ds[1]])) / m - 1e-9)
        out.append(({"ms": list(ms)}, _longest_make(c, list(ms), ds[0], max(fps, 1))))
    return out


def _longest_make(c, ms, j, fps):
    def make(rng):
        opts, a = opts4(rng, c.name[j], [c.name[k] for k in ms if k != j])
        names = ", ".join(str(k + 1) for k in ms[:-1]) + f" and {ms[-1] + 1}"
        return opts and L.question(2, "longest_of", f"Of markers {names}, which spends the most time in shadow in "
                                   "total?", c.name[j], opts, a, True, {"ms": ms, "min_fps": min(24, fps)})
    return make


# L3 ------------------------------------------------------------------------
def t_cover_time(c: Ctx):
    out = []
    for j in c.ever:
        t = c.first[j]
        f, ok = c.clock_safe(j, t)
        if not ok:
            continue
        gt = clock_of_frame(f)
        v = sim.clock_tenths(f) / 10
        L1 = c.un[j][0][1] - t

        def make(rng, j=j, gt=gt, v=v, L1=L1):
            opts, a = opts4(rng, gt, window(rng, v, 0.5, 0.0, c.N / sim.FPS, tsec))
            return opts and L.question(3, "cover_time", f"On the clock, when does {c.name[j]} first go into shadow?",
                                       gt, opts, a, True, {"j": j, "min_fps": max(4, fps_for([L1]))})
        out.append(({"j": j}, make))
    return out


def t_uncover_time(c: Ctx):
    out = []
    for j in c.ever:
        a0, b0 = c.un[j][0]
        if b0 >= c.T - 1e-9:
            continue
        f, ok = c.clock_safe(j, b0)
        if not ok:
            continue
        gt = clock_of_frame(f)
        v = sim.clock_tenths(f) / 10
        nxt = c.un[j][1][0] - b0 if len(c.un[j]) > 1 else INF

        def make(rng, j=j, gt=gt, v=v, g=[b0 - a0, nxt]):
            opts, a = opts4(rng, gt, window(rng, v, 0.5, 0.0, c.N / sim.FPS, tsec))
            return opts and L.question(3, "uncover_time", f"On the clock, when does {c.name[j]} first come back out of "
                                       "shadow?", gt, opts, a, True, {"j": j, "min_fps": max(4, fps_for(g))})
        out.append(({"j": j}, make))
    return out


def t_shade_dur(c: Ctx):
    out = []
    for j in c.ever:
        k = len(c.un[j])
        ok, v = dur_ok(c.dur[j])
        if k > 2 or not ok:
            continue

        def make(rng, j=j, v=v, k=k):
            opts, a = opts4(rng, tsec(v), window(rng, v, 1.0, 0.5, c.T, tsec))
            fps = max(math.ceil(k / 0.4 - 1e-9), fps_for([b - a for a, b in c.un[j]]))
            return opts and L.question(3, "shade_dur", f"For how long in total is {c.name[j]} in shadow (to the "
                                       "nearest 0.5 s)?", tsec(v), opts, a, True, {"j": j, "min_fps": fps})
        out.append(({"j": j}, make))
    return out


def t_gap_time(c: Ctx):
    out = []
    for a_, b_ in itertools.combinations(sorted(c.first, key=c.first.get), 2):
        d = c.first[b_] - c.first[a_]
        ok, v = dur_ok(d)
        if not ok or d < 0.5:
            continue

        def make(rng, a_=a_, b_=b_, v=v):
            opts, a = opts4(rng, tsec(v), window(rng, v, 1.0, 0.5, c.T, tsec))
            fps = max(3, fps_for([c.un[a_][0][1] - c.un[a_][0][0], c.un[b_][0][1] - c.un[b_][0][0]]))
            return opts and L.question(3, "gap_time", f"How much time passes between {c.name[a_]} first going into "
                                       f"shadow and {c.name[b_]} first going into shadow (to the nearest 0.5 s)?",
                                       tsec(v), opts, a, True, {"a": a_, "b": b_, "min_fps": fps})
        out.append(({"a": a_, "b": b_}, make))
    return out


def t_n_entries(c: Ctx):
    out = []
    for j in c.ever:
        gt = len(c.un[j])
        gaps = [b - a for a, b in c.un[j]] + [c2 - b for (_, b), (c2, _) in zip(c.un[j], c.un[j][1:])]

        def make(rng, j=j, gt=gt, gaps=gaps):
            opts, a = opts4(rng, gt, window(rng, gt, 1, 1, 9, int))
            return opts and L.question(3, "n_entries", f"How many separate times does {c.name[j]} go into shadow?",
                                       gt, opts, a, True, {"j": j, "min_fps": fps_for(gaps)})
        out.append(({"j": j}, make))
    return out


# L4 ------------------------------------------------------------------------
def t_order(c: Ctx):
    out = []
    sh = sorted((t, j) for j, t in c.first.items())
    for k in (3, 4):
        for sub in itertools.combinations(sh, k):
            if any(b[0] - a[0] < M for a, b in zip(sub, sub[1:])):
                continue
            ms = sorted(j for _, j in sub)
            seq = [j + 1 for _, j in sub]
            out.append(({"ms": ms}, _order_make(c, ms, seq, min(b[0] - a[0] for a, b in zip(sub, sub[1:])))))
    return out


def _order_make(c, ms, seq, g):
    def make(rng):
        fmt = lambda s: " → ".join(map(str, s))  # noqa: E731
        gt = fmt(seq)
        wrong = set()
        for _ in range(60):
            alt = list(seq)
            rng.shuffle(alt)
            if alt != seq:
                wrong.add(fmt(alt))
        wrong = sorted(wrong)
        rng.shuffle(wrong)
        opts, a = opts4(rng, gt, wrong)
        names = ", ".join(str(k + 1) for k in ms[:-1]) + f" and {ms[-1] + 1}"
        return opts and L.question(4, "order", f"In what order do markers {names} first go into shadow?", gt, opts, a,
                                   True, {"ms": ms, "min_fps": fps_for([g])})
    return make


def t_most_least(c: Ctx):
    out = []
    ds = sorted(c.ever, key=lambda j: -c.dur[j])
    if len(ds) >= 2 and _dur_margin(c, ds[0], ds[1]):
        m = c.dur[ds[0]] - c.dur[ds[1]]
        fps = min(24, math.ceil((len(c.un[ds[0]]) + len(c.un[ds[1]])) / m - 1e-9))
        out.append(({"most": True}, _ml_make(c, "most_shaded", ds[0], ds, fps,
                                             "Which marker spends the longest time in shadow in total?")))
    if len(ds) >= 3 and _dur_margin(c, ds[-2], ds[-1]):
        m = c.dur[ds[-2]] - c.dur[ds[-1]]
        fps = min(24, math.ceil((len(c.un[ds[-1]]) + len(c.un[ds[-2]])) / m - 1e-9))
        out.append(({"most": False}, _ml_make(c, "least_shaded", ds[-1], ds, fps,
                                              "Of the markers that are in shadow at some point, which one spends the "
                                              "shortest time in shadow in total?")))
    return out


def _ml_make(c, key, j, ds, fps, text):
    def make(rng):
        opts, a = opts4(rng, c.name[j], [c.name[k] for k in ds if k != j] + [c.name[k] for k in c.never])
        return opts and L.question(4, key, text, c.name[j], opts, a, True, {"min_fps": max(1, fps)})
    return make


def t_n_casters_for(c: Ctx):
    out = []
    for j in c.ever:
        hit = [i for i in range(c.nc) if c.per[(i, j)]]
        lens = [b - a for i in hit for a, b in c.per[(i, j)]]

        def make(rng, j=j, gt=len(hit), lens=lens):
            opts, a = opts4(rng, gt, window(rng, gt, 1, 0, max(4, c.nc + 1), int))
            return opts and L.question(4, "n_casters_for", f"How many different objects' shadows pass over "
                                       f"{c.name[j]} during the clip?", gt, opts, a, True,
                                       {"j": j, "min_fps": fps_for(lens)})
        out.append(({"j": j}, make))
    return out


def t_caster_markers(c: Ctx):
    out = []
    for i in range(c.nc):
        hit = [j for j in range(c.nm) if c.per[(i, j)]]
        lens = [b - a for j in hit for a, b in c.per[(i, j)]]

        def make(rng, i=i, gt=len(hit), lens=lens):
            opts, a = opts4(rng, gt, window(rng, gt, 1, 0, c.nm, int))
            return opts and L.question(4, "caster_markers", f"How many of the markers does the shadow of {c.lbl[i]} "
                                       "pass over?", gt, opts, a, True, {"i": i, "min_fps": fps_for(lens)})
        out.append(({"i": i}, make))
    return out


def t_total_entries(c: Ctx):
    gt = sum(len(c.un[j]) for j in range(c.nm))
    gaps = [b - a for j in c.ever for a, b in c.un[j]] + \
        [c2 - b for j in c.ever for (_, b), (c2, _) in zip(c.un[j], c.un[j][1:])]

    def make(rng):
        opts, a = opts4(rng, gt, window(rng, gt, 1, 1, 30, int))
        return opts and L.question(4, "total_entries", "Counting every marker and every time, how many times in total "
                                   "does a marker go into shadow?", gt, opts, a, True, {"min_fps": fps_for(gaps)})
    return [({}, make)]


def _simul(c, shrink):
    pts = []
    for j in range(c.nm):
        for a, b in c.un[j]:
            a2, b2 = a + shrink, b - shrink
            if b2 > a2:
                pts += [(a2, 1), (b2, -1)]
    pts.sort(key=lambda p: (p[0], p[1]))
    best, cur, runs = 0, 0, []
    for k, (t, d) in enumerate(pts):
        cur += d
        nxt = pts[k + 1][0] if k + 1 < len(pts) else t
        if cur > best:
            best, runs = cur, []
        if cur == best and cur > 0:
            runs.append(nxt - t)
    return best, max(runs, default=0.0)


def t_max_simul(c: Ctx):
    best, run = _simul(c, 0.0)
    if best < 1 or _simul(c, 0.15)[0] != best or _simul(c, -0.15)[0] != best or run < M:
        return []

    def make(rng):
        opts, a = opts4(rng, best, window(rng, best, 1, 1, c.nm, int))
        return opts and L.question(4, "max_simul", "What is the largest number of markers that are in shadow at the "
                                   "same moment?", best, opts, a, True, {"min_fps": fps_for([run])})
    return [({}, make)]


# L5 ------------------------------------------------------------------------
def _alt_ok(alt):
    return not sim.event_errors(alt)


def t_cf_remove(c: Ctx):
    out = []
    for i in range(c.nc):
        alt = sim.without(c.sc, i)
        if not _alt_ok(alt):
            continue
        aun = sim.union(alt)
        lens = [b - a for j in range(c.nm) for a, b in c.per[(i, j)]] or [INF]
        what = f"If {c.lbl[i]} were taken away (everything else the same)"
        cnt = sum(1 for j in range(c.nm) if aun[j])

        def mk_count(rng, i=i, cnt=cnt, what=what, lens=lens):
            opts, a = opts4(rng, cnt, window(rng, cnt, 1, 0, c.nm, int))
            return opts and L.question(5, "cf_remove_count", f"{what}, how many markers would be in shadow at some "
                                       "point?", cnt, opts, a, True, {"x": i, "min_fps": fps_for(lens)})
        out.append(({"k": "count", "x": i}, mk_count))
        afirst = sorted((aun[j][0][0], j) for j in range(c.nm) if aun[j])
        if len(afirst) >= 2 and afirst[1][0] - afirst[0][0] >= M:
            j = afirst[0][1]

            def mk_first(rng, i=i, j=j, what=what, afirst=afirst):
                opts, a = opts4(rng, c.name[j], [c.name[k] for _, k in afirst[1:]] + [c.name[k] for k in range(c.nm)])
                return opts and L.question(5, "cf_remove_first", f"{what}, which marker would be the first to be in "
                                           "shadow?", c.name[j], opts, a, True,
                                           {"x": i, "min_fps": fps_for([afirst[1][0] - afirst[0][0]] + lens)})
            out.append(({"k": "first", "x": i}, mk_first))
        lost = [j for j in range(c.nm) if c.un[j] and not aun[j]]
        kept = [j for j in range(c.nm) if aun[j]]
        if lost and len(kept) >= 3:
            for j in lost:
                def mk_never(rng, i=i, j=j, what=what, kept=kept):
                    others = rng.sample(kept, 3)
                    opts, a = opts4(rng, c.name[j], [c.name[k] for k in others])
                    return opts and L.question(5, "cf_remove_never", f"{what}, which of these markers would never be "
                                               "in shadow?", c.name[j], opts, a, True,
                                               {"x": i, "ms": sorted(others + [j]), "min_fps": fps_for(lens)})
                out.append(({"k": "never", "x": i, "j": j}, mk_never))
        for j in range(c.nm):
            if not aun[j] or aun[j] == c.un[j] or len(aun[j]) > 2:
                continue
            ok, v = dur_ok(sum(b - a for a, b in aun[j]))
            if not ok:
                continue

            def mk_dur(rng, i=i, j=j, v=v, what=what, k=len(aun[j])):
                opts, a = opts4(rng, tsec(v), window(rng, v, 1.0, 0.5, c.T, tsec))
                return opts and L.question(5, "cf_remove_dur", f"{what}, for how long in total would {c.name[j]} be "
                                           "in shadow (to the nearest 0.5 s)?", tsec(v), opts, a, True,
                                           {"x": i, "j": j, "min_fps": max(math.ceil(k / 0.4 - 1e-9), fps_for(lens))})
            out.append(({"k": "dur", "x": i, "j": j}, mk_dur))
    return out


def _cf_first_count(c, alt, key, what, extra):
    out = []
    if not _alt_ok(alt):
        return out
    aun = sim.union(alt)
    afirst = sorted((aun[j][0][0], j) for j in range(c.nm) if aun[j])
    lens = [b - a for j in range(c.nm) for a, b in aun[j]]
    if len(afirst) >= 2 and afirst[1][0] - afirst[0][0] >= M:
        j = afirst[0][1]

        def mk_first(rng, j=j):
            opts, a = opts4(rng, c.name[j], [c.name[k] for _, k in afirst[1:]] + [c.name[k] for k in range(c.nm)])
            return opts and L.question(5, key + "_first", f"{what}, which marker would be the first to be in shadow?",
                                       c.name[j], opts, a, True,
                                       dict(extra, min_fps=fps_for([afirst[1][0] - afirst[0][0]])))
        out.append(({"k": key + "_first"}, mk_first))
    cnt = len(afirst)

    def mk_count(rng):
        opts, a = opts4(rng, cnt, window(rng, cnt, 1, 0, c.nm, int))
        return opts and L.question(5, key + "_count", f"{what}, how many markers would be in shadow at some point?",
                                   cnt, opts, a, True, dict(extra, min_fps=fps_for(lens)))
    if cnt != len(c.ever) or key == "cf_still":
        out.append(({"k": key + "_count"}, mk_count))
    return out


def t_cf_reverse(c: Ctx):
    what = f"If the {c.light} moved along the same path in the opposite direction (same speed)"
    return _cf_first_count(c, sim.reversed_light(c.sc), "cf_reverse", what, {})


def t_cf_still(c: Ctx):
    if len(c.movers) != 1:
        return []
    i = c.movers[0]
    what = f"If {c.lbl[i]} stayed where it starts instead of moving"
    return _cf_first_count(c, sim.stilled(c.sc), "cf_still", what, {"i": i})


LEVELS = {
    1: [t_n_casters, t_light_dir, t_first_marker, t_n_ever, t_never, t_shaded_at_t, t_n_shaded_at_t,
        t_moving_object, t_move_dir],
    2: [t_caster_of, t_who_covers_at_t, t_shadow_way, t_earliest_of, t_last_marker, t_longest_of],
    3: [t_cover_time, t_uncover_time, t_shade_dur, t_gap_time, t_n_entries],
    4: [t_order, t_most_least, t_n_casters_for, t_caster_markers, t_total_entries, t_max_simul],
    5: [t_cf_remove, t_cf_reverse, t_cf_still],
}


def candidates(scene) -> dict:
    c = Ctx(scene)
    out = {}
    for lv, temps in LEVELS.items():
        groups = []
        for tf in temps:
            items = tf(c)
            if items:
                groups.append((tf.__name__, items))
        out[lv] = groups
    return out


def build(scene: dict, rng: random.Random) -> list[dict]:
    cands = candidates(scene)
    avail = {lv: sum(len(it) for _, it in g) for lv, g in cands.items()}
    combos = [cc for cc in L._count_combos() if sum(cc) == L.TEN_TOTAL and all(cc[k] <= avail[k + 1] for k in range(5))]
    if not combos:
        raise ValueError(f"not enough question candidates: {avail}")
    counts = rng.choice(combos)
    qs = []
    for lv in range(1, 6):
        groups = [(name, list(items)) for name, items in cands[lv]]
        used_keys, got = set(), 0
        while got < counts[lv - 1]:
            live = [g for g in groups if g[1]]
            if not live:
                raise ValueError(f"level {lv} ran out of candidates")
            fresh = [g for g in live if g[0] not in used_keys] or live
            name, items = rng.choice(fresh)
            params, make = items.pop(rng.randrange(len(items)))
            q = make(rng)
            if not q:
                continue
            sig = (q["derivation_key"], repr(sorted(q["params"].items())))
            if any((p["derivation_key"], repr(sorted(p["params"].items()))) == sig for p in qs):
                continue
            used_keys.add(name)
            qs.append(q)
            got += 1
    return L.number(qs)
