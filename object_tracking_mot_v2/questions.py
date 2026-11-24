"""10-question ladder for object_tracking_mot_v2 (common/ladder5.py, mode "ten").

Objects are named by colour and shape ("the teal hexagon"; colours are unique
per clip); in the targets scenario the identical objects are referred to as
"the ringed objects" (ringed before the motion starts). Positions are the
arena's quadrants (split by the faint centre lines). Times are what the
clock shows. Answers come from sim.py's exact event solver; answers.py
re-derives them from the drawn frames and re-simulates the what-ifs.

Templates per level (derivation_key):
  L1 n_objects, n_targets, shape_of, pattern_of, largest
  L2 quadrant_at_t, leftmost_at_t, first_partner, first_wall, closest_at_end, targets_in_quad_at_t
  L3 untargeted_in_quad_at_t, n_walls, n_hits, time_first_hit, time_first_wall, time_hidden, quad_changes, total_hits
  L4 order_end, hidden_at_end, quadrant_end, targets_in_quad_end
  L5 cf_remove_quad, cf_speed_quad, cf_remove_hits, cf_delay_targets
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
from answers import name, fmt_clock, clock_tenths, frame_of_clock, QUADS, order_text  # noqa: E402

LADDER = "ten"
TEN_SPLITS = [c for c in itertools.product(*(range(lo, hi + 1) for lo, hi in L.TEN_RANGES)) if sum(c) == L.TEN_TOTAL]
FPS = sim.FPS
EV = 0.3
QM = 20                       # px: queried positions stay this far from the centre lines
WALLS = ("left", "right", "top", "bottom")


def fps_for(gaps, floor=1):
    g = min(gaps) if gaps else 99.0
    return int(min(24, max(floor, math.ceil(1.0 / g - 1e-9))))


def align(t):
    return 1 if abs(t - round(t)) < 1e-9 else 2


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


def clock_opts(rng, tenths, extra=()):
    cands = [tenths + d for d in (-15, -10, -5, 5, 10, 15, 20, -20)] + [round(e) for e in extra]
    rng.shuffle(cands)
    keep = []
    for c in cands:
        if c >= 0 and all(abs(c - u) >= 5 for u in keep + [tenths]):
            keep.append(c)
    return mc4(rng, tenths, keep, fmt_clock)


def event_ok(te):
    x = te * FPS
    fa = math.ceil(x - 1e-9)
    return 0.25 <= x - math.floor(x) <= 0.75 and clock_tenths(fa - 1) == clock_tenths(fa)


def Q(key, params, text, gt, opts, motion, fps):
    if opts is None:
        return None
    return (key, dict(params, min_fps=fps), text, gt, opts[0], opts[1], motion)


class Ctx:
    def __init__(self, scene):
        self.s = scene
        self.n = len(scene["objects"])
        self.dur = sim.duration(scene)
        self.t_last = (scene["frames"] - 1) / FPS
        self.scen = scene["scenario"]
        self.unique = self.scen != "targets"
        self.ev = [] if self.scen == "orbit" else [e for e in sim.motion(scene)["events"] if e[0] <= self.dur]
        self.grid = [k / 2 for k in range(2, int(2 * (self.dur - 1.0)) + 1) if k / 2 >= scene["start_t"] + 0.5]

    def pos(self, i, t):
        return sim.position(self.s, i, t)

    def nm(self, i):
        return name(self.s, i)

    def q(self, x, y):
        return sim.quadrant(self.s, x, y)

    def margin(self, x, y):
        return sim.quad_margin(self.s, x, y)

    def obj_events(self, i, kind=None):
        return [e for e in self.ev if i in e[2] and (kind is None or e[1] == kind)]

    def window_frames(self, t):
        f0 = frame_of_clock(t)
        return [f for f in range(f0, f0 + 4) if clock_tenths(f) == clock_tenths(f0)]

    def visible(self, i, t):
        return not sim.hidden(self.s, i, t)

    def quad_crossings(self, i):
        """Times the centre crosses a centre line (sampled finely)."""
        out, prev = [], None
        k = 0
        while k / 240 <= self.dur:
            qq = self.q(*self.pos(i, k / 240))
            if prev is not None and qq != prev:
                out.append(k / 240)
            prev = qq
            k += 1
        return out


# ------------------------------------------------------------------ L1
def level1(c: Ctx, rng):
    out = [Q("n_objects", {}, "How many moving objects are in the arena?", c.n, int_opts(rng, c.n, 1, 12), False, 1)]
    if c.scen == "targets":
        k = len(c.s["targets"])
        out.append(Q("n_targets", {}, "At the start some objects wear a ring. How many?", k, int_opts(rng, k, 0, 6),
                     False, 1))
        return out
    for i in rng.sample(range(c.n), min(2, c.n)):
        o = c.s["objects"][i]
        out.append(Q("shape_of", {"o": i}, f"What shape is the {o['colour']} object?", o["shape"],
                     mc4(rng, o["shape"], rng.sample([x for x in sim.SHAPES if x != o["shape"]], 4)), False, 1))
        out.append(Q("pattern_of", {"o": i}, f"Which pattern is on the {o['colour']} object?", o["pattern"],
                     mc4(rng, o["pattern"], [x for x in sim.PATTERNS if x != o["pattern"]]), False, 1))
    rs = sorted(c.s["objects"], key=lambda o: -o["r"])
    if c.n >= 4 and rs[0]["r"] >= 1.1 * rs[1]["r"]:
        out.append(Q("largest", {}, "Which object is the largest?", c.nm(rs[0]["id"]),
                     mc4(rng, c.nm(rs[0]["id"]), [c.nm(o["id"]) for o in rng.sample(rs[1:], 3)]), False, 1))
    return out


# ------------------------------------------------------------------ L2
def level2(c: Ctx, rng):
    out = []
    ts = list(c.grid)
    rng.shuffle(ts)
    if c.unique:
        made = 0
        for t in ts:
            if made == 2:
                break
            i = rng.randrange(c.n)
            fr = c.window_frames(t)
            ps = [c.pos(i, f / FPS) for f in fr]
            if min(c.margin(*p) for p in ps) < QM or not all(c.visible(i, f / FPS) for f in fr):
                continue
            gt = c.q(*ps[0])
            out.append(Q("quadrant_at_t", {"o": i, "t": t}, f"At {t:.1f} s on the clock, in which quadrant of the arena "
                         f"is {c.nm(i)}?", gt, mc4(rng, gt, [x for x in QUADS if x != gt]), True, align(t)))
            made += 1
        if c.n >= 4:
            for t in ts[:10]:
                fr = c.window_frames(t)
                ranks = [sorted(range(c.n), key=lambda i: c.pos(i, f / FPS)[0]) for f in fr]
                if any(r[0] != ranks[0][0] for r in ranks):
                    continue
                if any(c.pos(r[1], f / FPS)[0] - c.pos(r[0], f / FPS)[0] < 15 for r, f in zip(ranks, fr)):
                    continue
                if not all(c.visible(k, f / FPS) for k in range(c.n) for f in fr):
                    continue
                g = ranks[0][0]
                out.append(Q("leftmost_at_t", {"t": t}, f"At {t:.1f} s on the clock, which object is farthest to the "
                             "left?", c.nm(g), mc4(rng, c.nm(g), [c.nm(x) for x in ranks[0][1:4]]), True, align(t)))
                break
        for i in rng.sample(range(c.n), c.n):
            hits = c.obj_events(i, "hit")
            if hits and c.n >= 4 and sim.EDGE_S <= hits[0][0] <= c.dur - sim.EDGE_S:
                partner = [x for x in hits[0][2] if x != i][0]
                out.append(Q("first_partner", {"o": i}, f"Which object does {c.nm(i)} bump into first?", c.nm(partner),
                             mc4(rng, c.nm(partner), [c.nm(x) for x in rng.sample([k for k in range(c.n) if k not in (i, partner)], c.n - 2)]),
                             True, fps_for([EV])))
                break
        for i in rng.sample(range(c.n), c.n):
            w = c.obj_events(i, "wall")
            if w and sim.EDGE_S <= w[0][0] <= c.dur - sim.EDGE_S:
                gt = w[0][3]
                out.append(Q("first_wall", {"o": i}, f"Which wall of the arena does {c.nm(i)} bounce off first?", gt,
                             mc4(rng, gt, [x for x in WALLS if x != gt]), True, fps_for([EV])))
                break
        if c.n >= 4:
            i = rng.randrange(c.n)
            me = c.pos(i, c.t_last)
            ds = sorted((math.dist(c.pos(k, c.t_last), me), k) for k in range(c.n) if k != i)
            if ds[1][0] - ds[0][0] >= 15 and all(c.visible(k, c.t_last) for k in range(c.n)):
                g = ds[0][1]
                out.append(Q("closest_at_end", {"o": i}, f"At the end of the clip, which object is closest to "
                             f"{c.nm(i)}?", c.nm(g), mc4(rng, c.nm(g), [c.nm(k) for _, k in ds[1:4]]), True, 1))
    else:
        for t in ts[:12]:
            fr = c.window_frames(t)
            if any(c.margin(*c.pos(i, f / FPS)) < QM for i in c.s["targets"] for f in fr):
                continue
            q = rng.choice(QUADS)
            k = sum(1 for i in c.s["targets"] if c.q(*c.pos(i, t)) == q)
            out.append(Q("targets_in_quad_at_t", {"q": q, "t": t}, f"At {t:.1f} s on the clock, how many of the objects "
                         f"that were ringed at the start are in the {q} quadrant?", k,
                         int_opts(rng, k, 0, len(c.s["targets"]) + 2), True, align(t)))
            if len(out) == 3:
                break
    return out


# ------------------------------------------------------------------ L3
def level3(c: Ctx, rng):
    out = []
    if not c.unique:
        return _target_l3(c, rng)
    wins = [(a, b) for a in c.grid for b in c.grid if b - a >= 2.0]
    rng.shuffle(wins)
    for key, kind, word in (("n_walls", "wall", "bounce off a wall"), ("n_hits", "hit", "bump into another object")):
        for (t0, t1), i in zip(wins[:30], itertools.cycle(rng.sample(range(c.n), c.n))):
            evs = c.obj_events(i, kind)
            allt = [e[0] for e in c.obj_events(i)]
            if any(abs(e - t0) < 0.25 or abs(e - t1) < 0.25 for e in allt):
                continue
            inside = [e for e in evs if t0 <= e[0] <= t1]
            if kind == "hit" and not inside:
                continue
            k = len(inside)
            out.append(Q(key, {"o": i, "t0": t0, "t1": t1}, f"Between {t0:.1f} s and {t1:.1f} s on the clock, how many "
                         f"times does {c.nm(i)} {word}?", k, int_opts(rng, k, 0, 30), True,
                         max(align(t0), align(t1), fps_for([EV] + [min(abs(e - t0), abs(e - t1)) for e in allt]))))
            break
    for key, kind, word in (("time_first_hit", "hit", "first bump into another object"),
                            ("time_first_wall", "wall", "first bounce off a wall")):
        for i in rng.sample(range(c.n), c.n):
            evs = c.obj_events(i, kind)
            if not evs or not sim.EDGE_S <= evs[0][0] <= c.dur - sim.EDGE_S or not event_ok(evs[0][0]):
                continue
            if any(not c.visible(j, evs[0][0]) for j in evs[0][2]):
                continue
            tenths = clock_tenths(math.ceil(evs[0][0] * FPS - 1e-9))
            out.append(Q(key, {"o": i}, f"When does {c.nm(i)} {word}? Give the clock reading.", fmt_clock(tenths),
                         clock_opts(rng, tenths), True, 4))
            break
    if c.scen == "occluders":
        for i in rng.sample(range(c.n), c.n):
            hid = [f for f in range(c.s["frames"]) if not c.visible(i, f / FPS)]
            if not hid:
                continue
            f = hid[0]
            # the moment it touches the screen sits well inside a frame interval with the same clock either side
            te = _hide_time(c, i, f)
            if te is None or not event_ok(te) or not sim.EDGE_S <= te <= c.dur - sim.EDGE_S:
                continue
            tenths = clock_tenths(f)
            out.append(Q("time_hidden", {"o": i}, f"When does {c.nm(i)} first start to go behind a screen? Give the "
                         "clock reading.", fmt_clock(tenths), clock_opts(rng, tenths), True, 4))
            break
    for i in rng.sample(range(c.n), c.n):
        cr = c.quad_crossings(i)
        if any(x < sim.EDGE_S or x > c.dur - sim.EDGE_S for x in cr):
            continue
        if any(b - a < EV for a, b in zip(cr, cr[1:])):
            continue
        if any(not c.visible(i, x + k / FPS) for x in cr for k in range(-3, 4)):
            continue                                   # every crossing is seen, with 3 frames either side
        k = len(cr)
        out.append(Q("quad_changes", {"o": i}, f"How many times does {c.nm(i)} move from one quadrant into another during "
                     "the clip?", k, int_opts(rng, k, 0, 30), True, fps_for([EV] + [b - a for a, b in zip(cr, cr[1:])])))
        break
    for t0, t1 in wins[:20]:
        hits = [e for e in c.ev if e[1] == "hit"]
        if any(abs(e[0] - t0) < 0.25 or abs(e[0] - t1) < 0.25 for e in hits):
            continue
        k = sum(1 for e in hits if t0 <= e[0] <= t1)
        if k == 0 or any(not c.visible(j, e[0]) for e in hits for j in e[2] if t0 <= e[0] <= t1):
            continue
        out.append(Q("total_hits", {"t0": t0, "t1": t1}, f"Between {t0:.1f} s and {t1:.1f} s on the clock, how many "
                     "collisions between two objects happen in total?", k, int_opts(rng, k, 0, 40), True,
                     max(align(t0), align(t1), fps_for([EV]))))
        break
    return out


def _hide_time(c, i, f):
    lo, hi = (f - 1) / FPS, f / FPS
    if not c.visible(i, lo):
        for _ in range(40):
            mid = (lo + hi) / 2
            if c.visible(i, mid):
                lo = mid
            else:
                hi = mid
        return hi
    return None


def _target_l3(c: Ctx, rng):
    out = []
    # counts of ringed objects per quadrant at two more times (measurement of where they are)
    ts = list(c.grid)
    rng.shuffle(ts)
    for t in ts[:12]:
        fr = c.window_frames(t)
        if any(c.margin(*c.pos(i, f / FPS)) < QM for i in range(c.n) for f in fr):
            continue
        q = rng.choice(QUADS)
        k = sum(1 for i in range(c.n) if c.q(*c.pos(i, t)) == q and i not in c.s["targets"])
        out.append(Q("untargeted_in_quad_at_t", {"q": q, "t": t}, f"At {t:.1f} s on the clock, how many objects that were "
                     f"NOT ringed at the start are in the {q} quadrant?", k, int_opts(rng, k, 0, c.n), True, align(t)))
        if len(out) == 3:
            break
    return out


# ------------------------------------------------------------------ L4
def level4(c: Ctx, rng):
    out = []
    tl = c.t_last
    if c.unique:
        for _ in range(3):
            os_ = rng.sample(range(c.n), 3)
            xs = sorted(os_, key=lambda i: c.pos(i, tl)[0])
            if all(c.visible(i, tl) for i in os_) and all(c.pos(b, tl)[0] - c.pos(a, tl)[0] >= 15 for a, b in zip(xs, xs[1:])):
                gt = order_text([c.nm(i) for i in xs])
                perms = [order_text([c.nm(i) for i in p]) for p in itertools.permutations(xs) if list(p) != xs]
                out.append(Q("order_end", {"os": sorted(os_)}, "At the end of the clip, list these from left to right: "
                             + ", ".join(c.nm(i) for i in os_) + ".", gt, mc4(rng, gt, rng.sample(perms, len(perms))),
                             True, 1))
                break
        for i in rng.sample(range(c.n), min(3, c.n)):
            p = c.pos(i, tl)
            if c.margin(*p) >= QM and c.visible(i, tl):
                gt = c.q(*p)
                out.append(Q("quadrant_end", {"o": i}, f"In which quadrant is {c.nm(i)} at the end of the clip?", gt,
                             mc4(rng, gt, [x for x in QUADS if x != gt]), True, 1))
        if c.scen == "occluders":
            hid = [i for i in range(c.n) if not c.visible(i, tl)]
            if len(hid) == 1 and c.n >= 4:
                g = hid[0]
                ok = all(not c.visible(g, f / FPS) for f in range(c.s["frames"] - 12, c.s["frames"]))
                if ok:
                    out.append(Q("hidden_at_end", {}, "At the end of the clip one object is behind a screen. Which one?",
                                 c.nm(g), mc4(rng, c.nm(g), [c.nm(k) for k in rng.sample([k for k in range(c.n) if k != g], c.n - 1)]),
                                 True, 2))
    else:
        tg = c.s["targets"]
        if all(c.margin(*c.pos(i, tl)) >= QM for i in tg):
            for q in rng.sample(QUADS, 2):
                k = sum(1 for i in tg if c.q(*c.pos(i, tl)) == q)
                out.append(Q("targets_in_quad_end", {"q": q}, f"At the end of the clip, how many of the objects that were "
                             f"ringed at the start are in the {q} quadrant?", k, int_opts(rng, k, 0, len(tg) + 2), True, 1))
    return out


# ------------------------------------------------------------------ L5
def _scene_with(c, **kw):
    sc = json.loads(json.dumps(c.s))
    if "remove" in kw:
        sc["objects"] = [o for o in sc["objects"] if o["id"] != kw["remove"]]
    if "speed" in kw:
        for o in sc["objects"]:
            if c.scen == "orbit":
                o["wx"] *= kw["speed"]
                o["wy"] *= kw["speed"]
            else:
                o["vx"] *= kw["speed"]
                o["vy"] *= kw["speed"]
    if "delay" in kw:
        sc["start_t"] = sc["start_t"] + kw["delay"]
    return sc


def _cf_pos(c, sc, i, t):
    if c.scen == "orbit":
        o = next(o for o in sc["objects"] if o["id"] == i)
        return sim.orbit_pos(o, t)
    objs = [{"id": o["id"], "x": o["x"], "y": o["y"], "vx": o["vx"], "vy": o["vy"], "r": o["r"]} for o in sc["objects"]]
    mo = sim.simulate(objs, sim.box_of(sc), c.dur + 1.0, sc["start_t"])
    return sim.seg_pos(mo["segs"][i], t), mo


def _cf_clean(mo, dur):
    """No two contacts within 4 ms of each other (the what-if integrator resolves them one at a time), and no
    grazes or glancing hits (tiny changes the two integrators could resolve differently)."""
    ts = sorted(e[0] for e in mo["events"] if e[0] <= dur)
    return all(b - a > 0.004 for a, b in zip(ts, ts[1:])) and not sim.clear_events(mo, dur)


def level5(c: Ctx, rng):
    out = []
    T = c.dur
    if c.scen == "orbit":
        for i in rng.sample(range(c.n), c.n):
            for sp in rng.sample([0.5, 2.0, 1.5], 3):
                sc = _scene_with(c, speed=sp)
                p = _cf_pos(c, sc, i, T)
                if c.margin(*p) >= QM:
                    gt = c.q(*p)
                    word = {0.5: "half as fast", 2.0: "twice as fast", 1.5: "1.5 times as fast"}[sp]
                    out.append(Q("cf_speed_quad", {"o": i, "speed": sp}, f"Suppose every object went round its path "
                                 f"{word}. In which quadrant would {c.nm(i)} be at the end of the clip?", gt,
                                 mc4(rng, gt, [x for x in QUADS if x != gt]), True, 4))
                    break
            if len(out) >= 3:
                break
        return out
    if c.unique:
        pairs = [(e[2][0], e[2][1]) for e in c.ev if e[1] == "hit"]
        cand = [(r, o) for a, b in pairs for r, o in ((a, b), (b, a))]
        rng.shuffle(cand)
        made = 0
        for rem, i in cand:
            if made == 2:
                break
            sc = _scene_with(c, remove=rem)
            p, mo = _cf_pos(c, sc, i, T)
            if c.margin(*p) < QM or not _cf_clean(mo, T):
                continue
            gt = c.q(*p)
            out.append(Q("cf_remove_quad", {"remove": rem, "o": i}, f"Suppose {c.nm(rem)} had not been in the arena at all. "
                         f"In which quadrant would {c.nm(i)} be at the end of the clip?", gt,
                         mc4(rng, gt, [x for x in QUADS if x != gt]), True, 4))
            # how many collisions would i have
            k = sum(1 for e in mo["events"] if e[1] == "hit" and i in e[2] and e[0] <= T)
            near_end = any(abs(e[0] - T) < 0.05 for e in mo["events"] if i in e[2])
            if not near_end:
                out.append(Q("cf_remove_hits", {"remove": rem, "o": i}, f"Suppose {c.nm(rem)} had not been in the arena at "
                             f"all. How many times would {c.nm(i)} bump into another object during the clip?", k,
                             int_opts(rng, k, 0, 30), True, 4))
            made += 1
        for sp in rng.sample([0.5, 1.5], 2):
            sc = _scene_with(c, speed=sp)
            i = rng.randrange(c.n)
            p, mo = _cf_pos(c, sc, i, T)
            if c.margin(*p) >= QM and _cf_clean(mo, T):
                gt = c.q(*p)
                out.append(Q("cf_speed_quad", {"o": i, "speed": sp}, f"Suppose every object had moved "
                             f"{'half as fast' if sp == 0.5 else '1.5 times as fast'} from the start. In which quadrant "
                             f"would {c.nm(i)} be at the end of the clip?", gt, mc4(rng, gt, [x for x in QUADS if x != gt]),
                             True, 4))
                break
    else:
        for d in rng.sample([0.5, 1.0, 1.5], 3):
            sc = _scene_with(c, delay=d)
            objs = [{"id": o["id"], "x": o["x"], "y": o["y"], "vx": o["vx"], "vy": o["vy"], "r": o["r"]}
                    for o in sc["objects"]]
            mo = sim.simulate(objs, sim.box_of(sc), T + 1.0, sc["start_t"])
            ps = {i: sim.seg_pos(mo["segs"][i], T) for i in c.s["targets"]}
            if any(c.margin(*p) < QM for p in ps.values()) or not _cf_clean(mo, T):
                continue
            q = rng.choice(QUADS)
            k = sum(1 for p in ps.values() if c.q(*p) == q)
            out.append(Q("cf_delay_targets", {"d": d, "q": q}, f"Suppose the objects had started moving {d:g} s later "
                         f"(same starting spots and directions). How many of the ringed objects would be in the {q} "
                         "quadrant at the end of the clip?", k, int_opts(rng, k, 0, len(c.s["targets"]) + 2), True, 4))
            if len(out) == 2:
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
    for attempt in range(20):
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
    counts = rng.choice(feasible)
    qs = []
    for lv in range(1, 6):
        for key, params, text, gt, opts, letter, motion in _pick(rng, made[lv], counts[lv - 1]):
            qs.append(L.question(lv, key, text, gt, opts, letter, motion, params, idx=len(qs) + 1))
    return qs
