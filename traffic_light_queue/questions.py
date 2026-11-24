"""10-question ladder for traffic_light_queue (common/ladder5.py, mode "ten").

Vehicles are named by colour and body ("the teal van"; colours are unique per
clip, the body shows as the length). Lights are named by their road ("the
light on the upper road") or their order along the road ("the first light").
Times are what the clock shows. Answers come from sim.py's events; answers.py
re-derives them from the drawn frames and re-simulates the what-ifs with its
own integrator.

Templates per level (derivation_key):
  L1 n_lights, n_on_road_end, light_end, longest_end
  L2 light_at_t, first_enter, first_stop, front_of_queue_at_t
  L3 n_stops, time_first_stop, time_first_green, n_crossed_by_t, n_still_at_t, n_turn_red
  L4 order_cross, n_crossed_end, never_cross, n_still_end
  L5 cf_always_green, cf_remove_stops, cf_late
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
from answers import name, light_name, fmt_clock, clock_tenths, frame_of_clock, order_text  # noqa: E402

LADDER = "ten"
TEN_SPLITS = [c for c in itertools.product(*(range(lo, hi + 1) for lo, hi in L.TEN_RANGES)) if sum(c) == L.TEN_TOTAL]
FPS = sim.FPS
EV = 0.3
COLOURS = ("red", "green", "yellow", "switched off")


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


def clock_opts(rng, tenths):
    cands = [tenths + d for d in (-15, -10, -5, 5, 10, 15, 20, -20)]
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
    def __init__(self, scene, res=None):
        self.s = scene
        self.dur = sim.duration(scene)
        self.t_last = (scene["frames"] - 1) / FPS
        self.res = res or sim.simulate(scene)
        self.ev = [e for e in self.res["events"] if e[0] <= self.dur]
        self.nl = len(scene["lights"])
        self.ch = [sim.light_changes(Lg, 0, self.dur) for Lg in scene["lights"]]
        self.entered = sorted({e[2] for e in self.ev if e[1] == "enter"})
        self.grid = [k / 2 for k in range(2, int(2 * (self.dur - 1.0)) + 1)]

    def nm(self, i):
        return name(self.s, i)

    def ln(self, li):
        return light_name(self.s, li)

    def car_ev(self, i, kind=None):
        return [e for e in self.ev if e[2] == i and (kind is None or e[1] == kind)]

    def red_at(self, li, t):
        return sim.light_red(self.s["lights"][li], t)

    def near_change(self, li, t, m):
        return any(abs(t - c) < m for c, _ in self.ch[li])

    def still_at(self, i, t):
        """Standing still at t (between a stop and the next go), or None when an event is within EV of t."""
        if any(abs(e[0] - t) < EV for e in self.car_ev(i) if e[1] in ("stop", "go", "enter", "leave")):
            return None
        on = [e for e in self.car_ev(i) if e[0] <= t]
        if not on or on[0][1] != "enter" or any(e[1] == "leave" for e in on):
            return False                                # not on the road
        st = [e[1] for e in on if e[1] in ("stop", "go")]
        return bool(st) and st[-1] == "stop"

    def on_road(self, i, t):
        on = [e[1] for e in self.car_ev(i) if e[0] <= t]
        return "enter" in on and "leave" not in on

    def crossed_by(self, li, t):
        return [e[2] for e in self.ev if e[1] == "cross" and e[3] == li and e[0] <= t]


# ------------------------------------------------------------------ L1 (one frame)
def level1(c: Ctx, rng):
    out = []
    if c.nl > 1:
        out.append(Q("n_lights", {}, "How many traffic lights are there?", c.nl, int_opts(rng, c.nl, 1, 5), False, 1))
    on = [i for i in c.entered if c.on_road(i, c.t_last)]
    k = len(on)
    out.append(Q("n_on_road_end", {}, "In the last frame of the clip, how many vehicles are on the road?", k,
                 int_opts(rng, k, 0, 12), False, 1))
    for li in range(c.nl):
        if not c.near_change(li, c.t_last, 0.5):
            gt = "red" if c.red_at(li, c.t_last) else "green"
            out.append(Q("light_end", {"l": li}, f"In the last frame of the clip, what colour is {c.ln(li)} showing?",
                         gt, mc4(rng, gt, [x for x in COLOURS if x != gt]), False, 1))
    fl = c.s["frames"] - 1
    whole = all(0 <= c.res["pos"][i][fl] - c.s["cars"][i]["len"] and
                c.res["pos"][i][fl] <= c.s["lanes"][c.s["cars"][i]["lane"]]["length"] for i in on)
    if len(on) >= 2 and whole:
        ls = sorted(on, key=lambda i: -c.s["cars"][i]["len"])
        if c.s["cars"][ls[0]]["len"] >= 1.1 * c.s["cars"][ls[1]]["len"]:
            others = [x for x in c.entered if x != ls[0] and c.s["cars"][x]["len"] < c.s["cars"][ls[0]]["len"] / 1.1]
            out.append(Q("longest_end", {}, "In the last frame of the clip, which vehicle on the road is the longest?",
                         c.nm(ls[0]), mc4(rng, c.nm(ls[0]), [c.nm(x) for x in ls[1:]] + [c.nm(x) for x in others]),
                         False, 1))
    return out


# ------------------------------------------------------------------ L2
def level2(c: Ctx, rng):
    out = []
    ts = list(c.grid)
    rng.shuffle(ts)
    made = 0
    for t in ts:
        if made == 2:
            break
        li = rng.randrange(c.nl)
        if c.near_change(li, t, 0.3):
            continue
        gt = "red" if c.red_at(li, t) else "green"
        out.append(Q("light_at_t", {"l": li, "t": t}, f"At {t:.1f} s on the clock, what colour is {c.ln(li)} showing?",
                     gt, mc4(rng, gt, [x for x in COLOURS if x != gt]), True, align(t)))
        made += 1
    ent = [e for e in c.ev if e[1] == "enter"]
    if len(ent) >= 4 and ent[1][0] - ent[0][0] >= EV:
        g = ent[0][2]
        out.append(Q("first_enter", {}, "Which vehicle comes onto the road first?", c.nm(g),
                     mc4(rng, c.nm(g), [c.nm(e[2]) for e in ent[1:]]), True, fps_for([ent[1][0] - ent[0][0]])))
    st = [e for e in c.ev if e[1] == "stop"]
    if len(st) >= 2 and st[1][0] - st[0][0] >= EV:
        g = st[0][2]
        others = [x for x in c.entered if x != g]
        rng.shuffle(others)
        out.append(Q("first_stop", {}, "Which vehicle is the first to come to a complete stop?", c.nm(g),
                     mc4(rng, c.nm(g), [c.nm(x) for x in others]), True, fps_for([st[1][0] - st[0][0]])))
    for t in ts[:12]:
        for li in rng.sample(range(c.nl), c.nl):
            at = [i for i in c.entered if c.still_at(i, t)
                  and _waiting_line(c, i, t) == li]
            if len(at) == 1 and len([i for i in c.entered if c.on_road(i, t)]) >= 4:
                g = at[0]
                others = [x for x in c.entered if x != g and c.on_road(x, t)]
                rng.shuffle(others)
                out.append(Q("front_of_queue_at_t", {"l": li, "t": t}, f"At {t:.1f} s on the clock, which vehicle is "
                             f"waiting first in line at the stop line of {c.ln(li)}?", c.nm(g),
                             mc4(rng, c.nm(g), [c.nm(x) for x in others]), True, align(t)))
                return out
    return out


def _waiting_line(c, i, t):
    st = [e for e in c.car_ev(i, "stop") if e[0] <= t]
    return st[-1][3] if st else None


# ------------------------------------------------------------------ L3
def level3(c: Ctx, rng):
    out = []
    for i in rng.sample(c.entered, len(c.entered)):
        ev = [e for e in c.car_ev(i) if e[1] in ("enter", "stop", "go", "leave")]
        if ev[-1][0] > c.dur - sim.EDGE_S:
            continue
        k = len(c.car_ev(i, "stop"))
        out.append(Q("n_stops", {"o": i}, f"How many times does {c.nm(i)} come to a complete stop during the clip?", k,
                     int_opts(rng, k, 0, 8), True, fps_for([EV] + [b[0] - a[0] for a, b in zip(ev, ev[1:])])))
        break
    for i in rng.sample(c.entered, len(c.entered)):
        st = c.car_ev(i, "stop")
        if st and event_ok(st[0][0]) and sim.EDGE_S <= st[0][0] <= c.dur - sim.EDGE_S:
            tenths = clock_tenths(math.ceil(st[0][0] * FPS - 1e-9))
            out.append(Q("time_first_stop", {"o": i}, f"When does {c.nm(i)} first come to a complete stop? Give the "
                         "clock reading.", fmt_clock(tenths), clock_opts(rng, tenths), True, 4))
            break
    for li in rng.sample(range(c.nl), c.nl):
        g = [t for t, k in c.ch[li] if k == "green"]
        if g and event_ok(g[0]) and sim.EDGE_S <= g[0] <= c.dur - sim.EDGE_S:
            tenths = clock_tenths(math.ceil(g[0] * FPS - 1e-9))
            out.append(Q("time_first_green", {"l": li}, f"When does {c.ln(li)} first turn green? Give the clock "
                         "reading.", fmt_clock(tenths), clock_opts(rng, tenths), True, 4))
            break
    ts = list(c.grid)
    rng.shuffle(ts)
    for t in ts[:12]:
        li = rng.randrange(c.nl)
        cr = [e[0] for e in c.ev if e[1] == "cross" and e[3] == li]
        if any(abs(x - t) < EV for x in cr):
            continue
        k = len(c.crossed_by(li, t))
        out.append(Q("n_crossed_by_t", {"l": li, "t": t}, f"By {t:.1f} s on the clock, how many vehicles have driven "
                     f"past the stop line of {c.ln(li)}?", k, int_opts(rng, k, 0, 10), True,
                     max(align(t), fps_for([EV] + [b - a for a, b in zip(cr, cr[1:]) if b <= t + EV]))))
        break
    for t in ts[12:24]:
        flags = [c.still_at(i, t) for i in c.entered]
        if None in flags or sum(1 for i in c.entered if c.on_road(i, t)) < 3:
            continue
        k = sum(flags)
        out.append(Q("n_still_at_t", {"t": t}, f"At {t:.1f} s on the clock, how many vehicles are standing still?", k,
                     int_opts(rng, k, 0, 10), True, max(align(t), fps_for([EV]))))
        break
    for li in rng.sample(range(c.nl), c.nl):
        rd = [t for t, k in c.ch[li] if k == "red"]
        allc = [t for t, _ in c.ch[li]]
        if any(t < sim.EDGE_S or t > c.dur - sim.EDGE_S for t in allc):
            continue
        k = len(rd)
        out.append(Q("n_turn_red", {"l": li}, f"How many times does {c.ln(li)} change from green to red during the "
                     "clip?", k, int_opts(rng, k, 0, 8), True, fps_for([EV] + [b - a for a, b in zip(allc, allc[1:])])))
        break
    return out


# ------------------------------------------------------------------ L4
def level4(c: Ctx, rng):
    out = []
    for li in rng.sample(range(c.nl), c.nl):
        cr = [e for e in c.ev if e[1] == "cross" and e[3] == li]
        if len(cr) < 3:
            continue
        for _ in range(4):
            pick = sorted(rng.sample(range(len(cr)), 3))
            ts = [cr[k][0] for k in pick]
            if min(b - a for a, b in zip(ts, ts[1:])) < EV:
                continue
            os_ = [cr[k][2] for k in pick]
            gt = order_text([c.nm(i) for i in os_])
            perms = [order_text([c.nm(i) for i in p]) for p in itertools.permutations(os_) if list(p) != os_]
            shown = rng.sample(os_, 3)
            out.append(Q("order_cross", {"l": li, "os": sorted(os_)}, f"In what order do these vehicles drive past the "
                         f"stop line of {c.ln(li)}: " + ", ".join(c.nm(i) for i in shown) + "?", gt,
                         mc4(rng, gt, rng.sample(perms, len(perms))), True, fps_for([min(b - a for a, b in zip(ts, ts[1:]))])))
            break
        break
    for li in rng.sample(range(c.nl), c.nl):
        cr = [e[0] for e in c.ev if e[1] == "cross" and e[3] == li]
        k = len(cr)
        out.append(Q("n_crossed_end", {"l": li}, f"During the whole clip, how many vehicles drive past the stop line of "
                     f"{c.ln(li)}?", k, int_opts(rng, k, 0, 10), True, fps_for([EV] + [b - a for a, b in zip(cr, cr[1:])])))
        break
    for li in rng.sample(range(c.nl), c.nl):
        lane_cars = [i for i in c.entered if li in c.s["lanes"][c.s["cars"][i]["lane"]]["lights"]]
        never = [i for i in lane_cars if i not in c.crossed_by(li, c.dur)]
        did = [i for i in lane_cars if i in c.crossed_by(li, c.dur)]
        if len(never) == 1 and len(did) >= 3:
            g = never[0]
            out.append(Q("never_cross", {"l": li}, f"Which of these vehicles never drives past the stop line of "
                         f"{c.ln(li)}?", c.nm(g), mc4(rng, c.nm(g), [c.nm(x) for x in rng.sample(did, len(did))]), True, 1))
            break
    tl = c.t_last
    flags = [c.still_at(i, tl - 0.2) for i in c.entered]
    if None not in flags and all(c.still_at(i, tl - 0.2) == _still_last(c, i) for i in c.entered):
        k = sum(flags)
        out.append(Q("n_still_end", {}, "At the end of the clip, how many vehicles are standing still?", k,
                     int_opts(rng, k, 0, 10), True, 2))
    return out


def _still_last(c, i):
    on = [e for e in c.car_ev(i) if e[0] <= c.t_last]
    if not on or any(e[1] == "leave" for e in on):
        return False
    st = [e[1] for e in on if e[1] in ("stop", "go")]
    return bool(st) and st[-1] == "stop"


# ------------------------------------------------------------------ L5 (what-ifs, re-simulated)
def _cf_clean(scene, res, dur):
    """The what-if is clear-cut: no crossing / line stop near a light change or near the clip end."""
    ch = [[t for t, _ in sim.light_changes(Lg, 0, dur + 1)] for Lg in scene["lights"]]
    for t, kind, i, extra in res["events"]:
        if t > dur + EV:
            continue
        if abs(t - dur) < EV and kind in ("cross", "stop", "go", "enter"):
            return False
        if kind in ("cross", "stop") and isinstance(extra, int) and any(abs(t - x) < 0.15 for x in ch[extra]):
            if not (kind == "cross" and _was_waiting(res, i, extra, t)):
                return False
    return True


def _was_waiting(res, i, li, t):
    st = [e for e in res["events"] if e[2] == i and e[1] == "stop" and e[0] < t]
    gos = [e for e in res["events"] if e[2] == i and e[1] == "go" and e[0] <= t + 1e-9]
    return bool(st) and st[-1][3] == li and bool(gos) and abs(gos[-1][0] - t) < 1e-6


def _scene_with(scene, **kw):
    sc = json.loads(json.dumps(scene))
    if "green" in kw:
        sc["lights"][kw["green"]]["red"] = 0.0
    if "remove" in kw:
        sc["cars"] = [c for c in sc["cars"] if c["id"] != kw["remove"]]
    if "late" in kw:
        for car in sc["cars"]:
            car["t_in"] = round(car["t_in"] + kw["late"], 3)
    return sc


def level5(c: Ctx, rng):
    out = []
    T = c.dur
    for li in rng.sample(range(c.nl), c.nl):
        if c.s["scenario"] == "crossing":
            break                                       # the roads share the junction: an always-green light is unsafe
        sc = _scene_with(c.s, green=li)
        res = sim.simulate(sc, t_end=T + 1.0)
        if not _cf_clean(sc, res, T):
            continue
        cnt = lambda r: sum(1 for e in r["events"] if e[1] == "cross" and e[3] == li and e[0] <= T)  # noqa: E731
        k = cnt(res)
        if not _stable(sc, cnt, k):
            continue
        out.append(Q("cf_always_green", {"l": li}, f"Suppose {c.ln(li)} had stayed green the whole time. How many "
                     f"vehicles would have driven past its stop line by the end of the clip?", k,
                     int_opts(rng, k, 0, 10), True, 4))
        break
    cand = [(r, i) for r in c.entered for i in c.entered if r != i
            and c.s["cars"][r]["lane"] == c.s["cars"][i]["lane"] and c.s["cars"][r]["t_in"] < c.s["cars"][i]["t_in"]]
    rng.shuffle(cand)
    for rem, i in cand[:6]:
        sc = _scene_with(c.s, remove=rem)
        res = sim.simulate(sc, t_end=T + 1.0)
        if not _cf_clean(sc, res, T) or not sim_ok_spacing(res, T):
            continue
        cnt = lambda r: sum(1 for e in r["events"] if e[1] == "stop" and e[2] == i and e[0] <= T)  # noqa: E731
        k = cnt(res)
        if not _stable(sc, cnt, k):
            continue
        out.append(Q("cf_remove_stops", {"remove": rem, "o": i}, f"Suppose {c.nm(rem)} had never come along. How many "
                     f"times would {c.nm(i)} have come to a complete stop during the clip?", k,
                     int_opts(rng, k, 0, 8), True, 4))
        break
    for d in rng.sample([0.5, 1.0, 1.5], 3):
        sc = _scene_with(c.s, late=d)
        res = sim.simulate(sc, t_end=T + 1.0)
        if not _cf_clean(sc, res, T):
            continue
        li = rng.randrange(c.nl)
        cnt = lambda r: sum(1 for e in r["events"] if e[1] == "cross" and e[3] == li and e[0] <= T)  # noqa: E731
        k = cnt(res)
        if not _stable(sc, cnt, k):
            continue
        out.append(Q("cf_late", {"d": d, "l": li}, f"Suppose every vehicle had arrived {d:g} s later than it did (the "
                     f"lights unchanged). How many vehicles would have driven past the stop line of {c.ln(li)} by the "
                     "end of the clip?", k, int_opts(rng, k, 0, 10), True, 4))
        break
    return out


def _stable(sc, cnt, k):
    """The what-if answer is not a near-tie: it is the same with every vehicle 3% slower or faster."""
    for f in (0.97, 1.03):
        alt = json.loads(json.dumps(sc))
        for car in alt["cars"]:
            car["v"] = car["v"] * f
        if cnt(sim.simulate(alt, t_end=sim.duration(sc) + 1.0)) != k:
            return False
    return True


def sim_ok_spacing(res, T):
    per = {}
    for t, kind, i, extra in res["events"]:
        if t <= T + EV and kind in ("stop", "go"):
            per.setdefault(i, []).append(t)
    return all(b - a >= EV for ts in per.values() for a, b in zip(ts, ts[1:]))


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
