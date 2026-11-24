"""10-question, 5-rung ladder for gear_train (common/ladder5.py, LADDER = "ten").

`candidates(scene)` lists every question this scene supports (deterministic,
no rng); `build` draws the per-level counts, then picks templates round-robin
in a random order so the mix varies clip to clip. Every candidate already
satisfies the margin rules (speed comparisons >= 15 % apart, sizes >= 10 %,
turn counts >= 0.2 turn from a whole number, clock readings >= 0.35 s from a
rounding boundary, still-counts >= 0.3 s from any start/stop) and stores
params["min_fps"], the sampling rate needed to see what it relies on.
Gears are named by colour only ("the red gear"); the timer is the only text.
"""
from __future__ import annotations
import itertools
import math
import random
import sys
from fractions import Fraction
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from common import ladder5 as L  # noqa: E402

import sim  # noqa: E402

LADDER = "ten"
DRIVER = "the driver gear (the one with the black hub)"
CW, CCW = "clockwise", "counter-clockwise"
SPEED_WORDS = ("faster than at the start", "slower than at the start", "at the same speed as at the start")
DIR_WORDS = ("in the same direction", "in the opposite direction")
FACTOR_TEXT = {2.0: "twice as fast", 3.0: "three times as fast", 0.5: "half as fast", 1 / 3: "one third as fast"}
CHANGE_OPTS = ("every turning gear speeds up", "every turning gear slows down",
               "only the driver speeds up; the other gears keep their speed",
               "only the driver slows down; the other gears keep their speed")
DOUBLE_OPTS = ("twice as fast as in the video", "half as fast as in the video",
               "at the same speed as in the video", "it would not turn")
SAME_AMOUNT = "they all turn the same amount"
NONE_OF = "none of them"
ROT_LIMIT = 2 * math.pi / 3        # rad per sampled frame: direction and turn counts stay unambiguous


def ceil1(x: float) -> float:
    return max(0.5, math.ceil(x * 10 - 1e-9) / 10)


def fps_gap(gap: float) -> float:
    return ceil1(1.0 / max(gap, 1e-3))


def dir_word(sign) -> str:
    return CW if sign < 0 else CCW


def set_text(cols) -> str:
    return ", ".join(sorted(cols)) if cols else "none"


class Ctx:
    def __init__(self, scene):
        self.s = scene
        self.gm = sim.gmap(scene)
        self.ids = [g["id"] for g in scene["gears"]]
        self.tr = sim.train_ids(scene)
        self.loose = sim.loose_id(scene)
        self.D = scene["duration"]
        self.END = sim.end_time(scene)          # time of the last frame
        self.log = scene["log"]
        self.win = self.log["windows"]
        self.sc = scene["scenario"]
        self.K = scene["K"]
        self.drv = scene["driver"]
        self.ch = sim.children(scene)

    def col(self, g):
        return self.gm[g]["colour"]

    def the(self, g):
        return f"the {self.col(g)} gear"

    def w0(self, g):
        """|omega| at the start (rad/s)."""
        return 0.0 if self.gm[g]["loose"] else self.K / self.gm[g]["z"]

    def dir_at(self, g, t):
        w = sim.omega(self.s, g, t)
        return 0 if abs(w) < 1e-9 else (1 if w > 0 else -1)

    def wmax(self, gids, t0, t1):
        n = max(2, int(math.ceil((t1 - t0) * 8)) + 1)
        return max(abs(sim.omega(self.s, g, t0 + (t1 - t0) * k / (n - 1))) for g in gids for k in range(n))

    def fps_rot(self, gids, t0, t1):
        return ceil1(max(self.wmax(gids, t0, t1) / ROT_LIMIT, 2.0 / max(t1 - t0, 1e-3)))

    def start(self):
        return self.win["start"]

    def end(self):
        return self.win["end"]

    def depth(self, g):
        d = 0
        while self.gm[g]["parent"]:
            g = self.gm[g]["parent"]
            d += 1
        return d

    def nb(self, g):
        out = list(self.ch[g])
        if self.gm[g]["parent"]:
            out.append(self.gm[g]["parent"])
        return out


def combo_text(c: Ctx, a, b, da, db) -> str:
    if da == db:
        return f"both {dir_word(da)}"
    return f"{c.col(a)} {dir_word(da)}, {c.col(b)} {dir_word(db)}"


def combo_opts(c: Ctx, a, b):
    return [combo_text(c, a, b, x, y) for x in (-1, 1) for y in (-1, 1)]


def _cand(level, key, params, text, gt, dist, motion, fmt=str):
    return {"level": level, "key": key, "params": params, "text": text, "gt": gt,
            "dist": dist, "motion": motion, "fmt": fmt}


def _pick_cols(c: Ctx, pool):
    cols = [c.col(g) for g in pool]
    return lambda rng: rng.sample(cols, len(cols))


def _set_dist(c: Ctx, members, universe):
    mem = [c.col(g) for g in members]
    uni = [c.col(g) for g in universe]

    def f(rng):
        outs = [u for u in uni if u not in mem]
        near = [set_text(mem + [o]) for o in outs] + [set_text([x for x in mem if x != y]) for y in mem]
        rng.shuffle(near)
        near += [set_text(uni), "none"]
        rest = []
        for r in (1, 2, 3):
            for cmb in itertools.combinations(sorted(uni), r):
                rest.append(set_text(list(cmb)))
        rng.shuffle(rest)
        return near + rest
    return f


def _int_dist(gt, lo=0, hi=99):
    return lambda rng: [v for v in rng.sample([gt - 2, gt - 1, gt + 1, gt + 2, gt + 3], 5) if lo <= v <= hi] + \
        sorted((v for v in range(lo, hi + 1) if v != gt), key=lambda v: (abs(v - gt), v))


def _turn_margin(x):
    f = x - math.floor(x)
    return min(f, 1 - f)


# ----------------------------------------------------------------- level 1
def _level1(c: Ctx):
    out = []
    n = len(c.ids)
    out.append(_cand(1, "gear_count", {"min_fps": 0.5},
                     "How many gears are in the video? Count every gear, whether it turns or not.",
                     n, _int_dist(n, 2, 12), False))
    if c.s["m"] >= 9.0:
        for g in c.ids:
            z = c.gm[g]["z"]
            if z <= 18:
                out.append(_cand(1, "teeth", {"g": g, "min_fps": 0.5}, f"How many teeth does {c.the(g)} have?",
                                 z, _int_dist(z, 6, 40), False))
    s0, s1 = c.start()
    for a, b in itertools.combinations(c.tr, 2):
        gt = combo_text(c, a, b, c.gm[a]["dir"], c.gm[b]["dir"])
        opts = combo_opts(c, a, b)
        out.append(_cand(1, "start_dirs", {"a": a, "b": b, "min_fps": c.fps_rot([a, b], s0, s1)},
                         f"At the start of the video, which way are {c.the(a)} and {c.the(b)} turning?",
                         gt, lambda rng, o=opts, g=gt: [x for x in o if x != g], True))
    if c.loose:
        others = [g for g in c.ids if g != c.loose]
        out.append(_cand(1, "loose_which", {"min_fps": 0.5}, "One gear never turns at any point in the video. Which one?",
                         c.col(c.loose), _pick_cols(c, others), True))
    zs = sorted(((c.gm[g]["z"], g) for g in c.ids), reverse=True)
    if zs[0][0] >= 1.1 * zs[1][0]:
        out.append(_cand(1, "largest", {"min_fps": 0.5}, "Which gear is the largest?", c.col(zs[0][1]),
                         _pick_cols(c, [g for _, g in zs[1:]]), False))
    if zs[-2][0] >= 1.1 * zs[-1][0]:
        out.append(_cand(1, "smallest", {"min_fps": 0.5}, "Which gear is the smallest?", c.col(zs[-1][1]),
                         _pick_cols(c, [g for _, g in zs[:-1]]), False))
    p = c.s["program"]
    if c.sc == "disengage":
        d = p["disengage"]
        others = [g for g in c.ids if g not in d["moving"]]
        if len(others) >= 3:
            out.append(_cand(1, "event_gear", {"min_fps": fps_gap(d["still1"] - d["still0"])},
                             "Partway through the video, one gear slides away from the gear it meshes with, "
                             "so a gap opens between their teeth. Which of these gears slides away?",
                             c.col(d["gear"]), _pick_cols(c, others), True))
    if c.sc == "brake":
        b = p["brake"]
        out.append(_cand(1, "event_gear", {"min_fps": fps_gap(b["release"] - b["t_touch"])},
                         "Partway through the video, a grey brake block presses against one gear's teeth. "
                         "Which gear?", c.col(b["gear"]), _pick_cols(c, [g for g in c.ids if g != b["gear"]]), True))
    if c.sc == "reversal":
        mids = [r["mid"] for r in p["reversals"]]
        gaps = [mids[0]] + [b - a for a, b in zip(mids, mids[1:])] + [c.D - mids[-1]]
        out.append(_cand(1, "reversal_count",
                         {"min_fps": max(c.fps_rot([c.drv], 0, c.D), fps_gap(min(gaps) / 2))},
                         f"How many times does {DRIVER} change its turning direction during the video?",
                         len(mids), lambda rng: [0, 1, 2, 3], True))
    if c.sc == "speed_change":
        f = p["speed"]["factor"]
        gt = CHANGE_OPTS[0] if f > 1 else CHANGE_OPTS[1]
        e0, e1 = c.end()
        out.append(_cand(1, "change_kind",
                         {"min_fps": max(c.fps_rot(c.tr, 0, c.start()[1]), c.fps_rot(c.tr, e0, e1))},
                         "Partway through the video the gears' speed changes. What happens?", gt,
                         lambda rng, g=gt: [x for x in CHANGE_OPTS if x != g], True))
    return out


# ----------------------------------------------------------------- level 2
def _level2(c: Ctx):
    out = []
    s0, s1 = c.start()
    for g in c.tr:
        same = [h for h in c.tr if h != g and c.gm[h]["dir"] == c.gm[g]["dir"]]
        opp = [h for h in c.tr if c.gm[h]["dir"] != c.gm[g]["dir"]] + ([c.loose] if c.loose else [])
        if len(opp) < 3:
            continue
        for h in same:
            out.append(_cand(2, "same_dir_as", {"g": g, "s": h, "min_fps": c.fps_rot([g, h] + c.tr, s0, s1)},
                             f"At the start of the video, which of these gears turns in the same direction as "
                             f"{c.the(g)}?", c.col(h), _pick_cols(c, opp), True))
    for tri in itertools.combinations(c.tr, 3):
        ws = sorted(((c.w0(g), g) for g in tri), reverse=True)
        names = ", ".join(c.col(g) for g in tri)
        cols = [c.col(g) for g in tri]
        fr = c.fps_rot(list(tri), s0, s1)
        if ws[0][0] >= 1.15 * ws[1][0]:
            gt = c.col(ws[0][1])
            out.append(_cand(2, "fastest_of", {"gs": list(tri), "min_fps": fr},
                             f"At the start of the video, which of these gears turns fastest: {names}?", gt,
                             lambda rng, cs=cols, g=gt: [x for x in cs if x != g] + ["all three turn equally fast"],
                             True))
        if ws[1][0] >= 1.15 * ws[2][0]:
            gt = c.col(ws[2][1])
            out.append(_cand(2, "slowest_of", {"gs": list(tri), "min_fps": fr},
                             f"At the start of the video, which of these gears turns slowest: {names}?", gt,
                             lambda rng, cs=cols, g=gt: [x for x in cs if x != g] + ["all three turn equally fast"],
                             True))
    for g in c.ids:
        if c.gm[g]["loose"]:
            continue
        non = [h for h in c.ids if h != g and h not in c.nb(g)]
        if len(non) < 3:
            continue
        for h in c.nb(g):
            out.append(_cand(2, "meshes_with", {"g": g, "nb": h, "min_fps": 0.5},
                             f"Which of these gears meshes directly with {c.the(g)} (their teeth interlock)?",
                             c.col(h), _pick_cols(c, non), False))
    p = c.s["program"]
    if c.sc in ("disengage", "brake"):
        s_0, s_1 = c.log["still"]
        fr = max(fps_gap((s_1 - s_0) / 2), c.fps_rot(c.tr, s_0, s_1))
        stopped = c.log["stopped"]
        turners = [g for g in c.tr if g not in stopped]
        idle = list(stopped) + ([c.loose] if c.loose else [])
        if c.sc == "disengage":
            x = p["disengage"]["gear"]
            lead = f"While {c.the(x)} is out of mesh and standing still"
        else:
            x = p["brake"]["gear"]
            lead = f"While the grey brake block holds {c.the(x)} still"
        q = f"{lead}, which of these gears keeps turning?"
        if turners:
            if len(idle) + 1 >= 3:
                for k in turners:
                    out.append(_cand(2, "keeps_turning", {"k": k, "min_fps": fr}, q, c.col(k),
                                     lambda rng, i=idle: rng.sample([c.col(h) for h in i], len(i))[:2] + [NONE_OF]
                                     + [c.col(h) for h in i], True))
        else:
            out.append(_cand(2, "keeps_turning", {"k": None, "min_fps": fr}, q, NONE_OF,
                             _pick_cols(c, stopped), True))
    if c.sc == "disengage":
        d = p["disengage"]
        x, par = d["gear"], d["parent"]
        others = [g for g in c.ids if g not in (x, par)]
        if len(others) >= 3:
            out.append(_cand(2, "slides_from", {"min_fps": fps_gap(d["still1"] - d["still0"])},
                             f"Partway through the video {c.the(x)} slides away so that its teeth no longer touch "
                             f"one of its neighbours. Which gear does it slide away from?", c.col(par),
                             _pick_cols(c, others), True))
    if c.sc in ("reversal", "speed_change"):
        e0, e1 = c.end()
        for g in c.tr:
            if c.sc == "speed_change":
                f = p["speed"]["factor"]
                gt = f"{SPEED_WORDS[0] if f > 1 else SPEED_WORDS[1]}, {DIR_WORDS[0]}"
            else:
                odd = len(p["reversals"]) % 2 == 1
                gt = f"{SPEED_WORDS[2]}, {DIR_WORDS[1] if odd else DIR_WORDS[0]}"
            allo = [f"{s}, {d}" for s in SPEED_WORDS for d in DIR_WORDS]
            out.append(_cand(2, "start_vs_end", {"g": g, "min_fps": max(c.fps_rot([g], s0, s1),
                                                                        c.fps_rot([g], e0, e1))},
                             f"Compare how {c.the(g)} turns at the end of the video with how it turns at the "
                             f"start. At the end it turns ...", gt,
                             lambda rng, a=allo, t=gt: rng.sample([x for x in a if x != t], 5), True))
    return out


# ----------------------------------------------------------------- level 3
def _no_ramp_inside(c: Ctx, t0, t1):
    if c.sc != "reversal":
        return True
    return all(b <= t0 or a >= t1 for a, b in c.win["events"])


def _level3(c: Ctx):
    out = []
    D = c.D
    s0, s1 = c.start()
    wins = []
    if c.sc != "reversal":
        wins.append((0.0, c.END, "from the start of the video to the end"))
    for T in range(2, int(math.floor(D - 0.5)) + 1):
        wins.append((0.0, float(T), f"from the start of the video until the clock reads {T}.0 s"))
    for T0 in range(1, int(D)):
        for T1 in range(T0 + 2, int(math.floor(D - 0.5)) + 1):
            wins.append((float(T0), float(T1), f"between the clock readings {T0}.0 s and {T1}.0 s"))
    for g in c.tr:
        for t0, t1, txt in wins:
            if not _no_ramp_inside(c, t0, t1):
                continue
            x = abs(sim.theta(c.s, g, t1) - sim.theta(c.s, g, t0)) / (2 * math.pi)
            if _turn_margin(x) < 0.2 or x > 6.5:
                continue
            wm = c.wmax([g], t0, t1)
            fr = max(c.fps_rot([g], t0, t1), fps_gap(_turn_margin(x) * 2 * math.pi / max(wm, 1e-6)))
            n = int(math.floor(x))
            out.append(_cand(3, "turns", {"g": g, "t0": t0, "t1": round(t1, 6), "min_fps": fr},
                             f"How many full turns does the white dot on {c.the(g)} make {txt}?",
                             n, _int_dist(n, 0, 12), True))
    for a, b in itertools.permutations(c.tr, 2):
        r = Fraction(c.gm[a]["z"], c.gm[b]["z"])
        if r == 1 or r.denominator > 4 or r.numerator > 6:
            continue
        gt = str(r)
        cands = {str(1 / r), str(r + Fraction(1, 2)), str(r * 2), str(r / 2), str(abs(r - Fraction(1, 2))),
                 str(r + 1), "1"}
        cands.discard(gt)
        cands.discard("0")
        out.append(_cand(3, "ratio", {"a": a, "b": b, "min_fps": c.fps_rot([a, b], s0, s1)},
                         f"While {c.the(a)} makes one full turn, how many turns does {c.the(b)} make?",
                         gt, lambda rng, cs=sorted(cands): rng.sample(cs, len(cs)), True))
    for g in c.tr:
        T = 2 * math.pi / c.w0(g)
        if 1.5 <= T <= s1 - s0 and 0.5 - _turn_margin(T) >= 0.25:
            k = round(T)
            out.append(_cand(3, "period", {"g": g, "min_fps": max(fps_gap(0.5 - _turn_margin(T)),
                                                                  c.fps_rot([g], s0, s1))},
                             f"At the start of the video, about how many seconds does {c.the(g)} take to make one "
                             f"full turn? Answer in whole seconds.", k, _int_dist(k, 1, 12), True))
    p = c.s["program"]
    ev = []
    if c.sc == "reversal":
        rv = p["reversals"]
        for i, r in enumerate(rv):
            which = "" if len(rv) == 1 else (" for the first time" if i == 0 else " for the second time")
            ev.append((i, r["mid"], f"At about which clock reading does {DRIVER} stop and start turning the "
                                    f"other way{which}? Answer in whole seconds."))
    elif c.sc == "speed_change":
        ev.append((0, p["speed"]["mid"], f"At about which clock reading does {DRIVER} change its speed? "
                                         f"Answer in whole seconds."))
    elif c.sc == "disengage":
        d = p["disengage"]
        ev.append((0, d["t_out"], f"At about which clock reading does {c.the(d['gear'])} start to slide away from "
                                  f"{c.the(d['parent'])}? Answer in whole seconds."))
    elif c.sc == "brake":
        b = p["brake"]
        ev.append((0, b["t_touch"], f"At about which clock reading does the grey brake block touch "
                                    f"{c.the(b['gear'])}? Answer in whole seconds."))
    for i, t, txt in ev:
        k = round(t)
        out.append(_cand(3, "event_time", {"i": i, "min_fps": fps_gap(0.5 - abs(t - k))}, txt, k,
                         _int_dist(k, 0, int(D)), True))
    if c.sc in ("disengage", "brake"):
        s_0, s_1 = c.log["still"]
        k = round(s_1 - s_0)
        for g in c.log["stopped"]:
            out.append(_cand(3, "stop_duration", {"g": g, "min_fps": fps_gap(0.5 - abs(s_1 - s_0 - k))},
                             f"Partway through the video {c.the(g)} stops turning for a while. For about how many "
                             f"seconds does it stand completely still? Answer in whole seconds.",
                             k, _int_dist(k, 1, 9), True))
    if c.sc == "speed_change":
        f = p["speed"]["factor"]
        gt = FACTOR_TEXT[min(FACTOR_TEXT, key=lambda v: abs(v - f))]
        e0, e1 = c.end()
        out.append(_cand(3, "speed_factor", {"min_fps": max(c.fps_rot([c.drv], s0, s1), c.fps_rot([c.drv], e0, e1))},
                         f"After its speed changes, how fast does {DRIVER} turn compared with before?", gt,
                         lambda rng, g=gt: [x for x in FACTOR_TEXT.values() if x != g], True))
    return out


# ----------------------------------------------------------------- level 4
def _transitions(c: Ctx):
    """Times around which a gear starts or stops turning."""
    p = c.s["program"]
    if c.sc == "disengage":
        d = p["disengage"]
        return [d["still0"] - d["ramp"], d["still0"], d["still1"], d["still1"] + d["ramp"]]
    if c.sc == "brake":
        b = p["brake"]
        return [b["t_touch"], b["still0"], b["release"], b["release"] + b["ramp"]]
    if c.sc == "reversal":
        return [x for r in p["reversals"] for x in (r["mid"] - r["ramp"] / 2, r["mid"] + r["ramp"] / 2)]
    return []


def _level4(c: Ctx):
    out = []
    D = c.D
    s0, s1 = c.start()
    p = c.s["program"]
    if c.sc == "reversal":
        e0, e1 = c.end()
        odd = len(p["reversals"]) % 2
        for a, b in itertools.combinations(c.tr, 2):
            da, db = c.gm[a]["dir"] * (-1) ** odd, c.gm[b]["dir"] * (-1) ** odd
            gt = combo_text(c, a, b, da, db)
            opts = combo_opts(c, a, b)
            out.append(_cand(4, "end_dirs", {"a": a, "b": b, "min_fps": c.fps_rot([a, b], e0, e1)},
                             f"At the end of the video, which way are {c.the(a)} and {c.the(b)} turning?", gt,
                             lambda rng, o=opts, g=gt: [x for x in o if x != g], True))
    if c.sc == "disengage":
        d = p["disengage"]
        x = d["gear"]
        mem = [g for g in d["moving"] if g != x]
        uni = [g for g in c.tr if g != x]
        out.append(_cand(4, "stopped_followers", {"min_fps": fps_gap((d["still1"] - d["still0"]) / 2)},
                         f"When {c.the(x)} slides out of mesh and stops, which OTHER gears stop turning with it? "
                         f"List their colours, or answer 'none'.", set_text([c.col(g) for g in mem]),
                         _set_dist(c, mem, uni), True))
    trans = _transitions(c)
    has_stops = c.sc in ("disengage", "brake")
    if has_stops or c.loose:
        for T in range(1, int(math.floor(D - 0.5)) + 1):
            gaps = [abs(T - t) for t in trans] + [T, D - T]
            if min(gaps) < 0.3:
                continue
            if c.sc == "reversal" and any(abs(T - r["mid"]) < 0.6 for r in p["reversals"]):
                continue
            n = sum(abs(sim.omega(c.s, g, float(T))) < 1e-9 for g in c.ids)
            if n == 0 and not has_stops:
                continue
            out.append(_cand(4, "still_count", {"T": T, "min_fps": fps_gap(min(gaps))},
                             f"When the clock reads {T}.0 s, how many gears are standing still?", n,
                             _int_dist(n, 0, len(c.ids)), True))
    for tri in itertools.combinations(c.tr, 3):
        order = sorted(tri, key=lambda g: -c.w0(g))
        if all(c.w0(a) >= 1.15 * c.w0(b) for a, b in zip(order, order[1:])):
            gt = ", ".join(c.col(g) for g in order)
            perms = [", ".join(c.col(g) for g in pm) for pm in itertools.permutations(order)]
            out.append(_cand(4, "speed_rank", {"gs": sorted(tri), "min_fps": c.fps_rot(list(tri), s0, s1)},
                             f"At the start of the video, order these gears from fastest to slowest: "
                             f"{', '.join(c.col(g) for g in tri)}.", gt,
                             lambda rng, ps=perms, g=gt: rng.sample([x for x in ps if x != g], 5), True))
        tt = c.log["turns_total"]
        tot = sorted(((tt[g], g) for g in tri), reverse=True)
        if tot[0][0] >= 1.1 * tot[1][0] and tot[0][0] - tot[1][0] >= 0.3 and tot[1][0] > 0:
            cols = [c.col(g) for g in tri]
            gt = c.col(tot[0][1])
            fr = max(c.fps_rot(list(tri), 0, c.END),
                     fps_gap(0.3 * 2 * math.pi / max(c.wmax(list(tri), 0, c.END), 1e-6)))
            out.append(_cand(4, "most_turns", {"gs": list(tri), "min_fps": fr},
                             f"Over the whole video, which of these gears turns through the most full turns: "
                             f"{', '.join(cols)}?", gt,
                             lambda rng, cs=cols, g=gt: [x for x in cs if x != g] + [SAME_AMOUNT], True))
    for g in c.tr:
        if c.depth(g) < 2:
            continue
        par = c.gm[g]["parent"]
        others = [h for h in c.ids if h not in (g, par)]
        out.append(_cand(4, "drive_path", {"g": g, "min_fps": 0.5},
                         f"The motion of {DRIVER} reaches {c.the(g)} through other gears. Which gear turns "
                         f"{c.the(g)} directly?", c.col(par), _pick_cols(c, others), False))
    whens = [("start", s0, s1, "At the start of the video")]
    if c.sc == "reversal":
        whens.append(("end",) + tuple(c.end()) + ("At the end of the video",))
    for when, t0, t1, lead in whens:
        tm = (t0 + t1) / 2
        n = sum(c.dir_at(g, tm) < 0 for g in c.tr)
        out.append(_cand(4, "cw_count", {"when": when, "min_fps": c.fps_rot(c.tr, t0, t1)},
                         f"{lead}, how many gears are turning clockwise?", n, _int_dist(n, 0, len(c.ids)), True))
    return out


# ----------------------------------------------------------------- level 5
def _level5(c: Ctx):
    out = []
    D = c.D
    s0, s1 = c.start()
    p = c.s["program"]
    for a, b in itertools.combinations(c.tr, 2):
        gt = combo_text(c, a, b, -c.gm[a]["dir"], -c.gm[b]["dir"])
        opts = combo_opts(c, a, b)
        out.append(_cand(5, "flip_driver", {"a": a, "b": b, "min_fps": c.fps_rot([a, b, c.drv], s0, s1)},
                         f"Suppose {DRIVER} had turned the other way from the very start. Which way would "
                         f"{c.the(a)} and {c.the(b)} turn at the start of the video?", gt,
                         lambda rng, o=opts, g=gt: [x for x in o if x != g], True))
    for g in c.tr:
        if g == c.drv:
            continue
        mem = [h for h in sim.subtree(c.s, g) if h != g]
        uni = [h for h in c.tr if h not in (g, c.drv)]
        if len(uni) < 2:
            continue
        out.append(_cand(5, "remove", {"g": g, "min_fps": 0.5},
                         f"Suppose {c.the(g)} had been taken out before the video started and nothing else moved. "
                         f"Which of the other gears that turn in the video would then not turn at all? List their "
                         f"colours, or answer 'none'.", set_text([c.col(h) for h in mem]), _set_dist(c, mem, uni),
                         False))
    if c.loose:
        zl = c.gm[c.loose]["z"]
        for g in c.tr:
            zg = c.gm[g]["z"]
            if max(zl, zg) < 1.1 * min(zl, zg):
                continue
            d = dir_word(-c.gm[g]["dir"])
            sp = "faster" if zl < zg else "slower"
            gt = f"{d}, {sp} than {c.the(g)}"
            allo = [f"{x}, {y} than {c.the(g)}" for x in (CW, CCW) for y in ("faster", "slower")]
            out.append(_cand(5, "loose_mesh", {"g": g, "min_fps": c.fps_rot([g], s0, s1)},
                             f"Suppose the {c.col(c.loose)} gear, which never turns, were moved so that its teeth "
                             f"meshed with {c.the(g)} only. At the start of the video, how would the "
                             f"{c.col(c.loose)} gear turn?", gt, lambda rng, a=allo, t=gt: [x for x in a if x != t],
                             True))
    if c.sc == "disengage":
        d = p["disengage"]
        for y in c.tr:
            if y in (c.drv, d["gear"]):
                continue
            mem = [h for h in sim.subtree(c.s, y) if h != y]
            uni = [h for h in c.tr if h != y]
            out.append(_cand(5, "disengage_instead", {"g": y, "min_fps": fps_gap((d["still1"] - d["still0"]) / 2)},
                             f"Suppose {c.the(y)} had slid out of mesh with the gear that turns it, instead of "
                             f"{c.the(d['gear'])} sliding out. Which OTHER gears would have stopped with it? List "
                             f"their colours, or answer 'none'.", set_text([c.col(h) for h in mem]),
                             _set_dist(c, mem, uni), True))
    if c.sc == "brake":
        b = p["brake"]
        n = len(c.tr)
        s_0, s_1 = c.log["still"]
        for y in c.tr:
            if y == b["gear"]:
                continue
            sub = len(sim.subtree(c.s, y))
            dists = [sub, 1, n - 1, n + 1, n - 2]
            out.append(_cand(5, "brake_other", {"g": y, "min_fps": fps_gap((s_1 - s_0) / 2)},
                             f"Suppose the grey brake block had pressed on {c.the(y)} instead of "
                             f"{c.the(b['gear'])}. How many gears would have stopped turning while it pressed?",
                             n, lambda rng, ds=dists, nn=n: [v for v in ds if v >= 1 and v != nn] +
                             [v for v in range(1, 10) if v != nn], True))
    if c.sc == "reversal":
        e0, e1 = c.end()
        odd = len(p["reversals"]) % 2
        for a, b in itertools.combinations(c.tr, 2):
            gt = combo_text(c, a, b, c.gm[a]["dir"], c.gm[b]["dir"])
            opts = combo_opts(c, a, b)
            out.append(_cand(5, "no_reversal", {"a": a, "b": b, "min_fps": c.fps_rot([a, b], s0, s1)},
                             f"Suppose {DRIVER} had never changed direction. Which way would {c.the(a)} and "
                             f"{c.the(b)} be turning at the end of the video?", gt,
                             lambda rng, o=opts, g=gt: [x for x in o if x != g], True))
        del odd, e0, e1
    if c.sc in ("disengage", "brake", "speed_change"):
        if c.sc == "disengage":
            lead = f"Suppose {c.the(p['disengage']['gear'])} had never slid out of mesh"
        elif c.sc == "brake":
            lead = "Suppose the grey brake block had never touched the gears"
        else:
            lead = f"Suppose {DRIVER} had kept its starting speed for the whole video"
        for g in c.tr:
            x = c.w0(g) * c.END / (2 * math.pi)
            if _turn_margin(x) < 0.2 or x > 12.5:
                continue
            k = int(math.floor(x))
            fr = max(c.fps_rot([g], s0, s1), fps_gap(_turn_margin(x) * 2 * math.pi / c.w0(g)))
            out.append(_cand(5, "turns_cf", {"g": g, "min_fps": fr},
                             f"{lead}. How many full turns would the white dot on {c.the(g)} have made from the "
                             f"start of the video to the end?", k, _int_dist(k, 0, 14), True))
    for x in c.tr:
        if c.gm[x]["z"] * 2 > 72:
            continue
        for y in c.tr:
            if x == c.drv:
                gt = DOUBLE_OPTS[2] if y == x else DOUBLE_OPTS[0]
            else:
                gt = DOUBLE_OPTS[1] if y == x else DOUBLE_OPTS[2]
            out.append(_cand(5, "teeth_double", {"x": x, "y": y, "min_fps": 0.5},
                             f"Suppose {c.the(x)} had twice as many teeth (with the gears moved so that "
                             f"everything still meshes), and the motor still turned {DRIVER} exactly as in the "
                             f"video. At the start, how fast would {c.the(y)} turn?", gt,
                             lambda rng, g=gt: [o for o in DOUBLE_OPTS if o != g], False))
    return out


def _enough_options(q) -> bool:
    gt = q["fmt"](q["gt"])
    return len({q["fmt"](d) for d in q["dist"](random.Random(0))} - {gt}) >= 3


def candidates(scene) -> list:
    c = Ctx(scene)
    return [q for q in _level1(c) + _level2(c) + _level3(c) + _level4(c) + _level5(c) if _enough_options(q)]


def _by_level(cands):
    lv = {i: {} for i in range(1, 6)}
    for q in cands:
        lv[q["level"]].setdefault(q["key"], []).append(q)
    return lv


def pool_ok(scene) -> bool:
    """At least 3 distinct candidates on every level and 2+ templates on L2-L5."""
    lv = _by_level(candidates(scene))
    return all(sum(len(v) for v in lv[i].values()) >= 3 for i in lv) and all(len(lv[i]) >= 2 for i in (2, 3, 4, 5))


def build(scene: dict, rng: random.Random) -> list:
    lv = _by_level(candidates(scene))
    counts = L.ten_counts(rng)
    qs = []
    for level, k in zip(range(1, 6), counts):
        keys = sorted(lv[level])
        rng.shuffle(keys)
        pools = {key: list(lv[level][key]) for key in keys}
        picked, i = [], 0
        while len(picked) < k:
            key = keys[i % len(keys)]
            i += 1
            if pools[key]:
                picked.append(pools[key].pop(rng.randrange(len(pools[key]))))
        for q in picked:
            opts, a = L.mc(rng, q["gt"], q["dist"](rng), fmt=q["fmt"])
            if len(opts) != 4:
                raise ValueError(f"{q['key']}: only {len(opts)} options")
            qs.append(L.question(level, q["key"], q["text"], q["gt"], opts, a, q["motion"], q["params"]))
    return L.number(qs)
