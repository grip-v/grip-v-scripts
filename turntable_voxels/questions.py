"""10-question, 5-rung ladder for turntable_voxels (common/ladder5.py, LADDER = "ten").

`candidates(scene)` lists every question the scene supports (no rng);
`build` draws the per-level counts with `L.ten_counts`, then picks templates
round-robin in a random order, so the mix changes from clip to clip. Every
question has 4 options and stores params["min_fps"].

Words used in the questions:
  * the START VIEW: the turntable standing still before it starts to turn;
  * the END VIEW: standing still after it has stopped for the last time;
  * a FRONT VIEW: the side facing you seen straight on, with no tilt; it is
    the tallest stack in each column, left to right;
  * the turning direction is told by the turntable's near edge (the edge
    closest to you), which moves left or right on screen.
Coloured cubes are named by colour ("the red cube"); the base-colour cubes
are only counted. The clock is the only text in the frame.
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
START = "before the turntable starts to turn"
END = "once the turntable has stopped for the last time"
FACING = ("the side that faced you at the start", "the side that was on your left at the start",
          "the side that was at the back at the start", "the side that was on your right at the start")
AMOUNT = {1: "a quarter turn", 2: "half a turn", 3: "three quarters of a turn", 4: "one full turn",
          5: "one and a quarter turns", 6: "one and a half turns", 7: "one and three quarter turns",
          8: "two full turns"}
EDGE = {1: "right", -1: "left"}
DIRS = ("always to the right", "always to the left", "first to the right, later to the left",
        "first to the left, later to the right")
VISTOP = ("visible, and it is the top cube of its stack", "visible, but it is not the top cube of its stack",
          "hidden, and it is the top cube of its stack", "hidden, and it is not the top cube of its stack")
ORD = ("first", "second", "third", "fourth")


def ceil1(x: float) -> float:
    return max(0.5, math.ceil(x * 10 - 1e-9) / 10)


def fps_gap(gap: float) -> float:
    return ceil1(1.0 / max(gap, 1e-3))


def fv_text(v) -> str:
    return ", ".join(map(str, v))


def set_text(cols) -> str:
    cols = sorted(cols)
    if not cols:
        return "none of them"
    if len(cols) == 1:
        return f"only the {cols[0]} cube"
    return "the " + ", ".join(cols[:-1]) + f" and {cols[-1]} cubes"


def dirs_text(signs) -> str:
    if len(set(signs)) == 1:
        return DIRS[0] if signs[0] > 0 else DIRS[1]
    return DIRS[2] if signs[0] > 0 else DIRS[3]


def leg_text(q: int, sign: int) -> str:
    return f"{AMOUNT[q]}, with its near edge moving {EDGE[sign]}"


def round_margin(t: float) -> float:
    return 0.5 - abs(t - round(t))


class Ctx:
    def __init__(self, s):
        self.s = s
        self.sc = s["scenario"]
        self.h0 = s["heights"]
        self.hend = sim.end_heights(s)
        self.n0 = sum(map(sum, self.h0))
        self.nend = sum(map(sum, self.hend))
        self.stacks0 = sum(1 for r in self.h0 for v in r if v)
        self.hs = sim.holds(s)
        self.a_end = self.hs[-1][2]
        self.k_end = sim.quarter(self.a_end)
        self.hold0 = self.hs[0][1] - self.hs[0][0]
        self.endlen = self.hs[-1][1] - self.hs[-1][0]
        self.vis0 = sim.visible(s, 0.0, self.h0)
        self.vis_end = sim.visible(s, self.a_end, self.hend)
        self.legs = s["legs"]
        self.signs = [1 if lg["a1"] > lg["a0"] else -1 for lg in self.legs]
        self.quarters = [int(round(abs(lg["a1"] - lg["a0"]) / 90)) for lg in self.legs]
        w = sim.peak_speed(s)
        self.rot90 = ceil1(w / 90.0)              # a sample at least every 90 degrees: every side is seen
        self.rot120 = ceil1(w / 120.0)            # < 1/3 turn between samples: direction / quarter counts unambiguous
        self.f_start = ceil1(1.0 / self.hold0)
        self.f_end = ceil1(1.0 / self.endlen)
        self.marked = s["marked"]
        self.evs = s["events"]
        if self.evs:
            kt = [t for name, t in sim.key_times(s) if name.startswith("event")]
            gaps = [b - a for a, b in zip(kt, kt[1:])]
            self.f_ev = ceil1(1.0 / min(gaps)) if gaps else 0.5
        else:
            self.f_ev = 0.5

    def top(self, m) -> bool:
        return m["z"] == self.h0[m["y"]][m["x"]] - 1

    def vis_m(self, m, vis) -> bool:
        return (m["x"], m["y"], m["z"]) in vis

    def m_end_present(self, m) -> bool:
        return not any(ev["kind"] == "lift" and (ev["x"], ev["y"], ev["z"]) == (m["x"], m["y"], m["z"])
                       for ev in self.evs)


def _cand(level, key, params, text, gt, dist, motion):
    return {"level": level, "key": key, "params": params, "text": text, "gt": gt, "dist": dist, "motion": motion}


def _ints(gt, extra=(), lo=0, hi=60):
    def f(rng):
        near = [gt - 2, gt - 1, gt + 1, gt + 2, gt + 3]
        rng.shuffle(near)
        out = [v for v in list(extra) + near if lo <= v <= hi and v != gt]
        return out + sorted((v for v in range(lo, hi + 1) if v != gt), key=lambda v: (abs(v - gt), v))
    return f


def _fixed(options, gt):
    return lambda rng: rng.sample([o for o in options if o != gt], len([o for o in options if o != gt]))


def _fv_dist(fv, extra=()):
    def f(rng):
        out = [fv_text(e) for e in extra]
        alts = []
        for k in range(len(fv)):
            for dv in (1, -1):
                a = list(fv)
                a[k] += dv
                if 0 <= a[k] <= 7:
                    alts.append(fv_text(a))
        rng.shuffle(alts)
        return out + [fv_text(fv[::-1])] + alts
    return f


def _set_dist(gt_cols, universe):
    import itertools

    def f(rng):
        subs = [set_text(c) for r in range(len(universe) + 1) for c in itertools.combinations(sorted(universe), r)]
        near = [set_text(sorted(set(gt_cols) ^ {u})) for u in universe]
        rng.shuffle(near)
        rest = list(subs)
        rng.shuffle(rest)
        return near + rest
    return f


def view_count_cf(s, phi, heights):
    """Visible count for a re-simulated still view, or None if a cube would be only partly visible."""
    if sim.ambiguous(s, phi, heights):
        return None
    return len(sim.visible(s, phi, heights))


# ----------------------------------------------------------------- level 1
def _level1(c: Ctx):
    s, out = c.s, []
    g = s["gx"] * s["gy"]
    out.append(_cand(1, "grid_squares", {"min_fps": 0.5}, "How many squares does the grid on the turntable have?",
                     g, _ints(g, [g + s["gx"], g - s["gy"], (s["gx"] + 1) * (s["gy"] + 1), g + 1], 4, 40), False))
    t = max(map(max, c.h0))
    out.append(_cand(1, "tallest_start", {"min_fps": c.f_start},
                     "At the start of the video, how many cubes tall is the tallest stack?",
                     t, _ints(t, [t - 1, t + 1], 1, 8), False))
    present = {m["colour"] for m in c.marked}
    absent = [k for k in sorted(sim.COLOURS) if k not in present and k not in sim.CONFUSABLE[s["theme"]["base"]]]
    if len(absent) >= 3:
        for m in c.marked:
            hid = not c.vis_m(m, c.vis0)
            out.append(_cand(1, "colour_present", {"c": m["colour"], "min_fps": c.rot90 if hid else c.f_start},
                             "One of these colours is on a cube in the video. Which one?", m["colour"],
                             lambda rng, a=absent: rng.sample(a, len(a)), hid))
    f = c.rot120
    if c.sc == "reverse":
        f = max(f, ceil1(2.0 / (c.legs[1]["t1"] - c.legs[1]["t0"])))
    gt = dirs_text(c.signs)
    out.append(_cand(1, "turn_dirs", {"min_fps": f}, "Whenever the turntable turns, which way does its near edge (the "
                     "edge closest to you) move?", gt,
                     _fixed(DIRS, gt), True))
    if c.evs:
        k = len(c.evs)
        what = ("How many cubes drop onto the structure during the video?" if c.sc == "build" else
                "How many cubes rise off the structure and vanish during the video?")
        out.append(_cand(1, "n_events", {"min_fps": c.f_ev}, what, k, _ints(k, [k + 1, k - 1], 1, 8), True))
    return out


# ----------------------------------------------------------------- level 2
def _level2(c: Ctx):
    out = []
    cols = [m["colour"] for m in c.marked]
    for m in c.marked:
        v, tp = c.vis_m(m, c.vis0), c.top(m)
        gt = VISTOP[(0 if v else 2) + (0 if tp else 1)]
        out.append(_cand(2, "marked_vis_top", {"c": m["colour"], "min_fps": c.f_start if v else max(c.f_start, c.rot90)},
                         f"{START.capitalize()}, can you see the {m['colour']} cube (even partly), and is it the "
                         f"top cube of its stack?", gt, _fixed(VISTOP, gt), not v))
        above = c.h0[m["y"]][m["x"]] - m["z"] - 1
        out.append(_cand(2, "above_marked", {"c": m["colour"], "min_fps": c.f_start if v else c.rot90},
                         f"At the start of the video, how many cubes are stacked on top of the {m['colour']} cube?",
                         above, _ints(above, [above + 1, above - 1], 0, 6), not v))
        out.append(_cand(2, "below_marked", {"c": m["colour"], "min_fps": c.f_start if v else c.rot90},
                         f"At the start of the video, how many cubes are under the {m['colour']} cube in its stack?",
                         m["z"], _ints(m["z"], [m["z"] + 1, m["z"] - 1], 0, 6), not v))
    if len(c.marked) >= 2:
        gt_cols = [m["colour"] for m in c.marked if c.vis_m(m, c.vis0)]
        out.append(_cand(2, "marked_set_start", {"min_fps": c.f_start},
                         f"{START.capitalize()}, which of the coloured cubes can you see (even partly)?",
                         set_text(gt_cols), _set_dist(gt_cols, cols), False))
        gt_cols = [m["colour"] for m in c.marked if c.m_end_present(m) and c.vis_m(m, c.vis_end)]
        out.append(_cand(2, "marked_set_end", {"min_fps": max(c.f_end, c.f_ev if c.evs else 0.5)},
                         f"{END.capitalize()}, which of the coloured cubes can you see (even partly)?",
                         set_text(gt_cols), _set_dist(gt_cols, cols), True))
    if len(c.marked) == 3:
        zs = sorted(((m["z"], m["colour"]) for m in c.marked), reverse=True)
        if zs[0][0] > zs[1][0]:
            hid = any(not c.vis_m(m, c.vis0) for m in c.marked)
            opts = cols + ["all three are at the same height"]
            out.append(_cand(2, "highest_marked", {"min_fps": c.rot90 if hid else c.f_start},
                             "At the start of the video, which coloured cube sits highest above the turntable?",
                             zs[0][1], _fixed(opts, zs[0][1]), hid))
    if c.sc == "stops":
        st = c.hs[1]
        gt = FACING[sim.quarter(st[2])]
        out.append(_cand(2, "facing_stop", {"min_fps": max(c.rot120, ceil1(1.0 / (st[1] - st[0])))},
                         "The first time the turntable stops partway, which side of the structure faces you?",
                         gt, _fixed(FACING, gt), True))
    return out


# ----------------------------------------------------------------- level 3
def _level3(c: Ctx):
    s, out = c.s, []
    out.append(_cand(3, "n_stacks", {"min_fps": c.rot90},
                     "At the start of the video, on how many squares of the grid does at least one cube stand?",
                     c.stacks0, _ints(c.stacks0, [c.stacks0 + 1, c.stacks0 - 1], 1, 25), True))
    nv = len(c.vis0)
    out.append(_cand(3, "visible_start", {"min_fps": c.f_start},
                     f"{START.capitalize()}, how many cubes can you see (even partly)?",
                     nv, _ints(nv, [c.n0, nv + 1, nv - 1], 1, 40), False))
    ne = len(c.vis_end)
    out.append(_cand(3, "visible_end", {"min_fps": max(c.f_end, c.f_ev if c.evs else 0.5)},
                     f"{END.capitalize()}, how many cubes can you see (even partly)?",
                     ne, _ints(ne, [c.nend, nv, ne + 1], 1, 40), True))
    out.append(_cand(3, "total_start", {"min_fps": c.rot90},
                     "At the start of the video, how many cubes are in the structure altogether?",
                     c.n0, _ints(c.n0, [nv, c.n0 + 1, c.n0 - 1], 1, 45), True))
    t0 = c.legs[0]["t0"]
    out.append(_cand(3, "t_start", {"min_fps": fps_gap(round_margin(t0))},
                     "At about which clock reading does the turntable start to turn? Answer in whole seconds.",
                     int(round(t0)), _ints(int(round(t0)), [int(round(t0)) + 1, int(round(t0)) - 1], 0, 10), True))
    q0, s0 = c.quarters[0], c.signs[0]
    gt = leg_text(q0, s0)
    lead = ("Before the turntable first stops or changes direction, how far does it turn?"
            if len(c.legs) > 1 else "How far does the turntable turn before it stops?")
    cands = [leg_text(q, sg) for q in (q0 - 1, q0 + 1, q0 - 2, q0 + 2, q0) for sg in (s0, -s0) if 1 <= q <= 8]
    out.append(_cand(3, "first_leg", {"min_fps": c.rot120}, lead, gt,
                     lambda rng, cc=cands, g=gt: [x for x in cc if x != g], True))
    tq = sum(c.quarters)
    out.append(_cand(3, "total_quarters", {"min_fps": c.rot120},
                     "Counting all of its turning (turning back counts too), through how many quarter turns does "
                     "the turntable turn during the video?", tq, _ints(tq, [tq - 1, tq + 1, tq + 2], 1, 12), True))
    for i, a in enumerate(c.marked):
        for b in c.marked:
            if a["z"] > b["z"]:
                d = a["z"] - b["z"]
                hid = not (c.vis_m(a, c.vis0) and c.vis_m(b, c.vis0))
                out.append(_cand(3, "layer_diff", {"a": a["colour"], "b": b["colour"],
                                                   "min_fps": c.rot90 if hid else c.f_start},
                                 f"At the start of the video, how many layers higher than the {b['colour']} cube is "
                                 f"the {a['colour']} cube?", d, _ints(d, [0, d + 1, d - 1], 0, 6), hid))
    if c.sc == "stops":
        st = c.hs[1]
        ts = st[0]
        out.append(_cand(3, "t_stop", {"min_fps": fps_gap(round_margin(ts))},
                         "At about which clock reading does the turntable first stop partway? Answer in whole seconds.",
                         int(round(ts)), _ints(int(round(ts)), [int(round(ts)) + 1, int(round(ts)) - 1], 0, 10), True))
        ln = st[1] - st[0]
        out.append(_cand(3, "stop_len", {"min_fps": fps_gap(round_margin(ln))},
                         "The first time the turntable stops partway, for about how many seconds does it stand "
                         "still? Answer in whole seconds.", int(round(ln)), _ints(int(round(ln)), [], 0, 6), True))
        nvs = view_count_cf(s, st[2], c.h0)
        out.append(_cand(3, "visible_stop", {"min_fps": ceil1(1.0 / ln)},
                         "The first time the turntable stops partway, how many cubes can you see (even partly)?",
                         nvs, _ints(nvs, [nv, c.n0, nvs + 1], 1, 40), True))
    if c.sc == "reverse":
        tr = c.legs[0]["t1"]
        out.append(_cand(3, "t_reverse", {"min_fps": fps_gap(round_margin(tr))},
                         "At about which clock reading does the turntable change direction? Answer in whole seconds.",
                         int(round(tr)), _ints(int(round(tr)), [int(round(tr)) + 1, int(round(tr)) - 1], 0, 10), True))
    if c.evs:
        kt = [t for name, t in sim.key_times(s) if name.startswith("event")]
        for i, t in enumerate(kt[:2]):
            what = (f"the {ORD[i]} dropped cube land on the structure" if c.sc == "build" else
                    f"the {ORD[i]} cube start to rise off the structure")
            out.append(_cand(3, "t_event", {"i": i, "min_fps": max(fps_gap(round_margin(t)), c.f_ev)},
                             f"At about which clock reading does {what}? Answer in whole seconds.", int(round(t)),
                             _ints(int(round(t)), [int(round(t)) + 1, int(round(t)) - 1], 0, 10), True))
    return out


# ----------------------------------------------------------------- level 4
def _level4(c: Ctx):
    s, out = c.s, []
    nv = len(c.vis0)
    hid = c.n0 - nv
    out.append(_cand(4, "hidden_start", {"min_fps": c.rot90},
                     f"How many cubes are completely hidden {START}?", hid, _ints(hid, [nv, hid + 1, hid - 1], 0, 40),
                     True))
    gt = FACING[c.k_end]
    out.append(_cand(4, "facing_end", {"min_fps": max(c.rot120, c.f_end)},
                     f"{END.capitalize()}, which side of the structure faces you?", gt, _fixed(FACING, gt), True))
    fv = sim.front_view(c.h0)
    front_row = c.h0[0]
    sums = [sum(c.h0[y][x] for y in range(s["gy"])) for x in range(s["gx"])]
    out.append(_cand(4, "front_view_start", {"min_fps": c.f_start},
                     "Look straight at the side that faces you at the start, with no tilt. How many cubes tall is "
                     "each column, from left to right?", fv_text(fv), _fv_dist(fv, [front_row, sums]), False))
    rh = sim.rotate_heights(c.hend, c.k_end)
    fve = sim.front_view(rh)
    out.append(_cand(4, "front_view_end", {"min_fps": max(c.f_end, c.rot120)},
                     f"{END.capitalize()}, look straight at the side that then faces you, with no tilt. How many "
                     f"cubes tall is each column, from left to right?", fv_text(fve),
                     _fv_dist(fve, [fv, rh[0]]), True))
    ne = len(c.vis_end)
    he = c.nend - ne
    out.append(_cand(4, "hidden_end", {"min_fps": max(c.rot90, c.f_end, c.f_ev if c.evs else 0.5)},
                     f"{END.capitalize()}, how many cubes are completely hidden?", he,
                     _ints(he, [hid, ne, he + 1], 0, 40), True))
    if c.evs:
        out.append(_cand(4, "total_end", {"min_fps": max(c.rot90, c.f_ev)},
                         "At the end of the video, how many cubes are in the structure altogether?", c.nend,
                         _ints(c.nend, [c.n0, c.nend + 1, c.nend - 1], 1, 45), True))
        te = max(map(max, c.hend))
        t0 = max(map(max, c.h0))
        if te != t0:
            out.append(_cand(4, "tallest_end", {"min_fps": max(c.f_end, c.f_ev)},
                             "At the end of the video, how many cubes tall is the tallest stack?", te,
                             _ints(te, [t0, te + 1, te - 1], 1, 8), True))
    return out


# ----------------------------------------------------------------- level 5
def _level5(c: Ctx):
    s, out = c.s, []
    g = c.n0 + c.stacks0
    out.append(_cand(5, "cf_add_top", {"min_fps": c.rot90},
                     "Suppose one more cube had been placed on top of every stack at the start. How many cubes would "
                     "the structure have had then?", g,
                     _ints(g, [c.n0 + s["gx"] * s["gy"], c.n0 + 1, g + 1, c.n0], 1, 70), True))
    for m in c.marked:
        h2 = [list(r) for r in c.h0]
        h2[m["y"]][m["x"]] = 0
        fv2 = sim.front_view(h2)
        hid = not c.vis_m(m, c.vis0)
        out.append(_cand(5, "cf_remove_fv", {"c": m["colour"], "min_fps": c.rot90 if hid else c.f_start},
                         f"Suppose the whole stack holding the {m['colour']} cube had been taken away before the video "
                         f"started. Looking straight at the side that faces you at the start, with no tilt, how many "
                         f"cubes tall would each column be, from left to right?", fv_text(fv2),
                         _fv_dist(fv2, [sim.front_view(c.h0)]), hid))
        nv2 = view_count_cf(s, 0.0, h2)
        if nv2 is not None:
            n2 = sum(map(sum, h2))
            hd2 = n2 - nv2
            hd = c.n0 - len(c.vis0)
            out.append(_cand(5, "cf_remove_hidden", {"c": m["colour"], "min_fps": c.rot90},
                             f"Suppose the whole stack holding the {m['colour']} cube had been taken away before the "
                             f"video started. How many cubes would be completely hidden {START}?", hd2,
                             _ints(hd2, [hd, hd2 + 1, hd2 - 1], 0, 40), True))
    gt = FACING[(-c.k_end) % 4]
    out.append(_cand(5, "cf_other_dir", {"min_fps": c.rot120},
                     "Suppose the turntable had made exactly the same turns, but each one in the opposite direction. "
                     "Once it had stopped for the last time, which side of the structure would face you?",
                     gt, _fixed(FACING, gt), True))
    a2 = c.a_end - c.signs[-1] * 90
    nv3 = view_count_cf(s, a2, c.hend)
    if nv3 is not None:
        out.append(_cand(5, "cf_stop_early", {"min_fps": max(c.rot90, c.f_end)},
                         "Suppose the turntable's last turn had ended a quarter turn sooner, so it stopped a quarter "
                         "turn short of where it really stops. How many cubes could you see (even partly) once it had "
                         "stopped?", nv3, _ints(nv3, [len(c.vis_end), nv3 + 1, nv3 - 1], 1, 40), True))
    if c.evs:
        nv4 = view_count_cf(s, c.a_end, c.h0)
        if nv4 is not None:
            what = "dropped onto" if c.sc == "build" else "lifted off"
            out.append(_cand(5, "cf_no_events", {"min_fps": max(c.rot90, c.f_end)},
                             f"Suppose no cubes had been {what} the structure. {END.capitalize()}, how many cubes "
                             f"could you see (even partly)?", nv4,
                             _ints(nv4, [len(c.vis_end), nv4 + 1, nv4 - 1], 1, 40), True))
    return out


def _enough_options(q) -> bool:
    gt = str(q["gt"])
    return len({str(d) for d in q["dist"](random.Random(0))} - {gt}) >= 3


def candidates(scene) -> list:
    c = Ctx(scene)
    return [q for q in _level1(c) + _level2(c) + _level3(c) + _level4(c) + _level5(c) if _enough_options(q)]


def _by_level(cands):
    lv = {i: {} for i in range(1, 6)}
    for q in cands:
        lv[q["level"]].setdefault(q["key"], []).append(q)
    return lv


def pool_ok(scene) -> bool:
    """At least 3 candidates on every level and 2+ templates on every level."""
    lv = _by_level(candidates(scene))
    return all(sum(len(v) for v in lv[i].values()) >= 3 and len(lv[i]) >= 2 for i in lv)


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
            opts, a = L.mc(rng, q["gt"], q["dist"](rng))
            if len(opts) != 4:
                raise ValueError(f"{q['key']}: only {len(opts)} options")
            qs.append(L.question(level, q["key"], q["text"], q["gt"], opts, a, q["motion"], q["params"]))
    return L.number(qs)
