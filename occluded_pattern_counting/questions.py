"""10-question ladder for occluded_pattern_counting (common/ladder5.py, LADDER = "ten").

Objects are identical except the odd-coloured ones (odd_one scenario), so
questions count objects or name them by colour. Occluders are named by
colour ("the brown occluder"). Every count question is asked about a hold
(the occluders are still), and "when the clock shows X" questions keep the
frames showing X >= 6 frames inside a hold. Slides last >= 0.6 s and holds
>= 1 s, so a 1-2 fps sample always lands inside every hold. What-ifs re-run
the occluder track with the change (sim.track(scene, mods)). Every question
has four distinct options.
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
FAMILY_TEXT = {"grid": "a rectangular grid", "ring": "a ring", "triangle": "a triangle", "honeycomb":
               "a honeycomb (offset rows)"}
DIRS = ("left", "right", "up", "down")
PLURAL = {"circle": "circles", "square": "squares", "triangle": "triangles", "star": "stars", "diamond": "diamonds",
          "hexagon": "hexagons"}


def fps_for(gaps, floor=1):
    g = min(gaps) if gaps else 99.0
    return int(min(24, max(floor, math.ceil(1.0 / max(g, 1e-6) - 1e-9))))


def fmt_clock(t):
    return f"{t / 10:.1f} s"


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
    near = [v for v in (gt + 1, gt - 1, gt + 2, gt - 2, gt + 3, gt - 3, gt + 4, gt + 5) if lo <= v <= hi]
    rng.shuffle(near)
    near.sort(key=lambda v: abs(v - gt))
    ex = [e for e in extra if lo <= e <= hi and e != gt]
    return mc4(rng, gt, ex[:2] + near)


def clock_opts(rng, tenths, step=6):
    keep = []
    for c in [tenths + d for d in (-12, -6, 6, 12, 18, -18, 9, -9)]:
        if c >= 0 and all(abs(c - u) >= step for u in keep + [tenths]):
            keep.append(c)
    rng.shuffle(keep)
    return mc4(rng, fmt_clock(tenths), [fmt_clock(c) for c in keep])


def Q(key, params, text, gt, opts, motion, fps):
    if opts is None:
        return None
    return (key, dict(params, min_fps=fps), text, gt, opts[0], opts[1], motion)


class Ctx:
    def __init__(self, scene):
        self.scene = scene
        self.objs = scene["objects"]
        self.n = len(self.objs)
        self.frames = scene["frames"]
        self.hs = sim.holds(scene)
        self.st = sim.states(scene)
        self.mid = [(a + b) // 2 for a, b in self.hs]
        self.hid = [[i for i in range(self.n) if self.st[m][i] == 2] for m in self.mid]
        self.base = max(sorted({o["color"] for o in self.objs}), key=lambda c: sum(o["color"] == c for o in self.objs))
        self.odd = [i for i in range(self.n) if self.objs[i]["color"] != self.base]
        self.oname = [f"the {o['color']} occluder" for o in scene["occluders"]]
        self.many = len(scene["occluders"]) > 1
        self.slide_pts = [b for a, b in self.hs[:-1]] + [a for a, b in self.hs[1:]]
        self.hold_gap = min(b - a for a, b in self.hs) / FPS
        self.slide_gap = min(c - b for (a, b), (c, d) in zip(self.hs, self.hs[1:])) / FPS

    def occ_word(self):
        return "an occluder" if self.many else "the occluder"

    def safe_tick(self, t, pad=6):
        fr = [f for f in range(self.frames) if sim.clock_tenths(f) == t]
        if not fr or fr[0] < sim.EDGE_F or fr[-1] > self.frames - 1 - sim.EDGE_F:
            return None
        for k, (a, b) in enumerate(self.hs):
            if a + pad <= fr[0] and fr[-1] <= b - pad:
                return k
        return None


def level1(c: Ctx, rng):
    out = []
    v0 = c.n - len(c.hid[0])
    out.append(Q("n_visible_start", {}, "At the start of the video, how many objects can you see (not hidden behind "
                 f"{c.occ_word()})?", v0, int_opts(rng, v0, 0, 60, [c.n]), False, 1))
    fam = c.scene["family"]
    out.append(Q("family", {}, "How are the objects arranged?", FAMILY_TEXT[fam],
                 mc4(rng, FAMILY_TEXT[fam], [FAMILY_TEXT[f] for f in sim.FAMILIES if f != fam]), False, 1))
    sh = c.scene["shape"]
    pool = [s for s in sim.SHAPES if s != sh]
    rng.shuffle(pool)
    out.append(Q("object_shape", {}, "What shape are the objects?", sh, mc4(rng, sh, pool), False, 1))
    pool = [k for k in sim.COLORS if k != c.base and all(o["color"] != k for o in c.objs)]
    rng.shuffle(pool)
    out.append(Q("main_color", {}, "What colour are most of the objects?", c.base, mc4(rng, c.base, pool), False, 1))
    if c.many:
        out.append(Q("n_occluders", {}, "How many occluders (solid blocks covering objects) are there?", 2,
                     int_opts(rng, 2, 0, 5), False, 1))
    return out


def first_dir(c: Ctx, j):
    kf = c.scene["occluders"][j]["kf"]
    if len(kf) < 3:
        return None
    dx, dy = kf[2][1] - kf[1][1], kf[2][2] - kf[1][2]
    if abs(dx) >= 2 * abs(dy) and abs(dx) >= 40:
        return "right" if dx > 0 else "left"
    if abs(dy) >= 2 * abs(dx) and abs(dy) >= 40:
        return "down" if dy > 0 else "up"
    return None


def level2(c: Ctx, rng):
    out = []
    v = [c.n - len(h) for h in c.hid]
    if v[0] != v[-1]:
        gt = "more at the start" if v[0] > v[-1] else "more at the end"
        out.append(Q("start_vs_end", {}, "Are more objects visible at the start or at the end of the video?", gt,
                     mc4(rng, gt, ["more at the start", "more at the end", "the same number",
                                   "no objects are visible at either time"]), True, fps_for([c.hold_gap])))
    for j in range(len(c.scene["occluders"])):
        d = first_dir(c, j)
        if d:
            who = c.oname[j] if c.many else "the occluder"
            out.append(Q("slide_dir", {"j": j}, f"Which way does {who} move the first time it moves?", d,
                         mc4(rng, d, [x for x in DIRS if x != d]), True, fps_for([c.slide_gap])))
    if len(c.odd) == 1:
        i = c.odd[0]
        col = c.objs[i]["color"]
        gt = "hidden" if c.st[-1][i] == 2 else "visible"
        out.append(Q("odd_end", {"x": i}, f"At the end of the video, is the {col} object visible or hidden?", gt,
                     mc4(rng, gt, ["visible", "hidden", "partly hidden", "there is no such object"]), True,
                     fps_for([c.hold_gap])))
        gt = "yes" if i in c.hid[0] else "no"
        out.append(Q("odd_start", {"x": i}, f"At the start of the video, is the {col} object hidden?", gt,
                     mc4(rng, gt, ["yes", "no", "only partly", "there is no such object"]), True,
                     fps_for([c.hold_gap])))
    if len(c.hs) >= 3:
        a, b = len(c.hid[0]), len(c.hid[1])
        if a != b:
            gt = "fewer" if b < a else "more"
            out.append(Q("after_first_move", {}, "After the first move, are more or fewer objects hidden than "
                         "at the start?", gt, mc4(rng, gt, ["more", "fewer", "the same number", "none are hidden"]),
                         True, fps_for([c.hold_gap])))
    if c.many:
        end = [sum(sim.cover(rc, o["x"], o["y"], o["r"]) == 2 for o in c.objs) for rc in sim.track(c.scene)[-1]]
        if end[0] != end[1]:
            gt = c.oname[0] if end[0] > end[1] else c.oname[1]
            out.append(Q("occ_more_end", {}, "At the end of the video, which occluder hides more objects?", gt,
                         mc4(rng, gt, c.oname + ["they hide the same number", "neither hides any object"]), True,
                         fps_for([c.hold_gap])))
    return out

ORDW = ("first", "second", "third", "fourth")


def cover_side(c: Ctx, f):
    """Side of the pattern that the occluder(s) cover at frame f, or None if not clear-cut."""
    xs = [o["x"] for o in c.objs]
    ys = [o["y"] for o in c.objs]
    mx, my = (min(xs) + max(xs)) / 2, (min(ys) + max(ys)) / 2
    hx, hy = max(1.0, (max(xs) - min(xs)) / 2), max(1.0, (max(ys) - min(ys)) / 2)
    hid = [i for i in range(c.n) if c.st[f][i] == 2]
    if not hid or c.many:
        return None
    dx = sum(c.objs[i]["x"] - mx for i in hid) / len(hid) / hx
    dy = sum(c.objs[i]["y"] - my for i in hid) / len(hid) / hy
    if abs(dx) >= 0.35 and abs(dx) >= 2 * abs(dy):
        return "right" if dx > 0 else "left"
    if abs(dy) >= 0.35 and abs(dy) >= 2 * abs(dx):
        return "bottom" if dy > 0 else "top"
    return None


def level2_more(c: Ctx, rng):
    out = []
    sides = ("left", "right", "top", "bottom")
    for key, f, when in (("side_start", 0, "at the start"), ("side_end", c.frames - 1, "at the end")):
        sd = cover_side(c, f)
        if sd:
            out.append(Q(key, {}, f"Which part of the pattern is hidden {when} of the video?", f"the {sd} part",
                         mc4(rng, f"the {sd} part", [f"the {x} part" for x in sides if x != sd]), key == "side_end",
                         fps_for([c.hold_gap])))
    hs = [len(h) for h in c.hid]
    if len(hs) >= 3 and sorted(hs)[-1] > sorted(hs)[-2] and not c.many:
        k = hs.index(max(hs))
        gt = f"its {ORDW[k]} position"
        opts = [f"its {ORDW[j]} position" for j in range(len(hs))] + ["it hides the same number everywhere"]
        out.append(Q("most_stop", {}, "At which of its resting positions does the occluder hide the most objects?",
                     gt, mc4(rng, gt, opts + [f"its {ORDW[j]} position" for j in range(len(hs), 4)]), True,
                     fps_for([c.hold_gap])))
    a, b = set(c.hid[0]), set(c.hid[1])
    unc, cov = len(a - b), len(b - a)
    if unc != cov:
        gt = "it uncovers more" if unc > cov else "it covers more"
        out.append(Q("first_move_net", {}, f"During the first move, {'do the occluders' if c.many else 'does the occluder'} uncover more "
                     f"objects than {'they newly cover' if c.many else 'it newly covers'}, or cover more?", gt,
                     mc4(rng, gt, ["it uncovers more", "it covers more", "the same number", "it does not move"]), True,
                     fps_for([c.slide_gap])))
    return out


def level3(c: Ctx, rng):
    out = []
    tticks = [t for t in range(5, sim.clock_tenths(c.frames - 1)) if c.safe_tick(t) is not None]
    for t in tticks:
        k = c.safe_tick(t)
        v = c.n - len(c.hid[k])
        out.append(Q("n_visible_at", {"t": t}, f"When the clock shows {fmt_clock(t)}, how many objects are visible?",
                     v, int_opts(rng, v, 0, 60, [c.n - len(h) for h in c.hid if len(h) != len(c.hid[k])]), True,
                     fps_for([c.hold_gap], 2)))
        h = len(c.hid[k])
        out.append(Q("n_hidden_at", {"t": t}, f"When the clock shows {fmt_clock(t)}, how many objects are hidden "
                     f"behind {'the occluders' if c.many else 'the occluder'}?", h,
                     int_opts(rng, h, 0, 60, [len(x) for x in c.hid if len(x) != h]), True, fps_for([c.hold_gap], 2)))
    f = c.hs[0][1] + 1
    t = sim.clock_tenths(f)
    out.append(Q("first_move_time", {}, f"What does the clock show when the occluder{'s first start' if c.many else ' first starts'}"
                 " to move?", fmt_clock(t), clock_opts(rng, t), True, fps_for([c.slide_gap], 2)))
    f = c.hs[-1][0]
    t = sim.clock_tenths(f)
    out.append(Q("last_stop_time", {}, "What does the clock show when the occluder"
                 f"{'s' if c.many else ''} stop{'' if c.many else 's'} moving for the last time?", fmt_clock(t),
                 clock_opts(rng, t), True, fps_for([c.slide_gap], 2)))
    d = sim.clock_tenths(c.hs[0][1] + 1) - sim.clock_tenths(c.hs[0][0])
    out.append(Q("first_hold_len", {}, f"For how long do{'' if c.many else 'es'} the occluder{'s' if c.many else ''} "
                 "stay still at the start before moving (read the clock)?", fmt_clock(d), clock_opts(rng, d), True,
                 fps_for([c.slide_gap], 2)))
    return out


def level4(c: Ctx, rng):
    out = []
    out.append(Q("n_total", {}, "Counting the ones that are hidden at times, how many objects are there in total?",
                 c.n, int_opts(rng, c.n, 1, 60, [c.n + len(c.scene["gaps"]), c.n - len(c.hid[-1])]), True,
                 fps_for([c.hold_gap])))
    rev = sum(1 for i in c.hid[0] if i not in c.hid[-1])
    out.append(Q("n_revealed", {}, "How many objects are hidden at the start but visible at the end?", rev,
                 int_opts(rng, rev, 0, 60, [len(c.hid[0])]), True, fps_for([c.hold_gap])))
    cov = sum(1 for i in c.hid[-1] if i not in c.hid[0])
    out.append(Q("n_covered", {}, "How many objects are visible at the start but hidden at the end?", cov,
                 int_opts(rng, cov, 0, 60, [len(c.hid[-1])]), True, fps_for([c.hold_gap])))
    ev = len({i for h in c.hid for i in h})
    if len(c.hs) >= 3:
        out.append(Q("n_ever_hidden", {}, "How many different objects are hidden at some point while the occluder"
                     f"{'s are' if c.many else ' is'} standing still?", ev,
                     int_opts(rng, ev, 0, 60, [len(c.hid[0]) + len(c.hid[-1])]), True, fps_for([c.hold_gap])))
    if c.scene["gaps"]:
        g = len(c.scene["gaps"])
        out.append(Q("n_gaps", {}, "The pattern has some empty places where an object is missing. How many?", g,
                     int_opts(rng, g, 1, 9), True, fps_for([c.hold_gap])))
    if c.odd:
        col = c.objs[c.odd[0]]["color"]
        k = len(c.odd)
        out.append(Q("n_odd", {}, f"How many {col} objects are there in total?", k, int_opts(rng, k, 1, 9), True,
                     fps_for([c.hold_gap])))
    hs = [len(h) for h in c.hid]
    if len(hs) >= 3 and sorted(hs)[-1] > sorted(hs)[-2]:
        m = max(hs)
        out.append(Q("max_hidden", {}, "What is the largest number of objects hidden at the same time while the "
                     f"occluder{'s are' if c.many else ' is'} standing still?", m,
                     int_opts(rng, m, 0, 60, [h for h in hs if h != m]), True, fps_for([c.hold_gap])))
    return out


def cf_mods(c: Ctx):
    out = [(("stay",), None)]
    if len(c.hs) >= 3:
        out.append((("stop_early",), None))
    if c.many:
        out += [(("remove", 0), 0), (("remove", 1), 1)]
    return out


def level5(c: Ctx, rng):
    out = []
    for m, x in cf_mods(c):
        mods = (m,)
        if sim.run_errors(c.scene, mods):
            continue
        cst = sim.states(c.scene, mods)
        hid_end = sum(s == 2 for s in cst[-1])
        real = len(c.hid[-1])
        params = {"tag": m[0], "x": x}
        if m[0] == "stay":
            lead = f"Suppose the occluder{'s' if c.many else ''} had never moved from where {'they' if c.many else 'it'}" \
                   " started (everything else the same). "
        elif m[0] == "stop_early":
            lead = f"Suppose the occluder{'s' if c.many else ''} had stopped one move early, staying at the " \
                   "second-to-last position (everything else the same). "
        else:
            lead = f"Suppose {c.oname[x]} were taken away (everything else the same). "
        out.append(Q("cf_hidden_end", params, lead + "At the end of the video, how many objects would be hidden?",
                     hid_end, int_opts(rng, hid_end, 0, 60, [real]), True, fps_for([c.hold_gap])))
        vis = c.n - hid_end
        out.append(Q("cf_visible_end", params, lead + "At the end of the video, how many objects would be visible?",
                     vis, int_opts(rng, vis, 0, 60, [c.n - real]), True, fps_for([c.hold_gap])))
        if m[0] == "stay":
            nev = sum(all(cst[mm][i] == 2 for mm in c.mid) for i in range(c.n))
            out.append(Q("cf_never_seen", params, lead + "How many objects would never be visible during the video?",
                         nev, int_opts(rng, nev, 0, 60, [0]), True, fps_for([c.hold_gap])))
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
        if any(cand[2] == ch[2] for ch in chosen):
            continue
        chosen.append(cand)
        used.add(k)
        keys.remove(k)
        keys.append(k)
    return chosen


MAKERS = {1: level1, 2: lambda c, rng: level2(c, rng) + level2_more(c, rng), 3: level3, 4: level4, 5: level5}


def candidates(scene, rng):
    c = Ctx(scene)
    made = {}
    for lv in range(1, 6):
        seen, ok = set(), []
        for q in MAKERS[lv](c, random.Random(rng.random())):
            if q is None:
                continue
            if q[0] != "object_shape":               # name the (shared) shape instead of "objects"
                sh = scene["shape"]
                q = (q[0], q[1], q[2].replace("objects", PLURAL[sh]).replace("object", sh)) + q[3:]
            sig = (q[0], json.dumps(q[1], sort_keys=True))
            if sig not in seen:
                seen.add(sig)
                ok.append(q)
        made[lv] = ok
    return made


def feasible(scene) -> bool:
    made = candidates(scene, random.Random(0))
    texts = {lv: len({cand[2] for cand in made[lv]}) for lv in range(1, 6)}
    return any(all(texts[lv] >= cmb[lv - 1] + (1 if lv > 1 else 0) for lv in range(1, 6)) for cmb in TEN_SPLITS)


def build(scene: dict, rng: random.Random) -> list[dict]:
    made = candidates(scene, rng)
    texts = {lv: len({cand[2] for cand in made[lv]}) for lv in range(1, 6)}
    ok = [cmb for cmb in TEN_SPLITS if all(texts[lv] >= cmb[lv - 1] for lv in range(1, 6))]
    if not ok:
        raise ValueError("not enough question candidates for this scene")
    counts = rng.choice(ok)
    qs = []
    for lv in range(1, 6):
        for key, params, text, gt, opts, letter, motion in _pick(rng, made[lv], counts[lv - 1]):
            qs.append(L.question(lv, key, text, gt, opts, letter, motion, params, idx=len(qs) + 1))
    return qs
