"""10-question ladder for glasses_fill (common/ladder5.py, mode "ten").

Glasses are named by the number printed under them ("glass 3"), jugs by their
body pattern ("the striped jug"), liquids by colour; all three are unique in
a clip. Amounts are read off the ml scale (a tick every 10 ml on every glass,
ruler numbers every 20 ml); every amount is a multiple of 10 ml and sits on a
tick, and wrong ml options are >= 20 ml from the answer and from each other.
Times are what the on-screen clock shows; clock options are >= 0.6 s apart.
"The end of the video" means the last frame.

Every question has four distinct options and stores params["min_fps"]: the
sampling rate needed to see what it relies on, 1 / (the shortest time a
needed cue is on screen, or the smallest gap between the needed events),
rounded up and clamped to 1..24. Clock-reading questions need 4 fps.

Each template builds its candidate (or None when the clip can't support it);
the per-level counts are drawn uniformly from the valid splits the feasible
templates can fill, then each level's questions are drawn by weight without
replacement. Scenario-specific templates carry extra weight.
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

LADDER = "ten"               # common/video5.py validates the 10-question ladder
FPS = sim.FPS
CAP, STEP = sim.CAP, sim.STEP
ML_SPACING = 20              # wrong ml options are at least this far from the answer and each other
CLOCK_SPACING = 6            # tenths of a second between clock-reading options
NTH = {1: "first", 2: "second", 3: "third", 4: "fourth"}


# ------------------------------------------------------------------ helpers
def G(g) -> str:
    return f"glass {g + 1}"


def ml(v) -> str:
    return f"{v} ml"


def secs(tenths: int) -> str:
    return f"{tenths / 10:.1f} s"


def fps_for(gap) -> int:
    if gap is None or gap <= 0:
        return FPS
    return int(min(FPS, max(1, math.ceil(1.0 / gap - 1e-9))))


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


def spread(rng, gt: int, pool, spacing: int, prefer=()) -> list[int] | None:
    """Three values from `pool`, >= spacing from gt and from each other, with gt's
    rank among the four drawn uniformly (other ranks as fallback). Values from
    `prefer` (other real readings in the clip) are tried first."""
    pool = sorted(set(pool))
    rank = rng.randrange(4)
    for r in [rank] + [x for x in range(4) if x != rank]:
        out = []
        for want, sign in ((r, -1), (3 - r, 1)):
            side = [v for v in pool if (v - gt) * sign >= spacing]
            cand = [v for v in prefer if v in side]
            rng.shuffle(cand)
            near = sorted(side, key=lambda v: abs(v - gt))[:max(want * 3, 1)]
            rng.shuffle(near)
            got = []
            for v in cand + near:
                if len(got) == want:
                    break
                if all(abs(v - u) >= spacing for u in got + out + [gt]):
                    got.append(v)
            if len(got) < want:
                break
            out += got
        else:
            return out
    return None


def ml_opts(rng, gt, hi=CAP, prefer=()):
    d = spread(rng, gt, range(0, hi + 1, STEP), ML_SPACING, prefer)
    return None if d is None else mc4(rng, gt, d, ml)


def count_opts(rng, gt, lo, hi, prefer=()):
    d = spread(rng, gt, range(lo, hi + 1), 1, prefer)
    return None if d is None else mc4(rng, gt, d)


def Q(key, text, gt, opts, motion, params, fps):
    return {"key": key, "text": text, "gt": gt, "opts": opts, "motion": motion, "params": dict(params, min_fps=fps)}


class Ctx:
    """Everything the templates read, from the drawn track (via sim's state functions)."""

    def __init__(self, scene):
        self.s = scene
        self.n = scene["glasses"]["n"]
        self.jugs, self.pours, self.leaks = scene["jugs"], scene["pours"], scene["leaks"]
        self.end_t = (scene["n_frames"] - 1) / FPS
        self.final_layers = [[[c, int(round(m))] for c, m in sim.glass_layers(scene, g, self.end_t)]
                             for g in range(self.n)]
        self.final = [sum(m for _, m in lay) for lay in self.final_layers]
        self.peak = [sim.peak_ml(scene, g) for g in range(self.n)]
        self.cracked = {lk["glass"]: lk for lk in self.leaks}
        self.used_jugs = sorted({p["jug"] for p in self.pours})

    def J(self, j) -> str:
        return f"the {self.jugs[j]['pattern']} jug"

    def jug_word(self, pattern) -> str:
        return f"the {pattern} jug"

    def jug_names(self, rng, j):
        """Pattern options: the other jugs in the clip first, then patterns not shown."""
        others = [jg["pattern"] for k, jg in enumerate(self.jugs) if k != j]
        rng.shuffle(others)
        extra = [p for p in sim.PATTERNS if p not in {jg["pattern"] for jg in self.jugs}]
        rng.shuffle(extra)
        return others + extra

    def colour_names(self, rng, colour):
        others = [jg["colour"] for jg in self.jugs if jg["colour"] != colour]
        rng.shuffle(others)
        extra = [c for c in sorted(sim.LIQUIDS) if c not in {jg["colour"] for jg in self.jugs}]
        rng.shuffle(extra)
        return others + extra

    def glass_names(self, rng, g, prefer=()):
        pref = [x for x in prefer if x != g]
        rng.shuffle(pref)
        rest = [x for x in range(self.n) if x != g and x not in pref]
        rng.shuffle(rest)
        return pref + rest

    def changes(self, g):
        """Time intervals [a, b] during which glass g's level changes."""
        out = [(p["t_pour"], p["t_flow_end"]) for p in self.pours if p["glass"] == g and p["got_ml"]]
        out += [(lk["t_crack"], lk["t_drain_end"]) for lk in self.leaks if lk["glass"] == g]
        return sorted(out)

    def settled(self, gs):
        """How long the end state of glasses gs has been on screen."""
        last = max([b for g in gs for _, b in self.changes(g)] + [0.0])
        return self.end_t - last

    def level(self, g, t) -> float:
        return sum(m for _, m in sim.glass_layers(self.s, g, t))

    def tilt_gap(self):
        """Shortest tilted phase or gap between two tilts (for questions that count/order tilts)."""
        gaps = [p["t_back"] - p["t_tilt"] for p in self.pours]
        gaps += [b["t_tilt"] - a["t_back"] for a, b in zip(self.pours, self.pours[1:])]
        return min(gaps)

    def when(self, i) -> str:
        """'' if jug and glass of pour i meet only once, else ' the second time' etc."""
        p = self.pours[i]
        same = [k for k, q in enumerate(self.pours) if q["jug"] == p["jug"] and q["glass"] == p["glass"]]
        return "" if len(same) == 1 else f" the {NTH[same.index(i) + 1]} time"


# ------------------------------------------------------------------ L1 observation
def t_n_glasses(cx, rng):
    return Q("n_glasses", "How many glasses stand on the table?", cx.n, count_opts(rng, cx.n, 2, 12), False, {}, 1)


def t_n_jugs(cx, rng):
    k = len(cx.jugs)
    return Q("n_jugs", "How many jugs stand on the shelf at the start of the video?", k,
             count_opts(rng, k, 1, 7), False, {}, fps_for(cx.pours[0]["t_move"]))


def t_jug_colour(cx, rng):
    wet = [j for j, jg in enumerate(cx.jugs) if jg["start_ml"] >= 40]   # colour clearly visible
    j = rng.choice(wet)
    c = cx.jugs[j]["colour"]
    return Q("jug_colour", f"What colour is the liquid in {cx.J(j)} at the start of the video?", c,
             mc4(rng, c, cx.colour_names(rng, c)), False, {"jug": j}, fps_for(cx.pours[0]["t_move"]))


def t_jug_of_colour(cx, rng):
    wet = [j for j, jg in enumerate(cx.jugs) if jg["start_ml"] >= 40]   # colour clearly visible
    j = rng.choice(wet)
    pat = cx.jugs[j]["pattern"]
    return Q("jug_of_colour", f"Which jug holds the {cx.jugs[j]['colour']} liquid at the start of the video?",
             cx.jug_word(pat), mc4(rng, pat, cx.jug_names(rng, j), cx.jug_word), False,
             {"colour": cx.jugs[j]["colour"]}, fps_for(cx.pours[0]["t_move"]))


def t_first_jug(cx, rng):
    j = cx.pours[0]["jug"]
    pat = cx.jugs[j]["pattern"]
    return Q("first_jug", "Which jug is the first to tilt over a glass?", cx.jug_word(pat),
             mc4(rng, pat, cx.jug_names(rng, j), cx.jug_word), True, {}, fps_for(cx.tilt_gap()))


def t_dry_jug(cx, rng):
    dry = [i for i, p in enumerate(cx.pours) if p["got_ml"] == 0]
    if len(dry) != 1:
        return None
    p = cx.pours[dry[0]]
    pat = cx.jugs[p["jug"]]["pattern"]
    return Q("dry_jug", "One jug tilts over a glass but nothing comes out of it. Which jug is it?",
             cx.jug_word(pat), mc4(rng, pat, cx.jug_names(rng, p["jug"]), cx.jug_word), True, {},
             fps_for(p["t_untilt"] - p["t_pour"]))


def t_cracked_one(cx, rng):
    if len(cx.leaks) != 1:
        return None
    g = cx.leaks[0]["glass"]
    return Q("cracked_one", "One glass cracks and liquid leaks out of it. Which glass is it?", G(g),
             mc4(rng, g, cx.glass_names(rng, g, [x for x in range(cx.n) if cx.peak[x]]), G), False, {},
             fps_for(cx.end_t - cx.leaks[0]["t_crack"]))


def t_n_cracked(cx, rng):
    if not cx.leaks:
        return None
    k = len(cx.leaks)
    return Q("n_cracked", "How many glasses crack and leak during the video?", k, count_opts(rng, k, 0, 4), False,
             {}, fps_for(cx.end_t - max(lk["t_crack"] for lk in cx.leaks)))


def t_unused_jug(cx, rng):
    idle = [j for j in range(len(cx.jugs)) if j not in cx.used_jugs]
    if len(idle) != 1:
        return None
    pat = cx.jugs[idle[0]]["pattern"]
    return Q("unused_jug", "Which jug never leaves the shelf?", cx.jug_word(pat),
             mc4(rng, pat, cx.jug_names(rng, idle[0]), cx.jug_word), True, {}, fps_for(cx.tilt_gap()))


# ------------------------------------------------------------------ L2 relational
def t_most_end(cx, rng):
    order = sorted(range(cx.n), key=lambda g: -cx.final[g])
    a, b = order[0], order[1] if cx.n > 1 else None
    if cx.final[a] == 0 or (b is not None and (cx.final[a] - cx.final[b] < ML_SPACING
                                               or cx.final[a] - cx.final[b] < 0.1 * cx.final[a])):
        return None
    return Q("most_end", "At the end of the video, which glass holds the most liquid?", G(a),
             mc4(rng, a, cx.glass_names(rng, a, order[1:4]), G), False, {}, fps_for(cx.settled(range(cx.n))))


def t_first_target(cx, rng):
    j = rng.choice(cx.used_jugs)
    i = next(k for k, p in enumerate(cx.pours) if p["jug"] == j)
    g = cx.pours[i]["glass"]
    p = cx.pours[i]
    return Q("first_target", f"On its first tilt, over which glass is {cx.J(j)} held?", G(g),
             mc4(rng, g, cx.glass_names(rng, g, [q["glass"] for q in cx.pours]), G), True, {"jug": j},
             fps_for(p["t_back"] - p["t_tilt"]))


def _layer_q(cx, rng, key, pos):
    multi = [g for g in range(cx.n) if len(cx.final_layers[g]) >= 2]
    if not multi:
        return None
    g = rng.choice(multi)
    c = cx.final_layers[g][pos][0]
    word = "BOTTOM" if pos == 0 else "TOP"
    return Q(key, f"At the end of the video, what colour is the {word} layer of liquid in {G(g)}?", c,
             mc4(rng, c, cx.colour_names(rng, c)), False, {"glass": g}, fps_for(cx.settled([g])))


def t_bottom_layer(cx, rng):
    return _layer_q(cx, rng, "bottom_layer", 0)


def t_top_layer(cx, rng):
    return _layer_q(cx, rng, "top_layer", -1)


def t_jug_into(cx, rng):
    one = [g for g in range(cx.n) if len({p["jug"] for p in cx.pours if p["glass"] == g and p["got_ml"]}) == 1]
    if not one:
        return None
    g = rng.choice(one)
    ps = [p for p in cx.pours if p["glass"] == g and p["got_ml"]]
    j = ps[0]["jug"]
    pat = cx.jugs[j]["pattern"]
    return Q("jug_into", f"Which jug pours the liquid that goes into {G(g)}?", cx.jug_word(pat),
             mc4(rng, pat, cx.jug_names(rng, j), cx.jug_word), True, {"glass": g},
             fps_for(min(p["t_flow_end"] - p["t_pour"] for p in ps)))


def t_jug_after(cx, rng):
    cands = [j for j in cx.used_jugs
             if next(k for k, p in enumerate(cx.pours) if p["jug"] == j) < len(cx.pours) - 1]
    if not cands:
        return None
    j = rng.choice(cands)
    i = next(k for k, p in enumerate(cx.pours) if p["jug"] == j)
    nj = cx.pours[i + 1]["jug"]
    pat = cx.jugs[nj]["pattern"]
    return Q("jug_after", f"Which jug tilts over a glass next, right after {cx.J(j)} tilts for the first time?",
             cx.jug_word(pat), mc4(rng, pat, cx.jug_names(rng, nj), cx.jug_word), True, {"jug": j},
             fps_for(cx.tilt_gap()))


def t_first_glass(cx, rng):
    got = []
    for p in cx.pours:
        if p["got_ml"] and p["glass"] not in got:
            got.append(p["glass"])
    if len(got) < 2:
        return None
    shown = rng.sample(got, min(len(got), 4))
    g = min(shown, key=got.index)
    rest = [x for x in shown if x != g] + cx.glass_names(rng, g)
    return Q("first_glass", "Which of these glasses is the first to receive any liquid?", G(g),
             mc4(rng, g, rest, G), True, {"cands": sorted(shown)}, fps_for(cx.tilt_gap()))


def t_ran_dry(cx, rng):
    part = [i for i, p in enumerate(cx.pours) if 0 < p["got_ml"] < p["ask_ml"]]
    if len(part) != 1:
        return None
    p = cx.pours[part[0]]
    pat = cx.jugs[p["jug"]]["pattern"]
    return Q("ran_dry", "Which jug runs out of liquid while it is still tilted over a glass, so the stream stops "
             "before the jug turns back up?", cx.jug_word(pat),
             mc4(rng, pat, cx.jug_names(rng, p["jug"]), cx.jug_word), True, {},
             fps_for(min(p["t_untilt"] - p["t_flow_end"], p["t_flow_end"] - p["t_pour"])))


def t_jug_more(cx, rng):
    tot = {j: sum(p["got_ml"] for p in cx.pours if p["jug"] == j) for j in cx.used_jugs}
    order = sorted(tot, key=lambda j: -tot[j])
    if len(order) < 2 or tot[order[0]] - tot[order[1]] < ML_SPACING or tot[order[0]] - tot[order[1]] < 0.1 * tot[order[0]]:
        return None
    j = order[0]
    pat = cx.jugs[j]["pattern"]
    ps = [p for p in cx.pours if p["got_ml"]]
    return Q("jug_more", "Which jug pours the most liquid in total over the whole video?", cx.jug_word(pat),
             mc4(rng, pat, [cx.jugs[k]["pattern"] for k in order[1:]] + cx.jug_names(rng, j), cx.jug_word), True, {},
             fps_for(min(p["t_flow_end"] - p["t_pour"] for p in ps)))


def t_last_glass(cx, rng):
    got = []
    for p in cx.pours:
        if p["got_ml"]:
            got = [g for g in got if g != p["glass"]] + [p["glass"]]
    if len(got) < 2:
        return None
    g = got[-1]
    return Q("last_glass", "Which glass is the last one to receive liquid in the video?", G(g),
             mc4(rng, g, cx.glass_names(rng, g, got[:-1]), G), True, {}, fps_for(cx.tilt_gap()))


# ------------------------------------------------------------------ L3 measurement
def t_level_end(cx, rng):
    g = rng.choice([x for x in range(cx.n) if cx.final[x]] or range(cx.n))
    return Q("level_end", f"Using the scale, how much liquid is in {G(g)} at the end of the video?", ml(cx.final[g]),
             ml_opts(rng, cx.final[g], prefer=cx.final), False, {"glass": g}, fps_for(cx.settled([g])))


def t_layer_ml(cx, rng):
    cands = [(g, c) for g in range(cx.n) if len(cx.final_layers[g]) >= 2 for c, _ in cx.final_layers[g]
             if sum(1 for cc, _ in cx.final_layers[g] if cc == c) == 1]
    if not cands:
        return None
    g, c = rng.choice(cands)
    v = next(m for cc, m in cx.final_layers[g] if cc == c)
    return Q("layer_ml", f"At the end of the video, how many ml of {c} liquid are in {G(g)}?", ml(v),
             ml_opts(rng, v, prefer=[m for _, m in cx.final_layers[g]]), False, {"glass": g, "colour": c},
             fps_for(cx.settled([g])))


def t_crack_mark(cx, rng):
    if not cx.leaks:
        return None
    lk = rng.choice(cx.leaks)
    v = lk["hole_ml"]
    return Q("crack_mark", f"{G(lk['glass']).capitalize()} cracks. At which ml mark of the scale is the crack?",
             ml(v), ml_opts(rng, v, prefer=[cx.peak[lk['glass']]]), False, {"glass": lk["glass"]},
             fps_for(cx.end_t - lk["t_crack"]))


def t_n_tilts(cx, rng):
    k = len(cx.pours)
    return Q("n_tilts", "How many times in total does a jug tilt over a glass to pour (count every tilt, even if "
             "nothing comes out)?", k, count_opts(rng, k, 1, 8), True, {}, fps_for(cx.tilt_gap()))


def t_n_empty_end(cx, rng):
    k = sum(1 for v in cx.final if v == 0)
    return Q("n_empty_end", "How many glasses are completely empty at the end of the video?", k,
             count_opts(rng, k, 0, cx.n), False, {}, fps_for(cx.settled(range(cx.n))))


def t_pour_ml(cx, rng):
    cands = [i for i, p in enumerate(cx.pours) if p["got_ml"]]
    i = rng.choice(cands)
    p = cx.pours[i]
    g = p["glass"]
    ch = cx.changes(g)
    k = ch.index((p["t_pour"], p["t_flow_end"]))
    before = p["t_pour"] - (ch[k - 1][1] if k else 0.0)
    after = (ch[k + 1][0] if k + 1 < len(ch) else cx.end_t) - p["t_flow_end"]
    v = p["got_ml"]
    return Q("pour_ml", f"How many ml does {cx.J(p['jug'])} pour into {G(g)}{cx.when(i)}?", ml(v),
             ml_opts(rng, v, prefer=[q["got_ml"] for q in cx.pours]), True, {"pour": i},
             fps_for(min(before, after)))


def t_level_at_clock(cx, rng):
    """Level of a glass at a clock reading inside a >= 1.2 s stretch where it doesn't change,
    different from its end level (so the answer can't be read off the last frame)."""
    cands = []
    for g in range(cx.n):
        ch = cx.changes(g)
        bounds = [0.0] + [x for ab in ch for x in ab] + [cx.end_t]
        for a, b in zip(bounds[0::2], bounds[1::2]):
            if b - a < 1.2:
                continue
            lv = int(round(cx.level(g, (a + b) / 2)))
            if lv == 0 or lv == cx.final[g]:
                continue
            for tc in range(math.ceil((a + 0.5) * 10), math.floor((b - 0.5) * 10) + 1):
                f0 = math.ceil(tc * FPS / 10 - 1e-9)           # first frame showing this reading
                f1 = math.ceil((tc + 1) * FPS / 10 - 1e-9) - 1
                if a + 0.5 <= f0 / FPS and f1 / FPS <= b - 0.5:
                    cands.append((g, tc, lv, b - a))
    if not cands:
        return None
    g, tc, lv, span = rng.choice(cands)
    return Q("level_at_clock", f"When the clock shows {secs(tc)}, how much liquid is in {G(g)}?", ml(lv),
             ml_opts(rng, lv, prefer=[cx.final[g]]), True, {"glass": g, "clock": tc}, fps_for(1.0))


def t_pour_clock(cx, rng):
    """When liquid first flows out of a jug that pours liquid exactly once."""
    cands = [i for i, p in enumerate(cx.pours) if p["got_ml"]
             and sum(q["jug"] == p["jug"] and q["got_ml"] > 0 for q in cx.pours) == 1]
    if not cands:
        return None
    i = rng.choice(cands)
    p = cx.pours[i]
    fs = math.floor(p["t_pour"] * FPS + 1e-9) + 1                 # first frame with the stream
    tc = sim.clock_tenths(fs)
    others = [sim.clock_tenths(math.floor(q["t_pour"] * FPS + 1e-9) + 1) for q in cx.pours if q is not p]
    d = spread(rng, tc, range(5, int(cx.end_t * 10) - 4), CLOCK_SPACING, others)
    return Q("pour_clock", f"At about what time on the clock does liquid first start flowing out of {cx.J(p['jug'])}?",
             secs(tc), None if d is None else mc4(rng, tc, d, secs), True, {"jug": p["jug"]}, 4)


# ------------------------------------------------------------------ L4 integration
def t_total_end(cx, rng):
    tot = sum(cx.final)
    return Q("total_end", "At the end of the video, how many ml of liquid are there in all the glasses together?",
             ml(tot), ml_opts(rng, tot, hi=cx.n * CAP, prefer=[sum(cx.peak)]), False, {},
             fps_for(cx.settled(range(cx.n))))


def t_jug_total(cx, rng):
    cands = [j for j in cx.used_jugs if sum(p["got_ml"] > 0 for p in cx.pours if p["jug"] == j) >= 1
             and not any(p["glass"] in cx.cracked for p in cx.pours if p["jug"] == j)]
    if not cands:
        return None
    j = rng.choice(cands)
    tot = sum(p["got_ml"] for p in cx.pours if p["jug"] == j)
    ps = [p for p in cx.pours if p["jug"] == j and p["got_ml"]]
    return Q("jug_total", f"How many ml does {cx.J(j)} pour in total, over all of its pours?", ml(tot),
             ml_opts(rng, tot, hi=sim.JUG_CAP, prefer=[p["got_ml"] for p in cx.pours]), True, {"jug": j},
             fps_for(min(p["t_flow_end"] - p["t_pour"] for p in ps)))


def t_leak_loss(cx, rng):
    if not cx.leaks:
        return None
    lk = rng.choice(cx.leaks)
    g = lk["glass"]
    v = cx.peak[g] - lk["hole_ml"]
    return Q("leak_loss", f"{G(g).capitalize()} cracks. How many ml of liquid leak out of it?", ml(v),
             ml_opts(rng, v, prefer=[cx.peak[g], lk["hole_ml"]]), True, {"glass": g},
             fps_for(min(lk["t_crack"] - max(b for _, b in cx.changes(g) if b <= lk["t_crack"]),
                         cx.end_t - lk["t_drain_end"])))


def t_diff_end(cx, rng):
    pairs = [(a, b) for a in range(cx.n) for b in range(cx.n) if cx.final[a] - cx.final[b] >= STEP]
    if not pairs:
        return None
    a, b = rng.choice(pairs)
    v = cx.final[a] - cx.final[b]
    return Q("diff_end", f"At the end of the video, how many more ml of liquid does {G(a)} hold than {G(b)}?", ml(v),
             ml_opts(rng, v, prefer=[cx.final[a], cx.final[b]]), False, {"a": a, "b": b},
             fps_for(cx.settled([a, b])))


def t_n_received(cx, rng):
    k = len({p["glass"] for p in cx.pours if p["got_ml"]})
    gone = any(cx.final[g] == 0 for g in cx.cracked)
    return Q("n_received", "How many different glasses receive liquid at some point during the video?", k,
             count_opts(rng, k, 1, cx.n), True, {}, fps_for(cx.tilt_gap()) if gone else fps_for(cx.settled(range(cx.n))))


def t_before_last(cx, rng):
    cands = [g for g in range(cx.n) if sum(1 for p in cx.pours if p["glass"] == g and p["got_ml"]) >= 2]
    if not cands:
        return None
    g = rng.choice(cands)
    last = [p for p in cx.pours if p["glass"] == g and p["got_ml"]][-1]
    v = int(round(cx.level(g, last["t_pour"])))
    ch = cx.changes(g)
    k = ch.index((last["t_pour"], last["t_flow_end"]))
    return Q("before_last", f"How much liquid is in {G(g)} just before the last pour into it begins?", ml(v),
             ml_opts(rng, v, prefer=[cx.final[g], cx.peak[g]]), True, {"glass": g},
             fps_for(last["t_pour"] - ch[k - 1][1]))


# ------------------------------------------------------------------ L5 counterfactual
def t_cf_no_crack(cx, rng):
    if not cx.leaks:
        return None
    lk = rng.choice(cx.leaks)
    g = lk["glass"]
    v = cx.peak[g]
    return Q("cf_no_crack", f"If {G(g)} had NOT cracked, how much liquid would it hold at the end of the video?",
             ml(v), ml_opts(rng, v, prefer=[cx.final[g]]), True, {"glass": g},
             fps_for(lk["t_crack"] - max(b for _, b in cx.changes(g) if b <= lk["t_crack"])))


def t_cf_no_crack_total(cx, rng):
    if not cx.leaks:
        return None
    v = sum(cx.peak)
    return Q("cf_no_crack_total", "If no glass had cracked, how many ml of liquid would there be in all the glasses "
             "together at the end of the video?", ml(v), ml_opts(rng, v, hi=cx.n * CAP, prefer=[sum(cx.final)]),
             True, {}, fps_for(min(lk["t_crack"] - max(b for _, b in cx.changes(lk["glass"]) if b <= lk["t_crack"])
                                   for lk in cx.leaks)))


def _resim(cx, skip=None, redirect=None):
    """Final level of every glass when pour `skip` doesn't happen or pour redirect[0]
    goes into glass redirect[1]; the jugs keep what they don't pour."""
    left = [jg["start_ml"] for jg in cx.jugs]
    lev = [0] * cx.n
    for i, p in enumerate(cx.pours):
        if i == skip:
            continue
        g = redirect[1] if redirect and redirect[0] == i else p["glass"]
        got = min(p["ask_ml"], left[p["jug"]])
        left[p["jug"]] -= got
        lev[g] += got
    for g, lk in cx.cracked.items():
        lev[g] = min(lev[g], lk["hole_ml"])
    return lev


def t_cf_skip(cx, rng):
    cands = [i for i, p in enumerate(cx.pours) if p["got_ml"]]
    i = rng.choice(cands)
    g = cx.pours[i]["glass"]
    v = _resim(cx, skip=i)[g]
    if v == cx.final[g]:
        return None
    return Q("cf_skip", f"If {cx.J(cx.pours[i]['jug'])} had not poured into {G(g)}{cx.when(i)}, with everything else "
             f"the same, how much liquid would {G(g)} hold at the end of the video?", ml(v), ml_opts(rng, v, prefer=[cx.final[g]]), True,
             {"pour": i, "glass": g}, fps_for(cx.pours[i]["t_flow_end"] - cx.pours[i]["t_pour"]))


def t_cf_redirect(cx, rng):
    cands = []
    for i, p in enumerate(cx.pours):
        if not p["got_ml"]:
            continue
        for b in range(cx.n):
            if b != p["glass"] and b not in cx.cracked and cx.final[b] + p["got_ml"] <= CAP:
                cands.append((i, b))
    if not cands:
        return None
    i, b = rng.choice(cands)
    v = _resim(cx, redirect=(i, b))[b]
    p = cx.pours[i]
    return Q("cf_redirect", f"Suppose {cx.J(p['jug'])} had poured into {G(b)} instead of {G(p['glass'])}{cx.when(i)}, "
             f"with everything else the same. How much liquid would {G(b)} hold at the end of the video?", ml(v),
             ml_opts(rng, v, prefer=[cx.final[b], cx.pours[i]["got_ml"]]), True, {"pour": i, "glass": b},
             fps_for(cx.pours[i]["t_flow_end"] - cx.pours[i]["t_pour"]))


def t_cf_swap(cx, rng):
    cands = [g for g in range(cx.n) if g not in cx.cracked and len(cx.final_layers[g]) == 2
             and sum(1 for p in cx.pours if p["glass"] == g and p["got_ml"]) == 2]
    if not cands:
        return None
    g = rng.choice(cands)
    c = cx.final_layers[g][1][0]
    return Q("cf_swap", f"{G(g).capitalize()} is poured into twice. If those two pours had happened in the opposite "
             "order, what colour would its BOTTOM layer be at the end of the video?", c,
             mc4(rng, c, cx.colour_names(rng, c)), True, {"glass": g}, fps_for(cx.tilt_gap()))


LEVELS = {
    1: [(1, t_n_glasses), (1, t_n_jugs), (2, t_jug_colour), (2, t_jug_of_colour), (2, t_first_jug),
        (4, t_dry_jug), (4, t_cracked_one), (2, t_n_cracked), (2, t_unused_jug)],
    2: [(1, t_most_end), (3, t_first_target), (2, t_bottom_layer), (2, t_top_layer), (3, t_jug_into),
        (3, t_jug_after), (3, t_first_glass), (5, t_ran_dry), (2, t_jug_more), (2, t_last_glass)],
    3: [(3, t_level_end), (3, t_layer_ml), (3, t_crack_mark), (2, t_n_tilts), (1, t_n_empty_end), (3, t_pour_ml),
        (3, t_level_at_clock), (3, t_pour_clock)],
    4: [(1, t_total_end), (3, t_jug_total), (4, t_leak_loss), (1, t_diff_end), (2, t_n_received), (4, t_before_last)],
    5: [(4, t_cf_no_crack), (2, t_cf_no_crack_total), (3, t_cf_skip), (3, t_cf_redirect), (4, t_cf_swap)],
}
TEN_SPLITS = [c for c in itertools.product(*(range(lo, hi + 1) for lo, hi in L.TEN_RANGES)) if sum(c) == L.TEN_TOTAL]


def build(scene: dict, rng: random.Random) -> list[dict]:
    cx = Ctx(scene)
    pools = {}
    for lv, temps in LEVELS.items():
        pools[lv] = []
        for w, fn in temps:
            q = fn(cx, rng)
            if q is not None and q["opts"] is not None:
                pools[lv].append((w, q))
    combos = [c for c in TEN_SPLITS if all(c[lv - 1] <= len(pools[lv]) for lv in range(1, 6))]
    if not combos:
        raise RuntimeError("not enough feasible questions for a 10-question ladder")
    counts = rng.choice(combos)
    picked = []
    for lv in range(1, 6):
        pool = pools[lv][:]
        for _ in range(counts[lv - 1]):
            r = rng.random() * sum(w for w, _ in pool)
            idx = len(pool) - 1
            for k, (w, _) in enumerate(pool):
                r -= w
                if r < 0:
                    idx = k
                    break
            picked.append((lv, pool.pop(idx)[1]))
    qs = []
    for k, (lv, q) in enumerate(picked, start=1):
        opts, letter = q["opts"]
        qs.append(L.question(lv, q["key"], q["text"], q["gt"], opts, letter, q["motion"], q["params"], idx=k))
    return qs
