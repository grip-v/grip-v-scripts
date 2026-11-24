"""10-question ladder for symmetry_pattern_video (common/ladder5.py, LADDER = "ten").

Shapes of one orbit look identical by design (that is what makes the
pattern symmetric), so questions name a whole orbit by colour + outline
("the red kites"; no two orbits share both), the odd shape by its role
("the shape that breaks the symmetry"), places by compass direction from
the centre ("top-left"; only asked >= 10 deg from a sector boundary) and
moments by the clock. Every question has 4 options and stores
derivation_key + params + min_fps.
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
ORD = ["first", "second", "third", "fourth", "fifth"]
TURN_OPTS = ["45°", "60°", "72°", "90°", "120°", "180°", "no turn short of a full turn"]
MOVE_NAMES = {("turn", 1): "a counter-clockwise turn", ("turn", -1): "a clockwise turn",
              ("flip", "vertical"): "a left-right flip", ("flip", "horizontal"): "an upside-down flip"}
OCT_MARGIN = 10.0


def fps_for(gap_frames: float) -> int:
    gap = gap_frames / sim.FPS
    for f in FPS_STEPS:
        if f * gap >= 1 - 1e-9:
            return f
    return 24


def turn_label(order: int) -> str:
    return TURN_OPTS[-1] if order == 1 else f"{sim.min_turn(order)}°"


class Facts:
    def __init__(self, scene):
        self.scene = scene
        self.sc = scene["scenario"]
        self.shapes = scene["shapes"]
        self.n = len(self.shapes)
        self.nf = scene["n_frames"]
        self.n_orb = 1 + max(s["orbit"] for s in scene["base"])
        self.orb = [next(s for s in scene["base"] if s["orbit"] == o) for o in range(self.n_orb)]
        self.br = scene["break"]
        self.order, self.mirrors = sim.detect(self.shapes)
        drawn = {(s["color"], s["type"]) for s in self.shapes}
        real = {(o["color"], o["type"]) for o in self.orb}
        cols = sorted({o["color"] for o in self.orb})
        types = sorted({o["type"] for o in self.orb})
        self.confusers = [f"the {c} {sim.PLURAL[t]}" for c in cols for t in types
                          if (c, t) not in drawn and (c, t) not in real]

    def members(self, o) -> list[int]:
        """Shapes drawn with group o's colour and outline (an odd shape with
        another colour or outline belongs to no group)."""
        c, t = self.orb[o]["color"], self.orb[o]["type"]
        return [i for i, s in enumerate(self.shapes) if s["color"] == c and s["type"] == t]

    def gparam(self, o):
        return [self.orb[o]["color"], self.orb[o]["type"]]

    def odd_ok(self) -> bool:
        """The odd shape can be pointed out (groups of 3 or more copies)."""
        return bool(self.br) and len(sim.group_elements(self.scene["group"])) >= 3

    def ref(self):
        """Shapes of the 'finished picture': at the start for turns, at the end otherwise."""
        return self.shapes

    def pic(self) -> str:
        return "the picture as it is at the start" if self.sc == "turns" else "the finished picture"

    def gname(self, o) -> str:
        return f"the {self.orb[o]['color']} {sim.PLURAL[self.orb[o]['type']]}"

    def group_opts(self, rng, correct_o):
        pool = [self.gname(o) for o in rng.sample(range(self.n_orb), self.n_orb) if o != correct_o]
        conf = self.confusers[:]
        rng.shuffle(conf)
        return _four(rng, self.gname(correct_o), pool + conf)


def _q(lv, key, text, gt, opts, a, motion, params):
    return L.question(lv, key, text, gt, opts, a, motion, params)


def _four(rng, correct, pool):
    ds = [d for d in dict.fromkeys(pool) if d != correct]
    if len(ds) < 3:
        return None
    return L.mc(rng, correct, ds[:3])


def _ints(rng, gt, lo=0, hi=99):
    pool = L.near_ints(gt, lo, hi)
    rng.shuffle(pool)
    return _four(rng, gt, pool)


def _clock(rng, k, hi):
    cands = sorted((k + d for d in (-15, -10, -5, 5, 10, 15) if 0 <= k + d <= hi), key=lambda c: abs(c - k))
    return _four(rng, sim.clock_text(k), [sim.clock_text(c) for c in cands[:4]])


def _oct_opts(rng, correct):
    i = sim.OCTANTS.index(correct)
    near = [sim.OCTANTS[(i + d) % 8] for d in (1, -1, 4, 2, -2)]
    return _four(rng, correct, near)


# ---------------------------------------------------------------- L1 observation

def q_n_shapes(F, rng):
    o = _ints(rng, F.n, 2, 30)
    return o and _q(1, "n_shapes", "How many shapes are in the picture at the end of the video?", F.n, *o, False,
                    {"min_fps": 1})


def q_n_type(F, rng):
    t = rng.choice(sorted({s["type"] for s in F.shapes}))
    gt = sum(1 for s in F.shapes if s["type"] == t)
    o = _ints(rng, gt, 0, 30)
    return o and _q(1, "n_type", f"How many {sim.PLURAL[t]} are in the picture at the end of the video?", gt, *o, False,
                    {"type": t, "min_fps": 1})


def q_n_colors(F, rng):
    gt = len({s["color"] for s in F.shapes})
    o = _ints(rng, gt, 1, 10)
    return o and _q(1, "n_colors", "How many different colours do the shapes have at the end of the video?", gt, *o,
                    False, {"min_fps": 1})


def q_n_waves(F, rng):
    if F.sc != "waves":
        return None
    tl = F.scene["timeline"]
    o = _ints(rng, F.n_orb, 1, 8)
    return o and _q(1, "n_waves", "The picture is drawn in waves: a group of shapes is traced at the same time, "
                    "then the next group. How many waves are there?", F.n_orb, *o, True,
                    {"min_fps": fps_for(tl["starts"][1] - tl["starts"][0] - tl["trace"])})


def q_first_group(F, rng):
    tl = F.scene["timeline"]
    if F.sc == "waves":
        o, text, gap = 0, "Which group of shapes is drawn first?", tl["starts"][1] - tl["starts"][0]
    elif F.sc == "reveal":
        first = min(range(F.n), key=lambda i: tl["appear"][i])
        o = next((k for k in range(F.n_orb) if first in F.members(k)), None)
        if o is None:
            return None
        ap = sorted(tl["appear"])
        text, gap = "The first shape to appear belongs to which group?", ap[1] - ap[0]
    elif F.sc == "assemble":
        o = min(range(F.n_orb), key=lambda k: tl["starts"][k])
        ss = sorted(tl["starts"])
        text, gap = "Which group of shapes is the first to start moving?", ss[1] - ss[0]
    else:
        return None
    opts = F.group_opts(rng, o)
    return opts and _q(1, "first_group", text, F.gname(o), *opts, True, {"min_fps": fps_for(gap)})


def q_n_moves(F, rng):
    if F.sc != "turns":
        return None
    ms = F.scene["timeline"]["moves"]
    kind = rng.choice(["moves", "flips"])
    if kind == "moves":
        gt, text = len(ms), "The disc moves several times, pausing in between. How many separate moves does it make?"
    else:
        gt, text = sum(1 for m in ms if m["kind"] == "flip"), "How many times is the disc flipped over?"
    o = _ints(rng, gt, 0, 8)
    gaps = [b["f0"] - a["f1"] for a, b in zip(ms, ms[1:])]
    return o and _q(1, "n_moves", text, gt, *o, True, {"what": kind, "min_fps": fps_for(min(gaps))})


# ---------------------------------------------------------------- L2 relational

def q_break_kind(F, rng):
    kind = F.br["kind"] if F.br else None
    allowed = list(sim.BREAKS[F.sc]) + [None]
    pool = [sim.BREAK_NAMES[k] for k in rng.sample(allowed, len(allowed)) if k != kind]
    o = _four(rng, sim.BREAK_NAMES[kind], pool)
    motion = F.sc == "waves"
    fps = fps_for(F.scene["timeline"]["trace"] / 2) if motion else 1
    return o and _q(2, "break_kind", "Compare every shape with its symmetric partners, including how it is drawn. "
                    "Which is true?", sim.BREAK_NAMES[kind], *o, motion, {"min_fps": fps})


def q_group_wave(F, rng):
    if F.sc != "waves":
        return None
    o = rng.randrange(F.n_orb)
    opts = _four(rng, ORD[o], [x for x in ORD[:max(4, F.n_orb)] if x != ORD[o]])
    tl = F.scene["timeline"]
    return opts and _q(2, "group_wave", f"In which wave are {F.gname(o)} drawn?", ORD[o], *opts, True,
                       {"g": F.gparam(o), "min_fps": fps_for(tl["starts"][1] - tl["starts"][0] - tl["trace"])})


def q_last_group(F, rng):
    tl = F.scene["timeline"]
    if F.sc == "reveal":
        done = [max(tl["appear"][i] for i in F.members(o)) for o in range(F.n_orb)]
        text = "Which group is the last to have all of its shapes on screen?"
    elif F.sc == "assemble":
        done = sim.settle_frames(F.scene)
        text = "Which group of shapes is the last to settle into place?"
    else:
        return None
    o = max(range(F.n_orb), key=lambda k: done[k])
    ds = sorted(done)
    opts = F.group_opts(rng, o)
    return opts and _q(2, "last_group", text, F.gname(o), *opts, True, {"min_fps": fps_for(ds[-1] - ds[-2])})


def q_kth_move(F, rng):
    if F.sc != "turns":
        return None
    ms = F.scene["timeline"]["moves"]
    k = rng.randrange(len(ms))
    m = ms[k]
    gt = MOVE_NAMES[("turn", 1 if m["angle"] > 0 else -1)] if m["kind"] == "turn" else MOVE_NAMES[("flip", m["axis"])]
    o = _four(rng, gt, list(MOVE_NAMES.values()))
    return o and _q(2, "kth_move", f"What is the {ORD[k]} move the disc makes?", gt, *o, True,
                    {"k": k + 1, "min_fps": fps_for((m["f1"] - m["f0"] + 1) / 2)})


def q_odd_place(F, rng):
    if not F.odd_ok() or F.sc == "turns":
        return None
    x, y = sim.centroid(sim.outline(F.shapes[F.br["index"]]))
    if math.hypot(x, y) < 40 or sim.octant_margin(x, y) < OCT_MARGIN:
        return None
    gt = sim.octant(x, y)
    o = _oct_opts(rng, gt)
    motion = F.br["kind"] == "direction"
    fps = fps_for(F.scene["timeline"]["trace"] / 2) if motion else 1
    return o and _q(2, "odd_place", "One shape breaks the symmetry. In which direction from the centre of the "
                    "picture is it?", gt, *o, motion, {"min_fps": fps})


def q_first_done(F, rng):
    tl = F.scene["timeline"]
    if F.sc == "reveal":
        done = [max(tl["appear"][i] for i in F.members(o)) for o in range(F.n_orb)]
        text = "Which group is the first to have all of its shapes on screen?"
    elif F.sc == "assemble":
        done = sim.settle_frames(F.scene)
        text = "Which group of shapes is the first to settle into place?"
    else:
        return None
    o = min(range(F.n_orb), key=lambda k: done[k])
    ds = sorted(done)
    opts = F.group_opts(rng, o)
    return opts and _q(2, "first_done", text, F.gname(o), *opts, True, {"min_fps": fps_for(ds[1] - ds[0])})


def q_earlier_group(F, rng):
    """Of four listed groups, which one shows up (starts) first?"""
    tl = F.scene["timeline"]
    if F.sc == "waves":
        first = list(tl["starts"])
        text = "Of these groups, which one starts being drawn first?"
    elif F.sc == "reveal":
        first = [min(tl["appear"][i] for i in F.members(o)) for o in range(F.n_orb)]
        text = "Of these groups, which one is the first to have a shape appear?"
    elif F.sc == "assemble":
        first = list(tl["starts"])
        text = "Of these groups, which one starts moving first?"
    else:
        return None
    if F.n_orb < 2:
        return None
    real = rng.sample(range(F.n_orb), min(F.n_orb, rng.randint(2, 4)))
    o = min(real, key=lambda k: first[k])
    others = [F.gname(k) for k in real if k != o]
    conf = F.confusers[:]
    rng.shuffle(conf)
    opts = _four(rng, F.gname(o), others + conf)
    if not opts:
        return None
    shown = sorted(opts[0])
    times = sorted(first[k] for k in real)
    return _q(2, "earlier_group", text, F.gname(o), *opts, True,
              {"options": shown, "min_fps": fps_for(times[1] - times[0])})


# ---------------------------------------------------------------- L3 measurement

def q_min_turn(F, rng):
    gt = turn_label(F.order)
    o = _four(rng, gt, rng.sample(TURN_OPTS, len(TURN_OPTS)))
    return o and _q(3, "min_turn", f"What is the smallest turn about the centre that makes {F.pic()} look "
                    "exactly the same as before?", gt, *o, False, {"min_fps": 1})


def q_mirrors(F, rng):
    o = _ints(rng, F.mirrors, 0, 8)
    return o and _q(3, "mirrors", f"How many mirror lines through the centre does {F.pic()} have?",
                    F.mirrors, *o, False, {"min_fps": 1})


def q_wave_time(F, rng):
    if F.sc != "waves":
        return None
    o = rng.randrange(F.n_orb)
    f0 = F.scene["timeline"]["starts"][o]
    k = sim.tenths(f0)
    opts = _clock(rng, k, sim.tenths(F.nf - 1))
    return opts and _q(3, "wave_time", f"What does the clock show when {F.gname(o)} start to be drawn?",
                       sim.clock_text(k), *opts, True, {"g": F.gparam(o), "min_fps": 4})


def q_done_time(F, rng):
    tl = F.scene["timeline"]
    if F.sc == "waves":
        f = sim.wave_frames(F.scene)[-1][1]
        text = "What does the clock show when the last wave has been traced all the way round?"
    elif F.sc == "reveal":
        f = max(tl["appear"])
        text = "What does the clock show when the last shape appears?"
    elif F.sc == "assemble":
        f = max(sim.settle_frames(F.scene))
        text = "What does the clock show when the last group settles into place?"
    else:
        f = tl["moves"][-1]["f1"]
        text = "What does the clock show when the disc finishes its last move?"
    k = sim.tenths(f)
    opts = _clock(rng, k, sim.tenths(F.nf - 1))
    return opts and _q(3, "done_time", text, sim.clock_text(k), *opts, True, {"min_fps": 4})


def _quiet(F, events, need_before=1):
    out = []
    for k in range(sim.tenths(sim.EDGE_PAD) + 1, sim.tenths(F.nf - 1 - sim.EDGE_PAD)):
        lo, hi = sim.frames_showing(k)
        if any(lo - sim.GAP < e < hi + sim.GAP for e in events):
            continue
        if sum(1 for e in events if e < lo) < need_before:
            continue
        out.append(k)
    return out


def q_count_at(F, rng):
    tl = F.scene["timeline"]
    if F.sc == "reveal":
        ev = list(tl["appear"])
        text = "How many shapes are on screen when the clock shows {t}?"
    elif F.sc == "assemble":
        ev = sim.settle_frames(F.scene)
        text = "How many groups of shapes have settled into place when the clock shows {t}?"
    elif F.sc == "waves":
        ev = [b for a, b in sim.wave_frames(F.scene)] + [a for a, b in sim.wave_frames(F.scene)]
        text = "How many shapes have been traced all the way round when the clock shows {t}?"
    else:
        return None
    ks = _quiet(F, ev)
    if not ks:
        return None
    k = rng.choice(ks)
    lo, hi = sim.frames_showing(k)
    if F.sc == "waves":
        gt = sum(1 for i, s in enumerate(F.shapes) if s["cap"] >= 1 and sim.wave_frames(F.scene)[s["orbit"]][1] < lo)
    else:
        gt = sum(1 for e in (tl["appear"] if F.sc == "reveal" else sim.settle_frames(F.scene)) if e < lo)
    prev = max(e for e in ev if e < lo)
    nxt = min([e for e in ev if e > hi] + [F.nf - 1])
    o = _ints(rng, gt, 0, 30)
    return o and _q(3, "count_at", text.format(t=sim.clock_text(k)), gt, *o, True,
                    {"t": k, "min_fps": fps_for(nxt - prev)})


def q_total_turn(F, rng):
    if F.sc != "turns":
        return None
    ms = F.scene["timeline"]["moves"]
    turns = [m for m in ms if m["kind"] == "turn"]
    if not turns:
        return None
    gt = sum(abs(m["angle"]) for m in turns)
    pool = [gt + d for d in (-90, -45, 45, 90, -30, 30, 135) if gt + d > 0]
    o = _four(rng, f"{gt}°", [f"{v}°" for v in pool])
    return o and _q(3, "total_turn", "Adding up all the turns the disc makes (flips don't count), how many degrees "
                    "does it turn in total?", f"{gt}°", *o, True,
                    {"min_fps": fps_for(min((m["f1"] - m["f0"] + 1) / 2 for m in turns))})


def q_notch_after(F, rng):
    if F.sc != "turns":
        return None
    ms = F.scene["timeline"]["moves"]
    cands = []
    for k, m in enumerate(ms[:-1]):
        x, y = sim.notch_point(F.scene, m["f1"])
        if sim.octant_margin(x, y) >= OCT_MARGIN:
            cands.append((k, sim.octant(x, y)))
    if not cands:
        return None
    k, gt = rng.choice(cands)
    o = _oct_opts(rng, gt)
    gap = ms[k + 1]["f0"] - ms[k]["f1"]
    return o and _q(3, "notch_after", f"Right after the disc's {ORD[k]} move, in which direction from the centre "
                    "does the white notch on its rim point?", gt, *o, True, {"k": k + 1, "min_fps": fps_for(gap)})


def q_size_ratio(F, rng):
    if not F.odd_ok() or F.br["kind"] != "size":
        return None
    i = F.br["index"]
    r = F.shapes[i]["size"] / F.scene["base"][i]["size"]
    gt = f"about {r:.1f} times"
    pool = [f"about {v:.1f} times" for v in sorted((0.5, 0.6, 0.7, 1.4, 1.6, 1.8, 2.0, 1.2), key=lambda v: abs(v - r))
            if abs(v - r) >= 0.15]
    o = _four(rng, gt, pool)
    if abs(r * 10 - round(r * 10)) > 0.3:          # stay away from the 0.05 rounding boundary
        return None
    return o and _q(3, "size_ratio", "One shape is a different size from its symmetric partners. About how many times "
                    "as wide as a partner is it?", gt, *o, False, {"min_fps": 1})


# ---------------------------------------------------------------- L4 integration

def q_odd_wave(F, rng):
    if F.sc != "waves" or not F.odd_ok():
        return None
    o = F.shapes[F.br["index"]]["orbit"]
    opts = _four(rng, ORD[o], [x for x in ORD[:max(4, F.n_orb)] if x != ORD[o]])
    tl = F.scene["timeline"]
    return opts and _q(4, "odd_wave", "Find the shape that breaks the symmetry. In which wave is it drawn?", ORD[o],
                       *opts, True, {"min_fps": fps_for(min(tl["trace"] / 2,
                                                            tl["starts"][1] - tl["starts"][0] - tl["trace"]))})


def q_before_wave(F, rng):
    if F.sc != "waves" or F.n_orb < 2:
        return None
    o = rng.randrange(1, F.n_orb)
    f0 = F.scene["timeline"]["starts"][o]
    gt = sum(1 for s in F.shapes if s["cap"] >= 1 and sim.wave_frames(F.scene)[s["orbit"]][1] < f0)
    opts = _ints(rng, gt, 0, 30)
    tl = F.scene["timeline"]
    return opts and _q(4, "before_wave", f"How many shapes have been traced all the way round by the moment "
                       f"{F.gname(o)} start to be drawn?", gt, *opts, True,
                       {"g": F.gparam(o), "min_fps": fps_for(tl["starts"][1] - tl["starts"][0] - tl["trace"])})


def q_dir_count(F, rng):
    if F.sc != "waves":
        return None
    d = F.scene["direction"]
    gt = sum(1 for s in F.shapes if s["dir"] == d)
    opts = _ints(rng, gt, 1, 30)
    return opts and _q(4, "dir_count", "Follow the dot at the tip of each outline as it is traced. How many shapes "
                       "are traced turning the same way (clockwise or counter-clockwise) as most of the others?", gt,
                       *opts, True, {"min_fps": fps_for(F.scene["timeline"]["trace"] / 2)})


def visible_after(F, i_set):
    return [F.shapes[i] for i in sorted(i_set)]


def q_first_sym(F, rng):
    if F.sc != "reveal":
        return None
    ap = F.scene["timeline"]["appear"]
    order = sorted(range(F.n), key=lambda i: ap[i])
    kinds = []
    for kind in ("half", "mirror"):
        for m in range(2, F.n + 1):
            o, mr = sim.detect([F.shapes[i] for i in order[:m]])
            if (kind == "half" and o % 2 == 0) or (kind == "mirror" and mr >= 1):
                kinds.append((kind, m))
                break
    if not kinds:
        return None
    kind, gt = rng.choice(kinds)
    what = ("looks exactly the same after a half turn (180°) about the centre" if kind == "half"
            else "has at least one mirror line through the centre")
    o = _ints(rng, gt, 2, F.n)
    gaps = [ap[order[k + 1]] - ap[order[k]] for k in range(F.n - 1)]
    return o and _q(4, "first_sym", f"As the shapes appear one by one, count how many are on screen the first time "
                    f"(with at least two shapes) the picture {what}.", gt, *o, True,
                    {"kind": kind, "min_fps": fps_for(min(gaps))})


def q_kth_settle(F, rng):
    if F.sc != "assemble":
        return None
    st = sim.settle_frames(F.scene)
    order = sorted(range(F.n_orb), key=lambda o: st[o])
    k = rng.randrange(F.n_orb - 1)
    o = order[k]
    opts = F.group_opts(rng, o)
    ds = sorted(st)
    return opts and _q(4, "kth_settle", f"Which group is the {ORD[k]} to settle into place?", F.gname(o), *opts, True,
                       {"k": k + 1, "min_fps": fps_for(min(b - a for a, b in zip(ds, ds[1:])))})


def q_moving_at(F, rng):
    if F.sc != "assemble":
        return None
    tl = F.scene["timeline"]
    st = sim.settle_frames(F.scene)
    ev = list(tl["starts"]) + st
    ks = _quiet(F, ev)
    if not ks:
        return None
    k = rng.choice(ks)
    lo, hi = sim.frames_showing(k)
    gt = sum(1 for o in range(F.n_orb) if tl["starts"][o] <= lo and st[o] >= hi)
    prev = max(e for e in ev if e < lo)
    nxt = min([e for e in ev if e > hi] + [F.nf - 1])
    o = _ints(rng, gt, 0, 6)
    return o and _q(4, "moving_at", f"How many groups of shapes are moving when the clock shows {sim.clock_text(k)}?",
                    gt, *o, True, {"t": k, "min_fps": fps_for(nxt - prev)})


def q_match_after(F, rng):
    if F.sc != "turns":
        return None
    ms = F.scene["timeline"]["moves"]
    first = None
    for k in range(1, len(ms) + 1):
        M = sim.net_matrix(F.scene, ms[:k])
        if sim.is_symmetry(F.shapes, sim.matrix_elem(M)):
            first = k
            break
    gt = "never" if first is None else f"after the {ORD[first - 1]} move"
    pool = [f"after the {ORD[k]} move" for k in range(len(ms))] + ["never", "after the fifth move"]
    o = _four(rng, gt, pool)
    gaps = [b["f0"] - a["f1"] for a, b in zip(ms, ms[1:])]
    return o and _q(4, "match_after", "Ignoring the disc and its notch, after which move do the shapes first look "
                    "exactly as they did at the start?", gt, *o, True, {"min_fps": fps_for(min(gaps))})


def q_n_match(F, rng):
    if F.sc != "turns":
        return None
    ms = F.scene["timeline"]["moves"]
    gt = sum(1 for k in range(1, len(ms) + 1)
             if sim.is_symmetry(F.shapes, sim.matrix_elem(sim.net_matrix(F.scene, ms[:k]))))
    o = _ints(rng, gt, 0, 6)
    gaps = [b["f0"] - a["f1"] for a, b in zip(ms, ms[1:])]
    return o and _q(4, "n_match", "Ignoring the disc and its notch, after how many of the disc's moves do the shapes "
                    "look exactly as they did at the start?", gt, *o, True, {"min_fps": fps_for(min(gaps))})


def q_odd_after(F, rng):
    if F.sc != "turns" or not F.odd_ok():
        return None
    ms = F.scene["timeline"]["moves"]
    c0 = sim.centroid(sim.outline(F.shapes[F.br["index"]]))
    cands = []
    for k, m in enumerate(ms[:-1]):
        x, y = sim.apply(sim.net_matrix(F.scene, ms[:k + 1]), c0)
        if math.hypot(x, y) >= 40 and sim.octant_margin(x, y) >= OCT_MARGIN:
            cands.append((k, sim.octant(x, y)))
    if not cands:
        return None
    k, gt = rng.choice(cands)
    o = _oct_opts(rng, gt)
    return o and _q(4, "odd_after", f"One shape breaks the symmetry. Right after the disc's {ORD[k]} move, in which "
                    "direction from the centre is that shape?", gt, *o, True,
                    {"k": k + 1, "min_fps": fps_for(ms[k + 1]["f0"] - ms[k]["f1"])})


def q_kth_shape(F, rng):
    if F.sc != "reveal":
        return None
    ap = F.scene["timeline"]["appear"]
    order = sorted(range(F.n), key=lambda i: ap[i])
    ks = [(k, o) for k in range(1, F.n) for o in range(F.n_orb) if order[k] in F.members(o)]
    k, o = rng.choice(ks)
    opts = F.group_opts(rng, o)
    words = ["first", "second", "third", "fourth", "fifth", "sixth", "seventh", "eighth", "ninth", "tenth",
             "eleventh", "twelfth"]
    gaps = [ap[order[j + 1]] - ap[order[j]] for j in range(F.n - 1)]
    return opts and _q(4, "kth_shape", f"Counting the shapes in the order they appear, which group does the "
                       f"{words[k]} shape belong to?", F.gname(o), *opts, True, {"k": k + 1, "min_fps": fps_for(min(gaps))})


def q_on_screen_when(F, rng):
    if F.sc != "reveal":
        return None
    ap = F.scene["timeline"]["appear"]
    o = rng.randrange(F.n_orb)
    done = max(ap[i] for i in F.members(o))
    gt = sum(1 for i in range(F.n) if ap[i] <= done)
    opts = _ints(rng, gt, 1, F.n)
    gaps = sorted(ap)
    return opts and _q(4, "on_screen_when", f"How many shapes are on screen at the moment the last of "
                       f"{F.gname(o)} appears?", gt, *opts, True,
                       {"g": F.gparam(o), "min_fps": fps_for(min(b - a for a, b in zip(gaps, gaps[1:])))})


def q_after_wave(F, rng):
    if F.sc != "waves" or F.n_orb < 2:
        return None
    o = rng.randrange(F.n_orb - 1)
    gt = sum(1 for s in F.shapes if s["orbit"] > o)
    opts = _ints(rng, gt, 1, 30)
    tl = F.scene["timeline"]
    return opts and _q(4, "after_wave", f"How many shapes are drawn after {F.gname(o)} have been finished?", gt,
                       *opts, True, {"g": F.gparam(o), "min_fps": fps_for(tl["starts"][1] - tl["starts"][0] - tl["trace"])})


def q_turn_until(F, rng):
    if F.sc != "turns":
        return None
    ms = F.scene["timeline"]["moves"]
    ks = [k for k in range(2, len(ms) + 1) if any(m["kind"] == "turn" for m in ms[:k])]
    if not ks:
        return None
    k = rng.choice(ks)
    gt = sum(abs(m["angle"]) for m in ms[:k] if m["kind"] == "turn")
    pool = [gt + d for d in (-90, -45, 45, 90, -30, 30, 135) if gt + d > 0]
    o = _four(rng, f"{gt}°", [f"{v}°" for v in pool])
    tm = [m for m in ms[:k] if m["kind"] == "turn"]
    return o and _q(4, "turn_until", f"By the end of the disc's {ORD[k - 1]} move, how many degrees has it turned in "
                    "total (flips don't count)?", f"{gt}°", *o, True,
                    {"k": k, "min_fps": fps_for(min((m["f1"] - m["f0"] + 1) / 2 for m in tm))})


# ---------------------------------------------------------------- L5 counterfactual

CF_MOVES = [("a quarter turn (90°)", (90.0, 0)), ("a half turn (180°)", (180.0, 0)), ("a 60° turn", (60.0, 0)),
            ("a 72° turn", (72.0, 0)), ("a 120° turn", (120.0, 0)), ("a 45° turn", (45.0, 0)),
            ("a left-to-right flip", (180.0, 1)), ("an upside-down flip", (0.0, 1))]
NONE_OF_THESE = "none of these"


def q_cf_rotate(F, rng):
    good = [name for name, e in CF_MOVES if sim.is_symmetry(F.shapes, e)]
    bad = [name for name, e in CF_MOVES if name not in good]
    gt = rng.choice(good) if good and rng.random() < 0.8 else NONE_OF_THESE
    shown = rng.sample(bad, 3)
    if gt == NONE_OF_THESE and good:
        return None
    o = _four(rng, gt, shown)
    return o and _q(5, "cf_rotate", f"Imagine applying each of these to {F.pic()} (about its centre). "
                    "Which one would leave it looking exactly the same?", gt, *o, False,
                    {"options": sorted(o[0]), "min_fps": 1})


def q_cf_remove_odd(F, rng):
    if not F.odd_ok() or F.br["kind"] == "direction":
        return None
    rest = [s for i, s in enumerate(F.shapes) if i != F.br["index"]]
    gt = sim.detect(rest)[1]
    o = _ints(rng, gt, 0, 8)
    return o and _q(5, "cf_remove_odd", "If the shape that breaks the symmetry were removed, how many mirror lines "
                    "through the centre would the rest of the picture have?", gt, *o, False, {"min_fps": 1})


def q_cf_more_wave(F, rng):
    if F.sc != "waves":
        return None
    k = len(sim.group_elements(F.scene["group"]))
    gt = F.n + k
    o = _four(rng, gt, [F.n + 1, F.n + k + 1, F.n + 2 * k, F.n + k - 1, F.n + 2])
    tl = F.scene["timeline"]
    return o and _q(5, "cf_more_wave", "Suppose one more wave were drawn that keeps the same symmetry as the waves you "
                    "saw. How many shapes would the picture have then?", gt, *o, True,
                    {"min_fps": fps_for(tl["starts"][1] - tl["starts"][0] - tl["trace"])})


def q_cf_slow(F, rng):
    if F.sc != "waves":
        return None
    tl = F.scene["timeline"]
    # same pauses, every trace twice as long: wave j starts later by j * trace frames
    last = tl["starts"][-1] + (F.n_orb - 1) * tl["trace"] + 2 * tl["trace"] - 1
    k = sim.tenths(last)
    o = _clock(rng, k, 200)
    return o and _q(5, "cf_slow", "Suppose every wave took twice as long to trace, with the same pauses between waves. "
                    "What would the clock show when the last wave had been traced all the way round?",
                    sim.clock_text(k), *o, True, {"min_fps": 4})


def q_cf_drop_last(F, rng):
    if F.sc != "reveal":
        return None
    ap = F.scene["timeline"]["appear"]
    last = max(range(F.n), key=lambda i: ap[i])
    rest = [s for i, s in enumerate(F.shapes) if i != last]
    order, mirrors = sim.detect(rest)
    kind = rng.choice(["mirrors", "turn"])
    if kind == "mirrors":
        gt = mirrors
        o = _ints(rng, gt, 0, 8)
        text = "how many mirror lines through the centre would the picture have at the end?"
    else:
        gt = turn_label(order)
        o = _four(rng, gt, rng.sample(TURN_OPTS, len(TURN_OPTS)))
        text = "what would be the smallest turn about the centre that makes the picture look exactly the same?"
    ds = sorted(ap)
    return o and _q(5, "cf_drop_last", f"If the last shape to appear had never appeared, {text}", gt, *o, True,
                    {"kind": kind, "min_fps": fps_for(ds[-1] - ds[-2])})


def q_cf_delay(F, rng):
    if F.sc != "assemble":
        return None
    st = sim.settle_frames(F.scene)
    for o in rng.sample(range(F.n_orb), F.n_orb):
        alt = st[:]
        alt[o] += sim.FPS
        srt = sorted(alt)
        if any(b - a < sim.GAP for a, b in zip(srt, srt[1:])):
            continue
        last = max(range(F.n_orb), key=lambda k: alt[k])
        opts = F.group_opts(rng, last)
        return opts and _q(5, "cf_delay", f"If {F.gname(o)} had started moving 1 second later (and moved the same "
                           "way), which group would be the last to settle into place?", F.gname(last), *opts, True,
                           {"g": F.gparam(o), "min_fps": fps_for(min(b - a for a, b in zip(sorted(st), sorted(st)[1:])))})
    return None


def q_cf_skip_move(F, rng):
    if F.sc != "turns":
        return None
    ms = F.scene["timeline"]["moves"]
    a = math.radians(F.scene["timeline"]["notch"])
    p0 = (math.cos(a), math.sin(a))
    cands = []
    for k in range(len(ms)):
        x, y = sim.apply(sim.net_matrix(F.scene, ms[:k] + ms[k + 1:]), p0)
        if sim.octant_margin(x, y) >= OCT_MARGIN:
            cands.append((k, sim.octant(x, y)))
    if not cands:
        return None
    k, gt = rng.choice(cands)
    o = _oct_opts(rng, gt)
    gaps = [b["f0"] - a["f1"] for a, b in zip(ms, ms[1:])]
    return o and _q(5, "cf_skip_move", f"If the disc had skipped its {ORD[k]} move and made the others as before, in "
                    "which direction from the centre would the white notch point at the end?", gt, *o, True,
                    {"k": k + 1, "min_fps": fps_for(min(gaps))})


TEMPLATES = {
    1: [q_n_shapes, q_n_type, q_n_colors, q_n_waves, q_first_group, q_n_moves],
    2: [q_break_kind, q_group_wave, q_last_group, q_kth_move, q_odd_place, q_first_done, q_earlier_group],
    3: [q_min_turn, q_mirrors, q_wave_time, q_done_time, q_count_at, q_total_turn, q_notch_after, q_size_ratio],
    4: [q_odd_wave, q_before_wave, q_dir_count, q_first_sym, q_kth_settle, q_moving_at, q_match_after, q_n_match, q_odd_after,
        q_kth_shape, q_on_screen_when, q_after_wave, q_turn_until],
    5: [q_cf_rotate, q_cf_remove_odd, q_cf_more_wave, q_cf_slow, q_cf_drop_last, q_cf_delay, q_cf_skip_move],
}


def _sig(q):
    return q["derivation_key"], tuple(sorted((k, str(v)) for k, v in q["params"].items()))


def pools(F, rng) -> dict:
    """Distinct candidate questions per level (several draws per template)."""
    out = {}
    for lv in range(1, 6):
        got = {}
        tpl = TEMPLATES[lv][:]
        rng.shuffle(tpl)
        for _ in range(5):
            for fn in tpl:
                q = fn(F, rng)
                if q and _sig(q) not in got:
                    got[_sig(q)] = q
        out[lv] = list(got.values())
    return out


def _pick(pool, want, rng):
    """`want` questions: different templates first, motion questions first,
    at most one single-frame question unless nothing else is left."""
    chosen, keys = [], {}
    rest = pool[:]
    rng.shuffle(rest)
    while len(chosen) < want:
        static_used = sum(1 for q in chosen if not q["requires_motion"])
        rest.sort(key=lambda q: (keys.get(q["derivation_key"], 0),
                                 (not q["requires_motion"]) and static_used >= 1, not q["requires_motion"]))
        q = rest.pop(0)
        chosen.append(q)
        keys[q["derivation_key"]] = keys.get(q["derivation_key"], 0) + 1
    return chosen


def build(scene: dict, rng: random.Random) -> list[dict]:
    F = Facts(scene)
    P = pools(F, rng)
    avail = [len(P[lv]) for lv in range(1, 6)]
    combos = [c for c in L._count_combos() if sum(c) == L.TEN_TOTAL and all(a >= k for a, k in zip(avail, c))]
    if not combos:
        raise RuntimeError(f"not enough questions per level {avail} for {scene['scenario']}")
    counts = rng.choice(combos)
    out = []
    for lv, want in zip(range(1, 6), counts):
        out += _pick(P[lv], want, rng)
    return L.number(out)
