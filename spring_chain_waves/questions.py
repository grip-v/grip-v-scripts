"""10-question ladder for spring_chain_waves (common/ladder5.py, LADDER = "ten").

Things are named by what is on screen: the ends by side and hardware ("the
right end", "the left paddle"), the chain's parts by bead thickness or by
ruler marks, pulses by launch order and shape ("the first pulse the paddle
sends", "a small dip"), moments by the clock. Every question stores
derivation_key + params + min_fps; ground truth here comes from the exact
pulse trace in sim.py, answers.py re-derives it from the drawn chain.
"""
from __future__ import annotations
import math
import random
import sys
from fractions import Fraction as Fr
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE))
from common import ladder5 as L  # noqa: E402

import sim  # noqa: E402

LADDER = "ten"
FPS_STEPS = (1, 2, 3, 4, 6, 8, 12, 24)
ORD = ["first", "second", "third", "fourth", "fifth", "sixth"]
HARDWARE = {"paddle": "a paddle", "clamped": "a clamp", "free": "a ring that slides on a pole"}
KIND_OPTS = ["a clamp", "a ring that slides on a pole", "a paddle", "nothing, the end hangs loose"]
BACK_OPTS = ["upside down", "the same way up", "it does not come back", "it splits into two smaller pulses"]
REL_OPTS = ["the same way up", "upside down", "split into two smaller pulses", "about half as tall"]
SPLIT_OPTS = ["a smaller upside-down pulse bounces back and the rest goes on",
              "a smaller pulse bounces back the same way up and the rest goes on",
              "all of it goes on and nothing bounces back", "all of it bounces back upside down"]
RATIO_WORDS = {Fr(1, 3): "about 1/3 as tall", Fr(1, 2): "about 1/2 as tall", Fr(2, 3): "about 2/3 as tall",
               Fr(1): "about as tall", Fr(4, 3): "about 4/3 as tall", Fr(3, 2): "about 3/2 as tall"}
SPEED_WORDS = {2: "about half", 3: "about a third"}
SPEED_OPTS = ["about half", "about a third", "about two thirds", "about the same"]
CF_GAP = 8          # frames: what-if events this far from the clip's end


def fps_for(gap_frames) -> int:
    gap = float(gap_frames) / sim.FPS
    for f in FPS_STEPS:
        if f * gap >= 1 - 1e-9:
            return f
    return 24


def pulse_word(sign, big, sized) -> str:
    shape = "bump up" if sign > 0 else "dip down"
    return f"a {'big' if big else 'small'} {shape}" if sized else f"a {shape}"


def short_word(sign, big, sized) -> str:
    shape = "bump" if sign > 0 else "dip"
    return f"{'big' if big else 'small'} {shape}" if sized else shape


class Facts:
    def __init__(self, scene):
        self.scene = scene
        self.sc = scene["scenario"]
        self.L = scene["L"]
        self.nf = scene["n_frames"]
        self.tl = sim.t_last(scene)
        self.segs = sim.visible(scene, sim.trace(scene))
        self.home = scene["home"]
        self.far = sim.far_side(scene)
        self.hx = None if self.home is None else sim.end_x(scene, self.home)
        self.fx = None if self.far is None else sim.end_x(scene, self.far)
        self.launches = scene["launches"]
        self.lseg = {s["origin"]: i for i, s in enumerate(self.segs) if s["kind"] == "launch"}
        bigs = {ln["big"] for ln in self.launches}
        self.sized = len(bigs) == 2
        self.c = sim.speed(scene)
        self._calm = None

    def arrivals(self, x):
        return sorted((i for i, s in enumerate(self.segs) if s["xe"] == x and s["te"] <= self.tl),
                      key=lambda i: self.segs[i]["te"])

    def dur(self, i):
        s = self.segs[i]
        return (min(s["te"], self.tl) - s["tb"]) * sim.FPS

    def child(self, i, kind=None):
        for j, s in enumerate(self.segs):
            if s["parent"] == i and (kind is None or s["kind"] == kind):
                return j
        return None

    def calm(self):
        if self._calm is None:
            self._calm = [sim.calm(self.scene, self.segs, f) for f in range(self.nf)]
        return self._calm

    def stable_moments(self):
        """[(k tenths, window frames)]: clock readings whose frames, with 7 frames
        either side, show only whole separate pulses and the same set of them."""
        cm = self.calm()
        out = []
        for k in range(1, 100):
            lo, hi = sim.frames_showing(k)
            a, b = lo - 7, hi + 7
            if a < sim.EDGE or b > self.nf - 1 - sim.EDGE:
                continue
            if not all(cm[a:b + 1]):
                continue
            act = {tuple(sim.active(self.segs, f)) for f in range(a, b + 1)}
            if len(act) != 1 or not next(iter(act)):
                continue
            # the whole run of frames with this set
            s0 = sim.active(self.segs, lo)
            ra, rb = lo, hi
            while ra > 0 and cm[ra - 1] and sim.active(self.segs, ra - 1) == s0:
                ra -= 1
            while rb < self.nf - 1 and cm[rb + 1] and sim.active(self.segs, rb + 1) == s0:
                rb += 1
            out.append((k, rb - ra + 1))
        return out

    def end_word(self, side):
        return f"the {side} end"

    def far_phrase(self):
        return f"the {self.far} end"


def _q(lv, key, text, gt, opts, a, motion, params):
    return L.question(lv, key, text, gt, opts, a, motion, params)


def _four(rng, correct, pool, n=4):
    ds = [d for d in dict.fromkeys(pool) if d != correct]
    if len(ds) < n - 1:
        return None
    return L.mc(rng, correct, ds[:n - 1])


def _ints(rng, gt, lo=0, hi=99):
    pool = L.near_ints(gt, lo, hi)
    rng.shuffle(pool)
    return _four(rng, gt, pool)


def _clock(rng, k):
    cands = sorted((k + d for d in (-15, -10, -5, 5, 10, 15) if 0 <= k + d <= 99), key=lambda c: abs(c - k))
    return _four(rng, sim.clock_text(k), [sim.clock_text(c) for c in cands[:4]])


def _num(v: Fr) -> str:
    return f"{float(v):g}"


# ---------------------------------------------------------------- L1 observation

def q_far_kind(F, rng):
    if F.far is None:
        return None
    gt = HARDWARE[F.scene["ends"][F.far]]
    o = _four(rng, gt, KIND_OPTS)
    return o and _q(1, "far_kind", f"What holds {F.far_phrase()} of the chain?", gt, *o, False, {"min_fps": 1})


def q_first_shape(F, rng):
    if F.sc == "collide":
        side = rng.choice(["left", "right"])
        ks = [k for k, ln in enumerate(F.launches) if ln["end"] == side]
        who = f"the first pulse the {side} paddle sends"
        params = {"side": side}
    else:
        ks = [0]
        who = "the first pulse the paddle sends"
        params = {}
    k = ks[0]
    ln = F.launches[k]
    if F.sized:
        gt = pulse_word(ln["sign"], ln["big"], True)
        pool = [pulse_word(s, b, True) for s in (1, -1) for b in (True, False)]
    else:
        gt = pulse_word(ln["sign"], True, False)
        pool = ["a bump up", "a dip down", "a bump up followed by a dip", "a dip followed by a bump up"]
    o = _four(rng, gt, pool)
    params["min_fps"] = fps_for(F.dur(F.lseg[k]))
    return o and _q(1, "first_shape", f"What does {who} look like?", gt, *o, True, params)


def q_n_launch(F, rng):
    if F.sc not in ("train", "collide"):
        return None
    n = len(F.launches)
    gaps = []
    for side in ("left", "right"):
        fs = [ln["f"] for ln in F.launches if ln["end"] == side]
        gaps += [b - a for a, b in zip(fs, fs[1:])]
    who = "the paddles send into the chain in total" if F.sc == "collide" else "the paddle send into the chain"
    o = _ints(rng, n, 1, 9)
    return o and _q(1, "n_launch", f"How many pulses do{'es' if F.sc == 'train' else ''} {who}?", n, *o, True,
                    {"min_fps": fps_for(min(gaps) if gaps else 24)})


def q_change_mark(F, rng):
    if F.sc != "junction":
        return None
    secs = F.scene["sections"]
    if len(secs) == 2:
        j = secs[0][1]
        gt = str(j)
        pool = [str(v) for v in (j + 1, j - 1, j + 2, j - 2, j + 3) if 1 <= v <= F.L - 1]
        o = _four(rng, gt, pool)
        return o and _q(1, "change_mark", "At which ruler mark do the beads change between thin and thick?", gt,
                        *o, False, {"min_fps": 1})
    a, b = secs[1][0], secs[1][1]
    gt = f"{a} and {b}"
    pool = [f"{x} and {y}" for x, y in ((a - 1, b - 1), (a + 1, b + 1), (a - 1, b + 1), (a + 1, b - 1), (a, b + 1),
                                         (a - 1, b)) if 0 <= x < y <= F.L and y - x >= 1]
    o = _four(rng, gt, pool)
    return o and _q(1, "change_mark", "Between which two ruler marks are the beads thick?", gt, *o, False,
                    {"min_fps": 1})


def q_first_sender(F, rng):
    if F.sc != "collide":
        return None
    fl = min(ln["f"] for ln in F.launches if ln["end"] == "left")
    fr = min(ln["f"] for ln in F.launches if ln["end"] == "right")
    if fl == fr:
        gt, fps = "both at the same moment", 2
    elif abs(fl - fr) >= 8:
        gt, fps = ("the left paddle" if fl < fr else "the right paddle"), fps_for(abs(fl - fr))
    else:
        return None
    o = _four(rng, gt, ["the left paddle", "the right paddle", "both at the same moment"], n=3)
    return o and _q(1, "first_sender", "Which paddle sends its first pulse first?", gt, *o, True, {"min_fps": fps})


def q_end_count(F, rng):
    cm = F.calm()
    a = F.nf - 12
    if not all(cm[a:]):
        return None
    sets = {tuple(sim.active(F.segs, f)) for f in range(a, F.nf)}
    if len(sets) != 1:
        return None
    n = len(next(iter(sets)))
    o = _ints(rng, n, 0, 9)
    return o and _q(1, "end_count", "How many separate bumps or dips are on the chain at the very end of the video?",
                    n, *o, False, {"min_fps": 2})


# ---------------------------------------------------------------- L2 relational

def q_far_bounce(F, rng):
    if F.far is None:
        return None
    arr = F.arrivals(F.fx)
    if not arr:
        return None
    i = arr[0]
    j = F.child(i)
    if j is None or F.segs[j]["tb"] > F.tl - Fr(1, 2):
        return None
    gt = "upside down" if F.scene["ends"][F.far] == "clamped" else "the same way up"
    o = _four(rng, gt, BACK_OPTS)
    return o and _q(2, "far_bounce", f"When a pulse bounces off {F.far_phrase()}, how does it come back, compared "
                    f"with how it arrived?", gt, *o, True, {"min_fps": fps_for(min(F.dur(i), F.dur(j)))})


def _first_home(F, origin=0, nth=1):
    """Index of the nth arrival back at the paddle of pulses from launch `origin`
    that went via the far end."""
    hits = [i for i in F.arrivals(F.hx) if F.segs[i]["origin"] == origin and F.segs[i]["via_far"]]
    return hits[nth - 1] if len(hits) >= nth else None


def _chain_of(F, i):
    out = []
    while i >= 0:
        out.append(i)
        i = F.segs[i]["parent"]
    return out[::-1]


def q_first_back(F, rng):
    if F.sc not in ("reflect", "train"):
        return None
    i = _first_home(F)
    if i is None:
        return None
    rel = F.segs[i]["amp"] * F.launches[0]["sign"]
    gt = "the same way up" if rel > 0 else "upside down"
    o = _four(rng, gt, REL_OPTS)
    fps = fps_for(min(F.dur(j) for j in _chain_of(F, i)))
    return o and _q(2, "first_back", "When the first pulse comes back to the paddle, how does it look compared with "
                    "how the paddle sent it?", gt, *o, True, {"min_fps": fps})


def _first_split(F):
    """(incoming, back, on) indices at the launched pulse's first thickness change."""
    i = F.lseg[0]
    s = F.segs[i]
    if s["xe"] in (0, F.L):
        return None
    back, on = F.child(i, "split_back"), F.child(i, "split_on")
    if back is None or on is None:
        return None
    return i, back, on


def q_junction_split(F, rng):
    if F.sc != "junction":
        return None
    sp = _first_split(F)
    if not sp:
        return None
    i, b, n = sp
    gt = SPLIT_OPTS[0] if F.segs[b]["amp"] * F.segs[i]["amp"] < 0 else SPLIT_OPTS[1]
    o = _four(rng, gt, SPLIT_OPTS)
    return o and _q(2, "junction_split", "When the pulse first reaches a place where the beads change thickness, "
                    "what happens?", gt, *o, True, {"min_fps": fps_for(min(F.dur(i), F.dur(b), F.dur(n)))})


def _rng_word(lo, hi):
    return f"between marks {lo} and {hi}"


def q_slow_part(F, rng):
    if F.sc != "junction":
        return None
    secs = F.scene["sections"]
    gt = next(_rng_word(lo, hi) for lo, hi, hv in secs if hv)
    pool = [_rng_word(lo, hi) for lo, hi, hv in secs if not hv] + ["the same speed everywhere"]
    o = _four(rng, gt, pool, n=len(pool) + 1)
    durs = [F.dur(i) for i, s in enumerate(F.segs) if s["tb"] <= F.tl]
    return o and _q(2, "slow_part", "Where on the chain do pulses move most slowly?", gt, *o, True,
                    {"min_fps": fps_for(min(durs) / 2)})


def q_first_to_end(F, rng):
    if F.sc != "junction":
        return None
    sp = _first_split(F)
    if not sp:
        return None
    i, b, n = sp
    ta = F.segs[b]["te"] if F.segs[b]["xe"] == F.hx else None
    far = F.arrivals(F.fx)
    tb_ = F.segs[far[0]]["te"] if far else None
    ta = ta if ta is not None and ta <= F.tl else None
    opts = ["the part that bounced back reaches the paddle first",
            f"the part that went on reaches {F.far_phrase()} first", "both at the same moment",
            "neither within the video"]
    if ta is None and tb_ is None:
        return None
    if ta is not None and tb_ is not None:
        gap = abs(ta - tb_) * sim.FPS
        if gap < 8:
            return None
        gt = opts[0] if ta < tb_ else opts[1]
    else:
        gt = opts[0] if ta is not None else opts[1]
        gap = 24
    o = _four(rng, gt, opts)
    return o and _q(2, "first_to_end", "After the pulse first splits where the beads change, which happens first?",
                    gt, *o, True, {"min_fps": fps_for(gap)})


def q_trans_taller(F, rng):
    if F.sc != "junction":
        return None
    sp = _first_split(F)
    if not sp:
        return None
    i, b, n = sp
    gt = "taller" if abs(F.segs[n]["amp"]) > abs(F.segs[i]["amp"]) else "shorter"
    o = _four(rng, gt, ["taller", "shorter", "the same height", "upside down"])
    return o and _q(2, "trans_taller", "Compared with the pulse arriving at the first change of beads, the pulse that "
                    "goes on past it is:", gt, *o, True, {"min_fps": fps_for(min(F.dur(i), F.dur(n)))})


def _seq_text(items, sized):
    return ", then ".join(short_word(s, b, sized) for s, b in items)


def q_order_back(F, rng):
    if F.sc != "train":
        return None
    n = len(F.launches)
    far_hits = [i for i in F.arrivals(F.fx)]
    if len({F.segs[i]["origin"] for i in far_hits}) < n:
        return None
    r = -1 if F.scene["ends"][F.far] == "clamped" else 1
    seq = [(ln["sign"] * r, ln["big"]) for ln in F.launches]
    gt = _seq_text(seq, F.sized)
    alts = [seq[::-1], [(-s, b) for s, b in seq], [(-s, b) for s, b in seq[::-1]]]
    pool = [_seq_text(a, F.sized) for a in alts]
    if len({gt, *pool}) < 4:
        return None
    o = _four(rng, gt, pool)
    gaps = [b["f"] - a["f"] for a, b in zip(F.launches, F.launches[1:])]
    return o and _q(2, "order_back", f"In what order do the pulses come back toward the paddle after bouncing off "
                    f"{F.far_phrase()}?", gt, *o, True, {"min_fps": fps_for(min(gaps))})


def _first_of(F, side):
    return next(k for k, ln in enumerate(F.launches) if ln["end"] == side)


def _arrive_other(F, k):
    """First arrival of launch k at the opposite paddle."""
    side = F.launches[k]["end"]
    x = sim.end_x(F.scene, "right" if side == "left" else "left")
    hits = [i for i in F.arrivals(x) if F.segs[i]["origin"] == k]
    return hits[0] if hits else None


def q_which_first_far(F, rng):
    if F.sc != "collide":
        return None
    kl, kr = _first_of(F, "left"), _first_of(F, "right")
    al, ar = _arrive_other(F, kl), _arrive_other(F, kr)
    opts = ["the pulse from the left paddle", "the pulse from the right paddle", "both at the same moment",
            "neither gets there within the video"]
    if al is None or ar is None:
        return None
    ta, tb_ = F.segs[al]["te"], F.segs[ar]["te"]
    if ta == tb_:
        gt, gap = opts[2], 24
    elif abs(ta - tb_) * sim.FPS >= 8:
        gt, gap = (opts[0] if ta < tb_ else opts[1]), abs(ta - tb_) * sim.FPS
    else:
        return None
    o = _four(rng, gt, opts)
    return o and _q(2, "which_first_far", "The first pulse from each paddle travels to the other paddle. Which gets "
                    "there first?", gt, *o, True, {"min_fps": fps_for(gap)})


def _meets(F, ka, kb):
    return [z for z in sim.crossings(F.segs) if {F.segs[z[2]]["origin"], F.segs[z[3]]["origin"]} == {ka, kb}
            and z[0] <= F.tl]


def q_meet_side(F, rng):
    if F.sc != "collide":
        return None
    ms = _meets(F, _first_of(F, "left"), _first_of(F, "right"))
    if not ms:
        return None
    x = ms[0][1]
    half = Fr(F.L, 2)
    opts = ["nearer the left paddle", "nearer the right paddle", "in the middle of the chain",
            "they never meet"]
    if abs(x - half) < Fr(3, 20):
        gt = opts[2]
    elif abs(x - half) >= Fr(4, 5):
        gt = opts[0] if x < half else opts[1]
    else:
        return None
    o = _four(rng, gt, opts)
    return o and _q(2, "meet_side", "Where do the first pulses from the two paddles first pass through each other?",
                    gt, *o, True, {"min_fps": 2})


def q_bigger_sender(F, rng):
    if F.sc != "collide":
        return None
    kl, kr = _first_of(F, "left"), _first_of(F, "right")
    bl, br = F.launches[kl]["big"], F.launches[kr]["big"]
    gt = "the same height" if bl == br else ("the left paddle's" if bl else "the right paddle's")
    o = _four(rng, gt, ["the left paddle's", "the right paddle's", "the same height"], n=3)
    fps = fps_for(min(F.dur(F.lseg[kl]), F.dur(F.lseg[kr])))
    return o and _q(2, "bigger_sender", "Which paddle's first pulse is taller?", gt, *o, True, {"min_fps": fps})


def q_dir_at(F, rng):
    if F.sc != "reflect":
        return None
    ms = F.stable_moments()
    if not ms:
        return None
    k, win = rng.choice(ms)
    f = sim.frames_showing(k)[0]
    act = sim.active(F.segs, f)
    if len(act) != 1:
        return None
    s = F.segs[act[0]]
    gt = "toward the right end" if s["d"] > 0 else "toward the left end"
    o = _four(rng, gt, ["toward the right end", "toward the left end", "it is standing still",
                        "it is spreading out both ways"])
    return o and _q(2, "dir_at", f"Which way is the pulse moving when the clock shows {sim.clock_text(k)}?", gt, *o,
                    True, {"k": k, "min_fps": fps_for(win / 2)})


# ---------------------------------------------------------------- L3 measurement

def q_speed(F, rng):
    durs = [F.dur(i) for i, s in enumerate(F.segs) if s["tb"] <= F.tl and s["c"] == F.c]
    if not durs:
        return None
    gt = _num(F.c)
    pool = [_num(F.c + d) for d in (Fr(1, 2), -Fr(1, 2), Fr(1), -Fr(1), Fr(3, 2), -Fr(3, 2)) if F.c + d >= Fr(1, 2)]
    pool = pool[:2] + sorted(pool[2:], key=lambda _: rng.random())
    o = _four(rng, gt, pool)
    where = "where the beads are thin" if F.sc == "junction" else "along the chain"
    return o and _q(3, "speed", f"About how many ruler marks per second does a pulse move {where}?", gt, *o, True,
                    {"min_fps": fps_for(max(durs) / 2)})


def q_first_far_time(F, rng):
    if F.far is None:
        return None
    arr = F.arrivals(F.fx)
    if not arr:
        return None
    k = sim.tenths(sim.frame_of(F.segs[arr[0]]["te"]))
    o = _clock(rng, k)
    return o and _q(3, "first_far_time", f"At what time on the clock does a pulse first reach {F.far_phrase()}?",
                    sim.clock_text(k), *o, True, {"min_fps": 4})


def q_return_time(F, rng):
    if F.home is None:
        return None
    i = _first_home(F)
    if i is None:
        return None
    k = sim.tenths(sim.frame_of(F.segs[i]["te"]))
    o = _clock(rng, k)
    return o and _q(3, "return_time", f"The first pulse bounces off {F.far_phrase()}. At what time on the clock does it "
                    f"get back to the paddle?", sim.clock_text(k), *o, True, {"min_fps": 4})


def q_meet_mark(F, rng):
    if F.sc != "collide":
        return None
    ms = _meets(F, _first_of(F, "left"), _first_of(F, "right"))
    if not ms:
        return None
    x = ms[0][1]
    m = round(x)
    if abs(x - m) > Fr(3, 20):
        return None
    pool = [str(v) for v in (m + 1, m - 1, m + 2, m - 2, m + 3) if 0 <= v <= F.L]
    o = _four(rng, str(m), pool)
    return o and _q(3, "meet_mark", "At which ruler mark do the first pulses from the two paddles first pass through "
                    "each other?", str(m), *o, True, {"min_fps": 2})


def q_amp_ratio(F, rng):
    if F.sc != "junction":
        return None
    sp = _first_split(F)
    if not sp:
        return None
    i, b, n = sp
    v = abs(F.segs[n]["amp"] / F.segs[i]["amp"])
    gt = RATIO_WORDS[v]
    pool = [RATIO_WORDS[x] for x in sorted(RATIO_WORDS, key=lambda x: abs(x - v)) if x != v]
    o = _four(rng, gt, pool)
    return o and _q(3, "amp_ratio", "How tall is the pulse that goes on past the first change of beads, compared with "
                    "the pulse that arrived there?", gt, *o, True, {"min_fps": fps_for(min(F.dur(i), F.dur(n)))})


def q_refl_ratio(F, rng):
    if F.sc != "junction":
        return None
    sp = _first_split(F)
    if not sp:
        return None
    i, b, n = sp
    v = abs(F.segs[b]["amp"] / F.segs[i]["amp"])
    gt = RATIO_WORDS[v]
    o = _four(rng, gt, [RATIO_WORDS[x] for x in (Fr(1, 3), Fr(1, 2), Fr(2, 3), Fr(1))])
    return o and _q(3, "refl_ratio", "How tall is the pulse that bounces back from the first change of beads, compared "
                    "with the pulse that arrived there?", gt, *o, True, {"min_fps": fps_for(min(F.dur(i), F.dur(b)))})


def q_speed_ratio(F, rng):
    if F.sc != "junction":
        return None
    gt = SPEED_WORDS[F.scene["ratio"]]
    o = _four(rng, gt, SPEED_OPTS)
    durs = [F.dur(i) for i, s in enumerate(F.segs) if s["tb"] <= F.tl]
    return o and _q(3, "speed_ratio", "Where the beads are thick, pulses move at about what fraction of their speed "
                    "where the beads are thin?", gt, *o, True, {"min_fps": fps_for(min(durs) / 2)})


def q_count_at(F, rng):
    ms = F.stable_moments()
    if not ms:
        return None
    k, win = rng.choice(ms)
    n = len(sim.active(F.segs, sim.frames_showing(k)[0]))
    o = _ints(rng, n, 0, 9)
    return o and _q(3, "count_at", f"How many separate bumps or dips are on the chain when the clock shows "
                    f"{sim.clock_text(k)}?", n, *o, False, {"k": k, "min_fps": fps_for(win)})


def q_launch_gap(F, rng):
    if F.sc != "train":
        return None
    g = Fr(F.launches[1]["f"] - F.launches[0]["f"], sim.FPS)
    gt = f"{float(g):g} s"
    pool = [f"{float(g + d):g} s" for d in (Fr(1, 2), -Fr(1, 2), Fr(1), Fr(3, 2), -Fr(1)) if g + d > 0]
    o = _four(rng, gt, pool)
    return o and _q(3, "launch_gap", "How much time passes between the paddle sending its first and its second pulse?",
                    gt, *o, True, {"min_fps": 4})


def q_spacing(F, rng):
    if F.sc != "train":
        return None
    a, b = F.lseg[0], F.lseg[1]
    v = F.c * Fr(F.launches[1]["f"] - F.launches[0]["f"], sim.FPS)
    if abs(2 * v - round(2 * v)) > Fr(1, 10):
        return None
    both = [f for f in sim.iso_frames(F.scene, F.segs, a) if f in set(sim.iso_frames(F.scene, F.segs, b))]
    if len(both) < 6:
        return None
    m = Fr(round(2 * v), 2)
    gt = f"{float(m):g} marks"
    pool = [f"{float(m + d):g} marks" for d in (Fr(1, 2), -Fr(1, 2), Fr(1), -Fr(1), Fr(3, 2)) if m + d > 0]
    o = _four(rng, gt, pool)
    return o and _q(3, "spacing", "While the first two pulses travel out from the paddle, about how far apart are their "
                    "peaks, in ruler marks?", gt, *o, False, {"min_fps": fps_for(len(both))})


def q_cross_time(F, rng):
    if F.sc != "collide":
        return None
    side = rng.choice(["left", "right"])
    i = _arrive_other(F, _first_of(F, side))
    if i is None:
        return None
    other = "right" if side == "left" else "left"
    k = sim.tenths(sim.frame_of(F.segs[i]["te"]))
    o = _clock(rng, k)
    return o and _q(3, "cross_time", f"At what time on the clock does the first pulse from the {side} paddle reach the "
                    f"{other} paddle?", sim.clock_text(k), *o, True, {"side": side, "min_fps": 4})


# ---------------------------------------------------------------- L4 integration

def _hit_fps(F, idx):
    if not idx:
        return 1
    gaps = [(F.segs[b]["te"] - F.segs[a]["te"]) * sim.FPS for a, b in zip(idx, idx[1:])]
    return fps_for(min(gaps + [F.dur(i) for i in idx]))


def q_far_hits(F, rng):
    if F.far is None:
        return None
    arr = F.arrivals(F.fx)
    n = len(arr)
    o = _ints(rng, n, 0, 12)
    return o and _q(4, "far_hits", f"Counting every pulse, big or small, how many times does a pulse reach "
                    f"{F.far_phrase()} during the video?", n, *o, True, {"min_fps": _hit_fps(F, arr)})


def q_paddle_hits(F, rng):
    side = rng.choice(["left", "right"]) if F.sc == "collide" else F.home
    arr = F.arrivals(sim.end_x(F.scene, side))
    if not arr and F.sc != "collide":
        return None
    n = len(arr)
    who = f"the {side} paddle" if F.sc == "collide" else "the paddle"
    o = _ints(rng, n, 0, 12)
    return o and _q(4, "paddle_hits", f"Not counting the pulses it sends, how many times does a pulse arrive at {who} "
                    f"during the video?", n, *o, True, {"side": side, "min_fps": _hit_fps(F, arr)})


def q_kth_far(F, rng):
    if F.sc != "train" or not F.sized:
        return None
    arr = F.arrivals(F.fx)
    if len(arr) < 2:
        return None
    k = rng.randint(2, len(arr))
    s = F.segs[arr[k - 1]]
    gt = pulse_word(1 if s["amp"] > 0 else -1, abs(s["amp"]) == 1, True)
    o = _four(rng, gt, [pulse_word(a, b, True) for a in (1, -1) for b in (True, False)])
    return o and _q(4, "kth_far", f"What is the {ORD[k - 1]} pulse to reach {F.far_phrase()}?", gt, *o, True,
                    {"k": k, "min_fps": _hit_fps(F, arr)})


def q_time_kth_far(F, rng):
    if F.sc not in ("reflect", "train"):
        return None
    arr = F.arrivals(F.fx)
    if len(arr) < 2:
        return None
    k = rng.randint(2, min(len(arr), 4))
    kt = sim.tenths(sim.frame_of(F.segs[arr[k - 1]]["te"]))
    o = _clock(rng, kt)
    return o and _q(4, "time_kth_far", f"At what time on the clock does a pulse reach {F.far_phrase()} for the "
                    f"{ORD[k - 1]} time?", sim.clock_text(kt), *o, True, {"k": k, "min_fps": 4})


def q_meet_count(F, rng):
    if F.sc != "collide":
        return None
    xs = [z for z in sim.crossings(F.segs) if z[0] <= F.tl]
    n = len(xs)
    gaps = [(b[0] - a[0]) * sim.FPS for a, b in zip(xs, xs[1:])]
    o = _ints(rng, n, 0, 12)
    return o and _q(4, "meet_count", "How many times during the video do two pulses pass through each other?", n, *o,
                    True, {"min_fps": fps_for(min(gaps + [24]))})


def q_second_meet(F, rng):
    if F.sc != "collide":
        return None
    ms = _meets(F, _first_of(F, "left"), _first_of(F, "right"))
    if len(ms) < 2:
        return None
    x = ms[1][1]
    m = round(x)
    if abs(x - m) > Fr(3, 20):
        return None
    pool = [str(v) for v in (m + 1, m - 1, m + 2, m - 2, F.L - m) if 0 <= v <= F.L]
    o = _four(rng, str(m), pool)
    return o and _q(4, "second_meet", "The first pulses from the two paddles pass through each other, bounce off the "
                    "paddles and meet again. At which ruler mark do they meet the second time?", str(m), *o, True,
                    {"min_fps": 2})


def q_toward_at(F, rng):
    ms = F.stable_moments()
    if not ms:
        return None
    k, win = rng.choice(ms)
    side = rng.choice(["left", "right"])
    act = sim.active(F.segs, sim.frames_showing(k)[0])
    n = sum(1 for i in act if (F.segs[i]["d"] < 0) == (side == "left"))
    o = _ints(rng, n, 0, 9)
    return o and _q(4, "toward_at", f"When the clock shows {sim.clock_text(k)}, how many pulses are moving toward "
                    f"the {side} end?", n, *o, True, {"k": k, "side": side, "min_fps": fps_for(win / 2)})


def q_junction_return(F, rng):
    if F.sc != "junction":
        return None
    sp = _first_split(F)
    if not sp:
        return None
    s = F.segs[sp[1]]
    if s["xe"] != F.hx or s["te"] > F.tl:
        return None
    k = sim.tenths(sim.frame_of(s["te"]))
    o = _clock(rng, k)
    return o and _q(4, "junction_return", "Part of the pulse bounces back where the beads first change. At what time on "
                    "the clock does that part reach the paddle?", sim.clock_text(k), *o, True, {"min_fps": 4})


def q_second_back(F, rng):
    if F.sc != "reflect":
        return None
    i = _first_home(F, nth=2)
    if i is None:
        return None
    rel = F.segs[i]["amp"] * F.launches[0]["sign"]
    gt = "the same way up" if rel > 0 else "upside down"
    o = _four(rng, gt, REL_OPTS)
    fps = fps_for(min(F.dur(j) for j in _chain_of(F, i)))
    return o and _q(4, "second_back", "When the pulse gets back to the paddle for the second time, how does it look "
                    "compared with how the paddle first sent it?", gt, *o, True, {"min_fps": fps})


# ---------------------------------------------------------------- L5 counterfactual (re-traced)

def _other_end(kind):
    return "free" if kind == "clamped" else "clamped"


def _cf_first_home(F, segs):
    hx = F.hx
    hits = sorted((s for s in segs if s["origin"] == 0 and s["via_far"] and s["xe"] == hx), key=lambda s: s["te"])
    return hits[0] if hits else None


def q_cf_switch_far(F, rng):
    if F.far is None:
        return None
    ends = dict(F.scene["ends"])
    ends[F.far] = _other_end(ends[F.far])
    s = _cf_first_home(F, sim.trace(F.scene, ends=ends))
    if s is None:
        return None
    gt = "the same way up" if s["amp"] * F.launches[0]["sign"] > 0 else "upside down"
    o = _four(rng, gt, ["the same way up", "upside down", "split into two smaller pulses", "it would not come back"])
    hw = HARDWARE[ends[F.far]]
    return o and _q(5, "cf_switch_far", f"If {F.far_phrase()} were held by {hw} instead, how would the first pulse look "
                    f"when it first gets back to the paddle after bouncing off that end, compared with how the paddle "
                    f"sent it?", gt, *o, True, {"min_fps": fps_for(F.dur(F.lseg[0]))})


def _clean_count(times, tl):
    return all(abs(t - tl) * sim.FPS >= CF_GAP for t in times)


def q_cf_double_speed(F, rng):
    if F.far is None:
        return None
    segs = sim.trace(F.scene, secs=sim.sections(F.scene, k=Fr(2)))
    ts = [s["te"] for s in segs if s["xe"] == F.fx]
    if not _clean_count(ts, F.tl):
        return None
    n = sum(1 for t in ts if t <= F.tl)
    o = _ints(rng, n, 0, 20)
    durs = [F.dur(i) for i, s in enumerate(F.segs) if s["tb"] <= F.tl]
    return o and _q(5, "cf_double_speed", f"If pulses moved twice as fast everywhere (same launches, same video length), "
                    f"how many times would a pulse reach {F.far_phrase()}?", n, *o, True,
                    {"min_fps": fps_for(max(durs) / 2)})


def q_cf_uniform(F, rng):
    if F.sc != "junction":
        return None
    s = _cf_first_home(F, sim.trace(F.scene, secs=sim.sections(F.scene, uniform=True)))
    if s is None or sim.frame_of(s["te"]) > 239:
        return None
    k = sim.tenths(sim.frame_of(s["te"]))
    o = _clock(rng, k)
    durs = [F.dur(i) for i, s_ in enumerate(F.segs) if s_["tb"] <= F.tl and s_["c"] == F.c]
    return o and _q(5, "cf_uniform", f"If the whole chain had thin beads, at what time on the clock would the first "
                    f"pulse get back to the paddle after bouncing off {F.far_phrase()}?", sim.clock_text(k), *o, True,
                    {"min_fps": fps_for(max(durs) / 2)})


def q_cf_heavy_swap(F, rng):
    if F.sc != "junction":
        return None
    segs = sim.trace(F.scene, secs=sim.sections(F.scene, swap=True))
    first = next(s for s in segs if s["kind"] == "launch")
    i = segs.index(first)
    if first["xe"] in (0, F.L):
        return None
    back = next((s for s in segs if s["parent"] == i and s["kind"] == "split_back"), None)
    if back is None:
        return None
    gt = SPLIT_OPTS[0] if back["amp"] * first["amp"] < 0 else SPLIT_OPTS[1]
    o = _four(rng, gt, SPLIT_OPTS)
    return o and _q(5, "cf_heavy_swap", "If the thin and thick beads swapped places, what would happen when the pulse "
                    "first reaches a place where the beads change thickness?", gt, *o, True, {"min_fps": 1})


def q_cf_shorter(F, rng):
    if F.sc not in ("reflect", "train"):
        return None
    m = rng.randint(2, F.L - 2)
    length = m if F.home == "left" else F.L - m
    t = Fr(F.launches[0]["f"], sim.FPS) + 2 * Fr(length) / F.c
    f = sim.frame_of(t)
    if f > 239:
        return None
    k = sim.tenths(f)
    o = _clock(rng, k)
    hw = HARDWARE[F.scene["ends"][F.far]]
    return o and _q(5, "cf_shorter", f"If the chain ended at the {m} mark, held there by {hw} as now, at what time on "
                    f"the clock would the first pulse first get back to the paddle?", sim.clock_text(k), *o, True,
                    {"m": m, "min_fps": fps_for(F.dur(F.lseg[0]) / 2)})


def q_cf_slow_far(F, rng):
    if F.sc not in ("reflect", "train"):
        return None
    t = Fr(F.launches[0]["f"], sim.FPS) + Fr(F.L) / (F.c / 2)
    f = sim.frame_of(t)
    if f > 239:
        return None
    k = sim.tenths(f)
    o = _clock(rng, k)
    return o and _q(5, "cf_slow_far", f"If pulses moved half as fast, at what time on the clock would the first pulse "
                    f"reach {F.far_phrase()}?", sim.clock_text(k), *o, True,
                    {"min_fps": fps_for(F.dur(F.lseg[0]) / 2)})


def q_cf_later(F, rng):
    if F.sc != "collide":
        return None
    side = rng.choice(["left", "right"])
    dt = rng.choice([Fr(1, 2), Fr(1)])
    k = _first_of(F, side)
    ls = [dict(ln) for ln in F.launches]
    ls[k]["f"] += int(dt * sim.FPS)
    segs = sim.trace(F.scene, launches=ls)
    other = _first_of(F, "right" if side == "left" else "left")
    xs = [z for z in sim.crossings(segs) if {segs[z[2]]["origin"], segs[z[3]]["origin"]} == {k, other}]
    if not xs:
        return None
    x = xs[0][1]
    m = round(x)
    if abs(x - m) > Fr(3, 20) or not 1 <= m <= F.L - 1:
        return None
    pool = [str(v) for v in (m + 1, m - 1, m + 2, m - 2, m + 3) if 0 <= v <= F.L]
    o = _four(rng, str(m), pool)
    return o and _q(5, "cf_later", f"If the {side} paddle had sent its first pulse {float(dt):g} s later, at which ruler "
                    f"mark would the first pulses from the two paddles first pass through each other?", str(m), *o,
                    True, {"side": side, "dt": str(dt), "min_fps": 2})


def q_cf_end_swap(F, rng):
    if F.sc != "collide":
        return None
    side = rng.choice(["left", "right"])
    other = "right" if side == "left" else "left"
    kind = rng.choice(["free", "clamped"])
    ends = {side: "paddle", other: kind}
    k = _first_of(F, side)
    segs = sim.trace(F.scene, ends=ends)
    hx = sim.end_x(F.scene, side)
    back = sorted((s for s in segs if s["origin"] == k and s["xe"] == hx), key=lambda s: s["te"])
    if not back:
        return None
    gt = "the same way up" if back[0]["amp"] * F.launches[k]["sign"] > 0 else "upside down"
    o = _four(rng, gt, ["the same way up", "upside down", "split into two smaller pulses", "it would not come back"])
    return o and _q(5, "cf_end_swap", f"If the {other} paddle were replaced by {HARDWARE[kind]}, how would the first "
                    f"pulse from the {side} paddle look when it gets back to the {side} paddle, compared with how it was "
                    f"sent?", gt, *o, True, {"side": side, "kind": kind,
                                             "min_fps": fps_for(F.dur(F.lseg[k]))})


def q_cf_drop_last(F, rng):
    if F.sc != "train":
        return None
    segs = sim.trace(F.scene, launches=F.launches[:-1])
    ts = [s["te"] for s in segs if s["xe"] == F.fx]
    if not _clean_count(ts, F.tl):
        return None
    n = sum(1 for t in ts if t <= F.tl)
    o = _ints(rng, n, 0, 20)
    arr = F.arrivals(F.fx)
    return o and _q(5, "cf_drop_last", f"If the paddle had not sent its last pulse, how many times would a pulse reach "
                    f"{F.far_phrase()} during the video?", n, *o, True, {"min_fps": _hit_fps(F, arr)})


def q_cf_double_meet(F, rng):
    if F.sc != "collide":
        return None
    segs = sim.trace(F.scene, secs=sim.sections(F.scene, k=Fr(2)))
    xs = [z[0] for z in sim.crossings(segs)]
    if not _clean_count(xs, F.tl):
        return None
    n = sum(1 for t in xs if t <= F.tl)
    o = _ints(rng, n, 0, 20)
    durs = [F.dur(i) for i, s in enumerate(F.segs) if s["tb"] <= F.tl]
    return o and _q(5, "cf_double_meet", "If pulses moved twice as fast (same launches, same video length), how many "
                    "times would two pulses pass through each other?", n, *o, True,
                    {"min_fps": fps_for(max(durs) / 2)})


TEMPLATES = {
    1: [q_far_kind, q_first_shape, q_n_launch, q_change_mark, q_first_sender, q_end_count],
    2: [q_far_bounce, q_first_back, q_junction_split, q_slow_part, q_first_to_end, q_trans_taller, q_order_back,
        q_which_first_far, q_meet_side, q_bigger_sender, q_dir_at],
    3: [q_speed, q_first_far_time, q_return_time, q_meet_mark, q_amp_ratio, q_refl_ratio, q_speed_ratio, q_count_at,
        q_launch_gap, q_spacing, q_cross_time],
    4: [q_far_hits, q_paddle_hits, q_kth_far, q_time_kth_far, q_meet_count, q_second_meet, q_toward_at,
        q_junction_return, q_second_back],
    5: [q_cf_switch_far, q_cf_double_speed, q_cf_uniform, q_cf_heavy_swap, q_cf_shorter, q_cf_slow_far, q_cf_later,
        q_cf_end_swap, q_cf_drop_last, q_cf_double_meet],
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
        for _ in range(4):
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
