"""The 10-question ladder for inertia_race (common/ladder5.py, LADDER = "ten").

Per clip: 1-3 / 2-3 / 2-3 / 1-3 / 1-3 questions on levels 1-5. Candidates are
(derivation_key, params) pairs, so one pairwise template over n bodies yields
C(n,2) of them. Every template carries a precondition guard, and the per-level
counts are drawn only from the combinations the available candidates can fill
(`feasible`), so a clip never half-fills a rung.

Bodies are named by colour alone -- never by shape -- so that no prompt leaks
the answer to a shape question. Every mid-race template names a clock instant,
and that same instant is the one used to score it.
"""
from __future__ import annotations

import itertools
import math
import random

import sim
from common import ladder5 as L

LADDER = "ten"

# Templates whose answer cannot be had from a single frame.
MOTION = {
    "not_finished", "first_of_pair", "ahead_at_t", "closing_at_t", "more_turns_by_t",
    "passes_before_t", "same_finish", "pos_at_t", "finish_time", "turns_by_t",
    "release_time", "gap_at_t", "finish_order", "place_of", "win_margin",
    "leader_at_t", "crossed_by_t",
    "cf_gravity_time", "cf_from_top", "cf_as_shape", "cf_all_together", "cf_steeper_time",
}

SHAPE_WORD = {"slider": "sliding block", "solid_sphere": "solid ball",
              "solid_disc": "solid disc", "hollow_sphere": "hollow shell",
              "hoop": "thin ring"}
GRID = 0.05                      # ruler gridline spacing, in ramp fractions


def mf(frames: float) -> float:
    """min_fps for a window of `frames` frames: one sample must fall inside it."""
    return math.ceil(100 * sim.FPS / max(1, frames)) / 100


def name(r: dict) -> str:
    return f"the {r['colour']} body"


def tsec(f: int) -> str:
    return f"{sim.timer_tenths(f) / 10:.1f}"


def frac(x: float) -> str:
    return f"{x:.2f}"


def on_grid(x: float, tol: float = 0.004) -> bool:
    return abs(x / GRID - round(x / GRID)) * GRID <= tol


def snap(x: float) -> float:
    return round(round(x / GRID) * GRID, 2)


# --------------------------------------------------------------------------- context


class Ctx:
    """Everything the templates read, derived once per clip."""

    def __init__(self, scene: dict):
        self.scene = scene
        self.t0 = scene["t0"]
        self.end = scene["n_frames"] - 1
        self.rs = scene["racers"]
        self.by = {r["id"]: r for r in self.rs}
        self.n = len(self.rs)
        self.fin = {r["id"]: r["finish_f"] for r in self.rs}
        self.order = sim.order_of(scene)
        self.cross = sim.crossings(scene)
        self.clear = sim.clear_finishers(scene)
        # frames a question may name: inside the readable window and on the timer
        self.ask_f = [f for f in range(sim.EDGE_F, self.end - sim.EDGE_F + 1)
                      if sim.aligned(f)]
        self.pos = {r["id"]: [sim.pos_at(r, self.t0, f) for f in range(scene["n_frames"])]
                    for r in self.rs}
        self.turn = {r["id"]: [sim.spin_turns(r, self.t0, f) for f in range(scene["n_frames"])]
                     for r in self.rs}

    def rate(self, a: str, b: str, f: int) -> float:
        """How fast the gap between a and b is opening (+) or closing (-), per second."""
        if f + 1 >= self.scene["n_frames"]:
            return 0.0
        d0 = abs(self.pos[a][f] - self.pos[b][f])
        d1 = abs(self.pos[a][f + 1] - self.pos[b][f + 1])
        return (d1 - d0) * sim.FPS

    def live(self, rid: str, f: int) -> bool:
        """Still on the ramp and already released at frame f."""
        r = self.by[rid]
        return r["release_f"] < f and self.pos[rid][f] < 1.0

    def pairs(self):
        return list(itertools.combinations([r["id"] for r in self.rs], 2))


# --------------------------------------------------------------------------- candidates


def _sig(c: Ctx, key: str, p: dict):
    """The answer a candidate would carry, cheaply, for balancing the pool.

    A test asserts this agrees with the groundtruth `make` produces, so the two
    cannot drift apart.
    """
    by, pos, turn, fin = c.by, c.pos, c.turn, c.fin
    if key == "shape_of":
        return by[p["r"]]["shape"]
    if key == "n_racers":
        return c.n
    if key in ("not_finished", "widest", "starts_lowest", "leader_at_t",
               "cf_all_together", "first_of_pair", "ahead_at_t", "more_turns_by_t"):
        if key == "not_finished":
            return next(i for i in by if fin[i] is None)
        if key == "widest":
            return max(c.rs, key=lambda x: x["r_px"])["id"]
        if key == "starts_lowest":
            return max(c.rs, key=lambda x: x["start"])["id"]
        if key == "leader_at_t":
            return max(by, key=lambda i: pos[i][p["f"]])
        if key == "cf_all_together":
            return min(c.rs, key=lambda r: sim.TAU[r["k_shape"]])["id"]
        a, b = p["a"], p["b"]
        if key == "first_of_pair":
            return a if fin[a] < fin[b] else b
        if key == "ahead_at_t":
            return a if pos[a][p["f"]] > pos[b][p["f"]] else b
        return a if turn[a][p["f"]] > turn[b][p["f"]] else b
    if key == "same_finish":
        return fin[p["a"]] is not None and fin[p["a"]] == fin[p["b"]]
    if key == "passes_before_t":
        return any(x in (p["a"], p["b"]) and y in (p["a"], p["b"]) and g <= p["f"]
                   for x, y, g in c.cross)
    if key == "closing_at_t":
        return c.rate(p["a"], p["b"], p["f"]) > 0
    if key == "pos_at_t":
        return snap(pos[p["r"]][p["f"]])
    if key == "gap_at_t":
        return snap(abs(pos[p["a"]][p["f"]] - pos[p["b"]][p["f"]]))
    if key == "finish_time":
        return sim.timer_tenths(fin[p["r"]])
    if key == "release_time":
        return sim.timer_tenths(by[p["r"]]["release_f"])
    if key == "turns_by_t":
        return int(round(turn[p["r"]][p["f"]]))
    if key == "crossed_by_t":
        return sum(1 for i in by if fin[i] is not None and fin[i] <= p["f"])
    if key == "place_of":
        return c.clear.index(p["r"]) + 1
    if key in ("cf_gravity_time", "cf_steeper_time", "cf_from_top", "cf_as_shape"):
        r = by[p["r"]]
        if key == "cf_as_shape":
            v = _cf_time(c, r, shape=p["shape"])
        elif key == "cf_from_top":
            v = _cf_time(c, r, start=0.0)
        else:
            v = _cf_time(c, r, g_mul=p["x"] if key == "cf_gravity_time" else 1.0,
                         sin_mul=p["x"] if key == "cf_steeper_time" else 1.0)
        return round(v, 1)
    return None                       # single-candidate keys need no balancing


BALANCE_CAP = 3                       # most candidates kept per distinct answer


def _balance(c: Ctx, pool: list) -> list:
    """Even out a level's pool so each template offers each answer equally often.

    Several of these questions have an answer distribution that is lopsided by
    physics -- a quicker body nearly always pulls away, a body nearly always has
    turned once by mid-race -- and no precondition fixes that. Offering the same
    number of candidates for every answer a template can give in this clip does.
    """
    groups: dict = {}
    for key, p in pool:
        groups.setdefault(key, {}).setdefault(_sig(c, key, p), []).append((key, p))
    out = []
    for key, vals in groups.items():
        q = min(BALANCE_CAP, min(len(v) for v in vals.values()))
        for v in vals.values():
            out += v[:q]
    return out


def candidates(c: Ctx) -> dict[int, list]:
    k: dict[int, list] = {1: [], 2: [], 3: [], 4: [], 5: []}
    ids = [r["id"] for r in c.rs]

    # ---------------- L1 observation
    k[1] += [("shape_of", {"r": i}) for i in ids]
    k[1].append(("n_racers", {}))
    dnf = [i for i in ids if c.fin[i] is None]
    if len(dnf) == 1:
        k[1].append(("not_finished", {}))
    big = sorted(c.rs, key=lambda r: -r["r_px"])
    if c.n >= 2 and big[0]["r_px"] - big[1]["r_px"] >= 4:
        k[1].append(("widest", {}))
    if c.scene["scenario"] in ("handicap", "pursuit"):
        ah = sorted(c.rs, key=lambda r: -r["start"])
        if ah[0]["start"] - ah[1]["start"] >= 0.03:
            k[1].append(("starts_lowest", {}))

    # ---------------- L2 relational: four pairwise templates are time-grounded
    for a, b in c.pairs():
        near: list[int] = []
        fa, fb = c.fin[a], c.fin[b]
        if fa is not None and fb is not None and abs(fa - fb) >= sim.ORDER_GAP_F:
            k[2].append(("first_of_pair", {"a": a, "b": b}))
        elif fa is not None and fb is not None and fa == fb:
            k[2].append(("same_finish", {"a": a, "b": b}))     # the dead heat, "yes"
        for f in c.ask_f:
            if c.live(a, f) and c.live(b, f) and abs(c.pos[a][f] - c.pos[b][f]) >= 0.07:
                k[2].append(("ahead_at_t", {"a": a, "b": b, "f": f}))
            if (c.by[a]["shape"] != "slider" and c.by[b]["shape"] != "slider"
                    and c.live(a, f) and c.live(b, f)
                    and abs(c.turn[a][f] - c.turn[b][f]) >= 0.6):
                k[2].append(("more_turns_by_t", {"a": a, "b": b, "f": f}))
            # whether the gap is opening or closing *at that instant* turns on which
            # body is ahead as well as which is quicker, so unlike "has the gap grown
            # since the start" it is not answerable from the shapes alone
            if c.live(a, f) and c.live(b, f) and abs(c.rate(a, b, f)) >= 0.05:
                near.append(f)
        # only worth asking where the gap is seen doing both at some point: for a
        # pair that only ever draws apart, "opening" is right without watching
        if any(c.rate(a, b, f) > 0 for f in near) and any(c.rate(a, b, f) < 0 for f in near):
            k[2] += [("closing_at_t", {"a": a, "b": b, "f": f}) for f in near]
        # An overtake measured off the drawing can land a frame either side of the one
        # computed here, so never put the named instant next to a crossing; and take as
        # many instants before the overtake as after, so the key is not one-sided.
        gs = [g for x, y, g in c.cross if x in (a, b) and y in (a, b)]
        if gs:
            ok = [f for f in c.ask_f if c.live(a, f) and c.live(b, f)
                  and all(abs(g - f) >= 4 for g in gs)]
            pre = [f for f in ok if not any(g <= f for g in gs)]
            post = [f for f in ok if any(g <= f for g in gs)]
            take = min(len(pre), len(post), 4)
            for f in pre[-take:] + post[:take]:
                k[2].append(("passes_before_t", {"a": a, "b": b, "f": f}))

    # "do they arrive together" is only worth asking where a dead heat is on the
    # table at all, and then only against as many separated pairs as tied ones
    tied = [(x, y) for x, y in c.pairs()
            if c.fin[x] is not None and c.fin[x] == c.fin[y]]
    if tied:
        apart = [(x, y) for x, y in c.pairs()
                 if c.fin[x] is not None and c.fin[y] is not None
                 and abs(c.fin[x] - c.fin[y]) >= sim.ORDER_GAP_F]
        for x, y in apart[:len(tied)]:
            k[2].append(("same_finish", {"a": x, "b": y}))

    # ---------------- L3 measurement: read a number off the ruler or the clock
    for i in ids:
        for f in c.ask_f:
            if c.live(i, f) and on_grid(c.pos[i][f]):
                k[3].append(("pos_at_t", {"r": i, "f": f}))
        if c.fin[i] is not None and sim.aligned(c.fin[i]):
            k[3].append(("finish_time", {"r": i}))
        if c.by[i]["shape"] != "slider":
            hits = [(f, int(round(c.turn[i][f]))) for f in c.ask_f
                    if c.live(i, f) and abs(c.turn[i][f] - round(c.turn[i][f])) <= 0.12
                    and round(c.turn[i][f]) >= 1]
            # a body whose count is the same at every askable instant makes the
            # answer guessable from the template alone
            if len({v for _, v in hits}) >= 3:
                k[3] += [("turns_by_t", {"r": i, "f": f}) for f, _ in hits]
        if c.scene["scenario"] == "stagger" and c.by[i]["release_f"] >= sim.EDGE_F:
            k[3].append(("release_time", {"r": i}))
    for a, b in c.pairs():
        for f in c.ask_f:
            if c.live(a, f) and c.live(b, f):
                g = abs(c.pos[a][f] - c.pos[b][f])
                if g >= 0.10 and on_grid(g):
                    k[3].append(("gap_at_t", {"a": a, "b": b, "f": f}))

    # ---------------- L4 integration: combine events across the whole clip
    finishers = [i for i in ids if c.fin[i] is not None]
    # order and placing are only askable when no two finishers arrive together:
    # a viewer cannot rank a dead heat, and neither can answers.py
    if c.clear:
        k[4].append(("finish_order", {}))
        if len(c.clear) >= 3:        # with two finishers "first" is a coin toss
            k[4] += [("place_of", {"r": i}) for i in c.clear]
    if len(finishers) >= 2:
        ff = sorted(c.fin[i] for i in finishers)
        if ff[-1] - ff[0] >= sim.ORDER_GAP_F:
            k[4].append(("win_margin", {}))
    for f in c.ask_f:
        lead = sorted(ids, key=lambda i: -c.pos[i][f])
        if c.pos[lead[0]][f] - c.pos[lead[1]][f] >= 0.07:
            k[4].append(("leader_at_t", {"f": f}))
        # with two bodies the count at a mid-race instant is nearly always 1,
        # which is guessable without watching
        if c.n >= 3:
            k[4].append(("crossed_by_t", {"f": f}))

    # ---------------- L5 counterfactual: every one is re-derived, and numeric
    # A counterfactual about a body that is already breaking the rules is
    # ill-posed, and could not be re-derived from its drawn shape either.
    for i in [x for x in finishers if c.by[x]["violates"] == "none"]:
        k[5].append(("cf_gravity_time", {"r": i, "x": 2}))
        k[5].append(("cf_steeper_time", {"r": i, "x": 2}))
        if c.by[i]["start"] > 0.02:
            k[5].append(("cf_from_top", {"r": i}))
        for sh in sim.SHAPES:
            if sh != c.by[i]["shape"]:
                k[5].append(("cf_as_shape", {"r": i, "shape": sh}))
    # only askable when one body is strictly the quickest: with two bodies of equal
    # inertia the "winner" is a coin toss, and the two derivations break the tie
    # differently
    taus = sorted(sim.TAU[r["k_shape"]] for r in c.rs)
    if c.n >= 2 and taus[1] - taus[0] > 1e-9:
        k[5].append(("cf_all_together", {}))
    return {lv: _balance(c, v) for lv, v in k.items()}


# --------------------------------------------------------------------------- phrasing


# Two-body questions get a pair of options the guards make impossible, as the
# other categories on this pipeline do, so that a coin flip is not worth half.
PAIR_FOILS = ("they are level", "neither of them")


def _mcq(rng, lv, key, text, gt, distractors, params, fps, fmt=str):
    opts, a = L.mc(rng, gt, distractors, fmt=fmt)
    return L.question(lv, key, text, fmt(gt), opts, a, key in MOTION,
                      dict(params, min_fps=fps))


def _pair_mcq(rng, lv, key, text, gt, other, params, fps):
    return _mcq(rng, lv, key, text, gt, [other, *PAIR_FOILS], params, fps)


def make(c: Ctx, lv: int, key: str, p: dict, rng: random.Random) -> dict:
    by, pos, turn = c.by, c.pos, c.turn
    ids = [r["id"] for r in c.rs]
    others = lambda r: [name(x) for x in c.rs if x["id"] != r]

    if key == "shape_of":
        r = by[p["r"]]
        gt = SHAPE_WORD[r["shape"]]
        return _mcq(rng, lv, key, f"What kind of body is {name(r)}?", gt,
                    [SHAPE_WORD[s] for s in sim.SHAPES if s != r["shape"]],
                    p, mf(c.scene["n_frames"]))
    if key == "n_racers":
        return _mcq(rng, lv, key, "How many bodies race down the ramp?", c.n,
                    L.near_ints(c.n, 2, 6), p, mf(c.scene["n_frames"]))
    if key == "not_finished":
        r = next(i for i in ids if c.fin[i] is None)
        return _mcq(rng, lv, key,
                    "Which body had still not reached the finish line when the clip ended?",
                    name(by[r]), others(r), p, mf(sim.EDGE_F))
    if key == "widest":
        r = max(c.rs, key=lambda x: x["r_px"])
        return _mcq(rng, lv, key, "Which body is drawn with the largest radius?",
                    name(r), others(r["id"]), p, mf(c.scene["n_frames"]))
    if key == "starts_lowest":
        r = max(c.rs, key=lambda x: x["start"])
        return _mcq(rng, lv, key,
                    "Which body starts furthest down the ramp, closest to the finish line?",
                    name(r), others(r["id"]), p, mf(sim.EDGE_F))

    if key == "first_of_pair":
        a, b = p["a"], p["b"]
        w = a if c.fin[a] < c.fin[b] else b
        return _pair_mcq(rng, lv, key,
                         f"Which crosses the finish line first, {name(by[a])} or "
                         f"{name(by[b])}?", name(by[w]), name(by[b if w == a else a]),
                         p, mf(abs(c.fin[a] - c.fin[b])))
    if key == "same_finish":
        a, b = p["a"], p["b"]
        same = c.fin[a] is not None and c.fin[a] == c.fin[b]
        return _mcq(rng, lv, key,
                    f"Do {name(by[a])} and {name(by[b])} reach the finish line at the "
                    f"same moment?", "yes" if same else "no",
                    ["no" if same else "yes"], p, mf(sim.ORDER_GAP_F))
    if key == "ahead_at_t":
        a, b, f = p["a"], p["b"], p["f"]
        w = a if pos[a][f] > pos[b][f] else b
        return _pair_mcq(rng, lv, key,
                         f"At {tsec(f)} s on the clock, which is further down the ramp, "
                         f"{name(by[a])} or {name(by[b])}?", name(by[w]),
                         name(by[b if w == a else a]), p, mf(6))
    if key == "more_turns_by_t":
        a, b, f = p["a"], p["b"], p["f"]
        w = a if turn[a][f] > turn[b][f] else b
        return _pair_mcq(rng, lv, key,
                         f"By {tsec(f)} s on the clock, which has turned through more "
                         f"revolutions, {name(by[a])} or {name(by[b])}?", name(by[w]),
                         name(by[b if w == a else a]), p, mf(6))
    if key == "closing_at_t":
        a, b, f = p["a"], p["b"], p["f"]
        gt = "opening up" if c.rate(a, b, f) > 0 else "closing"
        return _mcq(rng, lv, key,
                    f"At {tsec(f)} s on the clock, is the gap along the ramp between "
                    f"{name(by[a])} and {name(by[b])} opening up or closing?", gt,
                    ["closing" if gt == "opening up" else "opening up"], p, mf(6))
    if key == "passes_before_t":
        a, b, f = p["a"], p["b"], p["f"]
        got = any(x in (a, b) and y in (a, b) and g <= f for x, y, g in c.cross)
        return _mcq(rng, lv, key,
                    f"Before the clock reaches {tsec(f)} s, does either of {name(by[a])} "
                    f"and {name(by[b])} overtake the other?",
                    "yes" if got else "no", ["no" if got else "yes"], p, mf(6))

    if key == "pos_at_t":
        r, f = p["r"], p["f"]
        g = snap(pos[r][f])
        ds = [snap(g + d) for d in (0.10, -0.10, 0.20, -0.20, 0.15)
              if 0.0 <= g + d <= 1.0]
        return _mcq(rng, lv, key,
                    f"Reading off the ruler, how far down the ramp is {name(by[r])} "
                    f"at {tsec(f)} s on the clock?", g, ds, p, mf(6), fmt=frac)
    if key == "finish_time":
        r = p["r"]
        t = sim.timer_tenths(c.fin[r]) / 10
        return _mcq(rng, lv, key,
                    f"What does the clock read when {name(by[r])} crosses the finish line?",
                    t, [round(t + d, 1) for d in (0.7, -0.7, 1.4, -1.4) if t + d > 0],
                    p, 2 / 0.7, fmt=lambda v: f"{v:.1f}")
    if key == "turns_by_t":
        r, f = p["r"], p["f"]
        g = int(round(turn[r][f]))
        return _mcq(rng, lv, key,
                    f"How many complete turns has {name(by[r])} made by {tsec(f)} s "
                    f"on the clock?", g, L.near_ints(g, 0, 20), p, mf(6))
    if key == "release_time":
        r = p["r"]
        t = sim.timer_tenths(by[r]["release_f"]) / 10
        return _mcq(rng, lv, key,
                    f"What does the clock read when {name(by[r])} is let go?", t,
                    [round(t + d, 1) for d in (0.7, -0.7, 1.4) if t + d >= 0],
                    p, 2 / 0.7, fmt=lambda v: f"{v:.1f}")
    if key == "gap_at_t":
        a, b, f = p["a"], p["b"], p["f"]
        g = snap(abs(pos[a][f] - pos[b][f]))
        return _mcq(rng, lv, key,
                    f"At {tsec(f)} s on the clock, how far apart along the ramp are "
                    f"{name(by[a])} and {name(by[b])}, in ruler units?", g,
                    [snap(g + d) for d in (0.10, -0.10, 0.20) if 0 < g + d < 1],
                    p, mf(6), fmt=frac)

    if key == "finish_order":
        seq = c.clear
        gt = " then ".join(by[i]["colour"] for i in seq)
        # every other ordering, not random shuffles: with two finishers a shuffle
        # can return the right order every time and leave the question one-sided
        alts = [" then ".join(by[i]["colour"] for i in q)
                for q in itertools.permutations(seq) if list(q) != list(seq)]
        rng.shuffle(alts)
        return _mcq(rng, lv, key,
                    "Give the order in which the bodies cross the finish line, "
                    "from first to last.", gt, alts, p, mf(sim.ORDER_GAP_F))
    if key == "place_of":
        r = p["r"]
        seq = c.clear
        g = seq.index(r) + 1
        return _mcq(rng, lv, key, f"In what place does {name(by[r])} finish?", g,
                    L.near_ints(g, 1, len(seq)), p, mf(sim.ORDER_GAP_F))
    if key == "win_margin":
        fs = sorted(c.fin[i] for i in ids if c.fin[i] is not None)
        t = round((fs[-1] - fs[0]) / sim.FPS, 1)
        return _mcq(rng, lv, key,
                    "How many seconds after the winner does the last body to finish "
                    "cross the line?", t,
                    [round(t + d, 1) for d in (0.7, -0.7, 1.4) if t + d > 0],
                    p, 2 / 0.7, fmt=lambda v: f"{v:.1f}")
    if key == "leader_at_t":
        f = p["f"]
        w = max(ids, key=lambda i: pos[i][f])
        return _mcq(rng, lv, key,
                    f"At {tsec(f)} s on the clock, which body is furthest down the ramp?",
                    name(by[w]), others(w), p, mf(6))
    if key == "crossed_by_t":
        f = p["f"]
        g = sum(1 for i in ids if c.fin[i] is not None and c.fin[i] <= f)
        return _mcq(rng, lv, key,
                    f"How many bodies have crossed the finish line by {tsec(f)} s "
                    f"on the clock?", g, L.near_ints(g, 0, c.n), p, mf(6))

    if key in ("cf_gravity_time", "cf_steeper_time", "cf_from_top", "cf_as_shape",
               "cf_all_together"):
        return _counterfactual(c, lv, key, p, rng)
    raise KeyError(key)


def _cf_time(c: Ctx, r: dict, *, g_mul=1.0, sin_mul=1.0, shape=None, start=None) -> float:
    """Crossing time under an altered world, re-derived from the closed form."""
    sh = shape or r["k_shape"]
    s0 = r["start"] if start is None else start
    t = sim.TAU[sh] * c.t0 * math.sqrt(max(0.0, 1.0 - s0)) / math.sqrt(g_mul * sin_mul)
    return t + r["release_f"] / sim.FPS


def _counterfactual(c: Ctx, lv: int, key: str, p: dict, rng: random.Random) -> dict:
    one = lambda v: f"{v:.1f}"
    if key == "cf_all_together":
        w = min(c.rs, key=lambda r: sim.TAU[r["k_shape"]])
        return _mcq(rng, lv, key,
                    "Suppose every body were held at the very top of the ramp and all "
                    "let go at the same moment. Which would reach the finish line first?",
                    name(w), [name(x) for x in c.rs if x["id"] != w["id"]],
                    p, mf(c.scene["n_frames"]))
    r = c.by[p["r"]]
    if key == "cf_gravity_time":
        gt = round(_cf_time(c, r, g_mul=p["x"]), 1)
        q = (f"Suppose gravity were {p['x']} times stronger, everything else unchanged. "
             f"How long after being let go would {name(r)} take to reach the line?")
    elif key == "cf_steeper_time":
        gt = round(_cf_time(c, r, sin_mul=p["x"]), 1)
        q = (f"Suppose the ramp were tilted so that it pulled the bodies along {p['x']} "
             f"times as hard. How long after being let go would {name(r)} take to reach "
             f"the line?")
    elif key == "cf_from_top":
        gt = round(_cf_time(c, r, start=0.0), 1)
        q = (f"Suppose {name(r)} had started at the very top of the ramp, at 0.00 on the "
             f"ruler, instead of its own start mark. How long after being let go would it "
             f"take to reach the line?")
    else:
        gt = round(_cf_time(c, r, shape=p["shape"]), 1)
        q = (f"Suppose {name(r)} were replaced by a {SHAPE_WORD[p['shape']]} of the same "
             f"size, released the same way. How long after being let go would it take to "
             f"reach the line?")
    ds = [round(gt + d, 1) for d in (0.8, -0.8, 1.6, -1.6, 2.4) if gt + d > 0.2]
    return _mcq(rng, lv, key, q, gt, ds, p, 2 / 0.7, fmt=one)


# --------------------------------------------------------------------------- the draw


def _draws(caps: list[int]) -> list[tuple]:
    """Per-level count combinations that sum to 10 and fit the available candidates."""
    return [cmb for cmb in L._count_combos()
            if sum(cmb) == L.TEN_TOTAL and all(k <= cap for k, cap in zip(cmb, caps))]


def _caps(cand: dict[int, list]) -> list[int]:
    """Distinct (key, params) pairs available on each level."""
    out = []
    for lv in range(1, 6):
        seen = {(k, tuple(sorted(p.items()))) for k, p in cand[lv]}
        out.append(len(seen))
    return out


def feasible(scene: dict) -> bool:
    """True when some legal per-level draw can be filled from this scene."""
    return bool(_draws(_caps(candidates(Ctx(scene)))))


def _pick(rng: random.Random, pool: list, k: int) -> list:
    """k distinct (key, params) pairs, spreading keys before repeating any."""
    out, used = [], set()
    by_key: dict[str, list] = {}
    for key, p in pool:
        by_key.setdefault(key, []).append((key, p))
    while len(out) < k:
        fresh = [key for key in by_key if key not in used and by_key[key]]
        bag = fresh or [key for key in by_key if by_key[key]]
        if not bag:
            raise ValueError("not enough question candidates")
        key = rng.choice(sorted(bag))
        cand = by_key[key]
        out.append(cand.pop(rng.randrange(len(cand))))
        used.add(key)
    return out


def build(scene: dict, rng: random.Random) -> list[dict]:
    c = Ctx(scene)
    cand = candidates(c)
    draws = _draws(_caps(cand))
    if not draws:
        raise ValueError("no feasible level counts for the available candidates")
    counts = list(rng.choice(sorted(draws)))
    qs = []
    for lv, k in zip(range(1, 6), counts):
        for key, p in _pick(rng, cand[lv], k):
            qs.append(make(c, lv, key, p, rng))
    return L.number(qs)