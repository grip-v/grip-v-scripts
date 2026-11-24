"""10-question ladder for robot_archery (common/ladder5.py, LADDER = "ten").

Robots are named by colour ("the red robot", unique per clip); a shot is
named by its shooter and its place among that robot's shots ("the red
robot's second shot"). Rings are numbered from the centre (ring 1) to the
edge (ring 10), as the ring numbers on the target show. Times are clock
readings.

Margins: every shot is >= 0.6 ring from the record it is judged against
(sim.MARGIN, >= 16 px); ring readings sit >= 0.2 ring inside their ring;
compared shots differ by >= 0.6 ring; clock options are >= 0.6 s apart;
"when the clock shows X" questions keep every landing >= 0.25 s (6 frames) outside
the frames that show X. What-ifs replay the game (sim.play) with every
other turn's arrow landing where it did and are only asked when the
replay finishes within the arrows shot in the clip. Every question has
four distinct options.
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
ORD = ("first", "second", "third", "fourth", "fifth", "sixth", "seventh", "eighth", "ninth", "tenth")
RING_NOTE = " (Ring 1 is the centre ring, ring 10 the outer edge.)"
RULE_TEXT = {
    "stay": "a robot that sets a new record stayed on the spot and shot again (instead of going to the back of the "
            "queue)",
    "back": "a robot that sets a new record went to the back of the queue (instead of shooting again)",
    "one_life": "every robot were eliminated by its first shot that is not a new record",
}


# ------------------------------------------------------------------ helpers
def fps_for(gaps, floor=1):
    g = min(gaps) if gaps else 99.0
    return int(min(24, max(floor, math.ceil(1.0 / g - 1e-9))))


def fmt_clock(tenths):
    return f"{tenths / 10:.1f} s"


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
    near = [v for v in (gt + 1, gt - 1, gt + 2, gt - 2, gt + 3, gt - 3, gt + 4) if lo <= v <= hi]
    rng.shuffle(near)
    near.sort(key=lambda v: abs(v - gt))
    ex = [e for e in extra if lo <= e <= hi and e != gt]
    return mc4(rng, gt, ex[:1] + near)


def clock_opts(rng, tenths, extra=()):
    cands = [round(e) for e in extra] + [tenths + d for d in (-12, -6, 6, 12, 18, -18, 9, -9)]
    keep = []
    for c in cands:
        if c >= 0 and all(abs(c - u) >= 6 for u in keep + [tenths]):
            keep.append(c)
    head, tail = keep[:len(extra)], keep[len(extra):]
    rng.shuffle(tail)
    return mc4(rng, fmt_clock(tenths), [fmt_clock(c) for c in head + tail])


def Q(key, params, text, gt, opts, motion, fps):
    if opts is None:
        return None
    return (key, dict(params, min_fps=fps), text, gt, opts[0], opts[1], motion)


def ring_safe(d, m=0.2):
    x = d - math.floor(d)
    return m <= x <= 1 - m


def rule_variants(scene):
    """What-if rule changes offered for this scene: (tag, lives, stay_on)."""
    sc = scene["scenario"]
    if sc == "stay_on":
        return [("back", 1, False)]
    if sc == "two_lives":
        return [("one_life", 1, False)]
    return [("stay", 1, True)]


# ------------------------------------------------------------------ context
class Ctx:
    def __init__(self, scene):
        self.s = scene
        self.turns = scene["turns"]
        self.n = len(scene["robots"])
        self.cols = [r["colour"] for r in scene["robots"]]
        self.tl = sim.t_last(scene)
        self.T = scene["T"]
        self.land = scene["land"]
        self.shots = {r: [t["turn"] for t in self.turns if t["shooter"] == r] for r in range(self.n)}
        self.out_turn = {t["shooter"]: t["turn"] for t in self.turns if t["out"]}
        self.winner = scene["winner"]
        self.ghost = sorted(set(sim.PALETTE) - set(self.cols))[0]

    def nm(self, r):
        return f"the {self.cols[r]} robot"

    def shot(self, k):
        r = self.turns[k]["shooter"]
        j = self.shots[r].index(k)
        return f"{self.nm(r)}'s {ORD[j]} shot" if len(self.shots[r]) > 1 else f"{self.nm(r)}'s only shot"

    def shot_params(self, k):
        r = self.turns[k]["shooter"]
        return {"robot": r, "nth": self.shots[r].index(k) + 1}

    def names(self, first=(), add=()):
        """Robot option texts: `first`, then the rest; with fewer than four robots a robot of a colour that is
        not in the clip fills the last option (the clip never shows it, so it is never the answer)."""
        order = list(first) + [r for r in range(self.n) if r not in first]
        out = [self.nm(r) for r in order] + list(add)
        return out + [f"the {self.ghost} robot"] if len(set(out)) < 4 else out

    def clock(self, t):
        return sim.clock_tenths(sim.first_frame(t))

    def windows(self):
        """Clock readings X whose frames are >= 0.25 s (6 frames) from every landing and >= 0.6 s inside the clip."""
        out = []
        F = self.s["frames"]
        for x in range(0, 101):
            fr = [f for f in range(F) if sim.clock_tenths(f) == x]
            if not fr or fr[0] < 0.6 * FPS or fr[-1] > F - 1 - 0.6 * FPS:
                continue
            gap = min(abs(lf - f / FPS) for lf in self.land for f in fr)
            if gap >= 0.25:
                out.append((x, fr[0], gap))
        return out

    def landed_by(self, f):
        return sum(1 for lt in self.land if lt <= f / FPS)


def _distinct_max(vals, margin=1):
    s = sorted(vals, reverse=True)
    return len(s) < 2 or s[0] >= s[1] + margin


# ------------------------------------------------------------------ level 1
def level1(c: Ctx, rng):
    out = []
    n_t = len(c.turns)
    out.append(Q("n_arrows", {}, "How many arrows are on the target at the end of the clip?", n_t,
                 int_opts(rng, n_t, 2, 16), False, 1))
    for r in range(c.n):
        k = len(c.shots[r])
        out.append(Q("shots_by", {"robot": r}, f"How many arrows does {c.nm(r)} shoot?", k,
                     int_opts(rng, k, 0, 10), False, 1))
    f0 = c.turns[0]["shooter"]
    out.append(Q("first_shooter", {}, "Which robot shoots the first arrow?", c.nm(f0),
                 mc4(rng, c.nm(f0), c.names([c.turns[1]["shooter"]])), True, fps_for([c.T])))
    w = c.winner
    out.append(Q("winner", {}, "Which robot is the last one left in the game?", c.nm(w),
                 mc4(rng, c.nm(w), c.names(list(reversed(c.s["eliminated"])))), False, 1))
    return out


# ------------------------------------------------------------------ level 2
def level2(c: Ctx, rng):
    out = []
    fps_t = fps_for([c.T])
    for t in c.turns:
        k = t["turn"]
        if t["record"] and t["prev_holder"] is not None:
            gt = c.nm(t["prev_holder"])
            out.append(Q("record_beaten", c.shot_params(k),
                         f"{c.shot(k)[0].upper()}{c.shot(k)[1:]} set a new record. Whose record did it beat? "
                         "(It can be the robot's own earlier record.)", gt,
                         mc4(rng, gt, c.names([t["shooter"]])), True, fps_t))
        if k + 1 < len(c.turns):
            gt = c.nm(c.turns[k + 1]["shooter"])
            out.append(Q("shooter_after", c.shot_params(k), f"Which robot shoots right after {c.shot(k)}?", gt,
                         mc4(rng, gt, c.names([t["shooter"]])), True, fps_t))
    el = c.s["eliminated"]
    for a, b in zip(el, el[1:]):
        out.append(Q("out_after", {"robot": a}, f"Which robot is eliminated right after {c.nm(a)}?", c.nm(b),
                     mc4(rng, c.nm(b), c.names([c.winner])), True, fps_t))
    for r, k in c.out_turn.items():
        h = c.turns[k]["prev_holder"]
        out.append(Q("holder_at_out", {"robot": r}, f"When {c.nm(r)} is eliminated, which robot holds the record?",
                     c.nm(h), mc4(rng, c.nm(h), c.names([r])), True, fps_t))
    # closest of four shots (by different or same robots), clear margins; which sets qualify doesn't depend on rng
    good = [pick for pick in itertools.combinations(range(len(c.turns)), 4)
            if (lambda ds: ds[1] - ds[0] >= 0.6)(sorted(c.turns[k]["d"] for k in pick))]
    for pick in (rng.sample(good, 1) if good else []):
        best = min(pick, key=lambda k: c.turns[k]["d"])
        out.append(Q("closest_of", {"shots": [c.shot_params(k) for k in pick]},
                     "Which of these shots landed closest to the centre of the target?", c.shot(best),
                     mc4(rng, c.shot(best), [c.shot(k) for k in pick]), True, fps_t))
    return out


# ------------------------------------------------------------------ level 3
def level3(c: Ctx, rng):
    out = []
    for t in c.turns:
        k = t["turn"]
        if ring_safe(t["d"]):
            g = sim.ring_of(t["d"])
            out.append(Q("ring_of", c.shot_params(k), f"In which ring did {c.shot(k)} land?{RING_NOTE}", g,
                         int_opts(rng, g, 1, 10), True, fps_for([c.T])))
        if t["record"] and t["prev_record"] is not None:
            imp = t["prev_record"] - t["d"]
            if abs(imp - round(imp)) <= 0.3 and round(imp) >= 1:
                g = round(imp)
                out.append(Q("record_gain", c.shot_params(k),
                             f"{c.shot(k)[0].upper()}{c.shot(k)[1:]} set a new record. By about how many rings did "
                             "the record move closer to the centre?", g, int_opts(rng, g, 0, 9), True, fps_for([c.T])))
    for r, k in c.out_turn.items():
        g = c.clock(c.land[k])
        out.append(Q("out_time", {"robot": r}, f"At about what time on the clock is {c.nm(r)} eliminated?",
                     fmt_clock(g), clock_opts(rng, g, [c.clock(c.land[j]) for j in (k - 1, k + 1)
                                                       if 0 <= j < len(c.land)]), True, 4))
    recs = [t["turn"] for t in c.turns if t["record"]]
    g = c.clock(c.land[recs[-1]])
    out.append(Q("last_record_time", {}, "At about what time on the clock is the last new record set?", fmt_clock(g),
                 clock_opts(rng, g, [c.clock(c.land[j]) for j in recs[-2:-1]]), True, 4))
    wins = c.windows()
    rng.shuffle(wins)
    for x, f, gap in wins[:4]:
        k = c.landed_by(f)
        out.append(Q("arrows_at", {"x": x}, f"How many arrows are on the target when the clock shows "
                     f"{fmt_clock(x)}?", k, int_opts(rng, k, 0, 16), True, fps_for([gap])))
        rec = min((t["d"] for t in c.turns[:k] if t["record"]), default=None)
        if rec is not None and ring_safe(rec):
            g = sim.ring_of(rec)
            out.append(Q("record_ring_at", {"x": x}, f"When the clock shows {fmt_clock(x)}, in which ring is the "
                         f"record shot so far?{RING_NOTE}", g, int_opts(rng, g, 1, 10), True, fps_for([gap])))
    return out


# ------------------------------------------------------------------ level 4
def _order_opts(rng, c, seq):
    gt = " → ".join(c.cols[r] for r in seq)
    alts = [p for p in itertools.permutations(seq) if list(p) != list(seq)]
    rng.shuffle(alts)
    subs = []
    for i in range(len(seq)):
        alt = list(seq)
        alt[i] = c.winner
        subs.append(tuple(alt))
    cands = alts[:2] + subs + alts[2:]
    return gt, mc4(rng, gt, [" → ".join(c.cols[r] for r in p) for p in cands])


def level4(c: Ctx, rng):
    out = []
    fps_t = fps_for([c.T])
    gt, opts = _order_opts(rng, c, c.s["eliminated"])
    out.append(Q("elim_order", {}, "In what order are the robots eliminated (by colour, first out first)?", gt,
                 opts, True, fps_t))
    recs = {r: sum(1 for t in c.turns if t["shooter"] == r and t["record"]) for r in range(c.n)}
    for r in range(c.n):
        out.append(Q("records_by", {"robot": r}, f"How many new records does {c.nm(r)} set?", recs[r],
                     int_opts(rng, recs[r], 0, 9), True, fps_t))
    if _distinct_max(list(recs.values())):
        b = max(recs, key=recs.get)
        out.append(Q("most_records", {}, "Which robot sets the most new records?", c.nm(b),
                     mc4(rng, c.nm(b), c.names(sorted(recs, key=lambda r: -recs[r]))), True, fps_t))
    nr = sum(recs.values())
    out.append(Q("n_records", {}, "How many of the shots set a new record?", nr, int_opts(rng, nr, 1, 14), True, fps_t))
    for r, k in c.out_turn.items():
        out.append(Q("arrows_at_out", {"robot": r}, f"How many arrows have been shot when {c.nm(r)} is eliminated "
                     "(counting the shot that eliminates it)?", k + 1, int_opts(rng, k + 1, 1, 16), True, fps_t))
    if c.s["lives"] > 1:
        for x, f, gap in c.windows()[::3]:
            st = sim.play(c.s["order"], c.s["draws"][:c.landed_by(f)], c.s["lives"], c.s["stay_on"])
            lost = {t["shooter"] for t in st["turns"] if not t["record"]}
            g = c.n - len(lost)
            out.append(Q("full_pips_at", {"x": x}, f"When the clock shows {fmt_clock(x)}, how many robots still "
                         "have both of their pips?", g, int_opts(rng, g, 0, c.n), True, fps_for([gap])))
    return out


# ------------------------------------------------------------------ level 5
def _replay_ok(c, order, draws, lives, stay_on):
    g = sim.play(order, draws, lives, stay_on)
    return g if g["winner"] is not None else None


def level5(c: Ctx, rng):
    out = []
    s = c.s
    draws = s["draws"]
    for t in c.turns:
        k = t["turn"]
        own = sim.ring_of(t["d"])
        for K in range(1, 11):
            if K == own or any(K - 1 <= d <= K for j, d in enumerate(draws) if j != k):
                continue
            new = list(draws)
            new[k] = K - 0.5
            g = _replay_ok(c, s["order"], new, s["lives"], s["stay_on"])
            if g is None:
                continue
            was = "a new record" if t["record"] else "not a new record"
            text = (f"{c.shot(k)[0].upper()}{c.shot(k)[1:]} was {was}. Suppose it had landed in ring {K} instead, "
                    f"and every other turn's arrow landed exactly where it did.{RING_NOTE}")
            p = dict(c.shot_params(k), ring=K)
            w = g["winner"]
            out.append(Q("cf_shot_winner", p, text + " Which robot would win?", c.nm(w),
                         mc4(rng, c.nm(w), c.names([c.winner, t["shooter"]])), True, fps_for([c.T])))
            m = len(g["turns"])
            out.append(Q("cf_shot_arrows", p, text + " How many arrows would be shot in the whole game?", m,
                         int_opts(rng, m, 2, 16, [len(c.turns)]), True, fps_for([c.T])))
    for a, b in itertools.combinations(range(c.n), 2):
        order = [b if r == a else a if r == b else r for r in s["order"]]
        g = _replay_ok(c, order, draws, s["lives"], s["stay_on"])
        if g is None:
            continue
        w = g["winner"]
        out.append(Q("cf_swap_winner", {"a": a, "b": b},
                     f"Suppose {c.nm(a)} and {c.nm(b)} had swapped places in the starting queue, and the arrow of "
                     "every turn landed exactly where it did. Which robot would win?", c.nm(w),
                     mc4(rng, c.nm(w), c.names([c.winner, a, b])), True, fps_for([c.T])))
    for tag, lives, stay in rule_variants(s):
        g = _replay_ok(c, s["order"], draws, lives, stay)
        if g is None:
            continue
        w = g["winner"]
        out.append(Q("cf_rule_winner", {"rule": tag},
                     f"Suppose {RULE_TEXT[tag]}, and the arrow of every turn landed exactly where it did. Which robot "
                     "would win?", c.nm(w), mc4(rng, c.nm(w), c.names([c.winner])), True, fps_for([c.T])))
        el = g["eliminated"]
        if el:
            out.append(Q("cf_rule_first_out", {"rule": tag},
                         f"Suppose {RULE_TEXT[tag]}, and the arrow of every turn landed exactly where it did. Which "
                         "robot would be eliminated first?", c.nm(el[0]),
                         mc4(rng, c.nm(el[0]), c.names([c.s["eliminated"][0], w])), True, fps_for([c.T])))
    return out


# ------------------------------------------------------------------ assembly
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
        if any(cand[2] == c[2] for c in chosen):                 # never the same question text twice in a clip
            continue
        chosen.append(cand)
        used.add(k)
        keys.remove(k)
        keys.append(k)
    return chosen


def build(scene: dict, rng: random.Random) -> list[dict]:
    c = Ctx(scene)
    makers = {1: level1, 2: level2, 3: level3, 4: level4, 5: level5}
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
    texts = {lv: len({cand[2] for cand in made[lv]}) for lv in range(1, 6)}
    feasible = [cmb for cmb in TEN_SPLITS if all(texts[lv] >= cmb[lv - 1] for lv in range(1, 6))]
    if not feasible:
        raise ValueError("not enough question candidates for this scene")
    counts = rng.choice(feasible)
    qs = []
    for lv in range(1, 6):
        for key, params, text, gt, opts, letter, motion in _pick(rng, made[lv], counts[lv - 1]):
            qs.append(L.question(lv, key, text, gt, opts, letter, motion, params, idx=len(qs) + 1))
    return qs
