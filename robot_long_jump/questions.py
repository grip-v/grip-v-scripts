"""10-question ladder for robot_long_jump (common/ladder5.py, LADDER = "ten").

Per clip: 1-3 / 2-3 / 2-3 / 1-3 / 1-3 questions on levels 1-5 (L5.ten_counts).
Templates are drawn first (motion templates weighted up), then their params;
every template is used once before any template repeats with new params, and
no derivation_key + params pair repeats within a clip.

Robots are named by colour ("the teal robot"); colours are unique within a
clip and no robot is red (red is the foul line). Positions are read off the
ruler (gridlines every 0.1, numbers at 0.0 / 0.5 / 1.0 / 1.5 / 2.0); times
are read off the on-screen timer. Every answer here comes from sim.py;
answers.py re-derives it from the renderer's per-frame geometry instead.
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

LADDER = "ten"
RULE = ("A robot's score is its take-off position plus its jump length, i.e. the spot where it lands; "
        "a robot that hops past the red foul line before jumping fouls and scores 0.")
PLACES = ("1st", "2nd", "3rd", "4th", "5th", "6th")
TIME_STEP = 7                    # tenths of a second between time options
ORDER_GAP_F = 8                  # events compared in time are >= 8 frames (0.33 s) apart
MOTION = {"first_land", "first_to_jump", "last_to_land", "land_time", "landed_by", "leader_after",
          "lead_changes", "landed_before", "cf_lead_changes", "takeoff_time", "cf_leader_after", "cf_wait_beat"}
NONE_FOULS = "none of them"


def name(r) -> str:
    return f"the {r['colour']} robot"


def mf(frames: float) -> float:
    """min_fps for a window of `frames` frames: one sample must fall inside it."""
    return math.ceil(100 * sim.FPS / max(1, frames)) / 100


def t_txt(tenths: int) -> str:
    return f"{tenths // 10}.{tenths % 10} s"


class Ctx:
    def __init__(self, scene):
        self.scene = scene
        self.n = scene["n_frames"]
        self.rs = scene["robots"]
        self.by = {r["id"]: r for r in self.rs}
        self.jumpers = [r for r in self.rs if not r["fouled"]]
        self.fouled = [r for r in self.rs if r["fouled"]]
        self.feet = {r["id"]: sim.feet(r) for r in self.rs}
        self.score = {r["id"]: sim.score(r) for r in self.rs}
        self.land = sim.landing_frames(scene)
        tip = scene["timing"]["tip_f"]
        self.done = {r["id"]: r["end_frame"] + (tip if r["fouled"] else 0) for r in self.rs}
        self.all_done = max(self.done.values())
        self.seq = scene["scenario"] in sim.SEQUENTIAL

    def win(self, ids=None) -> float:
        """min_fps of a static answer: visible from when the robots' attempts are over to the end."""
        last = max(self.done[i] for i in ids) if ids else self.all_done
        return mf(self.n - last)

    def land_gap(self, i) -> int:
        """Frames between robot i's sand landing and the nearest other sand landing."""
        return min((abs(self.land[i] - f) for j, f in self.land.items() if j != i), default=10 ** 6)

    def place(self, i, scores=None) -> int:
        sc = scores or self.score
        return 1 + sum(1 for j in sc if sc[j] > sc[i])


def _mcq(rng, lv, key, text, gt, distractors, params, fps, fmt=str):
    opts, a = L.mc(rng, gt, distractors, fmt=fmt)
    return L.question(lv, key, text, fmt(gt), opts, a, key in MOTION, dict(params, min_fps=fps))


def _near_pos(rng, gt, semantic=(), lo=0, hi=20):
    near = [gt + d for d in (-2, -1, 1, 2, 3, -3) if lo <= gt + d <= hi]
    rng.shuffle(near)
    return [s for s in semantic if s is not None and lo <= s <= hi and s != gt] + near


def _int_q(rng, lv, key, text, gt, params, fps, lo=0, hi=10 ** 6):
    return _mcq(rng, lv, key, text, gt, L.near_ints(gt, lo, hi), params, fps)


def _pos_q(rng, lv, key, text, gt, params, fps, semantic=(), lo=0, hi=20):
    return _mcq(rng, lv, key, text, gt, _near_pos(rng, gt, semantic, lo, hi), params, fps, fmt=sim.fmt)


def _robot_q(rng, lv, key, text, gt_r, pool, params, fps, pad):
    others = [name(r) for r in pool if r["id"] != gt_r["id"]]
    rng.shuffle(others)
    return _mcq(rng, lv, key, text, name(gt_r), others + [pad], params, fps)


def _pair_q(rng, lv, key, text, gt_r, a, b, same, neither, fps):
    opts = [name(a), name(b), same, neither]
    rng.shuffle(opts)
    return L.question(lv, key, text, name(gt_r), opts, L.LETTERS[opts.index(name(gt_r))], key in MOTION,
                      {"a": a["id"], "b": b["id"], "min_fps": fps})


# --------------------------------------------------------------------------- candidates


def candidates(c: Ctx) -> dict:
    sc = c.scene
    J = c.jumpers
    cand = {lv: [] for lv in range(1, 6)}
    # L1 observation
    cand[1] += [("hops_jump", {"r": r["id"]}) for r in J]
    cand[1] += [("hops_foul", {"r": r["id"]}) for r in c.fouled]
    cand[1] += [("n_fouls", {})]
    cand[1] += [("shape", {"r": r["id"]}) for r in c.rs]
    if len(c.fouled) <= 1:
        cand[1] += [("who_fouls", {})]
    # L2 relational
    for a, b in itertools.combinations(J, 2):
        cand[2].append(("farther_landing", {"a": a["id"], "b": b["id"]}))
        if sim.takeoff(a) != sim.takeoff(b):
            cand[2].append(("farther_takeoff", {"a": a["id"], "b": b["id"]}))
        if a["jump"] != b["jump"]:
            cand[2].append(("longer_jump", {"a": a["id"], "b": b["id"]}))
        if abs(c.land[a["id"]] - c.land[b["id"]]) >= ORDER_GAP_F:
            cand[2].append(("first_land", {"a": a["id"], "b": b["id"]}))
    for a, b in itertools.permutations(c.rs, 2):
        if len(a["hops"]) > len(b["hops"]):
            cand[2].append(("hop_diff", {"a": a["id"], "b": b["id"]}))
    cand[2].append(("top_scorer", {}))
    lands = sorted(c.land.values())
    if not c.seq and len(lands) >= 2:
        if lands[1] - lands[0] >= ORDER_GAP_F:
            cand[2].append(("first_to_jump", {}))
        if lands[-1] - lands[-2] >= ORDER_GAP_F:
            cand[2].append(("last_to_land", {}))
    # L3 measurement
    cand[3] += [("takeoff", {"r": r["id"]}) for r in J]
    cand[3] += [("jump_length", {"r": r["id"]}) for r in J]
    cand[3] += [("longest_hop", {"r": r["id"]}) for r in c.rs if len(r["hops"]) >= 2]
    if sc["scenario"] == "staggered":
        cand[3] += [("start_mark", {"r": r["id"]}) for r in c.rs]
    cand[3] += [("land_time", {"r": r["id"]}) for r in J if sim.aligned(c.land[r["id"]])]
    # take-off: the robot still stands on its spot at jump_frame and is airborne from the next frame;
    # both frames show the same timer reading
    cand[3] += [("takeoff_time", {"r": r["id"]}) for r in J if sim.aligned(r["jump_frame"]) and
                sim.timer_tenths(r["jump_frame"]) == sim.timer_tenths(r["jump_frame"] + 1)]
    for tt in range(5, sim.timer_tenths(c.n - 1)):
        g = next(f for f in range(c.n) if sim.timer_tenths(f) == tt)
        before = sum(1 for f in c.land.values() if f <= g)
        if 1 <= before < len(c.land) and min(abs(f - g) for f in c.land.values()) >= ORDER_GAP_F and tt % 5 == 0:
            cand[3].append(("landed_by", {"t": tt}))
    # L4 integration
    cand[4] += [("score", {"r": r["id"]}) for r in J]
    cand[4] += [("place", {"r": r["id"]}) for r in J]
    cand[4].append(("win_margin", {}))
    trios = [t for t in itertools.combinations(c.rs, 3) if sum(r["fouled"] for r in t) <= 1]
    cand[4] += [("ranking", {"ids": [r["id"] for r in t]}) for t in trios]
    if len(c.land) >= 3 and sim.lands_spread(sc) >= ORDER_GAP_F:
        cand[4].append(("lead_changes", {}))
    for r in J:
        if c.land_gap(r["id"]) >= ORDER_GAP_F and len(c.land) >= 2:
            cand[4].append(("leader_after", {"r": r["id"]}))
        jf = r["jump_frame"]
        if all(abs(f - jf) >= ORDER_GAP_F for j, f in c.land.items() if j != r["id"]):
            cand[4].append(("landed_before", {"r": r["id"]}))
    # L5 counterfactual (re-simulated in sim / re-derived from geometry in answers.py)
    cand[5] += [("cf_jump_earlier", {"r": r["id"]}) for r in J if len(r["hops"]) >= 2]
    for r in J:
        ft = c.feet[r["id"]]
        for p in range(max(r["start"] + 1, 4), sim.takeoff(r)):
            if next(x for x in ft if x >= p) not in ft[-2:]:          # >= 2 hops early (cf_jump_earlier is 1 hop)
                cand[5].append(("cf_threshold", {"r": r["id"], "p": p}))
    every_foot = {x for ft in c.feet.values() for x in ft}
    for line in (6, 7, 8, 9):
        if line not in every_foot:
            cand[5].append(("cf_foul_line", {"line": line}))
    cand[5] += [("cf_swap", {"a": a["id"], "b": b["id"]}) for a, b in itertools.permutations(J, 2)
                if a["jump"] != b["jump"]]
    for r in J:
        for d in (2, 3, 4, 5):
            new = {**c.score, r["id"]: c.score[r["id"]] + d}
            if all(new[r["id"]] != v for j, v in new.items() if j != r["id"]) and c.place(r["id"], new) != c.place(r["id"]):
                cand[5].append(("cf_longer_place", {"r": r["id"], "d": d}))
    if len(c.land) >= 3 and sim.lands_spread(sc) >= ORDER_GAP_F:
        cand[5] += [("cf_lead_changes", {"r": r["id"]}) for r in J]
    if len(c.land) >= 3:
        for r, s2 in itertools.permutations(J, 2):
            others = [f for j, f in c.land.items() if j not in (r["id"], s2["id"])]
            if min(abs(c.land[s2["id"]] - f) for f in others) >= ORDER_GAP_F:
                cand[5].append(("cf_leader_after", {"r": r["id"], "s": s2["id"]}))
    if not c.seq and len(c.land) >= 2:
        first = min(c.land, key=c.land.get)
        new = sorted(sim.landing_frames(sc, wait={first: 1}).values())
        if new[1] - new[0] >= ORDER_GAP_F:
            cand[5].append(("cf_wait_beat", {"r": first}))
    return cand


# --------------------------------------------------------------------------- question makers


def make(c: Ctx, lv: int, key: str, p: dict, rng: random.Random) -> dict:
    by, sc = c.by, c.scene
    r = by.get(p.get("r"))
    if key == "hops_jump":
        n = len(r["hops"])
        return _int_q(rng, lv, key, f"How many short hops does {name(r)} make before its long jump into the sand "
                      f"(not counting the jump itself)?", n, p, c.win([r["id"]]), lo=1)
    if key == "hops_foul":
        n = len(r["hops"])
        return _int_q(rng, lv, key, f"How many hops does {name(r)} make in total, counting the hop that carries "
                      f"it past the red foul line?", n, p, c.win([r["id"]]), lo=1)
    if key == "n_fouls":
        return _int_q(rng, lv, key, "How many robots foul, i.e. hop past the red foul line before they jump?",
                      len(c.fouled), p, c.win(), lo=0, hi=len(c.rs))
    if key == "shape":
        others = [s for s in sim.SHAPES if s != r["shape"]]
        rng.shuffle(others)
        return _mcq(rng, lv, key, f"What shape is the body of {name(r)}?", r["shape"], others, p, c.win([r["id"]]))
    if key == "who_fouls":
        gt = name(c.fouled[0]) if c.fouled else NONE_FOULS
        pool = [name(x) for x in c.jumpers]
        rng.shuffle(pool)
        return _mcq(rng, lv, key, "Which robot fouls by hopping past the red foul line before it jumps?", gt,
                    ([NONE_FOULS] if c.fouled else []) + pool, p, c.win())
    if key in ("farther_landing", "farther_takeoff", "longer_jump", "first_land"):
        a, b = by[p["a"]], by[p["b"]]
        if rng.random() < 0.5:
            a, b = b, a
        if key == "farther_landing":
            gt = a if c.score[a["id"]] > c.score[b["id"]] else b
            text = f"Which robot lands farther along the sand pit: {name(a)} or {name(b)}?"
            same, neither = "they land on the same spot", "neither of them reaches the sand pit"
            fps = c.win([a["id"], b["id"]])
        elif key == "farther_takeoff":
            gt = a if sim.takeoff(a) > sim.takeoff(b) else b
            text = f"Which robot takes off for its long jump from farther along the track: {name(a)} or {name(b)}?"
            same, neither = "they take off from the same spot", "neither of them jumps"
            fps = c.win([a["id"], b["id"]])
        elif key == "longer_jump":
            gt = a if a["jump"] > b["jump"] else b
            text = f"Whose long jump is longer, from take-off spot to landing spot: {name(a)}'s or {name(b)}'s?"
            same, neither = "the two jumps are the same length", "neither of them jumps"
            fps = c.win([a["id"], b["id"]])
        else:
            gt = a if c.land[a["id"]] < c.land[b["id"]] else b
            text = f"Which robot lands in the sand pit first: {name(a)} or {name(b)}?"
            same, neither = "they land at the same moment", "neither of them lands in the sand pit"
            fps = mf(abs(c.land[a["id"]] - c.land[b["id"]]))
        return _pair_q(rng, lv, key, text, gt, a, b, same, neither, fps)
    if key == "hop_diff":
        a, b = by[p["a"]], by[p["b"]]
        d = len(a["hops"]) - len(b["hops"])
        return _int_q(rng, lv, key, f"How many more hops does {name(a)} make than {name(b)} (count every short hop, "
                      f"not the long jump)?", d, p, c.win([a["id"], b["id"]]), lo=1)
    if key == "top_scorer":
        gt = max(c.rs, key=lambda x: c.score[x["id"]])
        return _robot_q(rng, lv, key, f"{RULE} Which robot scores the most?", gt, c.rs, p, c.win(),
                        "two robots tie for the top score")
    if key in ("first_to_jump", "last_to_land"):
        order = sorted(c.land, key=c.land.get)
        gt = by[order[0] if key == "first_to_jump" else order[-1]]
        k2 = (order[1], order[0]) if key == "first_to_jump" else (order[-2], order[-1])
        text = ("Which robot is the first to land in the sand pit?" if key == "first_to_jump"
                else "Which robot is the last to land in the sand pit?")
        return _robot_q(rng, lv, key, text, gt, c.rs, p, mf(abs(c.land[k2[0]] - c.land[k2[1]])),
                        "two robots land at the same moment")
    if key == "takeoff":
        to = sim.takeoff(r)
        return _pos_q(rng, lv, key, f"At what ruler position does {name(r)} take off for its long jump?", to, p,
                      c.win([r["id"]]), semantic=[c.feet[r["id"]][-2] if len(r["hops"]) > 1 else None], hi=9)
    if key == "jump_length":
        to = sim.takeoff(r)
        return _pos_q(rng, lv, key, f"How long is the long jump of {name(r)}, from its take-off spot to its landing "
                      f"spot, in ruler units?", r["jump"], p, c.win([r["id"]]),
                      semantic=[to + r["jump"] - sim.FOUL], lo=1, hi=12)
    if key == "longest_hop":
        hl = r["hops"]
        return _pos_q(rng, lv, key, f"How long is the longest short hop of {name(r)} (not counting the long jump), "
                      f"in ruler units?", max(hl), p, c.win([r["id"]]),
                      semantic=sorted(set(hl) - {max(hl)}), lo=1, hi=6)
    if key == "start_mark":
        return _pos_q(rng, lv, key, f"At which ruler position does {name(r)} start, before its first hop?",
                      r["start"], p, c.win([r["id"]]), hi=6)
    if key == "land_time":
        t = sim.timer_tenths(c.land[r["id"]])
        hi = sim.timer_tenths(c.n - 1)
        near = [t + k * TIME_STEP for k in (-1, 1, -2, 2, 3, -3) if 0 <= t + k * TIME_STEP <= hi]
        rng.shuffle(near)
        return _mcq(rng, lv, key, f"What does the clock read when {name(r)} lands in the sand pit?", t, near, p,
                    math.ceil(100 * 2 / (TIME_STEP / 10)) / 100, fmt=t_txt)
    if key == "landed_by":
        g = next(f for f in range(c.n) if sim.timer_tenths(f) == p["t"])
        n = sum(1 for f in c.land.values() if f <= g)
        return _int_q(rng, lv, key, f"How many robots have already landed in the sand pit when the clock first "
                      f"reads {t_txt(p['t'])}?", n, p, mf(min(abs(f - g) for f in c.land.values())), lo=0,
                      hi=len(c.rs))
    if key == "score":
        return _pos_q(rng, lv, key, f"{RULE} What is the score of {name(r)}?", c.score[r["id"]], p, c.win([r["id"]]),
                      semantic=[r["jump"] + sim.FOUL], lo=1, hi=20)
    if key == "place":
        n = c.place(r["id"])
        opts = [x for x in PLACES[:max(4, len(c.rs))] if x != PLACES[n - 1]]
        rng.shuffle(opts)
        return _mcq(rng, lv, key, f"{RULE} In what place does {name(r)} finish (1st = highest score)?",
                    PLACES[n - 1], opts, p, c.win())
    if key == "win_margin":
        top = sorted(c.score.values(), reverse=True)
        d = top[0] - top[1]
        return _pos_q(rng, lv, key, f"{RULE} By how much does the winner's score beat the second-best score?", d, p,
                      c.win(), lo=1, hi=18)
    if key == "ranking":
        trio = [by[i] for i in p["ids"]]
        ranked = sorted(trio, key=lambda x: c.score[x["id"]], reverse=True)
        gt = " > ".join(x["colour"] for x in ranked)
        perms = [" > ".join(x["colour"] for x in q) for q in itertools.permutations(ranked)]
        wrong = [q for q in perms if q != gt]
        rng.shuffle(wrong)
        names = ", ".join(name(x) for x in trio[:-1]) + f" and {name(trio[-1])}"
        return _mcq(rng, lv, key, f"{RULE} Rank {names} from highest to lowest score.", gt, wrong, p,
                    c.win(p["ids"]))
    if key == "lead_changes":
        n = sim.lead_changes(sc)
        return _int_q(rng, lv, key, f"{RULE} The leader is the robot with the best score among those that have landed "
                      f"in the sand so far. How many times does the leader change after the first landing?", n, p, mf(sim.lands_spread(sc)), lo=0,
                      hi=len(c.land))
    if key == "leader_after":
        f = c.land[r["id"]]
        gt = by[next(i for g, i in sim.leaders(sc) if g == f)]
        return _robot_q(rng, lv, key, f"{RULE} Among the robots that have landed in the sand so far, which one has the "
                        f"best score right after {name(r)} lands?", gt, c.rs, p, mf(c.land_gap(r["id"])),
                        "no robot has a score yet")
    if key == "landed_before":
        jf = r["jump_frame"]
        n = sum(1 for j, f in c.land.items() if j != r["id"] and f < jf)
        gap = min((abs(f - jf) for j, f in c.land.items() if j != r["id"]), default=c.n)
        return _int_q(rng, lv, key, f"How many other robots have already landed in the sand pit at the moment "
                      f"{name(r)} takes off for its long jump?", n, p, mf(gap), lo=0, hi=len(c.rs) - 1)
    if key == "cf_jump_earlier":
        ft = c.feet[r["id"]]
        v = ft[-2] + r["jump"]
        return _pos_q(rng, lv, key, f"{RULE} If {name(r)} had taken its long jump one hop earlier, with the same jump "
                      f"length, what would its score have been?", v, p, c.win([r["id"]]),
                      semantic=[c.score[r["id"]], ft[-2]], lo=1, hi=20)
    if key == "cf_threshold":
        ft = c.feet[r["id"]]
        to = next(x for x in ft if x >= p["p"])
        v = to + r["jump"]
        return _pos_q(rng, lv, key, f"{RULE} Suppose {name(r)} had taken its long jump as soon as it stood at "
                      f"{sim.fmt(p['p'])} or beyond, with the same jump length. What would its score have been?", v, p,
                      c.win([r["id"]]), semantic=[c.score[r["id"]], p["p"] + r["jump"]], lo=1, hi=20)
    if key == "cf_foul_line":
        n = sum(1 for x in c.rs if sim.run(x["tau"], x["start"], x["hops"], foul=p["line"])["fouled"])
        return _int_q(rng, lv, key, f"Suppose the foul line had been at {sim.fmt(p['line'])} instead of 1.0, and "
                      f"every robot made exactly the same hops. How many robots would have fouled?", n, p, c.win(),
                      lo=0, hi=len(c.rs))
    if key == "cf_swap":
        a, b = by[p["a"]], by[p["b"]]
        v = sim.takeoff(a) + b["jump"]
        return _pos_q(rng, lv, key, f"{RULE} If {name(a)} had jumped as far as {name(b)} did, from its own take-off "
                      f"spot, what would the score of {name(a)} have been?", v, p, c.win([a["id"], b["id"]]),
                      semantic=[c.score[a["id"]], c.score[b["id"]]], lo=1, hi=20)
    if key == "cf_longer_place":
        new = {**c.score, r["id"]: c.score[r["id"]] + p["d"]}
        n = c.place(r["id"], new)
        opts = [x for x in PLACES[:max(4, len(c.rs))] if x != PLACES[n - 1]]
        rng.shuffle(opts)
        return _mcq(rng, lv, key, f"{RULE} If the long jump of {name(r)} had been {sim.fmt(p['d'])} longer, in what "
                    f"place would it have finished (1st = highest score)?", PLACES[n - 1],
                    [PLACES[c.place(r["id"]) - 1]] + opts, p, c.win())
    if key == "cf_leader_after":
        s2 = by[p["s"]]
        f = c.land[s2["id"]]
        gt = by[next(i for g, i in sim.leaders(sc, drop=r["id"]) if g == f)]
        return _robot_q(rng, lv, key, f"{RULE} If {name(r)} had fouled instead of jumping, which robot would have had "
                        f"the best score among the robots landed in the sand so far, right after {name(s2)} lands?", gt,
                        c.rs, p, mf(min(abs(f - g) for j, g in c.land.items() if j not in (s2["id"], r["id"]))),
                        "no robot would have a score yet")
    if key == "cf_wait_beat":
        lands = sim.landing_frames(sc, wait={r["id"]: 1})
        order = sorted(lands, key=lands.get)
        beat = sc["timing"]["hop_f"] + sc["timing"]["rest_f"]
        return _robot_q(rng, lv, key, f"The robots hop together, one hop per beat. If {name(r)} had stood still for "
                        f"one extra beat before taking off for its long jump (everything else unchanged), which robot "
                        f"would have been the first to land in the sand pit?", by[order[0]], c.rs, p,
                        mf(min(lands[order[1]] - lands[order[0]], beat)), "two robots would land at the same moment")
    if key == "takeoff_time":
        t = sim.timer_tenths(r["jump_frame"])
        hi = sim.timer_tenths(c.n - 1)
        near = [t + k * TIME_STEP for k in (-1, 1, -2, 2, 3, -3) if 0 <= t + k * TIME_STEP <= hi]
        rng.shuffle(near)
        return _mcq(rng, lv, key, f"What does the clock read at the moment {name(r)} takes off for its long jump?", t,
                    near, p, math.ceil(100 * 2 / (TIME_STEP / 10)) / 100, fmt=t_txt)
    if key == "cf_lead_changes":
        n = sim.lead_changes(sc, drop=r["id"])
        return _int_q(rng, lv, key, f"{RULE} The leader is the robot with the best score among those that have landed "
                      f"in the sand so far. If {name(r)} had fouled instead of jumping, how many times would the leader "
                      f"have changed after the first landing?", n, p, mf(sim.lands_spread(sc)), lo=0, hi=len(c.land))
    raise KeyError(key)


# --------------------------------------------------------------------------- the ladder


def _pick(rng, cands, k):
    keys = []
    for key, _ in cands:
        if key not in keys:
            keys.append(key)
    params = {key: [p for k2, p in cands if k2 == key] for key in keys}
    chosen, pool = [], list(keys)
    while pool and len(chosen) < k:
        key = rng.choices(pool, weights=[4 if x in MOTION else 1 for x in pool])[0]
        pool.remove(key)
        chosen.append((key, rng.choice(params[key])))
    if len(chosen) < k:
        rest = [(key, p) for key, p in cands if (key, p) not in chosen]
        rng.shuffle(rest)
        chosen += rest[:k - len(chosen)]
    if len(chosen) < k:
        raise ValueError("not enough question candidates")
    return chosen


def build(scene: dict, rng: random.Random) -> list[dict]:
    c = Ctx(scene)
    counts = L.ten_counts(rng)
    cand = candidates(c)
    qs = []
    for lv, k in zip(range(1, 6), counts):
        for key, p in _pick(rng, cand[lv], k):
            qs.append(make(c, lv, key, p, rng))
    return L.number(qs)
