"""5-rung ladder for robot_capture_flag (see common/ladder5.py)."""
from __future__ import annotations
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from common import ladder5 as L  # noqa: E402

import sim  # noqa: E402


def build(scene: dict, rng: random.Random) -> list[dict]:
    rounds, d = scene["rounds"], scene["d"]
    n = len(rounds)
    qs = []

    # L1 observation
    k = rng.randrange(n)
    if rng.random() < 0.4:
        opts, a = L.mc(rng, n, L.near_ints(n, 1))
        qs.append(L.question(1, "n_rounds", "How many rounds are played?", n, opts, a, True))
    else:
        clear = [i for i, r in enumerate(rounds) if abs(r["r"] - 0.5) > 0.06]
        k = rng.choice(clear)
        gt = "inside" if rounds[k]["r"] < 0.5 else "outside"
        opts, a = L.mc(rng, gt, ["inside", "outside"])
        qs.append(L.question(1, "flag_inside_half",
                             f"In round {k + 1}, does the flag appear inside or outside the dashed half-way circle?",
                             gt, opts, a, True, {"round": k + 1}))

    # L2 relational
    k = rng.randrange(n)
    gt = rounds[k]["winner"]
    opts, a = L.mc(rng, gt, ["Aaron", "Erin"])
    qs.append(L.question(2, "round_winner", f"In round {k + 1}, which robot ends up closer to the flag?",
                         gt, opts, a, True, {"round": k + 1}))

    # L3 measurement: Erin's hidden fixed distance
    f = lambda v: f"{v:.2f}"
    cands = [round(d + s, 2) for s in (-0.3, -0.15, 0.15, 0.3, 0.45) if 0.05 <= round(d + s, 2) <= 1.0]
    rng.shuffle(cands)
    opts, a = L.mc(rng, f(d), [f(c) for c in cands])
    qs.append(L.question(3, "erin_distance",
                         "Erin runs the same distance every round. About how far is it, if the arena's radius is 1?",
                         f(d), opts, a, True))

    # L4 integration
    wins = [r["winner"] for r in rounds]
    if rng.random() < 0.6:
        gt = wins.count("Erin")
        opts, a = L.mc(rng, gt, L.near_ints(gt, 0, n))
        qs.append(L.question(4, "erin_wins", "Over all the rounds, how many does Erin win?", gt, opts, a, True))
    else:
        best = cur = 0
        for w in wins:
            cur = cur + 1 if w == "Aaron" else 0
            best = max(best, cur)
        opts, a = L.mc(rng, best, L.near_ints(best, 0, n))
        qs.append(L.question(4, "aaron_streak", "What is the longest run of consecutive rounds that Aaron wins?",
                             best, opts, a, True))

    # L5 counterfactual: Erin always runs a different distance
    new_d = round(rng.choice([v for v in (0.2, 0.35, 0.5, 0.65, 0.8, 0.95) if abs(v - d) >= 0.15]), 2)
    gt = sum(sim.round_result(rd, new_d)["winner"] == "Erin" for rd in rounds)
    opts, a = L.mc(rng, gt, [wins.count("Erin")] + L.near_ints(gt, 0, n))
    qs.append(L.question(5, "cf_erin_wins",
                         f"Suppose Erin had run {new_d:.2f} every round instead (Aaron and the flags unchanged). "
                         f"How many rounds would she have won?", gt, opts, a, True, {"new_d": new_d}))
    return qs
