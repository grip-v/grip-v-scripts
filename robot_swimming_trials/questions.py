"""5-rung ladder for robot_swimming_trials (see common/ladder5.py)."""
from __future__ import annotations
import itertools
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from common import ladder5 as L  # noqa: E402

import sim  # noqa: E402

pct = lambda b: f"{b / 100:.2f}"


def build(scene: dict, rng: random.Random) -> list[dict]:
    log, n_r = scene["log"], scene["n_races"]
    names = [r["name"] for r in scene["robots"]]
    winners = {e["race"]: e["robot"] for e in log if e["outcome"] == "wins"}
    qs = []

    # L1 observation
    if rng.random() < 0.6:
        cnt = sum(e["outcome"] != "wins" for e in log)
        opts, a = L.mc(rng, cnt, L.near_ints(cnt, 0))
        qs.append(L.question(1, "n_struck", "How many bids are crossed out before every race has a winner?",
                             cnt, opts, a, True))
    else:
        opts, a = L.mc(rng, len(names), L.near_ints(len(names), 2))
        qs.append(L.question(1, "n_robots", "How many robots are bidding?", len(names), opts, a, False))

    # L2 relational
    single = [k for k in range(1, n_r + 1)
              if sum(1 for e in log if e["race"] == k and e["outcome"] == "robot already qualified") == 1]
    if single and rng.random() < 0.5:
        k = rng.choice(single)
        gt = next(e["robot"] for e in log if e["race"] == k and e["outcome"] == "robot already qualified")
        opts, a = L.mc(rng, gt, [winners[j] for j in winners] + rng.sample(names, len(names)))
        qs.append(L.question(2, "qualified_skip",
                             f"In Race {k}, one robot's bid is crossed out because that robot has already won a "
                             f"different race. Which robot?", gt, opts, a, True, {"race": k}))
    else:
        k = rng.randint(1, n_r)
        gt = winners[k]
        opts, a = L.mc(rng, gt, [winners[j] for j in winners] + rng.sample(names, len(names)))
        qs.append(L.question(2, "race_winner", f"Which robot wins Race {k}?", gt, opts, a, False, {"race": k}))

    # L3 measurement
    if rng.random() < 0.5:
        k = rng.randint(1, n_r)
        b = next(e["bid"] for e in log if e["race"] == k and e["outcome"] == "wins")
        others = [pct(bs[k - 1]) for bs in scene["bids"].values() if bs[k - 1] != b]
        opts, a = L.mc(rng, pct(b), rng.sample(others, len(others)))
        qs.append(L.question(3, "winning_bid", f"What is the winning bid in Race {k}?", pct(b), opts, a, False,
                             {"race": k}))
    else:
        b = next(e["bid"] for e in log if e["outcome"] != "wins")
        others = [pct(e["bid"]) for e in log if e["bid"] != b]
        opts, a = L.mc(rng, pct(b), rng.sample(others, len(others)))
        qs.append(L.question(3, "top_losing_bid", "What is the largest bid that gets crossed out?",
                             pct(b), opts, a, True))

    # L4 integration
    if n_r == 3 and rng.random() < 0.5 or n_r == 2 and rng.random() < 0.2:
        order = [f"Race {e['race']}" for e in log if e["outcome"] == "wins"]
        gt = " → ".join(order)
        perms = [" → ".join(p) for p in itertools.permutations(order) if list(p) != order]
        opts, a = L.mc(rng, gt, rng.sample(perms, min(3, len(perms))))
        qs.append(L.question(4, "race_order", "In what order do the races get their winners?", gt, opts, a, True))
    else:
        k = rng.randint(1, n_r)
        cnt = next(i for i, e in enumerate(log) if e["race"] == k and e["outcome"] == "wins")
        opts, a = L.mc(rng, cnt, L.near_ints(cnt, 0))
        qs.append(L.question(4, "checked_before",
                             f"How many bids (in any race) are checked before Race {k} gets its winner?",
                             cnt, opts, a, True, {"race": k}))

    # L5 counterfactual: change one bid, keep every other bid the same
    k = rng.randint(1, n_r)
    win_b = next(e["bid"] for e in log if e["race"] == k and e["outcome"] == "wins")
    if rng.random() < 0.5:                 # raise a non-winner's bid above the winning bid
        loser = rng.choice([n for n in names if n != winners[k]])
        new_b = min(99, win_b + rng.randint(3, 15))
    else:                                  # lower the winner's own bid
        loser = winners[k]
        new_b = rng.randint(1, max(1, win_b - 5))
    bids = {r: list(bs) for r, bs in scene["bids"].items()}
    bids[loser][k - 1] = new_b
    gt = {e["race"]: e["robot"] for e in sim.resolve(bids, n_r) if e["outcome"] == "wins"}.get(k, "no winner")
    opts, a = L.mc(rng, gt, [winners[k], loser] + rng.sample(names, len(names)))
    qs.append(L.question(5, "cf_bid",
                         f"Suppose {loser}'s bid for Race {k} had been {pct(new_b)} instead (all other bids "
                         f"unchanged). Which robot would win Race {k}?", gt, opts, a, False,
                         {"robot": loser, "race": k, "new_bid": new_b}))
    return qs
