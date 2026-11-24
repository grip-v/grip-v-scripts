"""Robot Swimming Trials (after Jane Street, Oct 2021): 3N robots each split
one unit of fuel into bids for N races. Bids are resolved from the highest
down: a bid wins its race if that race is still open and the robot hasn't
already qualified; otherwise it is crossed out. It stops once every race has
a winner. The table shows all bids at once; the order they were resolved
in (and why each was crossed out) only appears in the video."""
from __future__ import annotations
import random

ROBOTS = [("Ari", "#E4553F"), ("Bex", "#3F8FE4"), ("Cyd", "#3FAF5B"), ("Dax", "#D99A1F"), ("Eon", "#9B5FE4"),
          ("Fei", "#2FB5B0"), ("Gus", "#C0507A"), ("Hal", "#7A8A30"), ("Ivo", "#5A6F8F")]


def resolve(bids: dict, n_races: int) -> list[dict]:
    """bids: {robot: [bid per race]} (hundredths). Returns the processed entries in order."""
    entries = sorted(((b, robot, race) for robot, bs in bids.items() for race, b in enumerate(bs) if b > 0),
                     key=lambda e: (-e[0], e[1], e[2]))
    won_race, qualified, log = {}, set(), []
    for b, robot, race in entries:
        if len(won_race) == n_races:
            break
        if race in won_race:
            outcome = "race already won"
        elif robot in qualified:
            outcome = "robot already qualified"
        else:
            outcome = "wins"
            won_race[race] = robot
            qualified.add(robot)
        log.append({"robot": robot, "race": race + 1, "bid": b, "outcome": outcome})
    return log


def build(rng: random.Random) -> dict:
    while True:
        n_races = rng.choice([2, 3])
        robots = rng.sample(ROBOTS, 3 * n_races)
        bids = {}
        for name, _ in robots:
            cuts = sorted(rng.sample(range(1, 100), n_races - 1))
            parts = [b - a for a, b in zip([0] + cuts, cuts + [100])]
            bids[name] = parts
        log = resolve(bids, n_races)
        vals = [e["bid"] for e in log]
        ties = len(vals) != len(set(vals))
        n_win = sum(e["outcome"] == "wins" for e in log)
        qual_skip = any(e["outcome"] == "robot already qualified" for e in log)
        if n_win == n_races and not ties and qual_skip and n_races + 2 <= len(log) <= 11:
            return {"n_races": n_races, "robots": [{"name": n, "color": c} for n, c in robots],
                    "bids": bids, "log": log}


def check(scene: dict) -> list[str]:
    errs = []
    if resolve(scene["bids"], scene["n_races"]) != scene["log"]:
        errs.append("stored resolution doesn't match a replay of the bids")
    if any(sum(b) != 100 for b in scene["bids"].values()):
        errs.append("a robot's bids don't add up to 1.00")
    return errs
