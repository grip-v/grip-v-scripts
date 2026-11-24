"""Robot Capture-the-Flag (after Jane Street, Apr 2024), played over many
rounds. Each round a flag appears at a random point of a unit-circle arena.
Aaron knows only how far the flag is (r) and runs exactly that far in a
random direction; Erin knows only its direction and always runs the same
fixed distance d along it -- d is never shown and has to be worked out from
the rounds. Whoever ends closer to the flag wins the round."""
from __future__ import annotations
import math
import random

N_ROUNDS = (10, 12)


def round_result(rd: dict, d: float) -> dict:
    fx, fy = rd["r"] * math.cos(math.radians(rd["theta"])), rd["r"] * math.sin(math.radians(rd["theta"]))
    ax, ay = rd["r"] * math.cos(math.radians(rd["phi"])), rd["r"] * math.sin(math.radians(rd["phi"]))
    ex, ey = d * math.cos(math.radians(rd["theta"])), d * math.sin(math.radians(rd["theta"]))
    da, de = math.hypot(fx - ax, fy - ay), math.hypot(fx - ex, fy - ey)
    return {"aaron_dist": round(da, 4), "erin_dist": round(de, 4), "winner": "Erin" if de < da else "Aaron"}


def build(rng: random.Random) -> dict:
    d = round(rng.uniform(0.25, 0.9) / 0.05) * 0.05
    d = round(d, 2)
    n_rounds = rng.randint(*N_ROUNDS)
    rounds = []
    while len(rounds) < n_rounds:
        rd = {"r": round(max(0.12, math.sqrt(rng.random())), 3), "theta": round(rng.uniform(0, 360), 1),
              "phi": round(rng.uniform(0, 360), 1)}
        res = round_result(rd, d)
        if abs(res["aaron_dist"] - res["erin_dist"]) < 0.08:
            continue                              # too close to call by eye
        rounds.append({**rd, **res})
    wins = [r["winner"] for r in rounds]
    if not ("Erin" in wins and "Aaron" in wins):
        return build(rng)
    return {"d": d, "rounds": rounds}


def check(scene: dict) -> list[str]:
    errs = []
    for k, rd in enumerate(scene["rounds"], 1):
        if {**rd, **round_result(rd, scene["d"])} != rd:
            errs.append(f"round {k} result doesn't match the geometry")
    return errs
