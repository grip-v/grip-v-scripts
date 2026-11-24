"""Independent answers for robot_capture_flag: recomputes each round's
positions with complex arithmetic straight from the stored r, theta, phi."""
from __future__ import annotations
import cmath
import math


def _winner(rd, d):
    flag = cmath.rect(rd["r"], math.radians(rd["theta"]))
    aaron = cmath.rect(rd["r"], math.radians(rd["phi"]))
    erin = cmath.rect(d, math.radians(rd["theta"]))
    return "Erin" if abs(flag - erin) < abs(flag - aaron) else "Aaron"


def answer(key: str, p: dict, scene: dict):
    rounds, d = scene["rounds"], scene["d"]
    wins = [_winner(rd, d) for rd in rounds]
    if key == "n_rounds":
        return len(rounds)
    if key == "flag_inside_half":
        return "inside" if rounds[p["round"] - 1]["r"] < 0.5 else "outside"
    if key == "round_winner":
        return wins[p["round"] - 1]
    if key == "erin_distance":
        return f"{d:.2f}"
    if key == "erin_wins":
        return wins.count("Erin")
    if key == "aaron_streak":
        best = cur = 0
        for w in wins:
            cur = cur + 1 if w == "Aaron" else 0
            best = max(best, cur)
        return best
    if key == "cf_erin_wins":
        return sum(_winner(rd, p["new_d"]) == "Erin" for rd in rounds)
    return None
