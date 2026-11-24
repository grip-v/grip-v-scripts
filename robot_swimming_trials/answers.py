"""Independent answers for robot_swimming_trials: a separate resolution loop
(repeatedly takes the largest remaining bid) over the stored bid table."""
from __future__ import annotations


def _resolve(bids, n_races):
    pool = [(b, r, k + 1) for r, bs in bids.items() for k, b in enumerate(bs) if b > 0]
    winners, done, steps = {}, set(), []
    while len(winners) < n_races and pool:
        top = max(pool, key=lambda e: (e[0], [-ord(ch) for ch in e[1]], -e[2]))
        pool.remove(top)
        b, r, race = top
        if race in winners:
            why = "race"
        elif r in done:
            why = "robot"
        else:
            why = "win"
            winners[race] = r
            done.add(r)
        steps.append((b, r, race, why))
    return steps, winners


def pct(b):
    return f"{b / 100:.2f}"


def answer(key: str, p: dict, scene: dict):
    steps, winners = _resolve(scene["bids"], scene["n_races"])
    if key == "n_struck":
        return sum(1 for s in steps if s[3] != "win")
    if key == "n_robots":
        return len(scene["bids"])
    if key == "race_winner":
        return winners[p["race"]]
    if key == "qualified_skip":
        hits = [r for b, r, race, why in steps if race == p["race"] and why == "robot"]
        return hits[0] if len(hits) == 1 else None
    if key == "winning_bid":
        return pct(next(b for b, r, race, why in steps if why == "win" and race == p["race"]))
    if key == "top_losing_bid":
        return pct(next(b for b, r, race, why in steps if why != "win"))
    if key == "race_order":
        return " → ".join(f"Race {race}" for b, r, race, why in steps if why == "win")
    if key == "checked_before":
        return next(i for i, (b, r, race, why) in enumerate(steps) if why == "win" and race == p["race"])
    if key == "cf_bid":
        bids = {r: list(bs) for r, bs in scene["bids"].items()}
        bids[p["robot"]][p["race"] - 1] = p["new_bid"]
        return _resolve(bids, scene["n_races"])[1].get(p["race"], "no winner")
    return None
