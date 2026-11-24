"""Independent answers for shell_game_tracking_v2: replays the swaps as a permutation of cups (cup id =
its starting slot) using only the stored scene."""
from __future__ import annotations


def _history(n, swaps):
    """history[k][slot] = cup id in that slot after k swaps."""
    row = list(range(n))
    hist = [list(row)]
    for s in swaps:
        row[s["i"]], row[s["j"]] = row[s["j"]], row[s["i"]]
        hist.append(list(row))
    return hist


def _slots(n, swaps, cup):
    return [h.index(cup) for h in _history(n, swaps)]


def _lab(s):
    return f"slot {s + 1}"


def answer(key: str, p: dict, scene: dict):
    n, balls, swaps = scene["n_cups"], scene["balls"], scene["swaps"]
    if key == "ball_start_slot":
        return _lab(balls[p["ball"]]["slot"])
    if key == "start_slot_ball":
        return next((b["name"] for b in balls if b["slot"] == p["slot"]), "nothing")
    if key == "ball_count":
        return len(balls)
    if key == "balls_adjacent":
        return "yes" if abs(balls[p["x"]]["slot"] - balls[p["y"]]["slot"]) == 1 else "no"
    if key == "swap_involves_ball":
        s = swaps[p["swap"] - 1]
        cur = _slots(n, swaps, balls[p["ball"]]["slot"])[p["swap"] - 1]
        return "yes" if cur in (s["i"], s["j"]) else "no"
    if key == "swap_count":
        return len(swaps)
    if key == "ball_moves":
        tr = _slots(n, swaps, balls[p["ball"]]["slot"])
        return sum(1 for a, b in zip(tr, tr[1:]) if a != b)
    if key == "ball_final_slot":
        return _lab(_slots(n, swaps, balls[p["ball"]]["slot"])[-1])
    if key == "final_slot_ball":
        last = _history(n, swaps)[-1][p["slot"]]          # cup id now in that slot
        return next((b["name"] for b in balls if b["slot"] == last), "nothing")
    if key == "cf_skip_swap":
        rest = [s for k, s in enumerate(swaps, 1) if k != p["swap"]]
        return _lab(_slots(n, rest, balls[p["ball"]]["slot"])[-1])
    if key == "cf_extra_swap":
        return _lab(_slots(n, swaps + [{"i": p["x"], "j": p["y"]}], balls[p["ball"]]["slot"])[-1])
    return None
