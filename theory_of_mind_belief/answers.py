"""Independent answers for theory_of_mind_belief, from the stored scene."""
from __future__ import annotations


def _val(actor, time, drop=None):
    v = actor["start"]
    for j, c in enumerate(actor["changes"]):
        if j == drop or not c["t"] < time:
            continue
        v = max(0, v + c["delta"]) if actor["attr"] == "number" else c["to"]
    return v


def _in_window(t, wins):
    return any(w["start_s"] <= t <= w["end_s"] for w in wins)


def _belief(actor, wins, t, drop=None):
    seen_until = t
    for w in wins:
        if w["start_s"] <= t <= w["end_s"]:
            seen_until = w["start_s"]
    return _val(actor, seen_until, drop)


def answer(key: str, p: dict, scene: dict):
    acts, wins, t = scene["actors"], scene["windows"], scene["query_t"]
    allc = [(a["index"], j, c["t"]) for a in acts for j, c in enumerate(a["changes"])]
    if key == "actor_changes":
        return len(acts[p["actor"]]["changes"])
    if key == "total_changes":
        return len(allc)
    if key == "watching_at_t":
        return "no" if _in_window(p["t"], wins) else "yes"
    if key == "watching_at_change":
        ct = acts[p["actor"]]["changes"][p["k"]]["t"]
        return "no" if _in_window(ct, wins) else "yes"
    if key == "unseen_changes":
        return sum(1 for _, _, ct in allc if _in_window(ct, wins))
    if key == "unseen_changes_actor":
        return sum(1 for i, _, ct in allc if i == p["actor"] and _in_window(ct, wins))
    a = acts[p["actor"]] if "actor" in p else None
    if key == "belief_state":
        return _belief(a, wins, t)
    if key == "belief_stale":
        return "yes" if _belief(a, wins, t) != _val(a, t) else "no"
    if key == "cf_remove_change":
        return _belief(a, wins, t, drop=p["k"])
    if key == "cf_shift_windows":
        d = scene["cf_shift_s"]
        moved = [{"start_s": w["start_s"] + d, "end_s": w["end_s"] + d} for w in wins]
        return _belief(a, moved, t)
    return None
