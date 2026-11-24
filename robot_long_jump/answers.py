"""Independent answers for robot_long_jump.

Never calls sim.py and never reads the generator's per-robot facts (tau,
hops, jump, fouled, frames). It replays the renderer's per-frame geometry
(`render.layout`, exactly what `render.frame` draws) over the whole clip and
works from that:

  * footprints: the footprint x positions in each frame; a new one appearing
    is a hop landing. x is converted to ruler units through the on-screen
    ruler labels ("0.0" and "1.0" in scene["labels"]) and must sit on a
    gridline (within 0.05);
  * start: where the robot stands in frame 0;
  * jump: the first frame the robot is in the "jump" state (take-off moment);
  * landing: the first frame it stands in the sand ("landed"), at the dent;
  * foul: the robot lies toppled ("fallen") at the end of the clip;
  * times: the timer string the renderer draws on that frame.

Counterfactuals re-run the rule on these drawn footprints (earlier take-off,
different foul line, swapped jumps, a robot removed from the leader race).
"""
from __future__ import annotations
import functools
import json

import render

PLACES = ("1st", "2nd", "3rd", "4th", "5th", "6th")


def _fmt(t: int) -> str:
    return f"{t // 10}.{t % 10}"


def _name(scene, i) -> str:
    return f"the {scene['robots'][i]['colour']} robot"


def _to_tenths(scene, x: float) -> int:
    lab = {lb["text"]: lb["x"] for lb in scene["labels"]}
    v = (x - lab["0.0"]) / (lab["1.0"] - lab["0.0"]) * 10
    k = round(v)
    if abs(v - k) > 0.05:
        raise ValueError(f"position {v:.3f} is not on a gridline")
    return k


@functools.lru_cache(maxsize=4)
def _track(key: str) -> dict:
    scene = json.loads(key)
    rob = {r["id"]: {"feet": [], "feet_f": [], "start": None, "jump_f": None, "land_f": None, "land": None,
                     "fallen": False} for r in scene["robots"]}
    last = None
    for f in range(scene["n_frames"]):
        lay = render.layout(scene, f)
        for i, st in lay["robots"].items():
            t = rob[i]
            if f == 0:
                t["start"] = _to_tenths(scene, st["contact"])
            ft = lay["feet"][i]
            if len(ft) > len(t["feet"]):
                for x in ft[len(t["feet"]):]:
                    t["feet"].append(_to_tenths(scene, x))
                    t["feet_f"].append(f)
            if st["state"] == "jump" and t["jump_f"] is None:
                t["jump_f"] = f
            if st["state"] == "landed" and t["land_f"] is None:
                t["land_f"] = f
                t["land"] = _to_tenths(scene, lay["dents"][i])
                if _to_tenths(scene, st["contact"]) != t["land"]:
                    raise ValueError("robot is not standing on its landing dent")
        last = lay
    for i, st in last["robots"].items():
        rob[i]["fallen"] = st["state"] == "fallen"
    for i, t in rob.items():
        t["score"] = t["land"] if t["land"] is not None else 0
        t["takeoff"] = t["feet"][-1] if t["land"] is not None else None
        t["jump"] = t["land"] - t["takeoff"] if t["land"] is not None else None
        t["hops"] = [b - a for a, b in zip([t["start"]] + t["feet"], t["feet"])]
    return rob


def _place(sc: dict, i) -> int:
    return 1 + sum(1 for j in sc if sc[j] > sc[i])


def _lead_changes(rob, drop=None) -> int:
    lands = sorted((t["land_f"], i) for i, t in rob.items() if t["land_f"] is not None and i != drop)
    best, seq = None, []
    for _, i in lands:
        if best is None or rob[i]["score"] > rob[best]["score"]:
            best = i
        seq.append(best)
    return sum(1 for a, b in zip(seq, seq[1:]) if a != b)


def answer(key: str, p: dict, scene: dict):
    rob = _track(json.dumps(scene, sort_keys=True))
    t = rob.get(p.get("r"))
    nm = lambda i: _name(scene, i)
    score = {i: x["score"] for i, x in rob.items()}
    foul_x = _to_tenths(scene, render.x_of(scene, scene["foul"]))
    if key in ("hops_jump", "hops_foul"):
        return len(t["feet"])
    if key == "n_fouls":
        return sum(1 for x in rob.values() if x["fallen"])
    if key == "shape":
        return scene["robots"][p["r"]]["shape"]
    if key == "who_fouls":
        bad = [i for i, x in rob.items() if x["fallen"]]
        return nm(bad[0]) if len(bad) == 1 else ("none of them" if not bad else None)
    if key in ("farther_landing", "farther_takeoff", "longer_jump", "first_land"):
        a, b = rob[p["a"]], rob[p["b"]]
        field = {"farther_landing": "land", "farther_takeoff": "takeoff", "longer_jump": "jump",
                 "first_land": "land_f"}[key]
        if a[field] == b[field]:
            return None
        pick_a = a[field] < b[field] if key == "first_land" else a[field] > b[field]
        return nm(p["a"] if pick_a else p["b"])
    if key == "hop_diff":
        return len(rob[p["a"]]["feet"]) - len(rob[p["b"]]["feet"])
    if key == "top_scorer":
        return nm(max(score, key=score.get))
    if key in ("first_to_jump", "last_to_land"):
        lands = sorted((x["land_f"], i) for i, x in rob.items() if x["land_f"] is not None)
        return nm(lands[0][1] if key == "first_to_jump" else lands[-1][1])
    if key == "takeoff":
        return _fmt(t["takeoff"])
    if key == "jump_length":
        return _fmt(t["jump"])
    if key == "longest_hop":
        return _fmt(max(t["hops"]))
    if key == "start_mark":
        return _fmt(t["start"])
    if key == "land_time":
        return render.timer_text(t["land_f"])
    if key == "landed_by":
        want = f"{p['t'] // 10}.{p['t'] % 10} s"
        g = next(f for f in range(scene["n_frames"]) if render.timer_text(f) == want)
        return sum(1 for x in rob.values() if x["land_f"] is not None and x["land_f"] <= g)
    if key == "score":
        return _fmt(t["score"])
    if key == "place":
        return PLACES[_place(score, p["r"]) - 1]
    if key == "win_margin":
        top = sorted(score.values(), reverse=True)
        return _fmt(top[0] - top[1])
    if key == "ranking":
        ids = sorted(p["ids"], key=lambda i: score[i], reverse=True)
        return " > ".join(scene["robots"][i]["colour"] for i in ids)
    if key == "lead_changes":
        return _lead_changes(rob)
    if key == "leader_after":
        f = t["land_f"]
        landed = [i for i, x in rob.items() if x["land_f"] is not None and x["land_f"] <= f]
        return nm(max(landed, key=lambda i: score[i]))
    if key == "landed_before":
        return sum(1 for i, x in rob.items() if i != p["r"] and x["land_f"] is not None and x["land_f"] < t["jump_f"])
    if key == "cf_jump_earlier":
        return _fmt(t["feet"][-2] + t["jump"])
    if key == "cf_threshold":
        return _fmt(next(x for x in t["feet"] if x >= p["p"]) + t["jump"])
    if key == "cf_foul_line":
        # with the line at `line`, a robot fouls iff one of its drawn footprints lies past it (all of a jumper's
        # footprints come before its take-off; a robot that fouled already had a footprint past 1.0)
        if any(x == p["line"] for r in rob.values() for x in r["feet"]):
            return None                                   # a footprint on the new line: not a fair question
        return sum(1 for x in rob.values() if any(y > p["line"] for y in x["feet"]) or
                   any(y > foul_x for y in x["feet"]))
    if key == "cf_swap":
        return _fmt(rob[p["a"]]["takeoff"] + rob[p["b"]]["jump"])
    if key == "cf_longer_place":
        new = {**score, p["r"]: score[p["r"]] + p["d"]}
        if any(v == new[p["r"]] for i, v in new.items() if i != p["r"]):
            return None
        return PLACES[_place(new, p["r"]) - 1]
    if key == "cf_leader_after":
        f = rob[p["s"]]["land_f"]
        landed = [i for i, x in rob.items() if x["land_f"] is not None and x["land_f"] <= f and i != p["r"]]
        return nm(max(landed, key=lambda i: score[i]))
    if key == "cf_wait_beat":
        # the shared beat = frames between two consecutive footprints of any robot
        beat = next(x["feet_f"][1] - x["feet_f"][0] for x in rob.values() if len(x["feet_f"]) >= 2)
        lands = {i: x["land_f"] + (beat if i == p["r"] else 0) for i, x in rob.items() if x["land_f"] is not None}
        return nm(min(lands, key=lands.get))
    if key == "takeoff_time":
        return render.timer_text(t["jump_f"])
    if key == "cf_lead_changes":
        return _lead_changes(rob, drop=p["r"])
    return None

