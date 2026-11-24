"""Independent answers for robot_road_trip, read off the stored event log
(waits = rejoin - pullover, finish order from finish events). The what-if
question re-runs the simulation without one car."""
from __future__ import annotations

import sim


def _waits(scene):
    out = {}
    for n, ev in scene["events"].items():
        w, t_off = 0.0, None
        for kind, t, _ in ev:
            if kind == "pullover":
                t_off = t
            elif kind == "rejoin":
                w += t - t_off
        out[n] = w
    return out


def answer(key: str, p: dict, scene: dict):
    fin = {n: ev[-1][1] for n, ev in scene["events"].items()}
    speeds = {c["name"]: c["speed"] for c in scene["cars"]}
    enter = {c["name"]: c["enter_t"] for c in scene["cars"]}
    if key == "n_pullovers":
        return sum(1 for ev in scene["events"].values() for e in ev if e[0] == "pullover")
    if key == "n_cars":
        return len(scene["cars"])
    if key == "first_passer":
        t_off = next(t for kind, t, _ in scene["events"][p["car"]] if kind == "pullover")
        return next(q["passer"] for q in scene["pullovers"] if q["car"] == p["car"] and q["t"] == t_off)
    if key == "first_finisher":
        return min(fin, key=fin.get)
    if key == "wait_time":
        return f"{round(_waits(scene)[p['car']] * 2) / 2:.1f} s"
    if key == "speed":
        return f"{speeds[p['car']]:.2f}"
    if key == "finish_order":
        return " → ".join(sorted(fin, key=fin.get))
    if key == "most_delayed":
        lost = {n: fin[n] - enter[n] - scene["road"] / speeds[n] for n in fin}
        return max(lost, key=lost.get)
    if key == "cf_remove_car":
        return len(sim.simulate(scene["cars"], skip=p["car"])["pullovers"])
    return None
