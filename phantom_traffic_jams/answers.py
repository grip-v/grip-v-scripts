"""Independent answers for phantom_traffic_jams. The generator steps an
occupancy array of the road; this replays the same deterministic
Nagel-Schreckenberg rule by tracking each car's distance driven and its
gap to the car ahead (cars never overtake, so the car ahead is fixed by
the starting order), then reads stops, laps and distances off that run."""
from __future__ import annotations
import functools
import json

L = 60
VMAX = 3
STEP_S = 0.3
NAMES = ["red", "blue", "green", "gold"]


@functools.lru_cache(maxsize=64)
def _run(key: str, skip: int):
    scene = json.loads(key)
    n = len(scene["pos"])
    order = sorted(range(n), key=lambda i: scene["pos"][i])
    ahead = {order[k]: order[(k + 1) % n] for k in range(n)}
    x = list(scene["pos"])                       # unwrapped distance along the road
    v = list(scene["vel"])
    hold = {}
    for t in scene["taps"]:
        if t["car"] != skip:
            for h in range(t["hold"]):
                hold[(t["step"] + h, t["car"])] = True
    speeds = []
    for s in range(scene["steps"]):
        nv = []
        for i in range(n):
            j = ahead[i]
            gap = (x[j] - x[i] - 1) % L if n > 1 else L
            w = min(v[i] + 1, VMAX, gap)
            if hold.get((s, i)):
                w = 0
            nv.append(w)
        for i in range(n):
            x[i] += nv[i]
        v = nv
        speeds.append(nv)
    return speeds


def _speeds(scene, skip=-1):
    return _run(json.dumps({k: scene[k] for k in ("pos", "vel", "taps", "steps")}, sort_keys=True), skip)


def _idx(scene, c):
    return scene["colors"].index(c)


def _episodes(sp, i):
    out, cur = [], 0
    for row in sp:
        if row[i] == 0:
            cur += 1
        elif cur:
            out.append(cur)
            cur = 0
    if cur:
        out.append(cur)
    return out


def _first(sp, i):
    return next((s for s, row in enumerate(sp) if row[i] == 0), None)


def answer(key: str, p: dict, scene: dict):
    sp = _speeds(scene)
    n = len(scene["pos"])
    car = lambda c: f"the {c} car"
    if key == "n_taps":
        return len(scene["taps"])
    if key == "stops_yesno":
        return "yes" if _episodes(sp, _idx(scene, p["c"])) else "no"
    if key == "n_col_stopped":
        return sum(1 for c in NAMES if _episodes(sp, _idx(scene, c)))
    if key == "first_stop":
        return car(min((p["x"], p["y"]), key=lambda c: _first(sp, _idx(scene, c))))
    if key == "farther":
        return car(max((p["x"], p["y"]), key=lambda c: sum(r[_idx(scene, c)] for r in sp)))
    if key == "laps":
        i = _idx(scene, p["c"])
        return (scene["pos"][i] + sum(r[i] for r in sp)) // L
    if key == "n_stops":
        return len(_episodes(sp, _idx(scene, p["c"])))
    if key == "stopped_time":
        return f"{sum(_episodes(sp, _idx(scene, p['c']))) * STEP_S:.1f} s"
    if key == "n_ever_stopped":
        return sum(1 for i in range(n) if _episodes(sp, i))
    if key == "stop_order":
        st = [c for c in NAMES if _first(sp, _idx(scene, c)) is not None]
        return ", then ".join(sorted(st, key=lambda c: _first(sp, _idx(scene, c))))
    if key == "most_stopped":
        return car(max(NAMES, key=lambda c: sum(_episodes(sp, _idx(scene, c)))))
    alt = _speeds(scene, _idx(scene, p["tap"]))
    if key == "cf_no_tap_yes":
        return "yes" if _episodes(alt, _idx(scene, p["c"])) else "no"
    if key == "cf_no_tap_count":
        return sum(1 for c in NAMES if _episodes(alt, _idx(scene, c)))
    if key == "cf_no_tap_all":
        return sum(1 for i in range(n) if _episodes(alt, i))
    return None
