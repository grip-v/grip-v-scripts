"""Independent answers for object_tracking_mot: re-evaluates the closed-form paths from the stored scene."""
from __future__ import annotations
import math

QUAD = ["top-left", "top-right", "bottom-left", "bottom-right"]


def _xy(d, t, c):
    return c + d["rx"] * math.cos(d["wx"] * t + d["phase"]), c + d["ry"] * math.sin(d["wy"] * t + d["phase"])


def _quad(x, y, c):
    return f"{'top' if y < c else 'bottom'}-{'left' if x < c else 'right'}"


def _pair_count(scene, i, j):
    c, hz, T = scene["canvas"] / 2, scene["sample_hz"], scene["duration"]
    steps = int(T * hz) + 1
    flags = []
    for k in range(steps):
        t = min(k / hz, T)
        a, b = _xy(scene["dots"][i], t, c), _xy(scene["dots"][j], t, c)
        flags.append((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2 < scene["encounter_dist"] ** 2)
    return sum(1 for k, f in enumerate(flags) if f and (k == 0 or not flags[k - 1]))


def answer(key: str, p: dict, scene: dict):
    dots, T, c = scene["dots"], scene["duration"], scene["canvas"] / 2
    n = len(dots)
    if key == "dot_count":
        return n
    if key == "start_quadrant":
        return _quad(*_xy(dots[p["dot"]], 0.0, c), c)
    if key == "pair_meet":
        return "yes" if _pair_count(scene, p["i"], p["j"]) else "no"
    if key == "closest_at_end":
        me = _xy(dots[p["dot"]], T, c)
        best = min((m for m in range(n) if m != p["dot"]),
                   key=lambda m: math.hypot(*(u - v for u, v in zip(_xy(dots[m], T, c), me))))
        return f"dot {best + 1}"
    if key == "encounter_count":
        return sum(_pair_count(scene, i, j) for i in range(n) for j in range(i + 1, n))
    if key == "midline_crossings":
        d, cnt, last = dots[p["dot"]], 0, None
        for s in range(int(T * 60) + 1):
            left = _xy(d, s / 60, c)[0] < c
            cnt += last is not None and left != last
            last = left
        return cnt
    if key == "final_quadrant":
        return _quad(*_xy(dots[p["dot"]], T, c), c)
    if key == "cf_speed":
        d = dict(dots[p["dot"]])
        d["wx"], d["wy"] = d["wx"] * p["factor"], d["wy"] * p["factor"]
        return _quad(*_xy(d, T, c), c)
    if key == "cf_freeze":
        return _quad(*_xy(dots[p["dot"]], min(T, p["t"]), c), c)
    return None
