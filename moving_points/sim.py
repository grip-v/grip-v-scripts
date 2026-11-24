"""Scene generation for moving_points (from the image benchmark's
coordinate_geometry category, made temporal).

A coordinate plane from -10 to 10 (gridlines every unit, axis numbers every
2) with 3-4 lettered points moving in straight lines at constant velocity.
Velocity components are in {-1, -0.5, 0, 0.5, 1} units per second and
starting coordinates are whole numbers, so every point sits on whole-number
coordinates at every even second. No trails are drawn: paths have to be
read from the motion. Points stay inside the plane for the whole clip.

Closest approach of two points has the closed form
t* = -(dp . dv) / |dv|^2 for relative position dp and relative velocity dv.
"""
from __future__ import annotations
import itertools
import math
import random

NAMES = "ABCD"
COLORS = {"A": (215, 60, 55), "B": (50, 105, 215), "C": (40, 150, 70), "D": (220, 140, 30)}
VEL = [-1.0, -0.5, 0.0, 0.5, 1.0]


def pos(p, t):
    return (p["x"] + p["vx"] * t, p["y"] + p["vy"] * t)


def speed(p):
    return math.hypot(p["vx"], p["vy"])


def closest(a, b):
    dx, dy = b["x"] - a["x"], b["y"] - a["y"]
    ux, uy = b["vx"] - a["vx"], b["vy"] - a["vy"]
    uu = ux * ux + uy * uy
    if uu == 0:
        return None
    return -(dx * ux + dy * uy) / uu


def axis_cross(p, axis, T):
    """Time point p crosses the y-axis (axis='y', x = 0) or x-axis (axis='x', y = 0) inside (0, T), else None."""
    c, v = (p["x"], p["vx"]) if axis == "y" else (p["y"], p["vy"])
    if v == 0 or c == 0:
        return None
    t = -c / v
    return t if 0 < t < T else None


def axis_clean(p, T) -> bool:
    """No ambiguity about axis crossings: never on an axis at the start or the end, never sliding along one, and
    any crossing at least 0.5 s from both ends."""
    for c, v in ((p["x"], p["vx"]), (p["y"], p["vy"])):
        if c == 0 or c + v * T == 0:
            return False
        if v != 0:
            t = -c / v
            if 0 < t < T and (t < 0.5 or t > T - 0.5):
                return False
    return True


def quadrant(x, y):
    if x == 0 or y == 0:
        return None
    return {(True, True): "top right", (False, True): "top left", (False, False): "bottom left",
            (True, False): "bottom right"}[(x > 0, y > 0)]


def check(scene) -> list:
    errs = []
    ps, T = scene["points"], scene["T"]
    for p in ps:
        if p["vx"] == 0 and p["vy"] == 0:
            errs.append("a point never moves")
        for t in (0, T):
            x, y = pos(p, t)
            if not (-9.5 <= x <= 9.5 and -9.5 <= y <= 9.5):
                errs.append("a point leaves the plane")
    for a, b in itertools.combinations(ps, 2):
        for t in [k * 0.25 for k in range(int(T * 4) + 1)]:
            if math.dist(pos(a, t), pos(b, t)) < 0.6:
                errs.append("two points overlap")
                break
    if len({round(speed(p), 6) for p in ps}) != len(ps):
        errs.append("two points share a speed")
    return errs


def build(rng: random.Random) -> dict:
    n = rng.randint(3, 4)
    while True:
        T = rng.choice([8, 10, 12])
        ps = []
        for k in range(n):
            ps.append({"name": NAMES[k], "x": rng.randint(-8, 8), "y": rng.randint(-8, 8),
                       "vx": rng.choice(VEL), "vy": rng.choice(VEL)})
        scene = {"T": T, "points": ps}
        if not check(scene):
            return scene
