"""Scene generation for motion_parallax (from the image benchmark's
depth_height category, made temporal).

A camera slides sideways at a steady speed past 3-6 flat coloured balls
hanging at different distances. A ball at depth z drifts across the screen
at f*v/z pixels per second (nearer balls slide faster), so depth can be
read from the motion. On screen, each ball's size and height are chosen
independently of its depth, so a single frame gives misleading cues: a big
ball may be far away and a small one close. Real size = screen size x z/f.

Depths are a base depth times distinct multipliers from MULTS, so depth
ratios are tidy numbers (2x, 2.5x, ...). A faint dashed line marks the
centre of the screen.
"""
from __future__ import annotations
import itertools
import random

F = 900.0                       # focal length in pixels
CX, CY = 640, 380
MULTS = [1.0, 1.5, 2.0, 2.5, 3.0, 4.0]
COLORS = {"red": (215, 55, 50), "blue": (50, 100, 215), "green": (45, 150, 70), "orange": (235, 135, 30),
          "purple": (130, 75, 195), "teal": (25, 150, 150), "pink": (225, 95, 165), "yellow": (225, 195, 40)}
T_RANGE = (10, 14)
TIME_MARGIN = 0.03


def screen_x(scene, b, t):
    return CX + F * (b["x"] - scene["v"] * t) / b["z"]


def speed(scene, b) -> float:
    """Screen speed in px/s (all balls drift left)."""
    return F * scene["v"] / b["z"]


def real_size(b) -> float:
    return b["r"] * b["z"] / F


def leave_time(scene, b) -> float:
    """When the ball's right edge passes the left edge of the screen."""
    return (screen_x(scene, b, 0) + b["r"]) / speed(scene, b)


def center_time(scene, b):
    sx = screen_x(scene, b, 0)
    return (sx - CX) / speed(scene, b) if sx > CX else None


def check(scene) -> list:
    errs = []
    bs = scene["balls"]
    if not 3 <= len(bs) <= 6 or len({b["color"] for b in bs}) != len(bs):
        errs.append("ball count or colours")
    if len({b["m"] for b in bs}) != len(bs):
        errs.append("two balls at the same depth")
    for b in bs:
        if not b["r"] + 10 <= screen_x(scene, b, 0) <= 1270 - b["r"]:
            errs.append("a ball starts off screen")
        if leave_time(scene, b) < 3.0:
            errs.append("a ball leaves within 3 s")
    lts = sorted(leave_time(scene, b) for b in bs if leave_time(scene, b) < scene["T"])
    if any(b - a < 0.6 for a, b in zip(lts, lts[1:])):
        errs.append("two balls leave the screen too close together")
    sizes = sorted(real_size(b) for b in bs)
    # the static cue must mislead: some pair where the farther ball looks bigger
    if not any(x["z"] > y["z"] * 1.25 and x["r"] > y["r"] * 1.2 for x, y in itertools.permutations(bs, 2)):
        errs.append("size cue never misleads")
    return errs


def build(rng: random.Random) -> dict:
    n = rng.randint(3, 6)
    while True:
        T = rng.randint(*T_RANGE)
        v = round(rng.uniform(0.8, 1.3), 2)
        z0 = round(rng.uniform(4.0, 6.0), 2)
        ms = rng.sample(MULTS, n)
        cols = rng.sample(list(COLORS), n)
        balls = []
        for m, c in zip(ms, cols):
            z = round(z0 * m, 3)
            r = rng.randint(18, 52)
            sx = rng.uniform(260, 1200)
            sy = rng.uniform(150, 610)
            balls.append({"color": c, "m": m, "z": z, "r": r, "x": round((sx - CX) * z / F, 4), "sy": round(sy, 1)})
        scene = {"T": T, "v": v, "z0": z0, "balls": balls}
        if not check(scene):
            return scene
