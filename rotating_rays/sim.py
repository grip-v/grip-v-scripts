"""Scene generation for rotating_rays (from the image benchmark's
angle_estimation category, made temporal).

Two rays, a long red one and a shorter blue one, turn about a shared
centre inside a dial with tick marks every 15 degrees (bigger ticks every
90), like clock hands at steady but different angular speeds, possibly in
opposite directions. Start angles are multiples of 15 degrees; speeds are
whole degrees per second (positive = counter-clockwise on screen).

The rays LINE UP when they point the same way and are at a RIGHT ANGLE
when 90 degrees apart; both happen at closed-form times from the relative
angle. Every such event is at least EDGE seconds from the start and end
and EV_GAP seconds from other events, so counts are never borderline.
"""
from __future__ import annotations
import math
import random

SPEEDS = [15, 20, 24, 30, 36, 40, 45, 60, 72, 90]
EDGE = 0.4
EV_GAP = 0.4


def angle(ray, t):
    return (ray["a0"] + ray["w"] * t) % 360


def between(scene, t) -> float:
    """Angle between the rays, 0-180."""
    d = (angle(scene["red"], t) - angle(scene["blue"], t)) % 360
    return min(d, 360 - d)


def events(scene, target, red=None, blue=None):
    """Times in (0, T) when the relative angle (red - blue) passes target or -target (mod 360)."""
    red, blue = red or scene["red"], blue or scene["blue"]
    rel_w = red["w"] - blue["w"]
    rel0 = red["a0"] - blue["a0"]
    out = []
    if rel_w == 0:
        return out
    goals = {target % 360, (-target) % 360}
    for g in goals:
        # rel0 + rel_w * t = g + 360 k
        k_lo = math.floor((rel0 - g) / 360) - 5
        for k in range(k_lo, k_lo + 400):
            t = (g + 360 * k - rel0) / rel_w
            if 0 < t < scene["T"]:
                out.append(t)
    return sorted(set(round(t, 9) for t in out))


def on_event(scene, t, red=None, blue=None) -> bool:
    """True if at time t the rays are exactly lined up or exactly at a right angle."""
    red, blue = red or scene["red"], blue or scene["blue"]
    rel = (red["a0"] + red["w"] * t - blue["a0"] - blue["w"] * t) % 360
    return any(abs(rel - g) < 1e-6 or abs(rel - g - 360) < 1e-6 for g in (0, 90, 270))


def check(scene) -> list:
    errs = []
    T = scene["T"]
    r, b = scene["red"], scene["blue"]
    if abs(r["w"]) == abs(b["w"]):
        errs.append("same speed")
    ev = events(scene, 0) + events(scene, 90)
    ev.sort()
    if any(t < EDGE or t > T - EDGE for t in ev):
        errs.append("an event too near the start or end")
    if any(y - x < EV_GAP for x, y in zip(ev, ev[1:])):
        errs.append("two events too close together")
    if on_event(scene, 0) or on_event(scene, T):
        errs.append("the clip starts or ends exactly on a line-up or right angle")
    if len(events(scene, 0)) < 1:
        errs.append("the rays never line up")
    for ray in (r, b):
        turns = abs(ray["w"]) * T / 360
        if 0 < turns % 1 < 0.1 or turns % 1 > 0.9:
            errs.append("turn count borderline")
    return errs


def build(rng: random.Random) -> dict:
    while True:
        T = rng.randint(10, 14)
        wr = rng.choice(SPEEDS) * rng.choice([1, -1])
        wb = rng.choice(SPEEDS) * rng.choice([1, -1])
        scene = {"T": T, "red": {"a0": 15 * rng.randrange(24), "w": wr},
                 "blue": {"a0": 15 * rng.randrange(24), "w": wb}}
        if not check(scene):
            return scene
