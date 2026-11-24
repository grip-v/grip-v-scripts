"""Multiple-object tracking: 4 identical dots on independent Lissajous-style closed-form paths, numbered
1-4 for the first moments, then never labelled again. Positions are in a 420x420 logical box (render scales it).

pos(dot, t) = (cx + rx cos(wx t + phase), cy + ry sin(wy t + phase))."""
from __future__ import annotations
import math
import random

CANVAS = 420
N_DOTS = 4
DURATION = (5.5, 7.0)
LABEL_FADE_END = 1.6
LABEL_FADE_DUR = 0.5
MIN_START_SEP = 70
RX, RY = (70, 150), (60, 150)
WX, WY = (0.6, 1.7), (0.6, 1.7)
ENCOUNTER_DIST = 28          # = 2 x dot radius: the dot discs touch or overlap (visible in the video)
BOUNDARY_MARGIN = 16         # queried positions must be this far from the centre lines
ENC_BAND = 5                 # no pair may come within +-ENC_BAND of the touch distance
MIN_APPROACH = 6             # dots never sit exactly on top of each other
SAMPLE_HZ = 30
DOT_R_LOGICAL = 14
QUADRANTS = ["top-left", "top-right", "bottom-left", "bottom-right"]


def pos(d: dict, t: float, c: float = CANVAS / 2):
    return (c + d["rx"] * math.cos(d["wx"] * t + d["phase"]),
            c + d["ry"] * math.sin(d["wy"] * t + d["phase"]))


def _dot(rng):
    return {"rx": round(rng.uniform(*RX), 1), "ry": round(rng.uniform(*RY), 1),
            "wx": round(rng.uniform(*WX), 3), "wy": round(rng.uniform(*WY), 3),
            "phase": round(rng.uniform(0, 2 * math.pi), 3)}


def _unambiguous(dots, T):
    """Reject scenes whose answers hinge on a near-tie a viewer could not resolve by eye."""
    c = CANVAS / 2
    ts = [k / 60 for k in range(int(T * 60) + 1)]
    for d in dots:
        for t in (0.0, T):                                   # start/end quadrant must be clear
            x, y = pos(d, t)
            if min(abs(x - c), abs(y - c)) < BOUNDARY_MARGIN:
                return False
        xs = [pos(d, t)[0] - c for t in ts]                  # centre-line grazes / touching the line
        for a, b, e in zip(xs, xs[1:], xs[2:]):
            if abs(b) <= abs(a) and abs(b) <= abs(e) and abs(b) < DOT_R_LOGICAL and a * e > 0:
                return False
    for i in range(N_DOTS):
        for j in range(i + 1, N_DOTS):
            m = min(math.dist(pos(dots[i], t), pos(dots[j], t)) for t in ts)
            if abs(m - ENCOUNTER_DIST) < ENC_BAND or m < MIN_APPROACH:
                return False
    for k in range(N_DOTS):                                  # closest-at-end needs a clear winner
        d = sorted(math.dist(pos(dots[k], T), pos(dots[m], T)) for m in range(N_DOTS) if m != k)
        if d[1] - d[0] < 12:
            return False
    return True


def build(rng: random.Random) -> dict:
    duration = round(rng.uniform(*DURATION), 2)
    for _ in range(20000):
        dots = [_dot(rng) for _ in range(N_DOTS)]
        st = [pos(d, 0.0) for d in dots]
        if all(math.dist(st[i], st[j]) >= MIN_START_SEP for i in range(N_DOTS) for j in range(i + 1, N_DOTS)) \
                and _unambiguous(dots, duration):
            break
    else:
        raise RuntimeError("could not separate start positions")
    return {"canvas": CANVAS, "duration": duration, "dots": dots, "label_fade_end": LABEL_FADE_END,
            "label_fade_dur": LABEL_FADE_DUR, "encounter_dist": ENCOUNTER_DIST, "sample_hz": SAMPLE_HZ}


def check(scene: dict) -> list[str]:
    errs = []
    if len(scene["dots"]) != N_DOTS:
        errs.append("wrong dot count")
    if scene["duration"] <= scene["label_fade_end"] + scene["label_fade_dur"]:
        errs.append("clip too short for label fade")
    return errs
