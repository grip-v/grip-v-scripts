"""PIL renderer for phantom_traffic_jams: a ring road seen from above with
a white start line at the top; cars drive clockwise, sliding smoothly from
one step to the next. Coloured cars are red, blue, green and gold; the rest
are grey. A car braking by itself shows bright red brake lights. The only
text is the clock."""
from __future__ import annotations
import functools
import math
import sys
from pathlib import Path

from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from common.video5 import font  # noqa: E402

import sim  # noqa: E402

BG = (214, 226, 200)
ROAD = (92, 94, 102)
INK = (45, 47, 54)
CX, CY, R, WIDTH = 640, 375, 280, 50
CAR = {"red": (220, 50, 45), "blue": (45, 105, 225), "green": (40, 165, 70), "gold": (240, 190, 30),
       "gray": (175, 176, 182)}
INTRO_S, OUTRO_S = 1.0, 1.0


def duration(scene) -> float:
    return INTRO_S + scene["steps"] * sim.STEP_S + OUTRO_S


def _ang(x):
    return -math.pi / 2 + 2 * math.pi * x / sim.L


@functools.lru_cache(maxsize=1)
def _background() -> Image.Image:
    img = Image.new("RGB", (1280, 720), BG)
    d = ImageDraw.Draw(img)
    d.ellipse([CX - R - WIDTH / 2, CY - R - WIDTH / 2, CX + R + WIDTH / 2, CY + R + WIDTH / 2], fill=ROAD)
    d.ellipse([CX - R + WIDTH / 2, CY - R + WIDTH / 2, CX + R - WIDTH / 2, CY + R - WIDTH / 2], fill=BG)
    for k in range(sim.L):
        a = _ang(k + 0.5)
        if k % 2 == 0:
            d.line([(CX + (R - 1) * math.cos(a), CY + (R - 1) * math.sin(a)),
                    (CX + (R + 1) * math.cos(a + 0.03), CY + (R + 1) * math.sin(a + 0.03))], fill=(200, 200, 205),
                   width=1)
    a = _ang(-0.5)
    d.line([(CX + (R - WIDTH / 2) * math.cos(a), CY + (R - WIDTH / 2) * math.sin(a)),
            (CX + (R + WIDTH / 2) * math.cos(a), CY + (R + WIDTH / 2) * math.sin(a))], fill=(250, 250, 250), width=6)
    return img


def frame(scene, t: float) -> Image.Image:
    img = _background().copy()
    d = ImageDraw.Draw(img)
    P, V, B = sim.run(scene)
    st = min(max(0.0, t - INTRO_S), scene["steps"] * sim.STEP_S)
    s = min(scene["steps"] - 1, int(st / sim.STEP_S))
    u = min(1.0, st / sim.STEP_S - s)
    if st >= scene["steps"] * sim.STEP_S:
        s, u = scene["steps"] - 1, 1.0
    for i, c in enumerate(scene["colors"]):
        x = P[s][i] + V[s + 1][i] * u
        a = _ang(x)
        px, py = CX + R * math.cos(a), CY + R * math.sin(a)
        tx, ty = -math.sin(a), math.cos(a)                 # direction of travel (clockwise)
        nx, ny = math.cos(a), math.sin(a)
        hl, hw = 17, 10
        corners = [(px + tx * hl * sx + nx * hw * sy, py + ty * hl * sx + ny * hw * sy)
                   for sx, sy in ((1, 1), (1, -1), (-1, -1), (-1, 1))]
        d.polygon(corners, fill=CAR[c], outline=(30, 30, 35))
        # windscreen
        wx, wy = px + tx * 5, py + ty * 5
        d.polygon([(wx + nx * 6, wy + ny * 6), (wx - nx * 6, wy - ny * 6), (wx - nx * 5 + tx * 4, wy - ny * 5 + ty * 4),
                   (wx + nx * 5 + tx * 4, wy + ny * 5 + ty * 4)], fill=(205, 225, 240))
        if B[s + 1][i]:
            for sy in (1, -1):
                bx, by = px - tx * hl + nx * hw * 0.6 * sy, py - ty * hl + ny * hw * 0.6 * sy
                d.ellipse([bx - 5, by - 5, bx + 5, by + 5], fill=(255, 40, 30), outline=(255, 200, 190), width=2)
    d.text((1236, 40), f"{st:5.2f} s", fill=INK, font=font(30), anchor="rm")
    return img
