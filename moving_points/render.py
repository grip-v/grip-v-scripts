"""PIL renderer for moving_points: a coordinate plane from -10 to 10 with
unit gridlines, bold axes and axis numbers every 2 units; 3-4 coloured dots
labelled A-D move in straight lines with no trails. The only text is the
axis numbers, the point letters and the clock."""
from __future__ import annotations
import functools
import sys
from pathlib import Path

from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from common.video5 import font  # noqa: E402

import sim  # noqa: E402

BG = (246, 245, 240)
GRID = (222, 220, 212)
AXIS = (70, 72, 80)
INK = (45, 47, 54)
U, OX, OY = 31.0, 640, 360
INTRO_S, OUTRO_S = 1.0, 1.2


def duration(scene) -> float:
    return INTRO_S + scene["T"] + OUTRO_S


def px(x, y):
    return OX + U * x, OY - U * y


@functools.lru_cache(maxsize=1)
def _background() -> Image.Image:
    img = Image.new("RGB", (1280, 720), BG)
    d = ImageDraw.Draw(img)
    for k in range(-10, 11):
        d.line([px(k, -10), px(k, 10)], fill=GRID, width=1)
        d.line([px(-10, k), px(10, k)], fill=GRID, width=1)
    d.line([px(-10.4, 0), px(10.4, 0)], fill=AXIS, width=3)
    d.line([px(0, -10.4), px(0, 10.4)], fill=AXIS, width=3)
    f = font(14)
    for k in range(-10, 11, 2):
        if k:
            x, y = px(k, 0)
            d.text((x, y + 12), str(k), fill=INK, font=f, anchor="mm")
            x, y = px(0, k)
            d.text((x - 14, y), str(k), fill=INK, font=f, anchor="rm")
    d.text((OX - 10, OY + 12), "0", fill=INK, font=f, anchor="rm")
    return img


def frame(scene, t: float) -> Image.Image:
    img = _background().copy()
    d = ImageDraw.Draw(img)
    st = min(max(0.0, t - INTRO_S), scene["T"])
    f = font(20)
    for p in scene["points"]:
        x, y = px(*sim.pos(p, st))
        col = sim.COLORS[p["name"]]
        d.ellipse([x - 9, y - 9, x + 9, y + 9], fill=col, outline=(30, 30, 35), width=2)
        d.text((x + 12, y - 14), p["name"], fill=col, font=f, anchor="lm")
    d.text((1236, 40), f"{st:5.2f} s", fill=INK, font=font(30), anchor="rm")
    return img
