"""PIL renderer for rotating_rays: a round dial with a tick every 15
degrees (longer ticks every 90), a long red ray and a shorter blue ray
turning about the centre. The only text is the clock."""
from __future__ import annotations
import functools
import math
import sys
from pathlib import Path

from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from common.video5 import font  # noqa: E402

import sim  # noqa: E402

BG = (240, 238, 232)
INK = (45, 47, 54)
CX, CY, R = 640, 375, 280
INTRO_S, OUTRO_S = 1.0, 1.2
SS = 2


def duration(scene) -> float:
    return INTRO_S + scene["T"] + OUTRO_S


@functools.lru_cache(maxsize=1)
def _dial() -> Image.Image:
    img = Image.new("RGB", (1280 * SS, 720 * SS), BG)
    d = ImageDraw.Draw(img)
    c = (CX * SS, CY * SS)
    d.ellipse([c[0] - R * SS, c[1] - R * SS, c[0] + R * SS, c[1] + R * SS], fill=(252, 251, 247),
              outline=(120, 122, 130), width=3 * SS)
    for k in range(24):
        a = math.radians(15 * k)
        inner = R - (34 if k % 6 == 0 else 18)
        d.line([(c[0] + inner * SS * math.cos(a), c[1] - inner * SS * math.sin(a)),
                (c[0] + R * SS * math.cos(a), c[1] - R * SS * math.sin(a))], fill=(90, 92, 100),
               width=(4 if k % 6 == 0 else 2) * SS)
    return img


def frame(scene, t: float) -> Image.Image:
    img = _dial().copy()
    d = ImageDraw.Draw(img)
    st = min(max(0.0, t - INTRO_S), scene["T"])
    c = (CX * SS, CY * SS)
    for ray, col, length, w in ((scene["blue"], (55, 105, 215), 0.72, 9), (scene["red"], (215, 60, 55), 0.92, 7)):
        a = math.radians(sim.angle(ray, st))
        tip = (c[0] + R * length * SS * math.cos(a), c[1] - R * length * SS * math.sin(a))
        d.line([c, tip], fill=col, width=w * SS)
        d.ellipse([tip[0] - 7 * SS, tip[1] - 7 * SS, tip[0] + 7 * SS, tip[1] + 7 * SS], fill=col)
    d.ellipse([c[0] - 10 * SS, c[1] - 10 * SS, c[0] + 10 * SS, c[1] + 10 * SS], fill=INK)
    img = img.resize((1280, 720), Image.LANCZOS)
    ImageDraw.Draw(img).text((1236, 40), f"{st:5.2f} s", fill=INK, font=font(30), anchor="rm")
    return img
