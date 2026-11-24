"""PIL renderer for motion_parallax: flat coloured balls (no shading, so
no lighting depth cue) on a plain dusk-blue background, drifting left as
the camera slides right; nearer balls are drawn on top. A faint dashed
vertical line marks the centre of the screen. The only text is the
clock."""
from __future__ import annotations
import functools
import sys
from pathlib import Path

from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from common.video5 import font  # noqa: E402

import sim  # noqa: E402

TOP, BOT = (40, 52, 82), (78, 96, 132)
INK = (232, 234, 240)
INTRO_S, OUTRO_S = 1.0, 1.0


def duration(scene) -> float:
    return INTRO_S + scene["T"] + OUTRO_S


@functools.lru_cache(maxsize=1)
def _background() -> Image.Image:
    img = Image.new("RGB", (1280, 720))
    d = ImageDraw.Draw(img)
    for y in range(720):
        u = y / 720
        d.line([(0, y), (1280, y)], fill=tuple(int(a + (b - a) * u) for a, b in zip(TOP, BOT)))
    for y in range(0, 720, 22):
        d.line([(sim.CX, y), (sim.CX, y + 11)], fill=(150, 160, 185), width=2)
    return img


def frame(scene, t: float) -> Image.Image:
    img = _background().copy()
    d = ImageDraw.Draw(img)
    st = min(max(0.0, t - INTRO_S), scene["T"])
    for b in sorted(scene["balls"], key=lambda b: -b["z"]):
        x = sim.screen_x(scene, b, st)
        r, y = b["r"], b["sy"]
        if x + r < -2 or x - r > 1282:
            continue
        col = sim.COLORS[b["color"]]
        d.ellipse([x - r, y - r, x + r, y + r], fill=col, outline=tuple(int(c * 0.55) for c in col), width=2)
    d.text((1236, 40), f"{st:5.2f} s", fill=INK, font=font(30), anchor="rm")
    return img
