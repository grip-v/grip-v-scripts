"""PIL renderer for robot_road_trip: a one-lane road seen from above with a
shoulder underneath, unit markers 0..12 and a running clock. Positions are
rebuilt from the stored event log (sim.position)."""
from __future__ import annotations
import sys
from pathlib import Path

from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from common.video5 import font, hex_rgb  # noqa: E402

import sim  # noqa: E402

BG = (232, 236, 226)
INK = (40, 42, 46)
X0, X1 = 90, 1190
LANE_Y0, LANE_Y1, SH_Y1 = 330, 400, 452
INTRO_S, OUTRO_S = 1.0, 1.6


def duration(scene) -> float:
    return INTRO_S + max(scene["finish"].values()) + OUTRO_S


def _x(u):
    return X0 + (X1 - X0) * u / sim.ROAD


def frame(scene, t: float) -> Image.Image:
    img = Image.new("RGB", (1280, 720), BG)
    d = ImageDraw.Draw(img)
    d.text((60, 40), "Robot Road Trip", fill=INK, font=font(30))
    d.text((60, 84), "faster cars never slow down: the slower car pulls onto the shoulder to let them pass",
           fill=(96, 100, 96), font=font(20))
    ts = max(0.0, t - INTRO_S)
    d.text((1000, 40), f"t = {ts:5.1f} s", fill=INK, font=font(30))

    d.rectangle([X0, LANE_Y0, X1, LANE_Y1], fill=(78, 80, 86))
    d.rectangle([X0, LANE_Y1, X1, SH_Y1], fill=(170, 164, 150))
    d.line([X0, LANE_Y1, X1, LANE_Y1], fill=(240, 240, 240), width=3)
    for k in range(int(sim.ROAD) + 1):
        x = _x(k)
        d.line([x, LANE_Y0 - 16, x, LANE_Y0], fill=INK, width=2)
        d.text((x - 7 if k < 10 else x - 12, LANE_Y0 - 42), str(k), fill=INK, font=font(18))
    for j in range(12):                                   # finish line checkers
        c = INK if j % 2 else (250, 250, 250)
        d.rectangle([X1 - 6, LANE_Y0 + j * 10, X1 + 6, LANE_Y0 + j * 10 + 10], fill=c)
    d.text((X1 - 30, SH_Y1 + 10), "finish", fill=INK, font=font(18))
    d.text((X0, SH_Y1 + 10), "shoulder", fill=(110, 104, 92), font=font(16))

    wy = 520
    d.text((X0, wy), "not yet on the road:", fill=(96, 100, 96), font=font(18))
    wx = X0 + 190
    for c in scene["cars"]:
        n = c["name"]
        x, mode = sim.position(scene["events"][n], c["speed"], ts)
        col = hex_rgb(c["color"])
        if mode == "waiting":
            if ts >= 0:
                d.rounded_rectangle([wx, wy - 4, wx + 40, wy + 26], radius=6, fill=col, outline=INK, width=2)
                d.text((wx + 13, wy - 1), n, fill=(255, 255, 255), font=font(20))
                wx += 52
            continue
        if mode == "done":
            continue
        xl, xr = _x(max(0.0, x - sim.CAR)), _x(x)
        y0, y1 = (LANE_Y0 + 14, LANE_Y1 - 14) if mode == "lane" else (LANE_Y1 + 8, SH_Y1 - 8)
        d.rounded_rectangle([xl, y0, max(xl + 8, xr), y1], radius=8, fill=col, outline=INK, width=2)
        d.text(((xl + xr) / 2 - 7, (y0 + y1) / 2 - 12), n, fill=(255, 255, 255), font=font(20))
    return img
