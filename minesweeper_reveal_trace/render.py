"""PIL renderer for minesweeper_reveal_trace. The frame holds only the
game itself: the board, row numbers on the left and column numbers on top
(1..n, the labels every question uses), the mine total and a reveal-step
counter. One tile is uncovered per step and outlined in yellow for that
step; the clip ends on a held final board."""
from __future__ import annotations
import sys
from pathlib import Path

from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from common.video5 import font  # noqa: E402

BG = (220, 220, 220)
COVER, COVER_HI, COVER_LO = (192, 192, 192), (240, 240, 240), (128, 128, 128)
OPEN, GRID = (232, 232, 232), (156, 156, 156)
INK = (26, 26, 26)
DIGIT = {1: (0, 0, 255), 2: (0, 128, 0), 3: (255, 0, 0), 4: (0, 0, 128), 5: (128, 0, 0), 6: (0, 128, 128),
         7: (0, 0, 0), 8: (128, 128, 128)}
INTRO_S, STEP_S, HOLD_S = 0.8, 0.6, 1.6


def duration(scene) -> float:
    return INTRO_S + len(scene["reveals"]) * STEP_S + HOLD_S


def frame(scene, t: float) -> Image.Image:
    n, rv = scene["n"], scene["reveals"]
    steps = 0 if t < INTRO_S else min(len(rv), int((t - INTRO_S) / STEP_S) + 1)
    cell = min(560 / n, 64)
    size = cell * n
    x0, y0 = (1280 - size) / 2 + 20, 110 + (590 - size) / 2
    img = Image.new("RGB", (1280, 720), BG)
    d = ImageDraw.Draw(img)
    d.text((x0, 30), f"Mines: {scene['mines']}", fill=INK, font=font(30))
    d.text((x0 + size - 170, 30), f"Step: {steps}", fill=INK, font=font(30))
    lf = font(max(15, int(cell * 0.36)))
    for k in range(1, n + 1):
        d.text((x0 + (k - 0.5) * cell, y0 - 16), str(k), fill=INK, font=lf, anchor="mm")
        d.text((x0 - 20, y0 + (k - 0.5) * cell), str(k), fill=INK, font=lf, anchor="mm")
    shown = {(x["r"], x["c"]): x["shown"] for x in rv[:steps]}
    df = font(int(cell * 0.56))
    for r in range(1, n + 1):
        for c in range(1, n + 1):
            a, b = x0 + (c - 1) * cell, y0 + (r - 1) * cell
            box = [a, b, a + cell, b + cell]
            if (r, c) in shown:
                v = shown[(r, c)]
                d.rectangle(box, fill=(235, 90, 80) if v == "mine" else OPEN, outline=GRID, width=1)
                cx, cy = a + cell / 2, b + cell / 2
                if v == "mine":
                    rad = cell * 0.24
                    for dx, dy in ((1, 0), (0, 1), (0.7, 0.7), (0.7, -0.7)):
                        d.line([cx - dx * rad * 1.45, cy - dy * rad * 1.45, cx + dx * rad * 1.45, cy + dy * rad * 1.45],
                               fill=INK, width=max(2, int(cell / 22)))
                    d.ellipse([cx - rad, cy - rad, cx + rad, cy + rad], fill=INK)
                elif v:
                    d.text((cx, cy), str(v), fill=DIGIT[v], font=df, anchor="mm")
            else:
                w = max(2, int(cell / 14))
                d.rectangle(box, fill=COVER)
                d.polygon([(a, b), (a + cell, b), (a + cell - w, b + w), (a + w, b + w), (a + w, b + cell - w),
                           (a, b + cell)], fill=COVER_HI)
                d.polygon([(a + cell, b + cell), (a, b + cell), (a + w, b + cell - w), (a + cell - w, b + cell - w),
                           (a + cell - w, b + w), (a + cell, b)], fill=COVER_LO)
    d.rectangle([x0, y0, x0 + size, y0 + size], outline=(110, 110, 110), width=2)
    if 0 < steps and t < INTRO_S + len(rv) * STEP_S:
        cur = rv[steps - 1]
        a, b = x0 + (cur["c"] - 1) * cell, y0 + (cur["r"] - 1) * cell
        d.rectangle([a - 1, b - 1, a + cell + 1, b + cell + 1], outline=(255, 200, 0), width=max(3, int(cell / 12)))
    return img
