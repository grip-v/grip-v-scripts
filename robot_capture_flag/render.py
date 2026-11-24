"""PIL renderer for robot_capture_flag: a unit-circle arena seen from above.
Each round a flag pops up, both robots glide out from the center at once,
dashed lines show who ended closer, then the arena resets. No score tally
is shown, so win counts must be tracked across the video."""
from __future__ import annotations
import math
import sys
from pathlib import Path

from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from common.video5 import font  # noqa: E402

BG = (241, 243, 236)
INK = (44, 46, 50)
AARON, ERIN = (52, 110, 214), (226, 124, 36)
CX, CY, R = 560, 380, 300
INTRO_S, FLAG_S, GLIDE_S, RESULT_S, OUTRO_S = 1.2, 0.4, 0.8, 0.7, 1.4
ROUND_S = FLAG_S + GLIDE_S + RESULT_S


def duration(scene) -> float:
    return INTRO_S + len(scene["rounds"]) * ROUND_S + OUTRO_S


def P(x, y):
    return (CX + x * R, CY - y * R)


def _dashed(d, a, b, color, n=14):
    for i in range(0, n, 2):
        p = (a[0] + (b[0] - a[0]) * i / n, a[1] + (b[1] - a[1]) * i / n)
        q = (a[0] + (b[0] - a[0]) * (i + 1) / n, a[1] + (b[1] - a[1]) * (i + 1) / n)
        d.line([p, q], fill=color, width=3)


def _robot(d, xy, color, letter):
    x, y = xy
    d.ellipse([x - 16, y - 16, x + 16, y + 16], fill=color, outline=INK, width=2)
    d.text((x - 7, y - 12), letter, fill=(255, 255, 255), font=font(20))


def frame(scene, t: float) -> Image.Image:
    img = Image.new("RGB", (1280, 720), BG)
    d = ImageDraw.Draw(img)
    d.ellipse([CX - R, CY - R, CX + R, CY + R], fill=(252, 252, 248), outline=INK, width=3)
    for i in range(0, 60, 2):
        d.arc([CX - R / 2, CY - R / 2, CX + R / 2, CY + R / 2], i * 6, i * 6 + 6, fill=(150, 150, 156), width=2)
    d.text((CX + R / 2 + 6, CY - 4), "0.5", fill=(120, 120, 126), font=font(16))
    d.text((CX + R + 6, CY - 4), "1", fill=(120, 120, 126), font=font(16))
    d.ellipse([CX - 5, CY - 5, CX + 5, CY + 5], fill=INK)

    d.text((900, 60), "Capture the Flag", fill=INK, font=font(30))
    d.ellipse([900, 130, 926, 156], fill=AARON)
    d.text((940, 130), "Aaron: knows how far", fill=INK, font=font(20))
    d.ellipse([900, 170, 926, 196], fill=ERIN)
    d.text((940, 170), "Erin: knows which way", fill=INK, font=font(20))

    rounds = scene["rounds"]
    k = int((t - INTRO_S) // ROUND_S) if t >= INTRO_S else -1
    if 0 <= k < len(rounds):
        rd = rounds[k]
        u = (t - INTRO_S) - k * ROUND_S
        d.text((900, 250), f"Round {k + 1}", fill=INK, font=font(28))
        th, ph = math.radians(rd["theta"]), math.radians(rd["phi"])
        flag = (rd["r"] * math.cos(th), rd["r"] * math.sin(th))
        fx, fy = P(*flag)
        d.line([fx, fy, fx, fy - 34], fill=INK, width=3)
        d.polygon([(fx, fy - 34), (fx + 24, fy - 27), (fx, fy - 20)], fill=(200, 40, 40))
        g = min(1.0, max(0.0, (u - FLAG_S) / GLIDE_S))
        g = g * g * (3 - 2 * g)
        a = (rd["r"] * g * math.cos(ph), rd["r"] * g * math.sin(ph))
        e = (scene["d"] * g * math.cos(th), scene["d"] * g * math.sin(th))
        if u >= FLAG_S + GLIDE_S:
            _dashed(d, P(*a), (fx, fy), AARON)
            _dashed(d, P(*e), (fx, fy), ERIN)
            win = P(*(e if rd["winner"] == "Erin" else a))
            d.ellipse([win[0] - 26, win[1] - 26, win[0] + 26, win[1] + 26], outline=(40, 160, 70), width=4)
            d.text((900, 300), f"{rd['winner']} is closer", fill=(40, 140, 60), font=font(24))
        _robot(d, P(*a), AARON, "A")
        _robot(d, P(*e), ERIN, "E")
    elif k >= len(rounds):
        d.text((900, 250), "All rounds played", fill=INK, font=font(26))
    return img
