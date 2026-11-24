"""PIL renderer for theory_of_mind_belief (dark scene, eye-icon watcher)."""
from __future__ import annotations
import math
import sys
from pathlib import Path

from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from common.video5 import font, hex_rgb  # noqa: E402

BG = (20, 20, 20)
SS = 2
N_SIDES = {"square": 4, "triangle": 3, "pentagon": 5, "hexagon": 6}
GREEN, GREY = "#7FE47F", "#555555"
FIXED = "#4A5A78"


def duration(scene) -> float:
    return scene["duration_s"]


def _poly(shape, cx, cy, s):
    if shape in N_SIDES:
        n = N_SIDES[shape]
        return [(cx + s * math.cos(math.radians(-90 + i * 360 / n)), cy + s * math.sin(math.radians(-90 + i * 360 / n)))
                for i in range(n)]
    if shape in ("circle", "ellipse"):
        rx, ry = s, s * (0.62 if shape == "ellipse" else 1)
        return [(cx + rx * math.cos(2 * math.pi * i / 48), cy + ry * math.sin(2 * math.pi * i / 48)) for i in range(48)]
    pts = []
    for i in range(10):
        r = s if i % 2 == 0 else s * 0.42
        th = math.radians(i * 36 - 90)
        pts.append((cx + r * math.cos(th), cy + r * math.sin(th)))
    return pts


def _value(a, t):
    v = a["start"]
    for c in a["changes"]:
        if c["t"] <= t:
            v = max(0, v + c["delta"]) if a["attr"] == "number" else c["to"]
    return v


def frame(scene, t: float) -> Image.Image:
    img = Image.new("RGB", (1280 * SS, 720 * SS), BG)
    d = ImageDraw.Draw(img)
    pal = scene["palette"]
    n = len(scene["actors"])
    xs = [640 + (i - (n - 1) / 2) * 300 for i in range(n)]
    names = ["left", "middle", "right"] if n == 3 else ["left", "right"]
    for a, x in zip(scene["actors"], xs):
        v = _value(a, t)
        shape, color = a["shape"], a["color"]
        if a["attr"] == "shape":
            shape = v
        elif a["attr"] == "color":
            color = v
        cy, size = 340, 95
        d.polygon([(px * SS, py * SS) for px, py in _poly(shape, x, cy, size)], fill=hex_rgb(pal[color]),
                  outline=(255, 255, 255), width=3 * SS)
        if a["attr"] == "number":
            d.text((x * SS, cy * SS), str(v), fill=(255, 255, 255), font=font(80 * SS), anchor="mm")
        d.text((x * SS, (cy + 150) * SS), f"{names[a['index']]} shape", fill=(150, 150, 158), font=font(24 * SS), anchor="mm")
    watching = not any(w["start_s"] <= t <= w["end_s"] for w in scene["windows"])
    ring = hex_rgb(GREEN if watching else GREY)
    cx, cy, r = 120, 610, 44
    d.ellipse([(cx - r) * SS, (cy - r) * SS, (cx + r) * SS, (cy + r) * SS], outline=ring, width=6 * SS, fill=(35, 35, 40))
    if watching:
        pr = r * 0.34
        d.ellipse([(cx - pr) * SS, (cy - pr) * SS, (cx + pr) * SS, (cy + pr) * SS], fill=(255, 255, 255))
    else:
        d.line([((cx - r * 0.5) * SS, cy * SS), ((cx + r * 0.5) * SS, cy * SS)], fill=(230, 230, 230), width=6 * SS)
    d.text((cx * SS, (cy + r + 26) * SS), "watching" if watching else "away", fill=ring, font=font(26 * SS), anchor="mm")
    d.text((60 * SS, 40 * SS), "Watcher and shapes", fill=(235, 235, 240), font=font(32 * SS))
    d.text((1220 * SS, 40 * SS), f"t = {t:.2f} s", fill=(200, 200, 208), font=font(30 * SS), anchor="ra")
    return img.resize((1280, 720), Image.LANCZOS)
