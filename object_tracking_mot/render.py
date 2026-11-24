"""PIL renderer for object_tracking_mot: identical dots, numbered only until the label fade, drawn inside a
square arena with a faint centre cross marking the quadrants. Logical 420 px arena scaled to 1280x720."""
from __future__ import annotations
import math
import sys
from pathlib import Path

from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from common.video5 import font, hex_rgb  # noqa: E402

W, H, SS = 1280, 720, 2
SCALE = 1.6
DOT_R = 14
BG = hex_rgb("#141414")
DOT = hex_rgb("#DCDCE6")


def duration(scene) -> float:
    return scene["duration"]


def frame(scene, t: float) -> Image.Image:
    A = scene["canvas"]
    side = A * SCALE
    ox, oy = (W - side) / 2, (H - side) / 2
    img = Image.new("RGB", (W * SS, H * SS), BG)
    d = ImageDraw.Draw(img, "RGBA")
    d.rectangle([ox * SS, oy * SS, (ox + side) * SS, (oy + side) * SS], outline=(70, 70, 75), width=3 * SS)
    d.line([((ox + side / 2) * SS, oy * SS), ((ox + side / 2) * SS, (oy + side) * SS)], fill=(60, 60, 66), width=2 * SS)
    d.line([(ox * SS, (oy + side / 2) * SS), ((ox + side) * SS, (oy + side / 2) * SS)], fill=(60, 60, 66), width=2 * SS)
    fe, fd = scene["label_fade_end"], scene["label_fade_dur"]
    alpha = 255 if t <= fe else int(255 * (1 - (t - fe) / fd)) if t < fe + fd else 0
    r, c = DOT_R * SCALE * SS, A / 2
    f = font(int(22 * SS))
    for k, dot in enumerate(scene["dots"]):
        x = ox + (c + dot["rx"] * math.cos(dot["wx"] * t + dot["phase"])) * SCALE
        y = oy + (c + dot["ry"] * math.sin(dot["wy"] * t + dot["phase"])) * SCALE
        x, y = x * SS, y * SS
        d.ellipse([x - r, y - r, x + r, y + r], fill=DOT + (255,), outline=(255, 255, 255, 255), width=2 * SS)
        if alpha > 5:
            d.text((x, y), str(k + 1), fill=(20, 20, 20, alpha), font=f, anchor="mm")
    tf = font(int(20 * SS))
    d.text(((ox + side + 24) * SS, oy * SS), f"t = {t:.1f} s", fill=(190, 190, 195, 255), font=tf, anchor="la")
    return img.resize((W, H), Image.LANCZOS)
