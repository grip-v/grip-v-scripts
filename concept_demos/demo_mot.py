"""Multiple-object tracking under crossing paths (Pylyshyn & Storm MOT
paradigm). 4 identical-appearance dots move on independent closed-form
paths; labels are shown briefly at t=0 then fade out, dots cross paths
several times, labels reappear at the end so the true final identity
assignment is visually checkable. The real dataset would never re-show
labels -- that's the question."""
from __future__ import annotations
import math
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

from render_util import render_clip, save_key_frames, hex_to_rgb

CANVAS = (420, 420)
CX, CY = 210, 210
N_DOTS = 4
DOT_R = 14
DURATION_S = 5.0
FPS = 20
LABEL_FADE_END = 0.8
LABEL_REAPPEAR_START = 4.2

# each dot: independent circular/Lissajous path, closed-form
PATH_PARAMS = [
    {"rx": 140, "ry": 90, "wx": 1.3, "wy": 0.9, "phase": 0.0},
    {"rx": 100, "ry": 130, "wx": 0.8, "wy": 1.4, "phase": 1.2},
    {"rx": 150, "ry": 60, "wx": 1.1, "wy": 1.7, "phase": 2.5},
    {"rx": 70, "ry": 150, "wx": 1.6, "wy": 1.0, "phase": 4.0},
]


def dot_pos(dot_id: int, t: float):
    p = PATH_PARAMS[dot_id]
    x = CX + p["rx"] * math.cos(p["wx"] * t + p["phase"])
    y = CY + p["ry"] * math.sin(p["wy"] * t + p["phase"])
    return x, y


def render_frame(t):
    img = Image.new("RGB", CANVAS, hex_to_rgb("#141414"))
    draw = ImageDraw.Draw(img, "RGBA")
    show_labels = t <= LABEL_FADE_END or t >= LABEL_REAPPEAR_START
    label_alpha = 255
    if t > LABEL_FADE_END and t < LABEL_FADE_END + 0.4:
        label_alpha = int(255 * (1 - (t - LABEL_FADE_END) / 0.4))
        show_labels = label_alpha > 5
    font = ImageFont.load_default(size=14)
    for i in range(N_DOTS):
        x, y = dot_pos(i, t)
        draw.ellipse([x - DOT_R, y - DOT_R, x + DOT_R, y + DOT_R],
                     fill=(220, 220, 230, 255), outline=(255, 255, 255, 255), width=2)
        if show_labels:
            a = label_alpha if t < LABEL_REAPPEAR_START else 255
            draw.text((x - 4, y - 6), str(i + 1), fill=(20, 20, 20, a), font=font)
    draw.text((10, 10), f"t={t:.2f}s", fill=(255, 255, 255, 255))
    return img


if __name__ == "__main__":
    out_dir = Path(__file__).resolve().parent / "out" / "mot"
    render_clip(render_frame, DURATION_S, FPS, out_dir / "mot.mp4")
    key_times = [0.0, 0.8, 2.5, 4.5]
    save_key_frames(render_frame, key_times, out_dir / "frames", "mot")
    print("final positions (t=4.9, identity known):")
    for i in range(N_DOTS):
        print(f"  dot {i+1}: {dot_pos(i, 4.9)}")
    print("wrote", out_dir / "mot.mp4")
