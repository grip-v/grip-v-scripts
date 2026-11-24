"""Balance-scale weighing -- items land on a pan at scheduled times; beam
angle is closed-form from net torque (mass difference x fixed arm length,
clamped to the physical stop angle), eased smoothly between updates."""
from __future__ import annotations
import math
from pathlib import Path

from PIL import Image, ImageDraw

from render_util import render_clip, save_key_frames, hex_to_rgb

CANVAS = (460, 300)
FULCRUM = (230, 150)
ARM_LEN = 170
MAX_ANGLE_DEG = 16.0
ANGLE_PER_MASS_DIFF = 4.0   # degrees per unit of (right_mass - left_mass)
TRANSITION_S = 0.5
DURATION_S = 5.0
FPS = 20

WEIGHTS = [(0.4, "left", 2), (1.2, "right", 1), (2.0, "right", 3), (3.0, "left", 1)]


def clamp(v, lo, hi):
    return max(lo, min(hi, v))


def build_events():
    events = [(0.0, 0.0)]
    left_mass = right_mass = 0
    for (t, side, m) in sorted(WEIGHTS):
        if side == "left":
            left_mass += m
        else:
            right_mass += m
        target = clamp(ANGLE_PER_MASS_DIFF * (right_mass - left_mass), -MAX_ANGLE_DEG, MAX_ANGLE_DEG)
        events.append((t, target))
    return events


EVENTS = build_events()


def angle_at(t):
    idx = 0
    for i in range(len(EVENTS)):
        if EVENTS[i][0] <= t:
            idx = i
        else:
            break
    start_val = EVENTS[idx - 1][1] if idx > 0 else 0.0
    target_val = EVENTS[idx][1]
    t0 = EVENTS[idx][0]
    frac = min(1.0, (t - t0) / TRANSITION_S) if t >= t0 else 0.0
    return start_val + (target_val - start_val) * frac


def landed_masses(t, side):
    return [m for (dt, s, m) in WEIGHTS if s == side and dt <= t]


def render_frame(t):
    img = Image.new("RGB", CANVAS, hex_to_rgb("#141414"))
    draw = ImageDraw.Draw(img, "RGBA")
    fx, fy = FULCRUM
    draw.polygon([(fx - 20, fy + 90), (fx + 20, fy + 90), (fx, fy)], fill=(90, 90, 100, 255))
    ang = math.radians(angle_at(t))
    lx, ly = fx - ARM_LEN * math.cos(ang), fy - ARM_LEN * math.sin(ang)
    rx, ry = fx + ARM_LEN * math.cos(ang), fy + ARM_LEN * math.sin(ang)
    draw.line([(lx, ly), (rx, ry)], fill=(220, 220, 225, 255), width=5)
    draw.ellipse([fx - 6, fy - 6, fx + 6, fy + 6], fill=(255, 220, 100, 255))

    pan_drop = 40
    for (px, py), side in [((lx, ly), "left"), ((rx, ry), "right")]:
        pan_y = py + pan_drop
        draw.line([(px, py), (px, pan_y)], fill=(160, 160, 165, 255), width=2)
        draw.line([(px - 24, pan_y), (px + 24, pan_y)], fill=(200, 200, 205, 255), width=3)
        masses = landed_masses(t, side)
        for i, m in enumerate(masses):
            s = 14 + m * 3
            wy = pan_y - 6 - i * (s + 3)
            draw.rectangle([px - s / 2, wy - s, px + s / 2, wy], fill=(90, 170, 230, 255),
                           outline=(255, 255, 255, 255), width=1)
            draw.text((px - 4, wy - s + 2), str(m), fill=(20, 20, 20, 255))

    total_l = sum(landed_masses(t, "left"))
    total_r = sum(landed_masses(t, "right"))
    draw.text((10, 10), f"t={t:.2f}s  left_mass={total_l}  right_mass={total_r}  angle={angle_at(t):.1f}deg",
               fill=(255, 255, 255, 255))
    return img


if __name__ == "__main__":
    out_dir = Path(__file__).resolve().parent / "out" / "balance_scale"
    render_clip(render_frame, DURATION_S, FPS, out_dir / "balance_scale.mp4")
    save_key_frames(render_frame, [0.2, 0.9, 1.8, 2.6, 4.5], out_dir / "frames", "balance_scale")
    print("events (t, target_angle_deg):", EVENTS)
    print("wrote", out_dir / "balance_scale.mp4")
