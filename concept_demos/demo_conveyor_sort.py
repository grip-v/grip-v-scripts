"""Conveyor-belt sorting -- objects enter at scheduled times, move at
constant velocity, and divert into a color-matched bin at a fixed routing
point. Closed-form position at every t; bin membership is a simple color
rule a model would need to induce from watching items get routed."""
from __future__ import annotations
from pathlib import Path

from PIL import Image, ImageDraw

from render_util import render_clip, save_key_frames, hex_to_rgb

CANVAS = (460, 320)
X_START = -20
BELT_Y = 120
ROUTE_X = 260
DIVERT_LEN = 90
VX = 90.0   # px/s
TOP_BIN_Y = 40
BOT_BIN_Y = 240
DURATION_S = 8.0
FPS = 20

RULE = {"#3F8FE4": "top", "#E4553F": "bottom", "#4FBF6B": "top"}  # color -> bin

OBJECTS = [
    {"spawn_t": 0.3, "color": "#3F8FE4"},
    {"spawn_t": 1.1, "color": "#E4553F"},
    {"spawn_t": 1.9, "color": "#4FBF6B"},
    {"spawn_t": 2.6, "color": "#3F8FE4"},
    {"spawn_t": 3.3, "color": "#E4553F"},
]
for o in OBJECTS:
    o["bin"] = RULE[o["color"]]

SETTLE_DT = (ROUTE_X + DIVERT_LEN - X_START) / VX
for o in OBJECTS:
    o["settle_t"] = o["spawn_t"] + SETTLE_DT

# stacking rank within each bin, by settle order
for bin_name in ("top", "bottom"):
    members = sorted([o for o in OBJECTS if o["bin"] == bin_name], key=lambda o: o["settle_t"])
    for rank, o in enumerate(members):
        o["rank"] = rank


def obj_pos(o, t):
    if t < o["spawn_t"]:
        return None
    dt = t - o["spawn_t"]
    x = X_START + VX * dt
    if x < ROUTE_X:
        return x, BELT_Y
    frac = min(1.0, (x - ROUTE_X) / DIVERT_LEN)
    target_y = TOP_BIN_Y if o["bin"] == "top" else BOT_BIN_Y
    y = BELT_Y + (target_y - BELT_Y) * frac
    x = min(x, ROUTE_X + DIVERT_LEN + o["rank"] * 26)
    return x, y


def render_frame(t):
    img = Image.new("RGB", CANVAS, hex_to_rgb("#141414"))
    draw = ImageDraw.Draw(img, "RGBA")
    draw.line([(0, BELT_Y + 22), (ROUTE_X, BELT_Y + 22)], fill=(90, 90, 95, 255), width=3)
    draw.line([(ROUTE_X, 10), (ROUTE_X, 230)], fill=(255, 255, 255, 60), width=1)
    draw.rectangle([ROUTE_X + 10, TOP_BIN_Y - 25, CANVAS[0] - 10, TOP_BIN_Y + 25],
                   outline=(255, 255, 255, 150), width=2)
    draw.rectangle([ROUTE_X + 10, BOT_BIN_Y - 25, CANVAS[0] - 10, BOT_BIN_Y + 25],
                   outline=(255, 255, 255, 150), width=2)
    draw.text((ROUTE_X + 14, TOP_BIN_Y - 45), "bin: top", fill=(255, 255, 255, 200))
    draw.text((ROUTE_X + 14, BOT_BIN_Y + 30), "bin: bottom", fill=(255, 255, 255, 200))
    for o in OBJECTS:
        p = obj_pos(o, t)
        if p is None:
            continue
        x, y = p
        draw.ellipse([x - 13, y - 13, x + 13, y + 13], fill=hex_to_rgb(o["color"]) + (255,),
                     outline=(255, 255, 255, 255), width=2)
    draw.text((10, 10), f"t={t:.2f}s", fill=(255, 255, 255, 255))
    return img


if __name__ == "__main__":
    out_dir = Path(__file__).resolve().parent / "out" / "conveyor_sort"
    render_clip(render_frame, DURATION_S, FPS, out_dir / "conveyor_sort.mp4")
    save_key_frames(render_frame, [0.5, 2.5, 5.0, 7.5], out_dir / "frames", "conveyor_sort")
    for o in OBJECTS:
        print(o["color"], "->", o["bin"], "settle_t=", round(o["settle_t"], 2), "rank=", o["rank"])
    print("wrote", out_dir / "conveyor_sort.mp4")
