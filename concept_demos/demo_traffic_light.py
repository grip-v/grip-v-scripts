"""Traffic-light queue -- cars spawn on a schedule and move at constant
speed unless blocked by a red light or the car ahead. Fully determined by
the spawn schedule + light cycle (no randomness at render time); simulated
once at fine dt and sampled for rendering, same "precompute the closed-
form trajectory, animate it" discipline as the other demos."""
from __future__ import annotations
from pathlib import Path

from PIL import Image, ImageDraw

from render_util import render_clip, save_key_frames, hex_to_rgb

CANVAS = (460, 200)
ROAD_Y = 120
STOP_X = 300
CAR_SPACING = 34
CAR_SPEED = 70.0
X_START = -20
LIGHT_PERIOD = 4.0
RED_DURATION = 2.0
DURATION_S = 7.0
FPS = 20
DT = 0.02

CARS = [{"spawn_t": 0.2, "color": "#3F8FE4"}, {"spawn_t": 0.9, "color": "#E4553F"},
        {"spawn_t": 1.6, "color": "#4FBF6B"}, {"spawn_t": 2.3, "color": "#E4C43F"},
        {"spawn_t": 4.4, "color": "#B85FE4"}]


def light_state(t):
    phase = t % LIGHT_PERIOD
    return "red" if phase < RED_DURATION else "green"


def simulate():
    n_steps = int(DURATION_S / DT) + 2
    x = [None] * len(CARS)
    log = [[] for _ in CARS]
    for step in range(n_steps):
        t = step * DT
        light = light_state(t)
        for i, car in enumerate(CARS):
            if x[i] is None:
                if t >= car["spawn_t"]:
                    x[i] = X_START
                else:
                    log[i].append(None)
                    continue
            desired = x[i] + CAR_SPEED * DT
            limit = CANVAS[0] + 60
            if i > 0 and x[i - 1] is not None:
                limit = min(limit, x[i - 1] - CAR_SPACING)
            if light == "red" and x[i] < STOP_X:
                limit = min(limit, STOP_X)
            x[i] = min(desired, limit)
            log[i].append(x[i])
    return log


LOG = simulate()


def car_x(i, t):
    idx = min(len(LOG[i]) - 1, max(0, int(round(t / DT))))
    return LOG[i][idx]


def render_frame(t):
    img = Image.new("RGB", CANVAS, hex_to_rgb("#141414"))
    draw = ImageDraw.Draw(img, "RGBA")
    draw.line([(0, ROAD_Y + 20), (CANVAS[0], ROAD_Y + 20)], fill=(90, 90, 95, 255), width=2)
    draw.line([(0, ROAD_Y - 20), (CANVAS[0], ROAD_Y - 20)], fill=(90, 90, 95, 255), width=2)
    draw.line([(STOP_X, ROAD_Y - 20), (STOP_X, ROAD_Y + 20)], fill=(255, 255, 255, 200), width=2)
    light = light_state(t)
    light_color = (230, 60, 60, 255) if light == "red" else (80, 220, 100, 255)
    draw.ellipse([STOP_X - 10, ROAD_Y - 60, STOP_X + 10, ROAD_Y - 40], fill=light_color)
    for i, car in enumerate(CARS):
        x = car_x(i, t)
        if x is None:
            continue
        draw.rectangle([x - 14, ROAD_Y - 10, x + 14, ROAD_Y + 10],
                       fill=hex_to_rgb(car["color"]) + (255,), outline=(255, 255, 255, 255), width=2)
    n_queued = sum(1 for i in range(len(CARS)) if car_x(i, t) is not None and car_x(i, t) < STOP_X - 2
                   and light == "red")
    draw.text((10, 10), f"t={t:.2f}s  light={light}  queued={n_queued}", fill=(255, 255, 255, 255))
    return img


if __name__ == "__main__":
    out_dir = Path(__file__).resolve().parent / "out" / "traffic_light"
    render_clip(render_frame, DURATION_S, FPS, out_dir / "traffic_light.mp4")
    save_key_frames(render_frame, [0.5, 1.8, 3.0, 4.8, 6.5], out_dir / "frames", "traffic_light")
    print("wrote", out_dir / "traffic_light.mp4")
