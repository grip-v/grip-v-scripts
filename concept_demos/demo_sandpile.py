"""Cascading threshold / sandpile (chip-firing) -- discrete grains arrive
at scheduled times; a cell at/over threshold "fires," resetting and
pushing +1 to each neighbor, which may itself cascade. The full event log
is precomputed once via a priority-queue simulation (deterministic, no
randomness at render time) and then played back on a real clock so
cascades visibly ripple outward instead of happening instantly."""
from __future__ import annotations
import heapq
from pathlib import Path

from PIL import Image, ImageDraw

from render_util import render_clip, save_key_frames, hex_to_rgb

CANVAS = (460, 260)
N_CELLS = 6
THRESHOLD = 4
FIRE_DELAY = 0.35
ARRIVALS = [(0.3, 2), (0.9, 2), (1.5, 2), (2.1, 2), (2.7, 3), (3.3, 2)]
DURATION_S = 6.0
FPS = 20

CELL_W = 60
BASE_Y = 220
UNIT_H = 36


def simulate():
    height = [0] * N_CELLS
    log = []  # (t, cell, new_height, fired)
    pq = [(t, 0, "arrive", c) for i, (t, c) in enumerate(ARRIVALS)]
    heapq.heapify(pq)
    seq = len(pq)
    while pq:
        t, _, kind, c = heapq.heappop(pq)
        if kind == "arrive":
            height[c] += 1
            log.append((t, c, height[c], False))
            if height[c] >= THRESHOLD:
                heapq.heappush(pq, (t + FIRE_DELAY, seq, "fire", c)); seq += 1
        else:
            height[c] -= THRESHOLD
            log.append((t, c, height[c], True))
            for nb in (c - 1, c + 1):
                if 0 <= nb < N_CELLS:
                    height[nb] += 1
                    log.append((t, nb, height[nb], False))
                    if height[nb] >= THRESHOLD:
                        heapq.heappush(pq, (t + FIRE_DELAY, seq, "fire", nb)); seq += 1
    return sorted(log, key=lambda e: e[0])


LOG = simulate()
N_FIRES = sum(1 for e in LOG if e[3])


def height_at(cell, t):
    h = 0
    for (et, ec, eh, fired) in LOG:
        if et > t:
            break
        if ec == cell:
            h = eh
    return h


def recent_fire(t, window=0.2):
    for (et, ec, eh, fired) in LOG:
        if fired and et <= t <= et + window:
            return ec
    return None


def render_frame(t):
    img = Image.new("RGB", CANVAS, hex_to_rgb("#141414"))
    draw = ImageDraw.Draw(img, "RGBA")
    flash_cell = recent_fire(t)
    margin = (CANVAS[0] - N_CELLS * CELL_W) / 2
    thresh_y = BASE_Y - THRESHOLD * UNIT_H
    draw.line([(margin, thresh_y), (margin + N_CELLS * CELL_W, thresh_y)],
               fill=(255, 90, 90, 150), width=1)
    for c in range(N_CELLS):
        h = height_at(c, t)
        x0 = margin + c * CELL_W + 6
        x1 = margin + (c + 1) * CELL_W - 6
        y1 = BASE_Y
        y0 = BASE_Y - h * UNIT_H
        color = (255, 200, 80, 255) if c == flash_cell else (80, 170, 230, 255)
        draw.rectangle([x0, y0, x1, y1], fill=color, outline=(255, 255, 255, 255), width=2)
        draw.text((x0 + 6, BASE_Y + 6), str(c), fill=(200, 200, 200, 255))
    fires_so_far = sum(1 for e in LOG if e[3] and e[0] <= t)
    draw.text((10, 10), f"t={t:.2f}s  cascades_so_far={fires_so_far}", fill=(255, 255, 255, 255))
    return img


if __name__ == "__main__":
    out_dir = Path(__file__).resolve().parent / "out" / "sandpile"
    render_clip(render_frame, DURATION_S, FPS, out_dir / "sandpile.mp4")
    save_key_frames(render_frame, [0.5, 2.2, 3.0, 3.6, 5.5], out_dir / "frames", "sandpile")
    print(f"total cascades: {N_FIRES}")
    for e in LOG:
        print(e)
    print("wrote", out_dir / "sandpile.mp4")
