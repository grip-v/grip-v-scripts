"""Light-beam mirror reflection -- pure 2D grid geometry, closed-form
reflection off 45-degree mirrors (no lens/refraction rendering needed,
unlike a full optics_geometric dataset). The whole path is computed once
(closed form) then animated growing over time."""
from __future__ import annotations
from pathlib import Path

from PIL import Image, ImageDraw

from render_util import render_clip, save_key_frames, hex_to_rgb

CANVAS = (420, 420)
GRID_N = 6
MARGIN = 30
CELL = (CANVAS[0] - 2 * MARGIN) / GRID_N
DURATION_S = 3.5
FPS = 20

MIRRORS = {(2, 2): "\\", (4, 2): "/", (4, 0): "\\"}
START_CELL = (2, 0)
START_DIR = (1, 0)   # moving +col (rightward)


def compute_path(start_cell, start_dir, mirrors, grid_n, max_steps=40):
    path = [start_cell]
    dx, dy = start_dir
    cell = start_cell
    bounces = []
    for _ in range(max_steps):
        nxt = (cell[0] + dy, cell[1] + dx)   # (row, col): dy moves row, dx moves col
        if not (0 <= nxt[0] < grid_n and 0 <= nxt[1] < grid_n):
            break
        cell = nxt
        path.append(cell)
        if cell in mirrors:
            m = mirrors[cell]
            if m == "/":
                dx, dy = -dy, -dx
            else:
                dx, dy = dy, dx
            bounces.append(cell)
    return path, bounces


PATH, BOUNCES = compute_path(START_CELL, START_DIR, MIRRORS, GRID_N)


def cell_center(cell):
    row, col = cell
    return (MARGIN + (col + 0.5) * CELL, MARGIN + (row + 0.5) * CELL)


def render_frame(t):
    img = Image.new("RGB", CANVAS, hex_to_rgb("#141414"))
    draw = ImageDraw.Draw(img, "RGBA")
    for i in range(GRID_N + 1):
        x = MARGIN + i * CELL
        y = MARGIN + i * CELL
        draw.line([(x, MARGIN), (x, CANVAS[1] - MARGIN)], fill=(60, 60, 65, 255), width=1)
        draw.line([(MARGIN, y), (CANVAS[0] - MARGIN, y)], fill=(60, 60, 65, 255), width=1)
    for cell, kind in MIRRORS.items():
        cx, cy = cell_center(cell)
        h = CELL * 0.38
        if kind == "/":
            draw.line([(cx - h, cy + h), (cx + h, cy - h)], fill=(90, 200, 255, 255), width=4)
        else:
            draw.line([(cx - h, cy - h), (cx + h, cy + h)], fill=(90, 200, 255, 255), width=4)

    frac = min(1.0, t / (DURATION_S * 0.85))
    total_segs = len(PATH) - 1
    grown = frac * total_segs
    full_segs = int(grown)
    pts = [cell_center(c) for c in PATH[:full_segs + 1]]
    if full_segs < total_segs:
        p0 = cell_center(PATH[full_segs])
        p1 = cell_center(PATH[full_segs + 1])
        rem = grown - full_segs
        pts.append((p0[0] + (p1[0] - p0[0]) * rem, p0[1] + (p1[1] - p0[1]) * rem))
    if len(pts) >= 2:
        draw.line(pts, fill=(255, 210, 60, 255), width=4)
    for p in pts[:1]:
        draw.ellipse([p[0] - 5, p[1] - 5, p[0] + 5, p[1] + 5], fill=(255, 255, 200, 255))
    draw.text((10, 10), f"t={t:.2f}s  bounces_so_far={sum(1 for c in BOUNCES if c in PATH[:full_segs+1])}",
               fill=(255, 255, 255, 255))
    return img


if __name__ == "__main__":
    out_dir = Path(__file__).resolve().parent / "out" / "mirror_beam"
    render_clip(render_frame, DURATION_S, FPS, out_dir / "mirror_beam.mp4")
    save_key_frames(render_frame, [0.0, 1.0, 2.0, 3.0], out_dir / "frames", "mirror_beam")
    print("path (row,col):", PATH)
    print("bounces:", BOUNCES)
    print("exit cell:", PATH[-1])
    print("wrote", out_dir / "mirror_beam.mp4")
