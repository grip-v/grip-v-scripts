"""PIL + OpenCV raster rendering for one laser_mirror_trace clip.

Plain per-frame raster pipeline (no Manim) -- a static grid with visible
mirrors and goal arrows, plus an animated beam trail. All puzzle logic
(mirror layout, path, content_mode) is already finished before this
module is ever called.
"""
from __future__ import annotations
import numpy as np
import cv2
from PIL import Image, ImageDraw

import config as C
import puzzle as P

DELTA = {"up": (-1, 0), "down": (1, 0), "left": (0, -1), "right": (0, 1)}


def _letterbox_bgr(img: Image.Image) -> np.ndarray:
    """Center `img` (square content canvas) onto the fixed OUTPUT_W x OUTPUT_H
    output frame (pillarboxed with the background color) and return BGR."""
    if img.size == (C.OUTPUT_W, C.OUTPUT_H):
        return np.array(img)[:, :, ::-1].copy()
    out = Image.new("RGB", (C.OUTPUT_W, C.OUTPUT_H), C.BACKGROUND_HEX)
    ox = (C.OUTPUT_W - img.width) // 2
    oy = (C.OUTPUT_H - img.height) // 2
    out.paste(img, (ox, oy))
    return np.array(out)[:, :, ::-1].copy()


def _cell_xy(cell, margin, cell_size):
    r, c = cell
    return margin + (c + 0.5) * cell_size, margin + (r + 0.5) * cell_size


def _draw_static(d, n, margin, cell_size, mirror_layout, goals, goal_flash=None):
    for i in range(n + 1):
        x = margin + i * cell_size
        d.line([(x, margin), (x, margin + n * cell_size)], fill=C.GRID_LINE_HEX, width=1)
        y = margin + i * cell_size
        d.line([(margin, y), (margin + n * cell_size, y)], fill=C.GRID_LINE_HEX, width=1)

    for (r, c), orientation in mirror_layout.items():
        x, y = _cell_xy((r, c), margin, cell_size)
        half = cell_size * 0.36
        if orientation == "/":
            d.line([(x - half, y + half), (x + half, y - half)], fill=C.MIRROR_HEX, width=max(2, int(cell_size * 0.09)))
        else:
            d.line([(x - half, y - half), (x + half, y + half)], fill=C.MIRROR_HEX, width=max(2, int(cell_size * 0.09)))

    for g in goals:
        cell = tuple(g["cell_position"])
        x, y = _cell_xy(cell, margin, cell_size)
        dr, dc = DELTA[g["direction"]]
        ax, ay = x + dc * cell_size * 0.62, y + dr * cell_size * 0.62
        is_flashed = goal_flash is not None and tuple(goal_flash["cell_position"]) == cell and goal_flash["direction"] == g["direction"]
        color = C.GOAL_SUCCESS_HEX if is_flashed else C.GOAL_ARROW_HEX
        head = cell_size * 0.18
        d.line([(x, y), (ax, ay)], fill=color, width=max(2, int(cell_size * 0.1)))
        perp_dc, perp_dr = -dr, dc
        d.polygon([(ax, ay), (ax - dc * head + perp_dc * head * 0.6, ay - dr * head + perp_dr * head * 0.6),
                   (ax - dc * head - perp_dc * head * 0.6, ay - dr * head - perp_dr * head * 0.6)], fill=color)


def render_clip(scene: P.SceneData, canvas_px, fps, step_duration_s, out_path):
    n = scene.grid_size
    margin = canvas_px * 0.06
    cell_size = (canvas_px - 2 * margin) / n
    path = scene.displayed_path
    step_frames = max(1, round(step_duration_s * fps))

    goal_flash = scene.reached_goal_position if scene.outcome_type == "reached_goal" else None
    total_move_frames = step_frames * len(path)
    hold_frames = max(1, round(total_move_frames * C.FINAL_HOLD_FRAC / (1 - C.FINAL_HOLD_FRAC)))
    total_frames = total_move_frames + hold_frames

    writer = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (C.OUTPUT_W, C.OUTPUT_H))

    def frame(upto_idx, flash_reflection=False):
        img = Image.new("RGB", (canvas_px, canvas_px), C.BACKGROUND_HEX)
        d = ImageDraw.Draw(img)
        _draw_static(d, n, margin, cell_size, scene.mirror_layout, scene.goal_positions,
                     goal_flash if upto_idx >= len(path) - 1 else None)
        pts = [_cell_xy(tuple(p["cell_position"]), margin, cell_size) for p in path[:upto_idx + 1]]
        if len(pts) > 1:
            d.line(pts, fill=C.BEAM_TRAIL_HEX, width=max(2, int(cell_size * 0.08)))
        if pts:
            x, y = pts[-1]
            rad = cell_size * 0.16
            beam_color = C.REFLECTION_FLASH_HEX if flash_reflection else C.BEAM_HEX
            d.ellipse([x - rad, y - rad, x + rad, y + rad], fill=beam_color)
        return _letterbox_bgr(img)

    for i, p in enumerate(path):
        f = frame(i, flash_reflection=p["reflection_occurred"])
        for _ in range(step_frames):
            writer.write(f)
    last = frame(len(path) - 1)
    for _ in range(hold_frames):
        writer.write(last)
    writer.release()
    return {"duration_seconds": round(total_frames / fps, 3), "fps": fps, "frame_count": total_frames}


def extract_final_frame(scene: P.SceneData, canvas_px, out_png):
    n = scene.grid_size
    margin = canvas_px * 0.06
    cell_size = (canvas_px - 2 * margin) / n
    path = scene.displayed_path
    img = Image.new("RGB", (canvas_px, canvas_px), C.BACKGROUND_HEX)
    d = ImageDraw.Draw(img)
    goal_flash = scene.reached_goal_position if scene.outcome_type == "reached_goal" else None
    _draw_static(d, n, margin, cell_size, scene.mirror_layout, scene.goal_positions, goal_flash)
    pts = [_cell_xy(tuple(p["cell_position"]), margin, cell_size) for p in path]
    if len(pts) > 1:
        d.line(pts, fill=C.BEAM_TRAIL_HEX, width=max(2, int(cell_size * 0.08)))
    frame_bgr = _letterbox_bgr(img)
    cv2.imwrite(str(out_png), frame_bgr)
