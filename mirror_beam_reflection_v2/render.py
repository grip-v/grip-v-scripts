"""PIL renderer for mirror_beam_reflection: grid, mirrors, and the beam growing along the truncated path."""
from __future__ import annotations
import sys
from pathlib import Path

from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from common.video5 import font  # noqa: E402

import sim  # noqa: E402

W, H, SS = 1280, 720, 2


def duration(scene) -> float:
    return scene["duration"]


def frame(scene, t: float):
    n = scene["grid_n"]
    CELL = {6: 100, 7: 86, 8: 76}[n]
    k = CELL / 100
    ox, oy = (W - n * CELL) / 2, (H - n * CELL) / 2 + 30
    img = Image.new("RGB", (W * SS, H * SS), (20, 20, 20))
    d = ImageDraw.Draw(img, "RGBA")

    def ctr(cell):
        return ((ox + (cell[1] + 0.5) * CELL) * SS, (oy + (cell[0] + 0.5) * CELL) * SS)

    for i in range(n + 1):
        d.line([((ox + i * CELL) * SS, oy * SS), ((ox + i * CELL) * SS, (oy + n * CELL) * SS)],
               fill=(60, 60, 65, 255), width=SS)
        d.line([(ox * SS, (oy + i * CELL) * SS), ((ox + n * CELL) * SS, (oy + i * CELL) * SS)],
               fill=(60, 60, 65, 255), width=SS)
    d.text((W * SS / 2, 40 * SS), "Beam and mirrors", font=font(30 * SS), fill=(230, 230, 230), anchor="mm")
    path = sim.trace(scene["mirrors"], scene["start"], scene["dir"], n)[0][:scene["truncate_index"] + 1]
    segs = len(path) - 1
    grown = min(1.0, t / (len(path) * sim.SEGMENT_S)) * segs
    full = int(grown)
    pts = [ctr(c) for c in path[:full + 1]]
    e0 = ctr(path[0])                                   # stub from the grid edge shows the entry side
    stub = (e0[0] - scene["dir"][1] * CELL * 0.5 * SS, e0[1] - scene["dir"][0] * CELL * 0.5 * SS)
    pts = [stub] + pts
    if full < segs:
        p0, p1, r = ctr(path[full]), ctr(path[full + 1]), grown - full
        pts.append((p0[0] + (p1[0] - p0[0]) * r, p0[1] + (p1[1] - p0[1]) * r))
    if len(pts) >= 2:
        d.line(pts, fill=(255, 210, 60, 255), width=int(12 * k * SS), joint="curve")
    r5 = 10 * k * SS
    d.ellipse([pts[1][0] - r5, pts[1][1] - r5, pts[1][0] + r5, pts[1][1] + r5], fill=(255, 255, 200, 255))
    if len(pts) >= 3:                                   # bright head so the beam's advance is unmistakable
        hx, hy = pts[-1]
        rh = 16 * k * SS
        d.ellipse([hx - rh, hy - rh, hx + rh, hy + rh], fill=(255, 250, 220, 255))
    h = CELL * 0.36 * SS
    for key, kind in scene["mirrors"].items():
        cx, cy = ctr(tuple(int(v) for v in key.split("-")))
        seg = [(cx - h, cy + h), (cx + h, cy - h)] if kind == "/" else [(cx - h, cy - h), (cx + h, cy + h)]
        d.line(seg, fill=(90, 200, 255, 255), width=int(7 * SS))

    lab = font(16 * SS)
    for i in range(n):
        d.text(((ox + (i + 0.5) * CELL) * SS, (oy - 14) * SS), f"col {i}", font=lab, fill=(170, 170, 170), anchor="mm")
        d.text(((ox - 32) * SS, (oy + (i + 0.5) * CELL) * SS), f"row {i}", font=lab, fill=(170, 170, 170), anchor="mm")
    return img.resize((W, H), Image.LANCZOS)
