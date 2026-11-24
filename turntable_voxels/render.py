"""PIL renderer for turntable_voxels (1280x720, 24 fps through common/video5.py).

Layers, back to front: background (plain / vertical gradient / vignette /
floor), the turntable (a disc with a visible rim, the grid of squares and a
contrasting notch near its edge that shows how far it has turned), the cubes
(painter's order by depth, faces shaded by their direction to the camera,
dark outlines) and the clock. The clock box in the top-right corner is the
only text in the frame.
"""
from __future__ import annotations
import functools
import json
import math
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from common.video5 import font  # noqa: E402

import sim  # noqa: E402

NOTCH = (0.62, 0.08, 9.0)        # notch: inner radius below the rim, outer gap, half-width in degrees


def duration(scene) -> float:
    return scene["duration"]


def shade(n, nr) -> float:
    """Brightness factor of a face: top 1.0; sides by how squarely they face the camera."""
    if n[2]:
        return 1.0
    return 0.6 + 0.24 * max(0.0, -nr[1]) + 0.06 * nr[0]


def face_rgb(scene, colour_key, n, nr):
    base = scene["theme"]["base_rgb"] if colour_key == "base" else sim.COLOURS[colour_key]
    f = shade(n, nr)
    return tuple(int(round(v * f)) for v in base)


@functools.lru_cache(maxsize=8)
def _background(key: str) -> Image.Image:
    th = json.loads(key)
    bg = np.array(th["bg_rgb"], dtype=float)
    yy, xx = np.mgrid[0:sim.H, 0:sim.W].astype(float)
    sign = 1.0 if th["dark"] else -1.0
    if th["bg_style"] == "vgrad":
        img = bg[None, None, :] + sign * 26 * (yy / sim.H)[:, :, None]
    elif th["bg_style"] == "vignette":
        r = np.hypot((xx - sim.W / 2) / sim.W, (yy - sim.H / 2) / sim.H)
        img = bg[None, None, :] - (22 if not th["dark"] else -14) * (r ** 2 * 2.2)[:, :, None]
    elif th["bg_style"] == "floor":
        img = np.where((yy > sim.H * 0.62)[:, :, None], bg + sign * 18, bg[None, None, :])
    else:
        img = np.broadcast_to(bg, (sim.H, sim.W, 3)).copy()
    return Image.fromarray(np.clip(img, 0, 255).astype(np.uint8), "RGB")


def disc_colours(scene):
    dc = scene["theme"]["disc_rgb"]
    rim = tuple(int(v * 0.62) for v in dc)
    light = sim.luma(dc) >= 120
    grid = tuple(int(v * 0.7) for v in dc) if light else tuple(min(255, int(v * 1.45 + 30)) for v in dc)
    return tuple(dc), rim, grid


def draw_disc(img, scene, phi):
    d = ImageDraw.Draw(img)
    dc, rim, grid = disc_colours(scene)
    R = sim.disc_radius(scene)
    gx, gy = scene["gx"], scene["gy"]
    S = scene["S"]
    drop = S * sim.THICK * sim.cs(scene["elev"])[0]
    top = []
    for k in range(120):
        c, s = sim.cs(k * 3)
        top.append(sim.project(scene, 0.0, (gx / 2 + R * c, gy / 2 + R * s, 0))[:2])
    # rim band: the near half of the bottom outline joined to the far half of the top outline
    band = [p for p in top if p[1] <= scene["cy"]] + [(p[0], p[1] + drop) for p in top if p[1] > scene["cy"]]
    band.sort(key=lambda p: math.atan2(p[1] - scene["cy"], p[0] - scene["cx"]))
    d.polygon(band, fill=rim)
    d.polygon(top, fill=dc, outline=rim)
    w = scene["theme"]["grid_w"]
    for x in range(gx + 1):
        d.line([sim.project(scene, phi, (x, 0, 0))[:2], sim.project(scene, phi, (x, gy, 0))[:2]], fill=grid, width=w)
    for y in range(gy + 1):
        d.line([sim.project(scene, phi, (0, y, 0))[:2], sim.project(scene, phi, (gx, y, 0))[:2]], fill=grid, width=w)
    d.polygon(notch_poly(scene, phi), fill=tuple(scene["theme"]["marker_rgb"]))


def notch_poly(scene, phi):
    """The turntable notch (it starts at the front, straight towards the camera, and turns with the disc)."""
    R = sim.disc_radius(scene)
    r0, r1 = R - NOTCH[0], R - NOTCH[1]
    gx, gy = scene["gx"], scene["gy"]
    pts = []
    for r, angs in ((r1, range(-9, 10, 3)), (r0, range(9, -10, -3))):
        for a in angs:
            c, s = sim.cs(-90 + a * NOTCH[2] / 9)
            pts.append(sim.project(scene, phi, (gx / 2 + r * c, gy / 2 + r * s, 0))[:2])
    return pts


def notch_centre(scene, phi):
    R = sim.disc_radius(scene)
    c, s = sim.cs(-90)
    r = R - (NOTCH[0] + NOTCH[1]) / 2
    return sim.project(scene, phi, (scene["gx"] / 2 + r * c, scene["gy"] / 2 + r * s, 0))[:2]


def draw_cubes(img, scene, t):
    d = ImageDraw.Draw(img)
    phi = sim.angle(scene, t)
    cubes = sim.cubes_at(scene, t)
    w = scene["theme"]["outline_w"]
    for k, n, nr, poly in sim.faces(scene, phi, cubes):
        d.polygon(poly, fill=face_rgb(scene, cubes[k][4], n, nr), outline=sim.OUTLINE, width=w)


def draw_timer(img, t):
    d = ImageDraw.Draw(img)
    x0, y0, x1, y1 = sim.TIMER_BOX
    d.rounded_rectangle([x0, y0, x1, y1], radius=10, fill=(10, 10, 14), outline=(120, 120, 132), width=2)
    txt = f"{t:.1f} s"
    f = font(42)
    tw = d.textlength(txt, font=f)
    d.text((x1 - 16 - tw, y0 + 10), txt, fill=(255, 255, 255), font=f)


def frame(scene, t: float) -> Image.Image:
    img = _background(json.dumps(scene["theme"], sort_keys=True)).copy()
    draw_disc(img, scene, sim.angle(scene, t))
    draw_cubes(img, scene, t)
    draw_timer(img, t)
    return img
