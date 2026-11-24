"""PIL renderer for spinning_polyhedra (1280x720, 24 fps).

Each solid is a wireframe with every edge drawn, seen in perspective from
slightly above (camera elevation and distance vary per clip). The depth cue
varies per clip (`scene["line_style"]`):
  shade        nearer edges brighter and thicker
  dashed_back  edges behind the solid's centre dashed and dim
  flat         every edge the same (depth only from the motion)
Optional extras: translucent faces (`tint`) and small corner dots. Markers
(coloured dot / square / diamond with a dark or light rim) sit on marked
corners and are drawn on top, nearer markers last. Backgrounds are flat, a
vertical gradient or a vignette.

The only text is the timer (seconds to 0.1 s) in one fixed corner per clip;
sim.check keeps every solid's bounding circle clear of the timer box.
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

SS = 2


def duration(scene) -> float:
    return scene["n_frames"] / sim.FPS


def camera_coords(scene, b, f, body=None) -> np.ndarray:
    """Vertices of body b at frame f in camera space (x right, y up, z toward the viewer)."""
    body = body or scene["bodies"][b]
    v = sim.geometry(body["solid"])[0]
    world = v @ sim.orientation(body, f).T
    return world @ sim.rot("x", math.radians(scene["camera"]["elev"])).T


def to_screen(scene, b, p, body=None):
    body = body or scene["bodies"][b]
    cam = scene["camera"]["dist"]
    depth = cam - p[..., 2]
    sx = body["cx"] + body["scale"] * p[..., 0] * cam / depth
    sy = body["cy"] - body["scale"] * p[..., 1] * cam / depth
    return sx, sy


def project(scene, b, f, body=None):
    p = camera_coords(scene, b, f, body)
    sx, sy = to_screen(scene, b, p, body)
    return sx, sy, p[:, 2]


def mark_radius(mk, z) -> float:
    return mk["r"] * (0.85 + 0.3 * (z + 1) / 2)


def mark_screen(scene, m, f):
    """(x, y, radius, camera z) of marker m as drawn at frame f."""
    mk = scene["marks"][m]
    sx, sy, z = project(scene, mk["body"], f)
    k = mk["v"]
    return float(sx[k]), float(sy[k]), mark_radius(mk, float(z[k])), float(z[k])


@functools.lru_cache(maxsize=4)
def _background(key: str) -> Image.Image:
    sc = json.loads(key)
    w, h = sim.W * SS, sim.H * SS
    a, b = np.array(sc["bg"], float), np.array(sc["bg2"], float)
    if sc["bg_style"] == "flat":
        arr = np.broadcast_to(a, (h, w, 3))
    elif sc["bg_style"] == "gradient":
        u = np.linspace(0, 1, h)[:, None, None]
        arr = np.broadcast_to(a + (b - a) * u, (h, w, 3))
    else:
        yy, xx = np.mgrid[0:h, 0:w]
        d = np.hypot((xx - w / 2) / (w / 2), (yy - h / 2) / (h / 2)) / math.sqrt(2)
        arr = b + (a - b) * np.clip(d, 0, 1)[:, :, None]
    return Image.fromarray(np.ascontiguousarray(arr).round().astype(np.uint8), "RGB")


def bg_key(scene) -> str:
    return json.dumps({k: scene[k] for k in ("bg", "bg2", "bg_style")}, sort_keys=True)


def _dashed(d, p, q, col, w, dash=10 * SS, gap=8 * SS):
    L = math.hypot(q[0] - p[0], q[1] - p[1])
    if L < 1:
        return
    ux, uy = (q[0] - p[0]) / L, (q[1] - p[1]) / L
    s = 0.0
    while s < L:
        e = min(L, s + dash)
        d.line([(p[0] + ux * s, p[1] + uy * s), (p[0] + ux * e, p[1] + uy * e)], fill=col, width=w)
        s += dash + gap


def _marker(d, mk, x, y, r):
    rim = (20, 20, 24) if sum(mk["rgb"]) > 330 else (245, 245, 245)
    col = tuple(mk["rgb"])
    x, y, r = x * SS, y * SS, r * SS
    if mk["shape"] == "dot":
        d.ellipse([x - r - 2 * SS, y - r - 2 * SS, x + r + 2 * SS, y + r + 2 * SS], fill=rim)
        d.ellipse([x - r, y - r, x + r, y + r], fill=col)
    elif mk["shape"] == "square":
        q = r * 0.9
        d.rectangle([x - q - 2 * SS, y - q - 2 * SS, x + q + 2 * SS, y + q + 2 * SS], fill=rim)
        d.rectangle([x - q, y - q, x + q, y + q], fill=col)
    else:
        q = r * 1.2
        e = q + 3 * SS
        d.polygon([(x, y - e), (x + e, y), (x, y + e), (x - e, y)], fill=rim)
        d.polygon([(x, y - q), (x + q, y), (x, y + q), (x - q, y)], fill=col)


def _body(d, scene, b, f):
    body = scene["bodies"][b]
    bg = tuple(scene["bg"])
    bright = tuple(body["edge_rgb"])
    dim = sim.edge_dim(body, bg)
    sx, sy, z = project(scene, b, f)
    v, edges, faces = sim.geometry(body["solid"])
    P = [(sx[k] * SS, sy[k] * SS) for k in range(len(sx))]
    if scene["tint"] > 0:
        alpha = int(round(255 * scene["tint"]))
        for fc in sorted(faces, key=lambda fc: float(np.mean(z[fc]))):
            pts = np.array([P[k] for k in fc])
            c = pts.mean(axis=0)
            ang = np.arctan2(pts[:, 1] - c[1], pts[:, 0] - c[0])
            pts = pts[np.argsort(ang)]
            d.polygon([tuple(p) for p in pts], fill=bright + (alpha,))
    style = scene["line_style"]
    w0 = body["edge_w"]
    for a, c in sorted(edges, key=lambda e: float(z[e[0]] + z[e[1]])):
        zm = float(z[a] + z[c]) / 2
        u = (zm + 1) / 2
        if style == "shade":
            col, w = sim.mix(dim, bright, u), int(round(SS * (w0 + 2 * u)))
            d.line([P[a], P[c]], fill=col, width=w)
        elif style == "dashed_back":
            if zm < 0:
                _dashed(d, P[a], P[c], dim, SS * w0)
            else:
                d.line([P[a], P[c]], fill=bright, width=SS * (w0 + 1))
        else:
            d.line([P[a], P[c]], fill=bright, width=SS * (w0 + 1))
    if scene["corner_dots"]:
        for k in range(len(P)):
            r = SS * (w0 + 1.5)
            d.ellipse([P[k][0] - r, P[k][1] - r, P[k][0] + r, P[k][1] + r], fill=bright)


def timer_xy(scene):
    x0, y0, x1, y1 = sim.timer_box(scene)
    t = scene["timer"]
    return (x0 if t["corner"][1] == "l" else x1), y0, ("la" if t["corner"][1] == "l" else "ra")


def draw_timer(img, scene, f):
    x, y, anchor = timer_xy(scene)
    ImageDraw.Draw(img).text((x, y), sim.timer_text(f), fill=tuple(scene["timer"]["ink"]),
                             font=font(scene["timer"]["size"]), anchor=anchor)


def frame_at(scene, f: int) -> Image.Image:
    img = _background(bg_key(scene)).copy()
    d = ImageDraw.Draw(img, "RGBA")
    for b in range(len(scene["bodies"])):
        _body(d, scene, b, f)
    ms = [(i,) + mark_screen(scene, i, f) for i in range(len(scene["marks"]))]
    for i, x, y, r, z in sorted(ms, key=lambda t: t[4]):
        _marker(d, scene["marks"][i], x, y, r)
    img = img.resize((sim.W, sim.H), Image.LANCZOS)
    draw_timer(img, scene, f)
    return img


def frame(scene, t: float) -> Image.Image:
    return frame_at(scene, int(round(t * sim.FPS)))
