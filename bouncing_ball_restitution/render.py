"""PIL renderer for bouncing_ball_restitution (1280x720, 24 fps).

The background (gradient, overlay, floor, pads, shelves, walls) is drawn once
per scene; each frame pastes the object sprites at their positions and the
clock. The clock (top-left panel, "s.t s", truncated to 0.1 s) is the only
text in the frame; nothing else is ever drawn above y = sim.TOP_Y there.
"""
from __future__ import annotations
import functools
import json
import math
import sys
from pathlib import Path

from PIL import Image, ImageDraw

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE))
from common.video5 import font  # noqa: E402

import sim  # noqa: E402

W, H = sim.W, sim.H
SS = 4                                   # sprite supersampling
TIMER_BOX = (20, 16, 196, 70)            # fixed clock panel
TIMER_FONT = 34


def duration(scene) -> float:
    return scene["duration"]


def _mix(a, b, u):
    return tuple(int(round(a[i] + (b[i] - a[i]) * u)) for i in range(3))


def _floor_rgb(style):
    base = (205, 196, 178) if style["dark"] else (112, 104, 96)
    alt = (160, 170, 186) if style["dark"] else (86, 98, 112)
    return _mix(base, alt, style["floor_shade"])


def _ink(style):
    return (225, 228, 235) if style["dark"] else (48, 50, 58)


@functools.lru_cache(maxsize=4)
def _background(key: str) -> Image.Image:
    scene = json.loads(key)
    st = scene["style"]
    img = Image.new("RGB", (W, H))
    d = ImageDraw.Draw(img)
    top, bot = st["bg_top"], st["bg_bottom"]
    for y in range(H):
        d.line([(0, y), (W, y)], fill=_mix(top, bot, y / (H - 1)))
    faint = _mix(_mix(top, bot, 0.5), _ink(st), 0.13)
    ov = st["overlay"]
    if ov == "grid":
        for x in range(0, W, 40):
            d.line([(x, sim.TOP_Y - 10), (x, H)], fill=faint, width=1)
        for y in range(sim.TOP_Y - 10, H, 40):
            d.line([(0, y), (W, y)], fill=faint, width=1)
    elif ov == "dots":
        for x in range(20, W, 48):
            for y in range(sim.TOP_Y + 4, H, 48):
                d.ellipse([x - 2, y - 2, x + 2, y + 2], fill=faint)
    ink = _ink(st)
    fl = _floor_rgb(st)
    if scene["scenario"] == "throw":
        _shelves(d, scene, fl, ink)
    else:
        _floor(d, scene, fl, ink)
        if ov == "ruler":
            x = scene["wall_l"] - 14
            for k, y in enumerate(range(int(scene["floor_y"]), sim.TOP_Y + 10, -25)):
                ln = 18 if k % 4 == 0 else 9
                d.line([(x - ln, y), (x, y)], fill=ink, width=3)
            d.line([(x, sim.TOP_Y + 10), (x, scene["floor_y"])], fill=ink, width=3)
        elif ov == "lanes":
            n = len(scene["objects"])
            lane = (scene["wall_r"] - scene["wall_l"]) / n
            for k in range(1, n):
                x = scene["wall_l"] + lane * k
                for y in range(sim.TOP_Y + 10, int(scene["floor_y"]) - 4, 22):
                    d.line([(x, y), (x, y + 10)], fill=faint, width=3)
        if scene["scenario"] == "staggered":
            d.line([(scene["wall_l"], 96), (scene["wall_r"], 96)], fill=ink, width=6)
    return img


def _floor(d, scene, fl, ink):
    st = scene["style"]
    fy = scene["floor_y"]
    d.rectangle([0, fy, W, H], fill=fl)
    dark = _mix(fl, (0, 0, 0), 0.25)
    if st["floor"] == "hatched":
        for x in range(-H, W, 22):
            d.line([(x, H), (x + (H - fy), fy)], fill=dark, width=3)
    elif st["floor"] == "tiles":
        for k, x in enumerate(range(0, W, 64)):
            if k % 2:
                d.rectangle([x, fy + 6, x + 63, H], fill=dark)
    elif st["floor"] == "stripe":
        d.rectangle([0, fy + 10, W, fy + 18], fill=dark)
    for p in scene.get("pads", []):
        rgb = tuple(p["rgb"])
        d.rectangle([p["x0"], fy, p["x1"], fy + 22], fill=rgb)
        acc = _accent(rgb)   # neutral, so pad markings never pass for another colour
        if p["pattern"] == "hatched":
            for x in range(int(p["x0"]), int(p["x1"]) - 6, 14):
                d.line([(x, fy + 22), (x + 10, fy + 4)], fill=acc, width=3)
        elif p["pattern"] == "dotted":
            for x in range(int(p["x0"]) + 10, int(p["x1"]) - 4, 18):
                d.ellipse([x - 3, fy + 8, x + 3, fy + 14], fill=acc)
    d.line([(0, fy), (W, fy)], fill=ink, width=4)


def _shelves(d, scene, fl, ink):
    st = scene["style"]
    wl, wr = scene["wall_l"], scene["wall_r"]
    last = scene["shelves"][-1]["floor_y"]
    d.rectangle([0, last, W, H], fill=fl)
    for sh in scene["shelves"]:
        fy = sh["floor_y"]
        d.rectangle([wl, fy, wr, fy + 9], fill=fl)
        if st["floor"] in ("hatched", "tiles"):
            for x in range(int(wl), int(wr) - 8, 18):
                d.line([(x, fy + 9), (x + 8, fy + 1)], fill=_mix(fl, (0, 0, 0), 0.3), width=2)
        d.line([(wl, fy), (wr, fy)], fill=ink, width=4)
    top = scene["shelves"][0]["top_y"]
    d.rectangle([wl - 12, top, wl - 1, last + 9], fill=ink)
    d.rectangle([wr + 1, top, wr + 12, last + 9], fill=ink)


def _accent(rgb):
    lum = 0.299 * rgb[0] + 0.587 * rgb[1] + 0.114 * rgb[2]
    return (255, 255, 255) if lum < 150 else (34, 34, 40)


def _shape_poly(shape, cx, cy, r):
    if shape == "square":
        return [(cx - r, cy - r), (cx + r, cy - r), (cx + r, cy + r), (cx - r, cy + r)]
    if shape == "diamond":
        return [(cx, cy - r), (cx + r, cy), (cx, cy + r), (cx - r, cy)]
    if shape == "hexagon":
        return [(cx + r * math.cos(math.radians(90 + 60 * k)), cy - r * math.sin(math.radians(90 + 60 * k)))
                for k in range(6)]
    return None


@functools.lru_cache(maxsize=64)
def sprite(shape: str, r: float, rgb: tuple, pattern: str, outline: int, dark_bg: bool) -> Image.Image:
    """RGBA sprite of one object, centred; size 2*ceil(r)+4."""
    half = int(math.ceil(r)) + 2
    size = 2 * half * SS
    c, R = size / 2, r * SS
    mask = Image.new("L", (size, size), 0)
    md = ImageDraw.Draw(mask)
    poly = _shape_poly(shape, c, c, R)
    if poly is None:
        md.ellipse([c - R, c - R, c + R, c + R], fill=255)
    else:
        md.polygon(poly, fill=255)
    body = Image.new("RGB", (size, size), rgb)
    bd = ImageDraw.Draw(body)
    acc = _accent(rgb)
    if pattern == "striped":
        band = R * 2 / 7
        for k in (1, 3, 5):
            y0 = c - R + band * k
            bd.rectangle([0, y0, size, y0 + band], fill=acc)
    elif pattern == "dotted":
        q = R * 0.34
        bd.ellipse([c - q, c - q, c + q, c + q], fill=acc)
    elif pattern == "ringed":
        q1, q2 = R * 0.62, R * 0.36
        bd.ellipse([c - q1, c - q1, c + q1, c + q1], fill=acc)
        bd.ellipse([c - q2, c - q2, c + q2, c + q2], fill=rgb)
    elif pattern == "two-tone":
        bd.rectangle([0, c, size, size], fill=acc)   # accent, never a shade that could pass for another colour
    elif pattern == "crossed":
        w = R * 0.2
        bd.rectangle([c - w, 0, c + w, size], fill=acc)
        bd.rectangle([0, c - w, size, c + w], fill=acc)
    if outline:
        edge = (240, 240, 240) if dark_bg else (30, 30, 34)
        ring = Image.new("L", (size, size), 0)
        rd = ImageDraw.Draw(ring)
        if poly is None:
            rd.ellipse([c - R, c - R, c + R, c + R], outline=255, width=outline * SS)
        else:
            rd.polygon(poly, outline=255, width=outline * SS)
        body.paste(Image.new("RGB", (size, size), edge), (0, 0), ring)
    out = body.convert("RGBA")
    out.putalpha(mask)
    return out.resize((2 * half, 2 * half), Image.LANCZOS)


def object_centers(scene: dict, t: float) -> list[tuple[float, float]]:
    """Where each object's centre is drawn at time t (px)."""
    return [sim.center_at(scene, o, t) for o in scene["objects"]]


def _clamp(d, scene, o, t, ink):
    x = o["x"]
    ytop = sim.floor_y(scene, o) - o["h0"] - 2 * o["r"]
    d.line([(x, 96), (x, ytop - 2)], fill=ink, width=4)
    opened = t >= o["release"]
    hw = sim.half_w(o)
    for s in (-1, 1):
        if opened:
            d.line([(x, ytop - 2), (x + s * (hw + 12), ytop - 14)], fill=ink, width=4)
        else:
            d.line([(x, ytop - 2), (x + s * (hw + 4), ytop - 2)], fill=ink, width=4)
            d.line([(x + s * (hw + 4), ytop - 2), (x + s * (hw + 4), ytop + o["r"] * 0.8)], fill=ink, width=4)


def draw_timer(d, i: int):
    k = sim.timer_tenths(i)
    d.rounded_rectangle(TIMER_BOX, radius=10, fill=(26, 28, 34), outline=(200, 200, 205), width=2)
    d.text(((TIMER_BOX[0] + TIMER_BOX[2]) / 2, (TIMER_BOX[1] + TIMER_BOX[3]) / 2), f"{k // 10}.{k % 10} s",
           fill=(255, 255, 255), font=font(TIMER_FONT), anchor="mm")


def frame(scene, t: float) -> Image.Image:
    key = json.dumps(scene, sort_keys=True)
    img = _background(key).copy()
    d = ImageDraw.Draw(img)
    st = scene["style"]
    ink = _ink(st)
    if scene["scenario"] == "staggered":
        for o in scene["objects"]:
            _clamp(d, scene, o, t, ink)
    for o, (cx, cy) in zip(scene["objects"], object_centers(scene, t)):
        sp = sprite(o["shape"], o["r"], tuple(o["rgb"]), o["pattern"], st["outline"], st["dark"])
        img.paste(sp, (int(round(cx - sp.width / 2)), int(round(cy - sp.height / 2))), sp)
    draw_timer(d, int(round(t * sim.FPS)))
    return img
