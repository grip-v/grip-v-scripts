"""PIL renderer for shape_collision (1280x720, 24 fps).

Frame f shows the world at t = f / 24. A walled box (random size, place,
floor colour and faint floor texture) on a plain or gradient background;
patterned shapes, each inside a thin ring (solid or dashed) that marks its
collision disc; grey posts (posts), dark holes (holes) or thin rails
(rails). No trails, glow or impact flash: the only collision cue is the
contact itself and the pattern change. A shape that falls into a hole is
gone from the frame its centre passes over the hole.

The only text is the clock in a fixed panel at the top-left (seconds to
0.1 s); the box always starts below that panel, so nothing passes under it.

`track(scene)` is the per-frame list of what gets drawn (each shape's centre
and pattern, or None once it has fallen in); `frame_at` draws exactly that.
answers.py and frame_check.py read the track, never the event list.
"""
from __future__ import annotations
import functools
import json
import math
import sys
from pathlib import Path

from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from common.video5 import font  # noqa: E402
from common.shapes import unit_shape  # noqa: E402

import sim  # noqa: E402

SS = 3
TIMER_BG = (16, 18, 24)
TIMER_EDGE = (120, 126, 140)
TIMER_FG = (245, 246, 250)
HOLE = (8, 8, 10)


def duration(scene) -> float:
    return scene["n_frames"] / sim.FPS


def _mix(c, w, f):
    return tuple(int(round(a + (b - a) * f)) for a, b in zip(c, w))


@functools.lru_cache(maxsize=16)
def _track(key: str):
    scene = json.loads(key)
    out = []
    for f in range(scene["n_frames"]):
        t = f / sim.FPS
        pos = sim.positions(scene, t)
        alive = sim.alive_at(scene, t)
        pat = sim.patterns_at(scene, t)
        out.append([(pos[i][0], pos[i][1], pat[i]) if alive[i] else None for i in range(len(pos))])
    return out


def track(scene) -> list:
    """Per frame, per shape: (x, y, pattern) as drawn, or None if gone."""
    return _track(json.dumps(scene, sort_keys=True))


# ---------------------------------------------------------------- sprites

def _pattern(d: ImageDraw.ImageDraw, pat: str, size: int, col, sec, r: int):
    """Fill a size x size square (supersampled) with the pattern."""
    u = SS
    per = max(12, round(r * 0.42)) * u          # stripe / dot period scales with the shape
    w = max(5 * u, per * 0.45)
    c = size / 2
    if pat == "solid":
        d.rectangle([0, 0, size, size], fill=col)
        return
    if pat == "hollow":
        return
    d.rectangle([0, 0, size, size], fill=sec)
    if pat == "horizontal stripes":
        for y in range(int(c - per * 20) % per - per, size, per):
            d.rectangle([0, y, size, y + w], fill=col)
    elif pat == "vertical stripes":
        for x in range(int(c - per * 20) % per - per, size, per):
            d.rectangle([x, 0, x + w, size], fill=col)
    elif pat == "diagonal stripes":
        step = per * 1.0
        k = -size
        while k < 2 * size:
            d.polygon([(k, 0), (k + w * 1.4, 0), (k + w * 1.4 - size, size), (k - size, size)], fill=col)
            k += step * 1.4
    elif pat == "dots":
        rr = per * 0.28
        for j, y in enumerate(range(-per, size + per, per)):
            for x in range(-per + (per // 2 if j % 2 else 0), size + per, per):
                d.ellipse([x - rr, y - rr, x + rr, y + rr], fill=col)
    elif pat == "checks":
        q = per
        for i in range(size // q + 2):
            for j in range(size // q + 2):
                if (i + j) % 2 == 0:
                    d.rectangle([i * q, j * q, (i + 1) * q, (j + 1) * q], fill=col)
    elif pat == "grid":
        lw = max(3 * u, per * 0.22)
        for v in range(0, size + per, per):
            d.rectangle([v, 0, v + lw, size], fill=col)
            d.rectangle([0, v, size, v + lw], fill=col)
    elif pat == "rings":
        for rr in range(size, 0, -per):
            d.ellipse([c - rr, c - rr, c + rr, c + rr], fill=col)
            ri = rr - per * 0.5
            if ri > 0:
                d.ellipse([c - ri, c - ri, c + ri, c + ri], fill=sec)
    elif pat == "two-tone":
        d.rectangle([0, 0, size, c], fill=col)


@functools.lru_cache(maxsize=1024)
def sprite(shape: str, color: str, pat: str, r: int, ring: tuple, ring_style: str, edge_w: int) -> Image.Image:
    size = 2 * (r + 4) * SS
    c = size / 2
    col = sim.COLORS[color]
    sec = sim.second_colour(col)
    edge = _mix(col, (0, 0, 0), 0.4)
    poly = [(c + 0.93 * r * SS * x, c + 0.93 * r * SS * y) for x, y in unit_shape(shape, 1.0)]
    tex = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    _pattern(ImageDraw.Draw(tex), pat, size, col + (255,), sec + (255,), r)
    mask = Image.new("L", (size, size), 0)
    ImageDraw.Draw(mask).polygon(poly, fill=255)
    out = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    out.paste(tex, (0, 0), mask)
    d = ImageDraw.Draw(out)
    rr = r * SS - 1.5 * SS
    if ring_style == "solid":
        d.ellipse([c - rr, c - rr, c + rr, c + rr], outline=ring + (255,), width=3 * SS)
    else:
        n = max(12, int(2 * math.pi * r / 14))
        for k in range(n):
            a0, a1 = 360 * k / n, 360 * (k + 0.55) / n
            d.arc([c - rr, c - rr, c + rr, c + rr], a0, a1, fill=ring + (255,), width=3 * SS)
    if pat == "hollow":
        d.polygon(poly, outline=col + (255,), width=(edge_w + 3) * SS)
    else:
        d.polygon(poly, outline=edge + (255,), width=edge_w * SS)
    return out.resize((size // SS, size // SS), Image.LANCZOS)


# ---------------------------------------------------------------- static layer

@functools.lru_cache(maxsize=8)
def _static(key: str) -> Image.Image:
    scene = json.loads(key)
    st = scene["style"]
    img = Image.new("RGB", (sim.W, sim.H))
    d = ImageDraw.Draw(img)
    top, bot = st["bg"]
    for y in range(sim.H):
        d.line([(0, y), (sim.W, y)], fill=_mix(top, bot, y / (sim.H - 1)))
    L, T, R, B = scene["box"]
    ww = st["wall_w"]
    d.rectangle([L - ww, T - ww, R + ww - 1, B + ww - 1], fill=tuple(st["wall"]))
    floor = tuple(st["floor"])
    d.rectangle([L, T, R - 1, B - 1], fill=floor)
    alt = tuple(max(0, min(255, c + st["tex_shade"])) for c in floor)
    s = st["tex_step"]
    tex = st["floor_tex"]
    if tex == "grid":
        for x in range(L + s, R, s):
            d.line([(x, T), (x, B - 1)], fill=alt, width=2)
        for y in range(T + s, B, s):
            d.line([(L, y), (R - 1, y)], fill=alt, width=2)
    elif tex == "dots":
        for x in range(L + s // 2, R, s):
            for y in range(T + s // 2, B, s):
                d.ellipse([x - 3, y - 3, x + 3, y + 3], fill=alt)
    elif tex == "tiles":
        for i, x in enumerate(range(L, R, s)):
            for j, y in enumerate(range(T, B, s)):
                if (i + j) % 2:
                    d.rectangle([x, y, min(x + s, R) - 1, min(y + s, B) - 1], fill=alt)
    elif tex == "planks":
        for j, y in enumerate(range(T, B, s // 2)):
            if j % 2:
                d.rectangle([L, y, R - 1, min(y + s // 2, B) - 1], fill=alt)
    for y in scene["lanes"]:
        d.rectangle([L, round(y) - 2, R - 1, round(y) + 2], fill=tuple(st["rail"]))
    for h in scene["holes"]:
        x, y, r = h["x"], h["y"], h["r"]
        d.ellipse([x - r - 4, y - r - 4, x + r + 4, y + r + 4], fill=tuple(st["hole_rim"]))
        d.ellipse([x - r, y - r, x + r, y + r], fill=HOLE)
    for p in scene["posts"]:
        x, y, r = p["x"], p["y"], p["r"]
        pc = tuple(st["post"])
        d.ellipse([x - r, y - r, x + r, y + r], fill=_mix(pc, (0, 0, 0), 0.35))
        d.ellipse([x - r + 4, y - r + 4, x + r - 4, y + r - 4], fill=pc)
        d.ellipse([x - r * 0.45, y - r * 0.55, x - r * 0.05, y - r * 0.15], fill=_mix(pc, (255, 255, 255), 0.35))
    tb = scene["timer"]["box"]
    d.rounded_rectangle(tb, radius=10, fill=TIMER_BG, outline=TIMER_EDGE, width=2)
    return img


def _static_key(scene) -> str:
    return json.dumps({k: scene[k] for k in ("style", "box", "lanes", "holes", "posts", "timer")}, sort_keys=True)


def timer_text(f: int) -> str:
    return sim.clock_text(sim.tenths(f))


def frame_at(scene, f: int) -> Image.Image:
    img = _static(_static_key(scene)).copy()
    d = ImageDraw.Draw(img)
    tb = scene["timer"]["box"]
    d.text(((tb[0] + tb[2]) / 2, (tb[1] + tb[3]) / 2), timer_text(f), fill=TIMER_FG,
           font=font(scene["timer"]["size"]), anchor="mm")
    st = scene["style"]
    ring = tuple(st["ring"])
    for b, item in zip(scene["bodies"], track(scene)[f]):
        if item is None:
            continue
        x, y, pat = item
        sp = sprite(b["shape"], b["color"], pat, b["r"], ring, st["ring_style"], st["edge_w"])
        img.paste(sp, (round(x - sp.width / 2), round(y - sp.height / 2)), sp)
    return img


def frame(scene, t: float) -> Image.Image:
    return frame_at(scene, min(scene["n_frames"] - 1, int(round(t * sim.FPS))))
