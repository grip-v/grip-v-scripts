"""PIL renderer for overlapping_shapes (1280x720, 24 fps).

Every frame is the board (theme colour, optional unnumbered grid lines,
dot grid or paper grain) with the shapes composited in a fixed z-order:
arrival order for arrive / paper, index order for depart and the static
glide shapes, movers on top. Each shape tile holds a fill (translucent, or
opaque for paper), an optional fill pattern and a border (solid, dashed,
dotted or double). The only text is the clock in the top-right box, which
the board never reaches."""
from __future__ import annotations
import functools
import json
import math
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageChops, ImageDraw
from shapely.geometry import Polygon

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from common.video5 import font, hex_rgb  # noqa: E402

import sim  # noqa: E402

TIMER_FONT = 40


def duration(scene) -> float:
    return scene["n_frames"] / sim.FPS


# ------------------------------------------------------------- borders
def _dash(d, pts, on, off, width, col):
    ring = list(pts) + [pts[0]]
    cum = [0.0]
    for (x0, y0), (x1, y1) in zip(ring, ring[1:]):
        cum.append(cum[-1] + math.hypot(x1 - x0, y1 - y0))
    total = cum[-1]

    def at(u):
        k = next((i for i in range(len(cum) - 1) if cum[i + 1] >= u), len(cum) - 2)
        seg = cum[k + 1] - cum[k]
        f = 0.0 if seg == 0 else (u - cum[k]) / seg
        (x0, y0), (x1, y1) = ring[k], ring[k + 1]
        return (x0 + (x1 - x0) * f, y0 + (y1 - y0) * f), k

    start = 0.0
    while start < total:
        end = min(total, start + on)
        p0, k0 = at(start)
        p1, k1 = at(end)
        d.line([p0] + ring[k0 + 1:k1 + 1] + [p1], fill=col, width=width, joint="curve")
        start += on + off


def _dots(d, pts, gap, r, col):
    ring = list(pts) + [pts[0]]
    rem = 0.0
    for (x0, y0), (x1, y1) in zip(ring, ring[1:]):
        seg = math.hypot(x1 - x0, y1 - y0)
        pos = rem
        while pos <= seg and seg > 0:
            x, y = x0 + (x1 - x0) * pos / seg, y0 + (y1 - y0) * pos / seg
            d.ellipse([x - r, y - r, x + r, y + r], fill=col)
            pos += gap
        rem = pos - seg


def _pattern(size, kind, col):
    """RGBA tile of the fill pattern (lines and dots 3 px or more)."""
    w, h = size
    im = Image.new("RGBA", size, (0, 0, 0, 0))
    d = ImageDraw.Draw(im)
    if kind in ("stripes", "hatch"):
        for c in range(-h, w + h, 16):
            d.line([(c, 0), (c + h, h)], fill=col, width=3)
        if kind == "hatch":
            for c in range(-h, w + h, 16):
                d.line([(c, h), (c + h, 0)], fill=col, width=3)
    elif kind == "dots":
        for y in range(6, h, 16):
            for x in range(6 + (8 if (y // 16) % 2 else 0), w, 16):
                d.ellipse([x - 3, y - 3, x + 3, y + 3], fill=col)
    return im


def shape_tile(scene, sh: dict, x: float, y: float, rot: float, scale: float = 1.0, alpha: float = 1.0):
    """(RGBA tile, (left, top)) of one shape at a pose."""
    pts = sim.outline_at(sh, x, y, rot, scale)
    xs, ys = [p[0] for p in pts], [p[1] for p in pts]
    ox, oy = int(math.floor(min(xs))) - 8, int(math.floor(min(ys))) - 8
    size = (int(math.ceil(max(xs))) - ox + 9, int(math.ceil(max(ys))) - oy + 9)
    local = [(px - ox, py - oy) for px, py in pts]
    r, g, b = hex_rgb(sim.COLORS[sh["color"]])
    paper = scene["scenario"] == "paper"
    dark = scene["theme"]["dark"]
    if paper:
        fill = (int(r * 0.7 + 255 * 0.3), int(g * 0.7 + 255 * 0.3), int(b * 0.7 + 255 * 0.3), 255)
    else:
        fill = (r, g, b, scene["alpha"] + (20 if dark else 0))
    line = (r, g, b, 255) if dark else (int(r * 0.78), int(g * 0.78), int(b * 0.78), 255)
    mask = Image.new("L", size, 0)
    ImageDraw.Draw(mask).polygon(local, fill=255)
    body = Image.new("RGBA", size, fill)
    if sh["fill"] != "plain":
        pc = (int(r * 0.6), int(g * 0.6), int(b * 0.6), 255 if paper else 170)
        body.alpha_composite(_pattern(size, sh["fill"], pc))
    tile = Image.new("RGBA", size, (0, 0, 0, 0))
    tile.paste(body, (0, 0), mask)
    d = ImageDraw.Draw(tile)
    lw = scene["line_w"]
    style = sh["style"]
    if style == "solid":
        d.line(local + [local[0]], fill=line, width=lw, joint="curve")
    elif style == "dashed":
        _dash(d, local, 15, 9, lw, line)
    elif style == "dotted":
        _dots(d, local, 3 * lw + 3, lw * 0.75, line)
    else:
        d.line(local + [local[0]], fill=line, width=3, joint="curve")
        inner = Polygon(local).buffer(-7)
        if not inner.is_empty and inner.geom_type == "Polygon":
            d.line(list(inner.exterior.coords), fill=line, width=3, joint="curve")
    if alpha < 1.0:
        a = tile.getchannel("A").point(lambda v: int(v * alpha))
        tile.putalpha(a)
    return tile, (ox, oy)


# ---------------------------------------------------------------- board
@functools.lru_cache(maxsize=32)
def _board(key: str) -> Image.Image:
    sc = json.loads(key)
    th = sc["theme"]
    img = Image.new("RGBA", (sim.W, sim.H), tuple(th["bg"]) + (255,))
    d = ImageDraw.Draw(img)
    x0, y0, x1, y1 = sc["board"]
    edge = (90, 96, 110, 255) if th["dark"] else (200, 196, 186, 255)
    d.rounded_rectangle([x0, y0, x1, y1], radius=14, fill=tuple(th["board"]) + (255,), outline=edge, width=2)
    if th["grid"] == "lines":
        gc = (60, 70, 92, 255) if th["dark"] else (222, 226, 234, 255)
        g = th["grid_px"]
        for x in range(x0 + g, x1 - 4, g):
            d.line([(x, y0 + 3), (x, y1 - 3)], fill=gc, width=1)
        for y in range(y0 + g, y1 - 4, g):
            d.line([(x0 + 3, y), (x1 - 3, y)], fill=gc, width=1)
    elif th["grid"] == "dots":
        g = th["grid_px"]
        for y in range(y0 + g, y1 - 4, g):
            for x in range(x0 + g, x1 - 4, g):
                d.ellipse([x - 1.5, y - 1.5, x + 1.5, y + 1.5], fill=(200, 196, 186, 255))
    elif th["grid"] == "noise":
        rs = np.random.default_rng(th["seed"])
        h, w = y1 - y0 - 6, x1 - x0 - 6
        n = rs.integers(-7, 8, size=(h, w, 1))
        base = np.array(th["board"], dtype=np.int16)[None, None, :]
        arr = np.clip(base + n, 0, 255).astype(np.uint8)
        img.paste(Image.fromarray(arr, "RGB"), (x0 + 3, y0 + 3))
    return img


def draw_timer(img, f: int):
    d = ImageDraw.Draw(img)
    x0, y0, x1, y1 = sim.TIMER_BOX
    d.rounded_rectangle([x0, y0, x1, y1], radius=10, fill=(12, 12, 16), outline=(120, 120, 132), width=2)
    txt = sim.timer_text(f)
    fo = font(TIMER_FONT)
    tw = d.textlength(txt, font=fo)
    d.text((x1 - 16 - tw, y0 + 9), txt, fill=(255, 255, 255), font=fo)


# --------------------------------------------------------------- shapes
@functools.lru_cache(maxsize=256)
def _tile(key: str, i: int):
    sc = json.loads(key)
    sh = sc["shapes"][i]
    return shape_tile(sc, sh, sh["x"], sh["y"], sh["rot"])


def _ease(p):
    return 1 - (1 - p) ** 3


def _anim_tile(sc, sh, p, kind):
    """Tile of a shape that is p of the way into the scene (p=1 fully drawn)."""
    e = _ease(p)
    x, y, rot = sh["x"], sh["y"], sh["rot"]
    if kind in ("grow", "shrink"):
        return shape_tile(sc, sh, x, y, rot, 0.45 + 0.55 * e, e)
    if kind == "spin":
        return shape_tile(sc, sh, x, y, rot - 75 * (1 - e), 0.7 + 0.3 * e, e)
    if kind in ("drop", "lift"):
        return shape_tile(sc, sh, x, y - 40 * (1 - e), rot, 1.0, e)
    return shape_tile(sc, sh, x, y, rot, 1.0, e)


def progress(scene, i: int, f: int) -> float:
    """How far shape i is into the board at frame f: 0 absent, 1 fully drawn."""
    sc = scene["scenario"]
    if sc == "glide":
        return 1.0
    ev = {e["shape"]: e["f"] for e in scene["events"]}
    A = scene["anim_f"]
    if sc == "depart":
        if i not in ev or f < ev[i]:
            return 1.0
        return max(0.0, 1.0 - (f - ev[i] + 1) / A)
    if i not in ev or f < ev[i]:
        return 0.0
    return min(1.0, (f - ev[i] + 1) / A)


def z_order(scene) -> list[int]:
    if scene["scenario"] in ("arrive", "paper"):
        return [e["shape"] for e in scene["events"]]
    return sim.statics(scene)


def frame_at(scene, f: int) -> Image.Image:
    key = json.dumps(scene, sort_keys=True)
    img = _board(json.dumps({"theme": scene["theme"], "board": scene["board"]}, sort_keys=True)).copy()
    for i in z_order(scene):
        p = progress(scene, i, f)
        if p <= 0:
            continue
        if p >= 1:
            tile, off = _tile(key, i)
        else:
            tile, off = _anim_tile(scene, scene["shapes"][i], p, scene["anim"])
        img.alpha_composite(tile, dest=off)
    for m in sim.movers(scene):
        sh = scene["shapes"][m]
        x, y, rot = sim.pose(sh, f / sim.FPS)
        tile, off = shape_tile(scene, sh, x, y, rot)
        img.alpha_composite(tile, dest=off)
    img = img.convert("RGB")
    draw_timer(img, f)
    return img


def frame(scene, t: float) -> Image.Image:
    return frame_at(scene, int(round(t * sim.FPS)))
