"""PIL renderer for simple_harmonic_motion (1280x720, 24 fps).

Every oscillator is a coloured mass with its own unnumbered tick ruler; the
taller, darker tick is its rest mark. Springs (zigzag / coil / wave) hang
from a ceiling or run from a wall; pendulum strings hang from a bar; damped
springs carry a dashpot beside the spring; a held mass rests against a
small stop post that vanishes at its release.

The ONLY text in the frame is the clock in the top-left box (seconds to
0.1 s, truncated from the frame number). All action stays right of
x = 250 px. Pure function of (scene, frame).
"""
from __future__ import annotations
import functools
import json
import math
import sys
from pathlib import Path

from PIL import Image, ImageDraw, ImageFilter

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from common.video5 import font  # noqa: E402
import sim  # noqa: E402

FPS = sim.FPS
SIZE = (1280, 720)
CLOCK_BOX = sim.CLOCK_BOX


def duration(scene) -> float:
    return sim.duration(scene)


def n_frames(scene) -> int:
    return sim.n_frames(scene)


def clock_text(frame: int) -> str:
    return f"{sim.clock_tenths(frame) / 10:.1f} s"


def state(scene, f: int) -> list[dict]:
    """What frame f draws, per object: the mass centre (px), its rest-mark position, the
    anchor of its spring / string, and whether it is still held by a stop post."""
    t = f / FPS
    out = []
    for o in scene["objects"]:
        u = sim.u_at(o, t)
        rx, ry = o["rest"]
        ax, ay = o["anchor"]
        if scene["orient"] == "horizontal":
            c = (rx + u, ry)
        elif scene["orient"] == "vertical":
            c = (rx, ry + u)
        else:
            th = math.asin(u / o["L"])
            c = (ax + u, ay + o["L"] * math.cos(th))
        out.append({"colour": o["colour"], "centre": c, "rest": (rx, ry), "anchor": (ax, ay),
                    "held": o["mode"] == "held" and t <= o["t_rel"]})
    return out


# ------------------------------------------------------------------ static layer
def _shade(c, d):
    return tuple(max(0, min(255, v + d)) for v in c)


@functools.lru_cache(maxsize=8)
def _static(key: str) -> Image.Image:
    scene = json.loads(key)
    st = scene["style"]
    bg = tuple(st["bg"])
    img = Image.new("RGB", SIZE, bg)
    d = ImageDraw.Draw(img)
    dec = st["decor"] * (1 if st["theme"] == "dark" else -1)
    if st["bg_pattern"] == "gradient":
        for y in range(SIZE[1]):
            d.line([(0, y), (SIZE[0], y)], fill=_shade(bg, int(dec * y / SIZE[1])))
    elif st["bg_pattern"] == "grid":
        for x in range(0, SIZE[0], st["bg_step"]):
            d.line([(x, 0), (x, SIZE[1])], fill=_shade(bg, dec), width=1)
        for y in range(0, SIZE[1], st["bg_step"]):
            d.line([(0, y), (SIZE[0], y)], fill=_shade(bg, dec), width=1)
    elif st["bg_pattern"] == "dots":
        for x in range(st["bg_step"] // 2, SIZE[0], st["bg_step"]):
            for y in range(st["bg_step"] // 2, SIZE[1], st["bg_step"]):
                d.ellipse([x - 2, y - 2, x + 2, y + 2], fill=_shade(bg, dec))
    fx, fc = scene["fixture"], tuple(st["fixture"])
    if fx["kind"] == "wall":
        box = [fx["x"] - 16, fx["y0"] - 10, fx["x"], fx["y1"] + 10]
    elif fx["kind"] == "ceiling":
        box = [fx["x0"] - 10, fx["y"] - 16, fx["x1"] + 10, fx["y"]]
    else:
        box = [fx["x0"], fx["y"] - 12, fx["x1"], fx["y"]]
    d.rectangle(box, fill=fc)
    if st["fixture_style"] == "hatched":
        x0, y0, x1, y1 = box
        k = x0 - (y1 - y0)
        while k < x1:
            d.line([(max(k, x0), y1 - max(0, x0 - k)), (min(k + (y1 - y0), x1), y0 + max(0, k + (y1 - y0) - x1))],
                   fill=_shade(fc, -40), width=2)
            k += 12
    ink, rest_col, tl = tuple(st["ink"]), tuple(st["rest_col"]), st["tick_len"]
    tick = scene["tick"]
    for o in scene["objects"]:
        rx, ry = o["rest"]
        r = o["ruler"]
        n = o["n_ticks"]
        if scene["orient"] == "vertical":
            d.line([(r, ry - n * tick), (r, ry + n * tick)], fill=ink, width=3)
            for k in range(-n, n + 1):
                if k:
                    d.line([(r, ry + k * tick), (r + tl, ry + k * tick)], fill=ink, width=3)
            d.line([(r - 4, ry), (r + int(2.2 * tl), ry)], fill=rest_col, width=5)
        else:
            d.line([(rx - n * tick, r), (rx + n * tick, r)], fill=ink, width=3)
            for k in range(-n, n + 1):
                if k:
                    d.line([(rx + k * tick, r), (rx + k * tick, r + tl)], fill=ink, width=3)
            d.line([(rx, r - 4), (rx, r + int(2.2 * tl))], fill=rest_col, width=5)
        if scene["orient"] == "pendulum":
            ax, ay = o["anchor"]
            d.ellipse([ax - 6, ay - 6, ax + 6, ay + 6], fill=_shade(fc, -50))
    return img


# ------------------------------------------------------------------ sprites
def _poly(shape, s):
    h = s / 2
    if shape == "square":
        return [(0, 0), (s - 1, 0), (s - 1, s - 1), (0, s - 1)]
    if shape == "diamond":
        return [(h, 0), (s - 1, h), (h, s - 1), (0, h)]
    if shape == "triangle":
        return [(h, 0), (s - 1, s - 1), (0, s - 1)]
    if shape == "hexagon":
        return [(h + h * math.cos(math.pi / 3 * k), h + h * math.sin(math.pi / 3 * k)) for k in range(6)]
    return None


@functools.lru_cache(maxsize=64)
def sprite(colour: str, shape: str, pattern: str, s: int) -> Image.Image:
    """RGBA sprite, s x s. The outer band of the shape is always the plain fill colour."""
    fill = sim.COLOURS[colour]
    pat = (30, 30, 34) if colour == "yellow" else (255, 255, 255)
    mask = Image.new("L", (s, s), 0)
    md = ImageDraw.Draw(mask)
    if shape == "circle":
        md.ellipse([0, 0, s - 1, s - 1], fill=255)
    else:
        md.polygon(_poly(shape, s), fill=255)
    pad = Image.new("L", (s + 16, s + 16), 0)           # pad first: a shape that fills the sprite must erode too
    pad.paste(mask, (8, 8))
    for _ in range(6):                                  # erode 6 px: patterns stay inside a plain fill band
        pad = pad.filter(ImageFilter.MinFilter(3))
    inner = pad.crop((8, 8, s + 8, s + 8))
    img = Image.new("RGBA", (s, s), fill + (255,))
    pimg = Image.new("RGBA", (s, s), (0, 0, 0, 0))
    pd = ImageDraw.Draw(pimg)
    if pattern == "striped":
        for k in range(-s, 2 * s, 9):
            pd.line([(k, 0), (k + s, s)], fill=pat + (255,), width=3)
    elif pattern == "dotted":
        for x in range(4, s, 9):
            for y in range(4, s, 9):
                pd.ellipse([x - 2, y - 2, x + 2, y + 2], fill=pat + (255,))
    elif pattern == "ringed":
        c = s / 2
        for r in (s * 0.32, s * 0.16):
            pd.ellipse([c - r, c - r, c + r, c + r], outline=pat + (255,), width=3)
    img.paste(pimg, (0, 0), Image.composite(pimg.getchannel("A"), Image.new("L", (s, s), 0), inner))
    img.putalpha(mask)
    return img


# ------------------------------------------------------------------ dynamic parts
def _spring(d, a, b, sp, off=(0.0, 0.0), width=None):
    (x0, y0), (x1, y1) = a, b
    x0, y0, x1, y1 = x0 + off[0], y0 + off[1], x1 + off[0], y1 + off[1]
    L = math.hypot(x1 - x0, y1 - y0)
    if L < 1:
        return
    ux, uy = (x1 - x0) / L, (y1 - y0) / L
    nx, ny = -uy, ux
    w = (width or sp["width"]) / 2
    lead = min(14.0, L * 0.12)
    body = L - 2 * lead
    pts = [(x0, y0), (x0 + ux * lead, y0 + uy * lead)]
    n = sp["coils"]
    steps = n * 12
    for i in range(1, steps):
        f = i / steps
        if sp["style"] == "zigzag":
            ph = (f * n * 2) % 2
            o = w * (ph if ph < 1 else 2 - ph) * 2 - w
            if i % 6:
                continue
        elif sp["style"] == "wave":
            o = w * math.sin(2 * math.pi * n * f)
        else:  # coil: a prolate cycloid looks like a side view of a helix
            ang = 2 * math.pi * n * f
            o = w * math.sin(ang)
            f = f + 0.6 * w * math.cos(ang) / max(body, 1) * 0.5
        s = lead + body * f
        pts.append((x0 + ux * s + nx * o, y0 + uy * s + ny * o))
    pts += [(x1 - ux * lead, y1 - uy * lead), (x1, y1)]
    d.line(pts, fill=tuple(sp["col"]), width=sp["lw"], joint="curve")


def _dashpot(d, a, b, col, off):
    (x0, y0), (x1, y1) = a, b
    x0, y0, x1, y1 = x0 + off[0], y0 + off[1], x1 + off[0], y1 + off[1]
    L = math.hypot(x1 - x0, y1 - y0)
    ux, uy = (x1 - x0) / L, (y1 - y0) / L
    nx, ny = -uy, ux
    cyl = 60.0
    c0, c1 = (x0 + ux * 6, y0 + uy * 6), (x0 + ux * (6 + cyl), y0 + uy * (6 + cyl))
    hw = 7
    d.line([a if off == (0, 0) else (x0, y0), c0], fill=col, width=4)
    poly = [(c0[0] + nx * hw, c0[1] + ny * hw), (c1[0] + nx * hw, c1[1] + ny * hw),
            (c1[0] - nx * hw, c1[1] - ny * hw), (c0[0] - nx * hw, c0[1] - ny * hw)]
    d.polygon(poly, outline=col, width=3)
    pist = (x0 + ux * (6 + cyl * 0.55), y0 + uy * (6 + cyl * 0.55))
    d.line([(pist[0] + nx * (hw - 3), pist[1] + ny * (hw - 3)), (pist[0] - nx * (hw - 3), pist[1] - ny * (hw - 3))],
           fill=col, width=4)
    d.line([pist, (x1, y1)], fill=col, width=4)


def _clock(d, frame):
    x0, y0, x1, y1 = CLOCK_BOX
    d.rounded_rectangle(CLOCK_BOX, radius=12, fill=(18, 18, 22), outline=(235, 235, 235), width=3)
    d.text(((x0 + x1) / 2, (y0 + y1) / 2), clock_text(frame), fill=(255, 255, 255), font=font(44), anchor="mm")


def frame_at(scene, f: int) -> Image.Image:
    key = json.dumps({k: v for k, v in scene.items()}, sort_keys=True)
    img = _static(key).copy()
    d = ImageDraw.Draw(img)
    st = scene["style"]
    sp = st["spring"]
    fc = tuple(st["fixture"])
    damped = scene["scenario"] == "damped"
    for o, s in zip(scene["objects"], state(scene, f)):
        cx, cy = s["centre"]
        ax, ay = s["anchor"]
        h = o["size"] / 2
        if scene["orient"] == "horizontal":
            end = (cx - h, cy)
            if damped:
                _spring(d, (ax, ay), end, sp, (0, -11), width=min(sp["width"], 14))
                _dashpot(d, (ax, ay), end, tuple(sp["col"]), (0, 11))
            else:
                _spring(d, (ax, ay), end, sp)
        elif scene["orient"] == "vertical":
            end = (cx, cy - h)
            if damped:
                _spring(d, (ax, ay), end, sp, (-11, 0), width=min(sp["width"], 14))
                _dashpot(d, (ax, ay), end, tuple(sp["col"]), (11, 0))
            else:
                _spring(d, (ax, ay), end, sp)
        else:
            d.line([(ax, ay), (cx, cy)], fill=tuple(sp["col"]), width=3)
        if s["held"]:
            if scene["orient"] == "horizontal":
                px = cx - h - 11 if o["s"] > 0 else cx + h + 1
                d.rectangle([px, cy + 13, px + 10, o["ruler"]], fill=_shade(fc, -30))
            else:
                py = cy - h - 11 if o["s"] > 0 else cy + h + 1
                d.rectangle([cx + 13, py, o["ruler"], py + 10], fill=_shade(fc, -30))
        spr = sprite(o["colour"], o["shape"], o["pattern"], o["size"])
        img.paste(spr, (round(cx - h), round(cy - h)), spr)
    _clock(d, f)
    return img


def frame(scene, t: float) -> Image.Image:
    return frame_at(scene, int(round(t * FPS)))
