"""PIL renderer for moving_shadows (1280x720, 24 fps).

Side view: sky (gradient / flat / faint bands), a sun disc on its arc or a
lamp hanging from a rail, coloured objects (floating shapes with a fill
pattern, standing posts / pillars / huts), and the ground. The top of the
ground is a lighter surface strip; marker plates are set into it, and every
shadow darkens the strip by the same factor (overlapping shadows merge, so
all shadows look alike). Pixel column c of the strip is in shadow exactly
when some shadow interval contains c + 0.5, so a marker plate's centre
column is shaded exactly when the marker is in shadow.

Text in the frame: the clock (top-left box, seconds to 0.1 s, truncated
from the frame number) and the marker numbers 1..n under the strip, drawn
from scene["labels"]. Pure function of (scene, frame).
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
sys.path.insert(0, str(Path(__file__).resolve().parent))
from common.video5 import font  # noqa: E402
import sim  # noqa: E402

FPS = sim.FPS
SIZE = sim.SIZE
CLOCK_BOX = sim.CLOCK_BOX
PLATE_RGB = (246, 246, 238)
INK = (30, 30, 34)


def duration(scene) -> float:
    return sim.duration(scene)


def n_frames(scene) -> int:
    return sim.n_frames(scene)


def clock_text(f: int) -> str:
    return f"{sim.clock_tenths(f) / 10:.1f} s"


def state(scene, f: int) -> dict:
    """What frame f draws: the light centre, each object's outline, and the shadow intervals."""
    t = f / FPS
    return {"light": sim.light(scene, t),
            "casters": [sim.outline(o, t) for o in scene["casters"]],
            "shadows": sim.shadows(scene, t)}


def shaded_columns(scene, ivs) -> np.ndarray:
    cols = np.arange(SIZE[0]) + 0.5
    on = np.zeros(SIZE[0], dtype=bool)
    for lo, hi in ivs:
        on |= (lo <= cols) & (cols <= hi)
    return on


def _shade(c, d):
    return tuple(max(0, min(255, int(v + d))) for v in c)


def _mix(a, b, u):
    return tuple(int(round(x + (y - x) * u)) for x, y in zip(a, b))


def strip_rows(scene):
    return scene["ground"], scene["ground"] + scene["style"]["strip"]


def plate_box(scene, j):
    st = scene["style"]
    y0, y1 = strip_rows(scene)
    cx, cy = sim.mx(scene, j), (y0 + y1) / 2
    w = st["plate_w"]
    return cx - w / 2, cy - 8, cx + w / 2, cy + 8


# ------------------------------------------------------------------ objects
def _pattern_layer(o, box, fill):
    """RGBA layer (box-sized) with the object's pattern; the pattern colour contrasts with the fill."""
    x0, y0, x1, y1 = box
    w, h = x1 - x0, y1 - y0
    lum = 0.3 * fill[0] + 0.59 * fill[1] + 0.11 * fill[2]
    pc = (255, 255, 255, 255) if lum < 150 else (35, 35, 40, 255)
    lay = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    d = ImageDraw.Draw(lay)
    p = o["pattern"]
    if p == "striped":
        for k in range(-h, w + h, 11):
            d.line([(k, 0), (k + h, h)], fill=pc, width=3)
    elif p == "banded":
        for y in range(8, h, 18):
            d.line([(0, y), (w, y)], fill=pc, width=4)
    elif p == "dotted":
        for x in range(5, w, 11):
            for y in range(5, h, 11):
                d.ellipse([x - 2, y - 2, x + 2, y + 2], fill=pc)
    elif p == "checked":
        for x in range(0, w, 10):
            for y in range(0, h, 10):
                if (x // 10 + y // 10) % 2:
                    d.rectangle([x, y, x + 9, y + 9], fill=pc[:3] + (150,))
    elif p == "ringed":
        cx, cy = w / 2, h / 2
        r = min(w, h) / 2
        for k in (0.62, 0.32):
            d.ellipse([cx - r * k, cy - r * k, cx + r * k, cy + r * k], outline=pc, width=3)
    return lay


def draw_object(img: Image.Image, o: dict, pts: list, lw: int):
    fill = sim.COLOURS[o["color"]]
    xs, ys = [p[0] for p in pts], [p[1] for p in pts]
    box = (int(math.floor(min(xs))) - 2, int(math.floor(min(ys))) - 2,
           int(math.ceil(max(xs))) + 3, int(math.ceil(max(ys))) + 3)
    w, h = box[2] - box[0], box[3] - box[1]
    loc = [(x - box[0], y - box[1]) for x, y in pts]
    mask = Image.new("L", (w, h), 0)
    ImageDraw.Draw(mask).polygon(loc, fill=255)
    tile = Image.new("RGBA", (w, h), fill + (255,))
    if o["pattern"] != "solid":
        inner = Image.new("L", (w, h), 0)
        ImageDraw.Draw(inner).polygon(loc, fill=255, outline=0)
        ImageDraw.Draw(inner).line(loc + [loc[0]], fill=0, width=2 * lw + 4)
        lay = _pattern_layer(o, box, fill)
        a = np.minimum(np.asarray(lay.getchannel("A")), np.asarray(inner))
        tile.paste(lay, (0, 0), Image.fromarray(a.astype(np.uint8)))
    ImageDraw.Draw(tile).line(loc + [loc[0]], fill=_shade(fill, -80) + (255,), width=lw, joint="curve")
    img.paste(tile, box[:2], mask)


# ------------------------------------------------------------------ static layer
@functools.lru_cache(maxsize=8)
def _static(key: str) -> Image.Image:
    scene = json.loads(key)
    st = scene["style"]
    G = scene["ground"]
    img = Image.new("RGB", SIZE, tuple(st["sky_bot"]))
    d = ImageDraw.Draw(img)
    top, bot = tuple(st["sky_top"]), tuple(st["sky_bot"])
    if st["sky_pattern"] == "gradient":
        for y in range(G):
            d.line([(0, y), (SIZE[0], y)], fill=_mix(top, bot, y / G))
    elif st["sky_pattern"] == "bands":
        for k, y in enumerate(range(0, G, 60)):
            d.rectangle([0, y, SIZE[0], y + 59], fill=_mix(top, bot, min(1.0, (y + 30) / G)))
    else:
        d.rectangle([0, 0, SIZE[0], G], fill=_mix(top, bot, 0.6))
    gr = tuple(st["ground_rgb"])
    y0, y1 = strip_rows(scene)
    d.rectangle([0, y1, SIZE[0], SIZE[1]], fill=_shade(gr, -22))
    tex = st["ground_tex"]
    tc = _shade(gr, -34)
    if tex == "stripes":
        for y in range(y1 + 12, SIZE[1], 22):
            d.line([(0, y), (SIZE[0], y)], fill=tc, width=2)
    elif tex == "bricks":
        for r, y in enumerate(range(y1 + 4, SIZE[1], 26)):
            d.line([(0, y), (SIZE[0], y)], fill=tc, width=2)
            for x in range(-40 + (r % 2) * 40, SIZE[0], 80):
                d.line([(x, y), (x, y + 26)], fill=tc, width=2)
    elif tex == "speckle":
        for k in range(900):
            x = (k * 7919) % SIZE[0]
            y = y1 + 6 + (k * 104729) % max(1, SIZE[1] - y1 - 10)
            d.ellipse([x - 1, y - 1, x + 1, y + 1], fill=tc)
    d.rectangle([0, y0, SIZE[0], y1 - 1], fill=_shade(gr, 18))
    d.line([(0, y0), (SIZE[0], y0)], fill=_shade(gr, -60), width=3)
    for j in range(len(scene["markers"])):
        bx0, by0, bx1, by1 = plate_box(scene, j)
        if st["plate"] == "rect":
            d.rectangle([bx0, by0, bx1, by1], fill=PLATE_RGB, outline=INK, width=2)
        elif st["plate"] == "round":
            d.rounded_rectangle([bx0, by0, bx1, by1], radius=5, fill=PLATE_RGB, outline=INK, width=2)
        else:
            cx, cy = (bx0 + bx1) / 2, (by0 + by1) / 2
            d.polygon([(bx0, cy), (cx, by0 - 1), (bx1, cy), (cx, by1 + 1)], fill=PLATE_RGB, outline=INK, width=2)
    if scene["light"]["kind"] == "lamp":
        lt = scene["light"]
        ry = lt["y"] - st["sun_r"] - 26
        a, b = sorted((lt["x0"], lt["x1"]))
        d.line([(a - 40, ry), (b + 40, ry)], fill=(70, 70, 78), width=6)
    for o in scene["casters"]:
        if not (o.get("vx") or o.get("vy")):
            draw_object(img, o, sim.outline(o, 0.0), st["outline"])
    for lb in scene["labels"]:
        d.text((lb["x"], lb["y"]), lb["text"], fill=(255, 255, 255), font=font(lb["size"]), anchor="mm",
               stroke_width=3, stroke_fill=INK)
    return img


def _light(d, scene, xy):
    st = scene["style"]
    r = st["sun_r"]
    lx, ly = xy
    col = sim.SUNS[st["sun"]]
    edge = _shade(col, -60)
    if scene["light"]["kind"] == "lamp":
        ry = scene["light"]["y"] - r - 26
        d.line([(lx, ry), (lx, ly - r)], fill=(70, 70, 78), width=4)
        d.polygon([(lx - r - 6, ly - 2), (lx + r + 6, ly - 2), (lx + 8, ly - r - 8), (lx - 8, ly - r - 8)],
                  fill=(70, 70, 78))
    elif st["sun_style"] == "rays":
        for k in range(12):
            a = math.pi * k / 6
            d.line([(lx + (r + 4) * math.cos(a), ly + (r + 4) * math.sin(a)),
                    (lx + (r + 13) * math.cos(a), ly + (r + 13) * math.sin(a))], fill=col, width=3)
    elif st["sun_style"] == "ring":
        d.ellipse([lx - r - 9, ly - r - 9, lx + r + 9, ly + r + 9], outline=col, width=3)
    d.ellipse([lx - r, ly - r, lx + r, ly + r], fill=col, outline=edge, width=3)


def _clock(d, f):
    x0, y0, x1, y1 = CLOCK_BOX
    d.rounded_rectangle(CLOCK_BOX, radius=12, fill=(18, 18, 22), outline=(235, 235, 235), width=3)
    d.text(((x0 + x1) / 2, (y0 + y1) / 2), clock_text(f), fill=(255, 255, 255), font=font(44), anchor="mm")


def static_key(scene) -> str:
    return json.dumps({k: v for k, v in scene.items() if k != "build_attempts"}, sort_keys=True)


def frame_at(scene, f: int) -> Image.Image:
    s = state(scene, f)
    arr = np.asarray(_static(static_key(scene))).copy()
    y0, y1 = strip_rows(scene)
    on = shaded_columns(scene, s["shadows"])
    k = scene["style"]["shade"]
    band = arr[y0 + 2:y1, on].astype(np.float32)
    arr[y0 + 2:y1, on] = np.clip(band * k + np.array([4, 6, 14]) * (1 - k), 0, 255).astype(np.uint8)
    img = Image.fromarray(arr)
    d = ImageDraw.Draw(img)
    _light(d, scene, s["light"])
    for o, pts in zip(scene["casters"], s["casters"]):
        if o.get("vx") or o.get("vy"):
            draw_object(img, o, pts, scene["style"]["outline"])
    _clock(d, f)
    return img


def frame(scene, t: float) -> Image.Image:
    return frame_at(scene, int(round(t * FPS)))
