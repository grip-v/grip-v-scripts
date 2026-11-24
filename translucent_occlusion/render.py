"""PIL renderer for translucent_occlusion v3. Frame f draws sim.track(scene)[f]:
each shape at (x, y) with radius r, then the fixed translucent panels on top
(flat tint at the scene's alpha plus a light 3 px rim), then the clock. The
clock (top right, outside the arena) is the only text in the frame."""
from __future__ import annotations
import functools
import math
import sys
from pathlib import Path

from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from common.video5 import font  # noqa: E402

import sim  # noqa: E402

FPS = sim.FPS
CLOCK_BOX = sim.CLOCK_BOX
CLOCK_PX = 34
SS = 3


def n_frames(scene) -> int:
    return scene["frames"]


def duration(scene) -> float:
    return scene["frames"] / FPS


def clock_tenths(f: int) -> int:
    return sim.clock_tenths(f)


def clock_text(f: int) -> str:
    return f"{clock_tenths(f) / 10:.1f} s"


def _mix(c, w, u):
    return tuple(int(a + (b - a) * u) for a, b in zip(c, w))


def outline(shape, cx, cy, r):
    """Polygon points for a shape of circumradius r (circle: 48-gon)."""
    def ring(k, start, rad=r):
        return [(cx + rad * math.cos(math.radians(start + i * 360 / k)),
                 cy + rad * math.sin(math.radians(start + i * 360 / k))) for i in range(k)]
    if shape == "circle":
        return ring(48, 0)
    if shape == "square":
        return ring(4, 45, r * 0.95)
    if shape == "diamond":
        return ring(4, 0)
    if shape == "triangle":
        return [(cx + r * math.cos(math.radians(-90 + i * 120)), cy + 0.15 * r + r * math.sin(math.radians(-90 + i * 120)))
                for i in range(3)]
    if shape == "pentagon":
        return ring(5, -90)
    if shape == "hexagon":
        return ring(6, 0)
    if shape == "star":
        return [(cx + (r if i % 2 == 0 else r * 0.45) * math.cos(math.radians(-90 + i * 36)),
                 cy + (r if i % 2 == 0 else r * 0.45) * math.sin(math.radians(-90 + i * 36))) for i in range(10)]
    if shape == "cross":
        a, b = r * 0.36, r * 0.92
        return [(cx - a, cy - b), (cx + a, cy - b), (cx + a, cy - a), (cx + b, cy - a), (cx + b, cy + a),
                (cx + a, cy + a), (cx + a, cy + b), (cx - a, cy + b), (cx - a, cy + a), (cx - b, cy + a),
                (cx - b, cy - a), (cx - a, cy - a)]
    raise ValueError(shape)


@functools.lru_cache(maxsize=512)
def sprite(shape: str, color: str, r: int) -> Image.Image:
    size = (2 * r + 6) * SS
    c = size / 2
    img = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    col = sim.COLORS[color]
    d.polygon(outline(shape, c, c, r * SS), fill=col + (255,), outline=_mix(col, (255, 255, 255), 0.5) + (255,),
              width=2 * SS)
    return img.resize((size // SS, size // SS), Image.LANCZOS)


@functools.lru_cache(maxsize=8)
def _floor(bg: str, style: str) -> Image.Image:
    base = sim.BACKGROUNDS[bg]
    img = Image.new("RGB", (sim.W, sim.H), base)
    d = ImageDraw.Draw(img)
    line = _mix(base, (255, 255, 255), 0.07)
    if style == "grid":
        for x in range(40, sim.W, 80):
            d.line([(x, 70), (x, sim.H)], fill=line, width=1)
        for y in range(110, sim.H, 80):
            d.line([(0, y), (sim.W, y)], fill=line, width=1)
    elif style == "dots":
        for x in range(40, sim.W, 60):
            for y in range(100, sim.H, 60):
                d.ellipse([x - 2, y - 2, x + 2, y + 2], fill=_mix(base, (255, 255, 255), 0.12))
    return img


def _panel_layer(scene, panels):
    tint = sim.TINTS[scene["tint"]]
    layer = Image.new("RGBA", (sim.W, sim.H), (0, 0, 0, 0))
    d = ImageDraw.Draw(layer)
    a = int(round(scene["alpha"] * 255))
    rim = _mix(tint, (255, 255, 255), 0.45) + (255,)
    for p in panels:
        if p["kind"] == "rect":
            d.rectangle([p["x0"], p["y0"], p["x1"], p["y1"]], fill=tint + (a,))
            d.rectangle([p["x0"], p["y0"], p["x1"], p["y1"]], outline=rim, width=3)
        else:
            box = [p["cx"] - p["R"], p["cy"] - p["R"], p["cx"] + p["R"], p["cy"] + p["R"]]
            d.ellipse(box, fill=tint + (a,))
            d.ellipse(box, outline=rim, width=3)
    return layer


@functools.lru_cache(maxsize=4)
def _panels_cached(key: str):
    import json
    scene = json.loads(key)
    return _panel_layer(scene, scene["panels"])


def _clock(d, f):
    d.rounded_rectangle(CLOCK_BOX, radius=10, fill=(236, 236, 236))
    x0, y0, x1, y1 = CLOCK_BOX
    d.text(((x0 + x1) / 2, (y0 + y1) / 2), clock_text(f), fill=(15, 15, 20), font=font(CLOCK_PX), anchor="mm")


def frame_at(scene, f: int, mods=()) -> Image.Image:
    import json
    img = _floor(scene["background"], scene["floor"]).copy().convert("RGBA")
    for a in sim.track(scene, mods)[f]:
        if a is None:
            continue
        x, y, r, color, shape = a
        sp = sprite(shape, color, int(r))
        img.alpha_composite(sp, (round(x - sp.width / 2), round(y - sp.height / 2)))
    panels = sim.panels_for(scene, mods)
    if panels:
        layer = _panels_cached(json.dumps({k: scene[k] for k in ("panels", "tint", "alpha")}, sort_keys=True)) \
            if not mods else _panel_layer(scene, panels)
        img.alpha_composite(layer)
    img = img.convert("RGB")
    _clock(ImageDraw.Draw(img), f)
    return img


def frame(scene, t: float) -> Image.Image:
    return frame_at(scene, min(scene["frames"] - 1, int(round(t * FPS))))
