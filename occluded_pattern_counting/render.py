"""PIL renderer for occluded_pattern_counting v3. Frame f draws the static
pattern of objects, then the opaque occluders at sim.track(scene)[f] (flat
colour + texture + 4 px light rim), then the clock. The clock (top right,
outside the arena) is the only text in the frame."""
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




@functools.lru_cache(maxsize=4)
def _base(key: str) -> Image.Image:
    import json
    scene = json.loads(key)
    bg = sim.BACKGROUNDS[scene["background"]]
    img = Image.new("RGBA", (sim.W, sim.H), bg + (255,))
    for o in scene["objects"]:
        sp = sprite(scene["shape"], o["color"], int(o["r"]))
        img.alpha_composite(sp, (round(o["x"] - sp.width / 2), round(o["y"] - sp.height / 2)))
    return img


@functools.lru_cache(maxsize=64)
def _occ_tile(color: str, texture: str, w: int, h: int) -> Image.Image:
    base = sim.OCC_COLORS[color]
    img = Image.new("RGBA", (w, h), base + (255,))
    d = ImageDraw.Draw(img)
    ink = _mix(base, (0, 0, 0), 0.35) + (255,)
    if texture == "hatched":
        for k in range(-h, w, 18):
            d.line([(k, 0), (k + h, h)], fill=ink, width=4)
    elif texture == "dotted":
        for x in range(10, w, 20):
            for y in range(10, h, 20):
                d.ellipse([x - 3, y - 3, x + 3, y + 3], fill=ink)
    elif texture == "checker":
        for x in range(0, w, 24):
            for y in range(0, h, 24):
                if (x // 24 + y // 24) % 2 == 0:
                    d.rectangle([x, y, x + 23, y + 23], fill=ink)
    d.rectangle([0, 0, w - 1, h - 1], outline=_mix(base, (255, 255, 255), 0.45) + (255,), width=4)
    return img


def _clock(d, f):
    d.rounded_rectangle(CLOCK_BOX, radius=10, fill=(20, 20, 24))
    x0, y0, x1, y1 = CLOCK_BOX
    d.text(((x0 + x1) / 2, (y0 + y1) / 2), clock_text(f), fill=(255, 255, 255), font=font(CLOCK_PX), anchor="mm")


def frame_at(scene, f: int, mods=()) -> Image.Image:
    import json
    key = json.dumps({k: scene[k] for k in ("background", "objects", "shape")}, sort_keys=True)
    img = _base(key).copy()
    occs = sim._apply_mods(scene, mods)
    for o, rc in zip(occs, sim.track(scene, mods)[f]):
        x0, y0 = int(round(rc[0])), int(round(rc[1]))
        tile = _occ_tile(o["color"], o["texture"], int(round(rc[2])) - int(round(rc[0])), int(round(rc[3])) - int(round(rc[1])))
        img.alpha_composite(tile, (x0, y0))
    img = img.convert("RGB")
    _clock(ImageDraw.Draw(img), f)
    return img


def frame(scene, t: float) -> Image.Image:
    return frame_at(scene, min(scene["frames"] - 1, int(round(t * FPS))))
