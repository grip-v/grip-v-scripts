"""PIL renderer for buoyancy_tank: a side view of the tank. The back wall
has faint horizontal grid lines (no numbers); a dashed mark outside the
right wall shows the starting water level. Objects are drawn first and the
water over them as a translucent layer, so the submerged part of each
object is visibly tinted. The only text is the clock in the top-left box.

Scenario props: a rail with strings (drop), floor straps (rise), a tap
with a falling stream (fill), a drain pipe with an outflow stream (drain).
A string or strap vanishes on the frame its object is let go; a stream is
drawn exactly while the tap / drain runs.

`track`, `outline_cm`, `attached` and `flow_on` are the drawn geometry;
answers.py reads only these."""
from __future__ import annotations
import functools
import json
import math
import sys
from pathlib import Path

from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from common.video5 import font, hex_rgb  # noqa: E402

import sim  # noqa: E402

FPS = sim.FPS
TIMER_BOX = (16, 14, 206, 76)
INK = (34, 36, 40)


def duration(scene) -> float:
    return scene["n_frames"] / FPS


def track(scene) -> list:
    """[(water level cm, (object bottom heights cm, ...)), ...] for every frame drawn."""
    return list(sim.simulate(scene)["frames"][:scene["n_frames"]])


def attached(scene, i: int, k: int) -> bool:
    rel = scene["objects"][i].get("release")
    return rel is not None and k < rel


def flow_on(scene, k: int) -> bool:
    f = scene.get("flow")
    return bool(f) and f["on"] <= k < f["off"]


def outline_cm(o: dict) -> list[tuple[float, float]]:
    """Outline relative to the object's bottom-left corner, y up (cm)."""
    w, h, k = o["w"], o["h"], o["shape"]
    if k in sim.RECTS:
        return [(0, 0), (w, 0), (w, h), (0, h)]
    if k == "hexagon":
        return [(w / 4, 0), (3 * w / 4, 0), (w, h / 2), (3 * w / 4, h), (w / 4, h), (0, h / 2)]
    if k == "triangle":
        return [(0, 0), (w, 0), (w / 2, h)]
    r = h / 2
    return [(r + r * math.cos(2 * math.pi * j / 96), r + r * math.sin(2 * math.pi * j / 96)) for j in range(96)]


def timer_text(k: int) -> str:
    return sim.timer_text(k)


class _G:
    def __init__(self, scene):
        st = scene["style"]
        self.S, self.x0, self.floor = st["scale"], st["x0"], st["floor"]
        self.tw, self.th = scene["tank"]["w"], scene["tank"]["h"]

    def X(self, cm):
        return self.x0 + self.S * cm

    def Y(self, cm):
        return self.floor - self.S * cm


def _skey(scene) -> str:
    return json.dumps(scene, sort_keys=True)


def _shade(rgb):
    lum = 0.299 * rgb[0] + 0.587 * rgb[1] + 0.114 * rgb[2]
    if lum < 110:
        return tuple(int(c + (255 - c) * 0.5) for c in rgb)
    return tuple(int(c * 0.62) for c in rgb)


def _sprite(o: dict, S: float) -> Image.Image:
    W, H = int(math.ceil(o["w"] * S)) + 6, int(math.ceil(o["h"] * S)) + 6
    base = hex_rgb(sim.COLOURS[o["colour"]])
    sh = _shade(base)
    tex = Image.new("RGB", (W, H), base)
    d = ImageDraw.Draw(tex)
    if o["pattern"] == "striped":
        for c in range(-H, W + H, 16):
            d.line([(c, H), (c + H, 0)], fill=sh, width=5)
    elif o["pattern"] == "spotted":
        for yy in range(6, H, 16):
            for xx in range(6 + (8 if (yy // 16) % 2 else 0), W, 16):
                d.ellipse([xx - 4, yy - 4, xx + 4, yy + 4], fill=sh)
    elif o["pattern"] == "checked":
        for yy in range(0, H, 14):
            for xx in range(0, W, 14):
                if (xx // 14 + yy // 14) % 2:
                    d.rectangle([xx, yy, xx + 13, yy + 13], fill=sh)
    pts = [(3 + S * x, 3 + S * (o["h"] - y)) for x, y in outline_cm(o)]
    mask = Image.new("L", (W, H), 0)
    ImageDraw.Draw(mask).polygon(pts, fill=255)
    out = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    out.paste(tex, (0, 0), mask)
    ImageDraw.Draw(out).polygon(pts, outline=INK + (255,), width=3)
    return out


@functools.lru_cache(maxsize=4)
def _assets(key: str):
    scene = json.loads(key)
    g = _G(scene)
    st = scene["style"]
    img = Image.new("RGB", (1280, 720), hex_rgb(st["bg"]))
    d = ImageDraw.Draw(img)
    x0, x1 = g.X(0), g.X(g.tw)
    top = g.Y(g.th)
    d.rectangle([x0, top, x1, g.floor], fill=hex_rgb(st["back"]))
    back = hex_rgb(st["back"])
    gl = tuple(int(c * 0.80) for c in back)
    U = st["grid"]
    k = 1
    while k * U < g.th - 0.5:
        y = round(g.Y(k * U))
        if st["dashed"]:
            for xs in range(int(x0) + 4, int(x1) - 4, 26):
                d.line([xs, y, min(xs + 15, x1 - 4), y], fill=gl, width=3)
        else:
            d.line([x0 + 2, y, x1 - 2, y], fill=gl, width=3)
        k += 1
    wall = hex_rgb(st["wall"])
    # start-level mark outside the right wall
    L0, _ = sim.initial_state(scene)
    y0 = round(g.Y(L0))
    wc = tuple(int(c * 0.75) for c in st["water"])
    for xs in range(int(x1 + st["wall_px"] + 6), int(x1 + st["wall_px"] + 58), 14):
        d.line([xs, y0, xs + 8, y0], fill=wc, width=3)
    if scene["scenario"] == "drop":
        d.line([x0 - 10, top + 4, x1 + 10, top + 4], fill=wall, width=7)
    if scene["scenario"] == "fill":
        mx = g.X(g.tw - 4)
        py = top - 34
        d.rectangle([mx - 9, py - 9, x1 + 110, py + 9], fill=(120, 124, 132), outline=INK, width=3)
        d.rectangle([mx - 11, py, mx + 11, top - 6], fill=(120, 124, 132), outline=INK, width=3)
    sprites = [_sprite(o, g.S) for o in scene["objects"]]
    return img, sprites


def frame(scene, t: float) -> Image.Image:
    k = int(round(t * FPS))
    g = _G(scene)
    st = scene["style"]
    bg, sprites = _assets(_skey(scene))
    img = bg.copy()
    d = ImageDraw.Draw(img)
    fr = sim.simulate(scene)["frames"]
    L, ys = fr[min(k, len(fr) - 1)]
    top = g.Y(g.th)
    wall = hex_rgb(st["wall"])
    for i, (o, y) in enumerate(zip(scene["objects"], ys)):
        if attached(scene, i, k):
            cx = g.X(o["x"] + o["w"] / 2)
            if scene["scenario"] == "drop":
                d.line([cx, top + 6, cx, g.Y(y + o["h"]) + 2], fill=INK, width=3)
            else:
                a, b = g.X(o["x"] - 0.7), g.X(o["x"] + o["w"] + 0.7)
                yt = g.Y(o["h"] + 0.8)
                d.line([a, g.floor, a, yt, b, yt, b, g.floor], fill=(70, 70, 76), width=5, joint="curve")
    for o, y, sp in zip(scene["objects"], ys, sprites):
        img.paste(sp, (int(round(g.X(o["x"]))) - 3, int(round(g.Y(y + o["h"]))) - 3), sp)
    water = tuple(st["water"])
    box = (int(round(g.X(0))), int(round(g.Y(L))), int(round(g.X(g.tw))), int(g.floor))
    if box[3] > box[1]:
        region = img.crop(box).convert("RGBA")
        layer = Image.new("RGBA", region.size, water + (st["alpha"],))
        img.paste(Image.alpha_composite(region, layer).convert("RGB"), box[:2])
    d = ImageDraw.Draw(img)
    surf = tuple(int(c * 0.75) for c in water)
    d.line([box[0], box[1], box[2], box[1]], fill=surf, width=3)
    if flow_on(scene, k):
        if scene["scenario"] == "fill":
            mx = g.X(g.tw - 4)
            d.rectangle([mx - 6, top - 6, mx + 6, g.Y(L)], fill=surf)
        else:
            ex = g.X(g.tw) + 90
            d.rectangle([ex - 4, g.floor - 30, ex + 8, 720], fill=surf)
    wp = st["wall_px"]
    x0, x1 = g.X(0), g.X(g.tw)
    d.line([x0 - wp / 2, top, x0 - wp / 2, g.floor + wp / 2, x1 + wp / 2, g.floor + wp / 2, x1 + wp / 2, top],
           fill=wall, width=wp, joint="curve")
    if scene["scenario"] == "drain":
        py = g.floor - 22
        d.rectangle([x1 - 2, py - 9, x1 + 90, py + 9], fill=(120, 124, 132), outline=INK, width=3)
    d.rectangle(TIMER_BOX, fill=(28, 30, 34))
    d.text(((TIMER_BOX[0] + TIMER_BOX[2]) / 2, (TIMER_BOX[1] + TIMER_BOX[3]) / 2), timer_text(k),
           fill=(255, 255, 255), font=font(40), anchor="mm")
    return img
