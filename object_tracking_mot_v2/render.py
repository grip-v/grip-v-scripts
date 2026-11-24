"""PIL renderer for object_tracking_mot_v2 (1280x720, 24 fps).

A walled arena with faint unnumbered centre lines (quadrants), the moving
objects (coloured shapes with a pattern; or identical shapes in the targets
scenario, where the targets wear a thick ring until the motion starts) and,
in the occluders scenario, opaque hatched screens drawn over the objects.
The clock (top-left box) is the only text. Pure function of (scene, frame).
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


def fill_of(scene, o):
    return tuple(scene["style"]["same"]) if o["colour"] == "same" else sim.COLOURS[o["colour"]]


def state(scene, f: int) -> dict:
    """What frame f draws: objects [{id, x, y, visible}] (visible = not under a screen), and whether
    the target rings are shown."""
    t = f / FPS
    objs = []
    for o in scene["objects"]:
        x, y = sim.position(scene, o["id"], t)
        objs.append({"id": o["id"], "x": x, "y": y, "visible": not sim.hidden(scene, o["id"], t)})
    return {"objects": objs, "rings": scene["scenario"] == "targets" and t < scene["start_t"]}


def _dash(d, a, b, col, w, on, off):
    L = math.dist(a, b)
    ux, uy = (b[0] - a[0]) / L, (b[1] - a[1]) / L
    s = 0.0
    while s < L:
        e = min(L, s + on)
        d.line([(a[0] + ux * s, a[1] + uy * s), (a[0] + ux * e, a[1] + uy * e)], fill=col, width=w)
        s += on + off


@functools.lru_cache(maxsize=8)
def _static(key: str) -> Image.Image:
    scene = json.loads(key)
    st = scene["style"]
    a = scene["arena"]
    img = Image.new("RGB", SIZE, tuple(st["bg"]))
    d = ImageDraw.Draw(img)
    ar = tuple(st["arena"])
    d.rectangle([a["x0"], a["y0"], a["x1"], a["y1"]], fill=ar)
    alt = tuple(max(0, min(255, v + (8 if st["theme"] == "dark" else -8))) for v in ar)
    step = st["floor_step"]
    if st["floor"] == "grid":
        for x in range(a["x0"] + step, a["x1"], step):
            d.line([(x, a["y0"]), (x, a["y1"])], fill=alt, width=1)
        for y in range(a["y0"] + step, a["y1"], step):
            d.line([(a["x0"], y), (a["x1"], y)], fill=alt, width=1)
    elif st["floor"] == "dots":
        for x in range(a["x0"] + step // 2, a["x1"], step):
            for y in range(a["y0"] + step // 2, a["y1"], step):
                d.ellipse([x - 2, y - 2, x + 2, y + 2], fill=alt)
    cx, cy = (a["x0"] + a["x1"]) / 2, (a["y0"] + a["y1"]) / 2
    ln = tuple(st["line"])
    for p, q in (((cx, a["y0"]), (cx, a["y1"])), ((a["x0"], cy), (a["x1"], cy))):
        if st["line_style"] == "solid":
            d.line([p, q], fill=ln, width=3)
        else:
            _dash(d, p, q, ln, 3, 14 if st["line_style"] == "dashed" else 4, 9)
    w = st["wall_w"]
    d.rectangle([a["x0"] - w, a["y0"] - w, a["x1"] + w, a["y1"] + w], outline=tuple(st["wall"]), width=w)
    return img


def _poly(shape, s):
    h = s / 2
    if shape == "square":
        return [(h - h * 0.72, h - h * 0.72), (h + h * 0.72, h - h * 0.72), (h + h * 0.72, h + h * 0.72),
                (h - h * 0.72, h + h * 0.72)]
    if shape == "diamond":
        return [(h, 0), (s - 1, h), (h, s - 1), (0, h)]
    if shape == "triangle":
        return [(h + h * math.cos(math.radians(a)), h + h * math.sin(math.radians(a))) for a in (-90, 30, 150)]
    if shape == "hexagon":
        return [(h + h * math.cos(math.pi / 3 * k), h + h * math.sin(math.pi / 3 * k)) for k in range(6)]
    return None


@functools.lru_cache(maxsize=64)
def sprite(rgb: tuple, shape: str, pattern: str, s: int) -> Image.Image:
    mask = Image.new("L", (s, s), 0)
    md = ImageDraw.Draw(mask)
    if shape == "circle":
        md.ellipse([0, 0, s - 1, s - 1], fill=255)
    else:
        md.polygon(_poly(shape, s), fill=255)
    pad = Image.new("L", (s + 16, s + 16), 0)
    pad.paste(mask, (8, 8))
    for _ in range(5):
        pad = pad.filter(ImageFilter.MinFilter(3))
    inner = pad.crop((8, 8, s + 8, s + 8))
    pat = (30, 30, 34) if sum(rgb) > 520 else (255, 255, 255)
    img = Image.new("RGBA", (s, s), rgb + (255,))
    p = Image.new("RGBA", (s, s), (0, 0, 0, 0))
    pd = ImageDraw.Draw(p)
    if pattern == "striped":
        for k in range(-s, 2 * s, 9):
            pd.line([(k, 0), (k + s, s)], fill=pat + (255,), width=3)
    elif pattern == "dotted":
        for x in range(4, s, 9):
            for y in range(4, s, 9):
                pd.ellipse([x - 2, y - 2, x + 2, y + 2], fill=pat + (255,))
    elif pattern == "ringed":
        c = s / 2
        r = s * 0.22
        pd.ellipse([c - r, c - r, c + r, c + r], outline=pat + (255,), width=3)
    img.paste(p, (0, 0), Image.composite(p.getchannel("A"), Image.new("L", (s, s), 0), inner))
    img.putalpha(mask)
    return img


def _clock(d, frame):
    x0, y0, x1, y1 = CLOCK_BOX
    d.rounded_rectangle(CLOCK_BOX, radius=12, fill=(18, 18, 22), outline=(235, 235, 235), width=3)
    d.text(((x0 + x1) / 2, (y0 + y1) / 2), clock_text(frame), fill=(255, 255, 255), font=font(44), anchor="mm")


def frame_at(scene, f: int) -> Image.Image:
    img = _static(json.dumps(scene, sort_keys=True)).copy()
    d = ImageDraw.Draw(img)
    st = state(scene, f)
    for so in st["objects"]:
        o = scene["objects"][so["id"]]
        s = 2 * o["r"]
        spr = sprite(fill_of(scene, o), o["shape"], o["pattern"], s)
        img.paste(spr, (round(so["x"] - o["r"]), round(so["y"] - o["r"])), spr)
        if st["rings"] and o["id"] in scene["targets"]:
            r = o["r"] + 7
            d.ellipse([so["x"] - r, so["y"] - r, so["x"] + r, so["y"] + r], outline=tuple(scene["style"]["ring"]), width=4)
    for sc in scene["screens"]:
        col = sim.SCREENS[sc["name"]]
        d.rectangle([sc["x0"], sc["y0"], sc["x1"], sc["y1"]], fill=col, outline=tuple(max(0, v - 50) for v in col), width=3)
        k = sc["x0"] - (sc["y1"] - sc["y0"])
        while k < sc["x1"]:
            a = (max(k, sc["x0"]), sc["y1"] - max(0, sc["x0"] - k))
            b = (min(k + (sc["y1"] - sc["y0"]), sc["x1"]), sc["y0"] + max(0, k + (sc["y1"] - sc["y0"]) - sc["x1"]))
            d.line([a, b], fill=tuple(max(0, v - 25) for v in col), width=2)
            k += 14
    _clock(d, f)
    return img


def frame(scene, t: float) -> Image.Image:
    return frame_at(scene, int(round(t * FPS)))
