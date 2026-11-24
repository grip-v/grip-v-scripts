"""PIL renderer for vessels_and_clocks (1280x720, 24 fps).

A row of glass vessels on a bench. Each vessel has a coloured rim and
plinth (its name), tick marks on one wall at every mark, and blue water.
A tap vessel has a faucet pouring in; once full it overflows and water
runs down both outer walls. A drain has a spout at the bottom whose stream
thickness follows the outflow and stops when the vessel is empty. A joined
pair has a pipe at the bottom with a valve wheel: handle across the pipe =
closed (pipe dry), handle along the pipe = open (pipe full of water).

Text: the clock (top-left box) and the ruler numbers 0, 2, ..., 10
(sim.labels), nothing else. `state(scene, f)` is the per-frame geometry the
drawing uses; answers.py re-derives every answer from it. Pure function of
(scene, frame).
"""
from __future__ import annotations
import functools
import json
import sys
from pathlib import Path

from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from common.video5 import font  # noqa: E402
import sim  # noqa: E402

FPS = sim.FPS
SIZE = (1280, 720)
CLOCK_BOX = sim.CLOCK_BOX
CLOCK_PX = 44
FILM = 4                                  # a vessel with any water shows at least this many px of it


def duration(scene) -> float:
    return sim.duration(scene)


def n_frames(scene) -> int:
    return sim.n_frames(scene)


def clock_text(frame: int) -> str:
    return f"{sim.clock_tenths(frame) / 10:.1f} s"


def tap_width(v):
    return min(14.0, 6 + 1.5 * v["q"])


def stream_width(flow):
    return min(18.0, 5 + 2.5 * flow)


# ------------------------------------------------------------------ per-frame state
def state(scene, f: int) -> dict:
    """What frame f shows: per vessel the drawn water height (px), overflow, tap and drain streams; per
    pipe whether its valve is open."""
    t = f / FPS
    M = scene["layout"]["M"]
    out = []
    for i, v in enumerate(scene["vessels"]):
        h = sim.level(scene, i, t)
        px = 0.0 if h <= 0 else max(h * M, FILM)
        flow = sim.outflow(scene, i, t) if v["role"] in ("steady", "leaky") else 0.0
        out.append({"water_px": px, "spill": v["role"] == "tap" and h >= sim.H,
                    "pour": tap_width(v) if v["role"] == "tap" else 0.0,
                    "stream": stream_width(flow) if flow > 0 else 0.0})
    pipes = [{"a": a, "b": b, "open": t >= scene["vessels"][a]["valve"]} for a, b in scene["pairs"]]
    return {"vessels": out, "pipes": pipes}


def water_top(scene, px):
    """Row of the first water pixel for a drawn water height px (0 -> no water)."""
    return scene["layout"]["bottom"] - int(round(px)) if px > 0 else None


# ------------------------------------------------------------------ drawing helpers
def _shade(c, d):
    return tuple(max(0, min(255, x + d)) for x in c)


def _hline(d, x0, x1, y, fill, width, style):
    if style == "solid":
        d.line([(x0, y), (x1, y)], fill=fill, width=width)
        return
    on, off = (16, 10) if style == "dashed" else (6, 6)
    x = x0
    while x < x1:
        d.line([(x, y), (min(x1, x + on), y)], fill=fill, width=width)
        x += on + off


def pipe_geom(scene, a, b):
    lay, w = scene["layout"], scene["style"]["wall"]
    x0, x1 = lay["x"][a][1] + w, lay["x"][b][0] - w
    bot = lay["bottom"]
    return x0, x1, bot - 20, bot - 6, (x0 + x1) / 2, bot - 13


def spout_geom(scene, i):
    lay, w = scene["layout"], scene["style"]["wall"]
    x0, x1 = lay["x"][i]
    bot = lay["bottom"]
    if lay["spout"][i] == "right":
        return x1 + w, x1 + w + sim.SPOUT_L, bot - 14, bot - 4, "right"
    return x0 - w - sim.SPOUT_L, x0 - w, bot - 14, bot - 4, "left"


def bench_top(scene):
    st = scene["style"]
    return scene["layout"]["bottom"] + st["wall"] + st["plinth"]


def faucet_geom(scene, i):
    lay = scene["layout"]
    x0, x1 = lay["x"][i]
    cx, top = (x0 + x1) / 2, lay["top"]
    return cx, top - 66, top - 54, top - 36          # centre x, arm top, arm bottom, nozzle bottom


@functools.lru_cache(maxsize=16)
def _static(key: str) -> Image.Image:
    scene = json.loads(key)
    st, lay = scene["style"], scene["layout"]
    bg = tuple(st["bg"])
    img = Image.new("RGB", SIZE, bg)
    d = ImageDraw.Draw(img)
    dk = -14 if st["theme"] == "light" else 12
    if st["bg_pat"] == "gradient":
        for y in range(0, 720, 4):
            d.rectangle([0, y, 1279, y + 3], fill=_shade(bg, int(dk * y / 720)))
    elif st["bg_pat"] == "stripes":
        for x in range(0, 1280, 80):
            d.rectangle([x, 0, x + 39, 719], fill=_shade(bg, dk // 2))
    elif st["bg_pat"] == "dots":
        for y in range(20, 720, 40):
            for x in range(20 + (y // 40 % 2) * 20, 1280, 40):
                d.ellipse([x - 2, y - 2, x + 2, y + 2], fill=_shade(bg, dk))
    M, bot, top = lay["M"], lay["bottom"], lay["top"]
    gx0, gx1 = lay["grid_x"]
    for m in range(11):
        _hline(d, gx0, gx1, bot - m * M, tuple(st["grid"]), 3, st["grid_style"])
    # ruler + numbers
    ink = tuple(st["ink"])
    rx = lay["ruler_x"]
    d.line([(rx, top - 2), (rx, bot + 2)], fill=ink, width=5)
    sgn = 1 if lay["ruler_side"] == "left" else -1          # ticks point toward the vessels
    for m in range(11):
        y = bot - m * M
        d.line([(rx, y), (rx + sgn * (14 if m % 2 == 0 else 9), y)], fill=ink, width=3)
    for lab in scene["labels"]:
        d.text((lab["x"], lab["y"]), lab["text"], fill=ink, font=font(lab["size"]), anchor=lab["anchor"])
    # bench
    w = st["wall"]
    btop = bench_top(scene)
    bx0, bx1 = lay["x"][0][0] - w - 70, lay["x"][-1][1] + w + 70
    bench = tuple(st["bench"])
    d.rectangle([bx0, btop, bx1, btop + st["bench_h"] - 1], fill=bench)
    if st["bench_pat"] == "grain":
        for y in range(btop + 5, btop + st["bench_h"] - 3, 7):
            d.line([(bx0 + 6, y), (bx1 - 6, y)], fill=_shade(bench, -22), width=2)
    elif st["bench_pat"] == "tiles":
        for x in range(bx0 + 50, bx1, 60):
            d.line([(x, btop), (x, btop + st["bench_h"] - 1)], fill=_shade(bench, -28), width=2)
    # plinths (vessel colour), faucets, spouts, pipes
    for i, v in enumerate(scene["vessels"]):
        x0, x1 = lay["x"][i]
        col = tuple(sim.PALETTE[st["theme"]][v["colour"]])
        box = [x0 - w - 4, bot + w, x1 + w + 3, btop - 1]
        d.rectangle(box, fill=col)
        sh = _shade(col, -45 if sum(col) > 300 else 45)
        if st["plinth_pat"] == "stripes":
            for x in range(box[0] + 6, box[2] - 2, 12):
                d.line([(x, box[1] + 3), (x, box[3] - 3)], fill=sh, width=3)
        elif st["plinth_pat"] == "dots":
            for x in range(box[0] + 8, box[2] - 4, 14):
                yc = (box[1] + box[3]) // 2
                d.ellipse([x - 3, yc - 3, x + 3, yc + 3], fill=sh)
        metal = tuple(st["metal"])
        if v["role"] == "tap":
            cx, a0, a1, nb = faucet_geom(scene, i)
            d.rectangle([cx - 42, sim.TAP_TOP, cx - 30, a1], fill=metal, outline=_shade(metal, -50), width=2)
            d.rectangle([cx - 42, a0, cx + 8, a1], fill=metal, outline=_shade(metal, -50), width=2)
            d.rectangle([cx - 8, a1 - 2, cx + 8, nb], fill=metal, outline=_shade(metal, -50), width=2)
        if lay["spout"][i]:
            sx0, sx1, sy0, sy1, _ = spout_geom(scene, i)
            d.rectangle([sx0, sy0, sx1, sy1], fill=metal, outline=_shade(metal, -50), width=2)
    for a, b in scene["pairs"]:
        x0, x1, y0, y1, _, _ = pipe_geom(scene, a, b)
        d.rectangle([x0, y0, x1, y1], fill=_shade(tuple(st["metal"]), 50), outline=tuple(st["glass"]), width=3)
    return img


def _clock(d, f):
    x0, y0, x1, y1 = CLOCK_BOX
    d.rectangle([x0, y0, x1, y1], fill=(28, 30, 38), outline=(200, 200, 205), width=2)
    d.text(((x0 + x1) / 2, (y0 + y1) / 2), clock_text(f), fill=(255, 255, 255), font=font(CLOCK_PX), anchor="mm")


def frame_at(scene, f: int) -> Image.Image:
    key = json.dumps(scene, sort_keys=True)
    img = _static(key).copy()
    d = ImageDraw.Draw(img)
    s = state(scene, f)
    st, lay = scene["style"], scene["layout"]
    M, bot, top, w = lay["M"], lay["bottom"], lay["top"], st["wall"]
    water, wdark = tuple(st["water"]), tuple(st["water_dark"])
    btop = bench_top(scene)
    for p in s["pipes"]:
        x0, x1, y0, y1, cx, cy = pipe_geom(scene, p["a"], p["b"])
        if p["open"]:
            d.rectangle([x0, y0 + 3, x1, y1 - 3], fill=water)
    for i, (v, vs) in enumerate(zip(scene["vessels"], s["vessels"])):
        x0, x1 = lay["x"][i]
        yt = water_top(scene, vs["water_px"])
        if yt is not None:
            d.rectangle([x0, yt, x1 - 1, bot - 1], fill=water)
            d.rectangle([x0, yt, x1 - 1, min(bot - 1, yt + 2)], fill=wdark)
        if vs["pour"]:
            cx, _, _, nb = faucet_geom(scene, i)
            hw = vs["pour"] / 2
            d.rectangle([cx - hw, nb, cx + hw, (yt if yt is not None else bot) - 1], fill=water)
        if vs["stream"]:
            sx0, sx1, sy0, sy1, side = spout_geom(scene, i)
            sw = vs["stream"]
            d.rectangle([sx0 + 2, sy0 + 3, sx1 - 2, sy1 - 3], fill=water)
            if side == "right":
                d.rectangle([sx1, sy0 + 2, sx1 + sw, btop - 1], fill=water)
            else:
                d.rectangle([sx0 - sw, sy0 + 2, sx0, btop - 1], fill=water)
        if vs["spill"]:
            for xa, xb in ((x0 - w - sim.SPILL_W, x0 - w - 1), (x1 + w + 1, x1 + w + sim.SPILL_W)):
                d.rectangle([xa, top - 6, xb, btop - 1], fill=water)
        # glass walls, rim, ticks
        glass = tuple(st["glass"])
        d.rectangle([x0 - w, top, x0 - 1, bot + w - 1], fill=glass)
        d.rectangle([x1, top, x1 + w - 1, bot + w - 1], fill=glass)
        d.rectangle([x0 - w, bot, x1 + w - 1, bot + w - 1], fill=glass)
        col = tuple(sim.PALETTE[st["theme"]][v["colour"]])
        if st["rim"] == "band":
            d.rectangle([x0 - w - 4, top - 10, x1 + w + 3, top - 1], fill=col)
        else:
            d.rectangle([x0 - w - 4, top - 6, x0 + 3, top + 3], fill=col)
            d.rectangle([x1 - 4, top - 6, x1 + w + 3, top + 3], fill=col)
            d.rectangle([x0 - w, top - 3, x1 + w - 1, top - 1], fill=col)
        tl = st["tick_len"]
        for m in range(1, 10):
            y = bot - m * M
            if st["tick_side"] == "left":
                d.line([(x0, y), (x0 + tl, y)], fill=glass, width=3)
            else:
                d.line([(x1 - 1 - tl, y), (x1 - 1, y)], fill=glass, width=3)
        if vs["spill"]:
            for cxp in (x0 - w - sim.SPILL_W / 2, x1 + w + sim.SPILL_W / 2):
                d.ellipse([cxp - 16, btop - 4, cxp + 16, btop + 4], fill=water)
    for p in s["pipes"]:
        x0, x1, y0, y1, cx, cy = pipe_geom(scene, p["a"], p["b"])
        metal = tuple(st["metal"])
        d.ellipse([cx - 13, cy - 13, cx + 13, cy + 13], fill=metal, outline=_shade(metal, -60), width=2)
        hand = (40, 40, 46) if st["theme"] == "light" else (235, 235, 240)
        if p["open"]:
            d.rectangle([cx - 17, cy - 3, cx + 17, cy + 3], fill=hand)
        else:
            d.rectangle([cx - 3, cy - 17, cx + 3, cy + 17], fill=hand)
    _clock(d, f)
    return img


def frame(scene, t: float) -> Image.Image:
    return frame_at(scene, int(round(t * FPS)))


def scene_px(scene, px) -> int:
    """Water height in whole px as drawn (rows water_top .. bottom - 1)."""
    return int(round(px)) if px > 0 else 0
