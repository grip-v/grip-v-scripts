"""PIL renderer for stacking_collapse (1280x720, 24 fps).

`track(scene)` is what gets drawn: the outline (4 corners, px) of every block
and of the plank on every frame (NaN while not in the frame), each block's
look (colour, pattern), the pillars, and the grid (1-unit squares, origin).
`frame_at` draws exactly that; answers.py and frame_check.py read the track,
never the rule, the actions or the event list.

Text on screen: the clock only (fixed panel, top-left). The grid is
unnumbered.
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
from common.video5 import font  # noqa: E402

import sim  # noqa: E402

SS = 2
TIMER_BG = (16, 18, 24)
TIMER_EDGE = (120, 126, 140)
TIMER_FG = (245, 246, 250)
INK = (28, 28, 34)


def duration(scene) -> float:
    return scene["n_frames"] / sim.FPS


def _mix(a, b, u):
    return tuple(int(round(x + (y - x) * u)) for x, y in zip(a, b))


@functools.lru_cache(maxsize=8)
def _track(key: str) -> dict:
    scene = json.loads(key)
    tl = sim.timeline(scene)
    n = scene["n_frames"]
    S = scene["S"]
    bl = scene["blocks"]
    P = np.full((n, len(bl), 4, 2), np.nan)
    for f in range(n):
        for j, b in enumerate(bl):
            p = tl["pose"][f][j]
            if p is None:
                continue
            poly = sim.rect(p[0], p[1], p[2], b["w"] * S, b["h"] * S)
            if not sim._outside([poly]):
                P[f, j] = poly
    PL = None
    if scene["plank"]:
        PL = np.full((n, 4, 2), np.nan)
        for f in range(n):
            p = tl["plank"][f]
            if p is not None:
                poly = sim.rect(p[0], p[1], p[2], scene["plank"]["w"] * S, sim.PLANK_H * S)
                if not sim._outside([poly]):
                    PL[f] = poly
    return {
        "blocks": [{"colour": b["colour"], "pattern": b["pattern"], "rgb": list(sim.COLOURS[b["colour"]])}
                   for b in bl],
        "poly": P, "plank": PL, "n_frames": n, "unit": S, "origin": [scene["ox"], scene["oy"]],
        "pillars": [list(p) for p in sim.bases_px(scene)],
        "labels": [dict(lb) for lb in scene["labels"]],
    }


def track(scene) -> dict:
    return _track(json.dumps(scene, sort_keys=True))


@functools.lru_cache(maxsize=128)
def sprite(colour: str, pattern: str, wpx: float, hpx: float, outline: int, stripe: float,
           edge: tuple = INK) -> Image.Image:
    """RGBA block of exactly wpx x hpx (x SS), outline drawn inside the edge."""
    Wd, Hd = int(round(wpx * SS)), int(round(hpx * SS))
    rgb = sim.COLOURS[colour]
    img = Image.new("RGBA", (Wd, Hd), rgb + (255,))
    d = ImageDraw.Draw(img)
    mark = (255, 255, 255) if sum(rgb) < 400 else INK
    u = hpx * SS                                    # pattern scale follows the block height
    if pattern == "striped":
        step = 0.9 * u
        sw = max(3 * SS, int(round(stripe * step)))
        k = -Hd
        while k < Wd + Hd:
            d.line([(k, Hd), (k + Hd, 0)], fill=mark + (255,), width=sw)
            k += step
    elif pattern == "dotted":
        r = 0.17 * u
        nd = max(1, int(round(wpx / (0.9 * hpx))))
        for i in range(nd):
            cx = Wd * (i + 0.5) / nd
            d.ellipse([cx - r, Hd / 2 - r, cx + r, Hd / 2 + r], fill=mark + (255,))
    w = outline * SS
    d.rectangle([0, 0, Wd - 1, Hd - 1], outline=tuple(edge) + (255,), width=w)
    return img


@functools.lru_cache(maxsize=8)
def plank_sprite(rgb: tuple, wpx: float, hpx: float, outline: int, edge: tuple = INK) -> Image.Image:
    Wd, Hd = int(round(wpx * SS)), int(round(hpx * SS))
    img = Image.new("RGBA", (Wd, Hd), tuple(rgb) + (255,))
    d = ImageDraw.Draw(img)
    grain = _mix(tuple(rgb), INK, 0.25)
    for k in range(1, 4):
        x = Wd * k / 4
        d.line([(x, 3 * SS), (x, Hd - 3 * SS)], fill=grain + (255,), width=3 * SS)
    d.rectangle([0, 0, Wd - 1, Hd - 1], outline=tuple(edge) + (255,), width=outline * SS)
    return img


@functools.lru_cache(maxsize=8)
def _static(key: str) -> Image.Image:
    """Background, grid, pillars and the clock panel (SS scale)."""
    scene = json.loads(key)
    st = scene["style"]
    top, bot = (tuple(c) for c in st["bg"])
    img = Image.new("RGB", (sim.W, sim.H), top)
    d = ImageDraw.Draw(img)
    if st["bg_style"] == "gradient":
        for y in range(sim.H):
            d.line([(0, y), (sim.W, y)], fill=_mix(top, bot, y / (sim.H - 1)))
    elif st["bg_style"] == "vignette":
        for k in range(24, 0, -1):
            u = k / 24
            d.ellipse([sim.W / 2 - 900 * u, sim.H / 2 - 620 * u, sim.W / 2 + 900 * u, sim.H / 2 + 620 * u],
                      fill=_mix(top, bot, u))
    img = img.resize((sim.W * SS, sim.H * SS), Image.BILINEAR)
    d = ImageDraw.Draw(img)
    S, ox, oy = scene["S"], scene["ox"], scene["oy"]
    ink = INK if st["light"] else (235, 238, 245)
    gc = _mix(_mix(top, bot, 0.5), ink, st["grid_mix"])
    xs = [ox + S * k for k in range(-60, 61) if 0 <= ox + S * k <= sim.W]
    ys = [oy - S * k for k in range(-60, 61) if 0 <= oy - S * k <= sim.H]
    if st["grid"] == "lines":
        for x in xs:
            d.line([(x * SS, 0), (x * SS, sim.H * SS)], fill=gc, width=3 * SS)
        for y in ys:
            d.line([(0, y * SS), (sim.W * SS, y * SS)], fill=gc, width=3 * SS)
    else:
        r = 3.5 * SS
        for x in xs:
            for y in ys:
                d.ellipse([x * SS - r, y * SS - r, x * SS + r, y * SS + r], fill=gc)
    pc = tuple(st["pillar"])
    for x0, yt, x1 in sim.bases_px(scene):
        d.rectangle([x0 * SS, yt * SS, x1 * SS - 1, sim.H * SS + 10], fill=pc, outline=_mix(pc, INK, 0.55),
                    width=3 * SS)
        d.line([(x0 * SS, yt * SS + 2 * SS), (x1 * SS - 1, yt * SS + 2 * SS)], fill=_mix(pc, INK, 0.55),
               width=3 * SS)
    tb = sim.TIMER_BOX
    d.rounded_rectangle([v * SS for v in tb], radius=10 * SS, fill=TIMER_BG, outline=TIMER_EDGE, width=2 * SS)
    return img


def edge(st) -> tuple:
    """Outline colour: dark on light backgrounds, light on dark ones."""
    return INK if st["light"] else (232, 234, 240)


def _paste(img, sp, cx, cy, ang):
    if ang != 0:
        sp = sp.rotate(-math.degrees(ang), resample=Image.BICUBIC, expand=True)
    img.paste(sp, (int(round(cx * SS - sp.size[0] / 2)), int(round(cy * SS - sp.size[1] / 2))), sp)


def frame_at(scene, f: int) -> Image.Image:
    st = scene["style"]
    key = json.dumps(scene, sort_keys=True)
    tl = sim.timeline(scene)
    img = _static(key).copy()
    S = scene["S"]
    if scene["plank"] and tl["plank"][f] is not None:
        p = tl["plank"][f]
        sp = plank_sprite(tuple(st["plank"]), scene["plank"]["w"] * S, sim.PLANK_H * S, st["outline"], edge(st))
        _paste(img, sp, p[0], p[1], p[2])
    for j, b in enumerate(scene["blocks"]):
        p = tl["pose"][f][j]
        if p is None:
            continue
        sp = sprite(b["colour"], b["pattern"], b["w"] * S, b["h"] * S, st["outline"], st["stripe"], edge(st))
        _paste(img, sp, p[0], p[1], p[2])
    img = img.reduce(SS)
    d = ImageDraw.Draw(img)
    tb = sim.TIMER_BOX
    d.text(((tb[0] + tb[2]) / 2, (tb[1] + tb[3]) / 2), sim.clock_text(sim.tenths(f)), fill=TIMER_FG,
           font=font(sim.TIMER_FONT), anchor="mm")
    return img


def frame(scene, t: float) -> Image.Image:
    return frame_at(scene, min(scene["n_frames"] - 1, int(round(t * sim.FPS))))
