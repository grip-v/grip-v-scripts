"""PIL renderer for sandpile_cascade (1280x720, 24 fps).

`track(scene)` is what gets drawn: the centre and scale of every grain on
every frame (NaN while not drawn), the grain size, shape and colour, and the
board: every cell's centre, kind (cell / hole), size and the slot centres
grains rest on. `frame_at` draws exactly that; answers.py and frame_check.py
read the track, never the rule, the drops or the event list.

Text on screen: the clock only (fixed panel, top-left). Cells are named by
position, so nothing else is written.
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


def edge(st) -> tuple:
    return INK if st["light"] else (236, 238, 244)


@functools.lru_cache(maxsize=8)
def _track(key: str) -> dict:
    scene = json.loads(key)
    tl = sim.timeline(scene)
    G = sim.geometry(scene)
    st = scene["style"]
    return {
        "poses": tl["poses"], "n_frames": scene["n_frames"],
        "grain": {"size": G["g"], "shape": st["shape"], "rgb": list(sim.GRAINS[st["grain"]])},
        "cells": [{"centre": list(c["c"]), "kind": c["kind"], "size": G["size"],
                   "slots": [list(p) for p in c["slots"]]} for c in G["cells"]],
        "side_view": scene["scenario"] == "row",
        "links": _links(scene),
        "labels": [dict(lb) for lb in scene["labels"]],
    }


def track(scene) -> dict:
    return _track(json.dumps(scene, sort_keys=True))


def _links(scene):
    """Faint guide lines drawn under the cells: the ring track, or the pyramid's down-links."""
    G = sim.geometry(scene)
    cs = G["cells"]
    if scene["scenario"] == "ring":
        k = len(cs)
        return [[list(cs[i]["c"]), list(cs[(i + 1) % k]["c"])] for i in range(k)]
    if scene["scenario"] == "pyramid":
        out = []
        for i, tg in enumerate(G["out"]):
            for t in tg:
                p = cs[t]["c"] if t >= 0 else G["virt"][-1 - t]
                out.append([list(cs[i]["c"]), list(p)])
        return out
    return []


@functools.lru_cache(maxsize=64)
def grain_sprite(rgb: tuple, shape: str, px: float, outline: int, edge_rgb: tuple) -> Image.Image:
    d = max(2, int(round(px * SS)))
    img = Image.new("RGBA", (d, d), (0, 0, 0, 0))
    dr = ImageDraw.Draw(img)
    w = outline * SS
    if shape == "disc":
        dr.ellipse([0, 0, d - 1, d - 1], fill=tuple(rgb) + (255,), outline=tuple(edge_rgb) + (255,), width=w)
    elif shape == "rounded":
        dr.rounded_rectangle([0, 0, d - 1, d - 1], radius=d * 0.28, fill=tuple(rgb) + (255,),
                             outline=tuple(edge_rgb) + (255,), width=w)
    else:
        dr.rectangle([0, 0, d - 1, d - 1], fill=tuple(rgb) + (255,), outline=tuple(edge_rgb) + (255,), width=w)
    return img


@functools.lru_cache(maxsize=8)
def _static(key: str) -> Image.Image:
    """Background, board and the clock panel (SS scale)."""
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
    G = sim.geometry(scene)
    cell = tuple(st["cell"])
    rim = _mix(cell, INK, 0.45) if st["light"] else _mix(cell, (255, 255, 255), 0.35)
    guide = _mix(_mix(top, bot, 0.5), rim, 0.55)
    for a, b in _links(scene):
        d.line([(a[0] * SS, a[1] * SS), (b[0] * SS, b[1] * SS)], fill=guide, width=4 * SS)
    sz, S = G["size"], scene["S"]
    mark = _mix(cell, rim, 0.45)
    for c in G["cells"]:
        cx, cy = c["c"]
        if scene["scenario"] == "row":
            t = sim.TRAY_T * S
            d.rounded_rectangle([(cx - sz / 2) * SS, (cy - t / 2) * SS, (cx + sz / 2) * SS, (cy + t / 2) * SS],
                                radius=3 * SS, fill=cell, outline=rim, width=3 * SS)
            continue
        box = [(cx - sz / 2) * SS, (cy - sz / 2) * SS, (cx + sz / 2) * SS, (cy + sz / 2) * SS]
        if c["kind"] == "hole":
            d.ellipse(box, fill=(10, 10, 12), outline=rim, width=4 * SS)
            r = sz * 0.3
            d.ellipse([(cx - r) * SS, (cy - r) * SS, (cx + r) * SS, (cy + r) * SS], fill=(0, 0, 0))
            continue
        if scene["scenario"] == "ring":
            d.ellipse(box, fill=cell, outline=rim, width=3 * SS)
        else:
            d.rounded_rectangle(box, radius=sz * 0.12 * SS, fill=cell, outline=rim, width=3 * SS)
        if st["mark"] != "none":
            r = G["g"] * (0.16 if st["mark"] == "dot" else 0.42)
            for x, y in c["slots"]:
                bb = [(x - r) * SS, (y - r) * SS, (x + r) * SS, (y + r) * SS]
                if st["mark"] == "dot":
                    d.ellipse(bb, fill=mark)
                else:
                    d.ellipse(bb, outline=mark, width=2 * SS)
    tb = sim.TIMER_BOX
    d.rounded_rectangle([v * SS for v in tb], radius=10 * SS, fill=TIMER_BG, outline=TIMER_EDGE, width=2 * SS)
    return img


def frame_at(scene, f: int) -> Image.Image:
    st = scene["style"]
    key = json.dumps(scene, sort_keys=True)
    tr = track(scene)
    P = tr["poses"][f]
    img = _static(key).copy()
    g = tr["grain"]["size"]
    rgb = tuple(tr["grain"]["rgb"])
    order = sorted(range(len(P)), key=lambda i: (0 if P[i][2] == 1.0 else 1, i))   # moving grains on top
    for i in order:
        x, y, s = P[i]
        if math.isnan(x):
            continue
        sp = grain_sprite(rgb, st["shape"], round(g * s, 2), st["outline"], edge(st))
        img.paste(sp, (int(round(x * SS - sp.size[0] / 2)), int(round(y * SS - sp.size[1] / 2))), sp)
    img = img.reduce(SS)
    d = ImageDraw.Draw(img)
    tb = sim.TIMER_BOX
    d.text(((tb[0] + tb[2]) / 2, (tb[1] + tb[3]) / 2), sim.clock_text(sim.tenths(f)), fill=TIMER_FG,
           font=font(sim.TIMER_FONT), anchor="mm")
    return img


def frame(scene, t: float) -> Image.Image:
    return frame_at(scene, min(scene["n_frames"] - 1, int(round(t * sim.FPS))))
