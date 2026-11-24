"""PIL renderer for spring_chain_waves (1280x720, 24 fps).

`track(scene)` is what gets drawn: the chain's height at every 2 px along
the chain on every frame (screen pixels), which samples carry thick beads,
what holds each end, and the ruler labels. `frame_at` draws exactly that.
answers.py and frame_check.py read the track, never the pulse list.

Text on screen: the clock (fixed panel, top-left) and the ruler numbers
0..L under the chain (a measuring aid: positions and speeds are asked in
ruler units). Nothing else.
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


def duration(scene) -> float:
    return scene["n_frames"] / sim.FPS


@functools.lru_cache(maxsize=8)
def _track(key: str) -> dict:
    scene = json.loads(key)
    xs = sim.sample_xs(scene)
    h = sim.heights(scene, xs, np.arange(scene["n_frames"]))
    heavy = np.zeros(len(xs), dtype=bool)
    for lo, hi, hv in scene["sections"]:
        if hv:
            heavy |= (xs >= lo) & ((xs < hi) | ((hi == scene["L"]) & (xs <= hi)))
    return {
        "L": scene["L"], "x0": scene["x0"], "scale": scene["scale"], "y0": scene["y0"], "amp_px": scene["amp_px"],
        "xs": xs, "px": scene["x0"] + xs * scene["scale"],
        "dy": -scene["amp_px"] * h,                    # screen offset of the chain from its rest line, per frame
        "heavy": heavy, "ends": dict(scene["ends"]), "labels": [dict(lb) for lb in scene["labels"]],
        "n_frames": scene["n_frames"],
    }


def track(scene) -> dict:
    return _track(json.dumps(scene, sort_keys=True))


def _mix(a, b, u):
    return tuple(int(round(x + (y - x) * u)) for x, y in zip(a, b))


@functools.lru_cache(maxsize=8)
def _static(key: str) -> Image.Image:
    """Background, clock panel and ruler (supersampled)."""
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
    d.rounded_rectangle(sim.TIMER_BOX, radius=10, fill=TIMER_BG, outline=TIMER_EDGE, width=2)
    ink = tuple(st["ink"])
    ry = scene["ruler_y"]
    x0, sc, L = scene["x0"], scene["scale"], scene["L"]
    d.line([(x0, ry), (x0 + L * sc, ry)], fill=ink, width=3)
    for k in range(2 * L + 1):
        x = x0 + k * sc / 2
        d.line([(x, ry), (x, ry + (16 if k % 2 == 0 else 9))], fill=ink, width=3 if k % 2 == 0 else 2)
    for lb in scene["labels"]:
        d.text((lb["x"], lb["y"]), lb["text"], fill=ink, font=font(lb["size"]), anchor="mt")
    return img.resize((sim.W * SS, sim.H * SS), Image.LANCZOS)


def _bead(d, x, y, r, shape, fill, edge):
    if shape == "circle":
        d.ellipse([x - r, y - r, x + r, y + r], fill=fill, outline=edge, width=SS)
    elif shape == "square":
        r *= 0.9
        d.rectangle([x - r, y - r, x + r, y + r], fill=fill, outline=edge, width=SS)
    else:
        r *= 1.2
        d.polygon([(x, y - r), (x + r, y), (x, y + r), (x - r, y)], fill=fill, outline=edge)


def frame_at(scene, f: int) -> Image.Image:
    st = scene["style"]
    tr = track(scene)
    img = _static(json.dumps(scene, sort_keys=True)).copy()
    d = ImageDraw.Draw(img)
    ink = tuple(st["ink"])
    thin, thick = tuple(st["thin"]), tuple(st["thick"])
    y0 = scene["y0"]
    px = tr["px"]
    py = y0 + tr["dy"][f]
    hv = tr["heavy"]
    lw = st["line"] * SS
    # chain line, one run per thickness
    k = 0
    n = len(px)
    while k < n - 1:
        j = k
        while j < n - 1 and hv[j + 1] == hv[k]:
            j += 1
        pts = [(px[i] * SS, py[i] * SS) for i in range(k, min(j + 2, n))]
        if len(pts) >= 2:
            d.line(pts, fill=thick if hv[k] else thin, width=lw * (2 if hv[k] else 1), joint="curve")
        k = j + 1
    # beads
    bpu = st["beads_per_unit"]
    xs = tr["xs"]
    for b in range(scene["L"] * bpu + 1):
        x = b / bpu
        y = float(np.interp(x, xs, py))
        heavy = bool(hv[min(n - 1, int(round(x / scene["L"] * (n - 1))))])
        r = st["bead_r"] * (1.6 if heavy else 1.0) * SS
        _bead(d, (scene["x0"] + x * scene["scale"]) * SS, y * SS, r, st["bead"], thick if heavy else thin, ink)
    # ends
    metal = tuple(st["metal"])
    for side, i in (("left", 0), ("right", n - 1)):
        kind = tr["ends"][side]
        ex, ey = px[i], py[i]
        out = -1 if side == "left" else 1
        if kind == "paddle":
            pc = tuple(st["paddle"])
            d.rectangle([(ex + out * 2 - 9 + out * 9) * SS, (ey - 38) * SS, (ex + out * 2 + 9 + out * 9) * SS,
                         (ey + 38) * SS], fill=pc, outline=ink, width=SS)
            d.line([((ex + out * 20) * SS, ey * SS), ((ex + out * 62) * SS, ey * SS)], fill=pc, width=8 * SS)
        elif kind == "clamped":
            xa = ex + (0 if out > 0 else -26)
            d.rectangle([xa * SS, (y0 - 60) * SS, (xa + 26) * SS, (y0 + 60) * SS], fill=metal, outline=ink, width=SS)
            for yy in range(-52, 56, 16):
                d.line([(xa + 3) * SS, (y0 + yy) * SS, (xa + 23) * SS, (y0 + yy + 10) * SS], fill=ink, width=SS)
        else:
            eu, ed = scene["end_reach"][0 if side == "left" else 1]
            rr = st["ring_r"]
            top = y0 - scene["amp_px"] * eu - rr - 22
            bot = y0 + scene["amp_px"] * ed + rr + 22
            d.line([ex * SS, top * SS, ex * SS, bot * SS], fill=metal, width=5 * SS)
            d.ellipse([(ex - rr) * SS, (ey - rr) * SS, (ex + rr) * SS, (ey + rr) * SS], outline=ink, width=4 * SS)
    img = img.resize((sim.W, sim.H), Image.LANCZOS)
    d = ImageDraw.Draw(img)
    tb = sim.TIMER_BOX
    d.text(((tb[0] + tb[2]) / 2, (tb[1] + tb[3]) / 2), sim.clock_text(sim.tenths(f)), fill=TIMER_FG,
           font=font(sim.TIMER_FONT), anchor="mm")
    return img


def frame(scene, t: float) -> Image.Image:
    return frame_at(scene, min(scene["n_frames"] - 1, int(round(t * sim.FPS))))
