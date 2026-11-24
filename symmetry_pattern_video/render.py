"""PIL renderer for symmetry_pattern_video (1280x720, 24 fps).

`track(scene)` is the per-frame list of what gets drawn: for each shape its
outline (screen pixels, in tracing order), how much of it is drawn, the tip
dot, how far its fill has faded in, and its style; for the turns scenario
also the disc outline and the notch. `frame_at` draws exactly that.
answers.py and frame_check.py read the track, never the timeline.

The only text is the clock in a fixed panel at the top-left; the pattern
(and the disc) never reaches it.
"""
from __future__ import annotations
import functools
import json
import math
import sys
from pathlib import Path

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


def _scr(scene, p):
    cx, cy = scene["center"]
    return (cx + p[0], cy - p[1])


@functools.lru_cache(maxsize=8)
def _track(key: str):
    scene = json.loads(key)
    out = []
    disc = [(scene["disc_r"] * math.cos(2 * math.pi * k / 72), scene["disc_r"] * math.sin(2 * math.pi * k / 72))
            for k in range(72)]
    for f in range(scene["n_frames"]):
        items = []
        for i, s in enumerate(scene["shapes"]):
            st = sim.shape_state(scene, i, f)
            if st is None:
                items.append(None)
                continue
            pts, frac, tip, fill_a = st
            items.append({"pts": [_scr(scene, p) for p in pts], "frac": frac,
                          "tip": None if tip is None else _scr(scene, tip), "fill_a": fill_a,
                          "color": s["color"], "fill": s["fill"], "dash": s["dash"], "type": s["type"]})
        fr = {"shapes": items}
        if scene["scenario"] == "turns":
            M = sim.pattern_matrix(scene, f)
            fr["disc"] = [_scr(scene, sim.apply(M, p)) for p in disc]
            fr["notch"] = _scr(scene, sim.notch_point(scene, f))
        out.append(fr)
    return out


def track(scene) -> list:
    return _track(json.dumps(scene, sort_keys=True))


def _mix(a, b, u):
    return tuple(int(round(x + (y - x) * u)) for x, y in zip(a, b))


@functools.lru_cache(maxsize=8)
def _background(key: str) -> Image.Image:
    st = json.loads(key)
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
    tb = sim.TIMER_BOX
    d.rounded_rectangle(tb, radius=10, fill=TIMER_BG, outline=TIMER_EDGE, width=2)
    return img


def _path(pts, frac):
    """Polyline along the closed outline from vertex 0 covering `frac` of the perimeter."""
    segs = [(pts[k], pts[(k + 1) % len(pts)]) for k in range(len(pts))]
    total = sum(math.dist(a, b) for a, b in segs)
    goal = frac * total
    out = [pts[0]]
    for a, b in segs:
        ln = math.dist(a, b)
        if goal >= ln - 1e-9:
            out.append(b)
            goal -= ln
        else:
            if ln > 0 and goal > 0:
                u = goal / ln
                out.append((a[0] + (b[0] - a[0]) * u, a[1] + (b[1] - a[1]) * u))
            break
    return out


def _stroke(d, line, col, w, dash):
    if not dash:
        d.line(line, fill=col, width=w, joint="curve")
        for p in line:
            d.ellipse([p[0] - w / 2, p[1] - w / 2, p[0] + w / 2, p[1] + w / 2], fill=col)
        return
    on, off = 10 * SS, 7 * SS
    pos = 0.0
    for a, b in zip(line, line[1:]):
        ln = math.dist(a, b)
        t = 0.0
        while t < ln:
            phase = pos % (on + off)
            step = min(ln - t, (on - phase) if phase < on else (on + off - phase))
            if phase < on:
                u0, u1 = t / ln, (t + step) / ln
                d.line([(a[0] + (b[0] - a[0]) * u0, a[1] + (b[1] - a[1]) * u0),
                        (a[0] + (b[0] - a[0]) * u1, a[1] + (b[1] - a[1]) * u1)], fill=col, width=w)
            t += step
            pos += step


def frame_at(scene, f: int) -> Image.Image:
    st = scene["style"]
    base = _background(json.dumps(st, sort_keys=True))
    img = base.resize((sim.W * SS, sim.H * SS), Image.NEAREST)
    d = ImageDraw.Draw(img)
    fr = track(scene)[f]
    bg = tuple(st["bg"][0])
    w = st["stroke"] * SS
    if "disc" in fr:
        disc = [(x * SS, y * SS) for x, y in fr["disc"]]
        d.polygon(disc, fill=tuple(st["disc"]))
        d.line(disc + [disc[0]], fill=tuple(st["rim"]), width=3 * SS)
        nx, ny = fr["notch"]
        cx, cy = scene["center"]
        vx, vy = nx - cx, ny - cy
        ln = math.hypot(vx, vy) or 1.0
        ux, uy = vx / ln, vy / ln
        tip = ((nx - ux * 22) * SS, (ny - uy * 22) * SS)
        b1 = ((nx + ux * 2 - uy * 13) * SS, (ny + uy * 2 + ux * 13) * SS)
        b2 = ((nx + ux * 2 + uy * 13) * SS, (ny + uy * 2 - ux * 13) * SS)
        d.polygon([tip, b1, b2], fill=tuple(st["notch"]))
    for it in fr["shapes"]:
        if it is None:
            continue
        col = sim.COLORS[it["color"]]
        pts = [(x * SS, y * SS) for x, y in it["pts"]]
        a = it["fill_a"]
        if it["fill"] == "solid" and a > 0 and it["frac"] >= 1:
            d.polygon(pts, fill=_mix(bg, col, a))
        line = _path(pts, it["frac"])
        if len(line) >= 2:
            _stroke(d, line, col, w, it["dash"])
        cxp = sum(p[0] for p in pts) / len(pts)
        cyp = sum(p[1] for p in pts) / len(pts)
        if it["fill"] == "dot" and a > 0:
            r = max(5 * SS, 0.2 * max(math.dist((cxp, cyp), p) for p in pts))
            d.ellipse([cxp - r, cyp - r, cxp + r, cyp + r], fill=_mix(bg, col, a))
        if it["fill"] == "double" and a > 0:
            inner = [(cxp + (x - cxp) * 0.55, cyp + (y - cyp) * 0.55) for x, y in pts]
            _stroke(d, inner + [inner[0]], _mix(bg, col, a), max(2 * SS, w - SS), False)
        if it["tip"] is not None:
            tx, ty = it["tip"][0] * SS, it["tip"][1] * SS
            r = st["dot_r"] * SS
            d.ellipse([tx - r, ty - r, tx + r, ty + r], fill=col, outline=(255, 255, 255) if not st["light"]
                      else (20, 20, 20), width=SS)
    img = img.resize((sim.W, sim.H), Image.LANCZOS)
    d = ImageDraw.Draw(img)
    tb = sim.TIMER_BOX
    d.text(((tb[0] + tb[2]) / 2, (tb[1] + tb[3]) / 2), sim.clock_text(sim.tenths(f)), fill=TIMER_FG,
           font=font(sim.TIMER_FONT), anchor="mm")
    return img


def frame(scene, t: float) -> Image.Image:
    return frame_at(scene, min(scene["n_frames"] - 1, int(round(t * sim.FPS))))
