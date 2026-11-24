"""PIL renderer for nested_shapes (1280x720, 24 fps).

Each frame: a fixed background (theme colour + texture; the box in the drift
scenario), then every outline on screen from the outside in, in its border
colour and line style (solid / dashed / dotted / double), its tab dot if it
has one, then every mark on top (filled, upright, with a thin rim), then the
timer. The dashes, dots and the tab are fixed to the outline, so they turn
with it.

The only text is the timer: top-left on a dark plate, seconds to 0.1 s, 40 px
bold. sim.AREA (where the nest and the box stay) starts below the plate.
`state(scene, f)` is the per-frame drawing state (centre, size, rotation,
colour, visibility of every outline and mark, the box) that answers.py and
frame_check.py read instead of the event lists.
"""
from __future__ import annotations
import json
import math
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from common.shapes import unit_shape  # noqa: E402
from common.video5 import font  # noqa: E402

import sim  # noqa: E402

TIMER_XY = (30, 22)
TIMER_SIZE = 40
TIMER_PLATE = (14, 12, 196, 80)
TIMER_INK = (238, 238, 242)
TIMER_BG = (6, 6, 8)


def duration(scene) -> float:
    return scene["n_frames"] / sim.FPS


def _mix(a, b, u):
    return tuple(int(round(a[k] + (b[k] - a[k]) * u)) for k in range(3))


def bg_rgb(scene):
    st = scene["style"]
    return sim.THEMES[st["theme"]]["bgs"][st["bg_name"]]


def rim_rgb(scene):
    return (24, 24, 26) if scene["style"]["theme"] == "light" else (250, 250, 250)


def state(scene, f: int) -> dict:
    """What is drawn at frame f."""
    rings = []
    for k, r in enumerate(scene["rings"]):
        x, y = sim.ring_centre(scene, k, f)
        rings.append({"vis": sim.visible(r, f), "x": x, "y": y, "size": r["size"], "rot": sim.rot(r, f),
                      "shape": r["shape"], "aspect": r["aspect"], "rgb": sim.COLORS[sim.color_at(r, f)],
                      "width": r["width"], "line": r["line"], "tab": r["tab"]})
    marks = []
    for i, m in enumerate(scene["marks"]):
        x, y = sim.mark_pos(scene, m, f)
        marks.append({"id": i, "vis": sim.mark_visible(m, f) and sim.visible(scene["rings"][m["ring"]], f),
                      "x": x, "y": y, "kind": m["kind"], "rgb": sim.COLORS[m["color"]],
                      "r": scene["rings"][m["ring"]]["mark_r"]})
    return {"rings": rings, "marks": marks, "nest": sim.nest_pos(scene, f), "arena": scene["arena"]}


def ring_points(st_ring: dict, size: float | None = None) -> list[tuple[float, float]]:
    """Pixel outline of a drawn ring (optionally at another size)."""
    s = st_ring["size"] if size is None else size
    th = math.radians(st_ring["rot"])
    c, sn = math.cos(th), math.sin(th)
    return [(st_ring["x"] + s * (ux * c - uy * sn), st_ring["y"] + s * (ux * sn + uy * c))
            for ux, uy in unit_shape(st_ring["shape"], st_ring["aspect"])]


def tab_point(st_ring: dict) -> tuple[float, float]:
    return ring_points(st_ring)[0]


def _arc_pieces(pts, on: float, off: float):
    """Split a closed polyline into dashes of about `on` px every `on + off` px,
    starting at vertex 0 (so the dashes turn with the outline)."""
    closed = list(pts) + [pts[0]]
    cum = [0.0]
    for a, b in zip(closed, closed[1:]):
        cum.append(cum[-1] + math.dist(a, b))
    tot = cum[-1]
    n = max(1, round(tot / (on + off)))
    per = tot / n
    dash = per * on / (on + off)

    def at(s):
        lo, hi = 0, len(cum) - 1
        while hi - lo > 1:
            mid = (lo + hi) // 2
            if cum[mid] <= s:
                lo = mid
            else:
                hi = mid
        a, b = closed[lo], closed[lo + 1]
        t = (s - cum[lo]) / max(cum[lo + 1] - cum[lo], 1e-9)
        return a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t
    out = []
    for i in range(n):
        s0, s1 = i * per, i * per + dash
        out.append([at(s0)] + [closed[j] for j in range(1, len(closed)) if s0 < cum[j] < s1] + [at(s1)])
    return out


def _dots(pts, step: float):
    segs = list(zip(pts, pts[1:] + pts[:1]))
    lens = [math.dist(a, b) for a, b in segs]
    tot = sum(lens)
    n = max(3, round(tot / step))
    out = []
    for i in range(n):
        s = i * tot / n
        acc = 0.0
        for (a, b), L in zip(segs, lens):
            if acc + L >= s:
                t = (s - acc) / max(L, 1e-9)
                out.append((a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t))
                break
            acc += L
    return out


_BG_CACHE: dict = {}


def background(scene) -> Image.Image:
    st = scene["style"]
    key = json.dumps([st, scene["arena"]], sort_keys=True)
    if key in _BG_CACHE:
        return _BG_CACHE[key]
    bg = bg_rgb(scene)
    a = np.zeros((sim.H, sim.W, 3), dtype=np.float32) + np.array(bg, dtype=np.float32)
    amp, step = st["tex_amp"], st["tex_step"]
    sgn = -1 if st["theme"] == "light" else 1
    yy, xx = np.mgrid[0:sim.H, 0:sim.W]
    tex = st["texture"]
    if tex == "vignette":
        d = np.hypot((xx - 640) / 640, (yy - 400) / 400)
        a += (sgn * amp * 2.2 * np.clip(d, 0, 1.3))[..., None]
    elif tex == "dots":
        m = ((xx % step) - step / 2) ** 2 + ((yy % step) - step / 2) ** 2 <= 9
        a[m] += sgn * amp
    elif tex == "stripes":
        a[((xx + yy) % step) < 3] += sgn * amp
    elif tex == "grid":
        a[(xx % step < 2) | (yy % step < 2)] += sgn * amp
    elif tex == "noise":
        rnd = np.sin(xx * 12.9898 + yy * 78.233) * 43758.5453
        a += ((rnd - np.floor(rnd)) - 0.5)[..., None] * amp * 1.4
    img = Image.fromarray(np.clip(a, 0, 255).astype(np.uint8))
    if scene["arena"]:
        d = ImageDraw.Draw(img)
        x0, y0, x1, y1 = scene["arena"]
        ink = _mix(bg, rim_rgb(scene), 0.45)
        d.rectangle([x0 - 4, y0 - 4, x1 + 3, y1 + 3], outline=ink, width=4)
    if len(_BG_CACHE) > 8:
        _BG_CACHE.clear()
    _BG_CACHE[key] = img
    return img


def draw_ring(d: ImageDraw.ImageDraw, s: dict, scene: dict, k: int):
    r = scene["rings"][k]
    pts = ring_points(s)
    w, col = s["width"], s["rgb"]
    if s["line"] in ("solid", "double"):
        d.line(pts + [pts[0]], fill=col, width=w, joint="curve")
        if s["line"] == "double":
            m = sim.extents(s["shape"], s["aspect"])[1]
            inner = ring_points(s, s["size"] - (w + 3) / m)
            d.line(inner + [inner[0]], fill=col, width=w, joint="curve")
    elif s["line"] == "dashed":
        for piece in _arc_pieces(pts, 3.2 * w + 6, 1.6 * w + 4):
            d.line(piece, fill=col, width=w, joint="curve")
    else:
        rr = max(2.5, 0.75 * w)
        for x, y in _dots(pts, 2.7 * w + 4):
            d.ellipse([x - rr, y - rr, x + rr, y + rr], fill=col)
    if s["tab"]:
        x, y = pts[0]
        tr = r["tab_r"]
        d.ellipse([x - tr, y - tr, x + tr, y + tr], fill=col)


def draw_mark(d: ImageDraw.ImageDraw, m: dict, rim):
    unit = unit_shape(sim.MARK_KINDS[m["kind"]], 1.0)
    big = max(math.hypot(x, y) for x, y in unit)
    pts = [(m["x"] + m["r"] * x / big, m["y"] + m["r"] * y / big) for x, y in unit]
    d.polygon(pts, fill=m["rgb"], outline=rim)
    d.line(pts + [pts[0]], fill=rim, width=2)


def draw_timer(d: ImageDraw.ImageDraw, f: int):
    d.rounded_rectangle(TIMER_PLATE, radius=10, fill=TIMER_BG)
    d.text(TIMER_XY, sim.timer_text(f), fill=TIMER_INK, font=font(TIMER_SIZE))


def frame_at(scene, f: int) -> Image.Image:
    img = background(scene).copy()
    d = ImageDraw.Draw(img)
    st = state(scene, f)
    for k, s in enumerate(st["rings"]):
        if s["vis"]:
            draw_ring(d, s, scene, k)
    rim = rim_rgb(scene)
    for m in st["marks"]:
        if m["vis"]:
            draw_mark(d, m, rim)
    draw_timer(d, f)
    return img


def frame(scene, t: float) -> Image.Image:
    return frame_at(scene, int(round(t * sim.FPS)))
