"""PIL renderer for hidden_mirrors (1280x720, 24 fps).

A board of cells whose mirrors are never drawn (except the drawn ones of
the `partial` scenario), a port marker outside every edge cell, dark blocks
(`blocks`), and the pulses: each one is a large head in its own colour that
leaves a trail of the same colour; the port it was fired from lights up in
its colour, and so does the port it leaves through. A pulse stopped by a
block leaves a ring on it. Trails and lit ports stay for sim.HOLD_S after
the pulse ends, then fade out over sim.FADE_S.

Text: the clock (top-left box) and the cell labels (column letters above
the board, row numbers left of it; sim.labels), nothing else.
Pure function of (scene, frame).
"""
from __future__ import annotations
import functools
import json
import math
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
NORMAL = {"top": (0, -1), "bottom": (0, 1), "left": (-1, 0), "right": (1, 0)}


def duration(scene) -> float:
    return sim.duration(scene)


def n_frames(scene) -> int:
    return sim.n_frames(scene)


def clock_text(frame: int) -> str:
    return f"{sim.clock_tenths(frame) / 10:.1f} s"


def colour(scene, name):
    return sim.PULSES[scene["style"]["theme"]][name]


def px(scene, x, y):
    st = scene["style"]
    return st["x0"] + x * st["cell"], st["y0"] + y * st["cell"]


def port_anchor(scene, port):
    """Screen point on the board edge at the middle of the port, and the outward normal."""
    edge, i = port
    W, H = scene["cols"], scene["rows"]
    gx, gy = {"top": (i + 0.5, 0), "bottom": (i + 0.5, H), "left": (0, i + 0.5), "right": (W, i + 0.5)}[edge]
    return px(scene, gx, gy), NORMAL[edge]


def marker_poly(scene, port):
    """Polygon (px) of a port marker; 'notch' markers are returned as a circle (cx, cy, r)."""
    st = scene["style"]
    c = st["cell"]
    (ax, ay), (nx, ny) = port_anchor(scene, port)
    tx, ty = -ny, nx                                    # along the edge
    w = 0.19 * c
    if st["marker"] == "wedge":                         # points into the board
        a, b = 0.08 * c, sim.MARK_OUT * c
        return [(ax + nx * a, ay + ny * a), (ax + nx * b + tx * w, ay + ny * b + ty * w),
                (ax + nx * b - tx * w, ay + ny * b - ty * w)]
    if st["marker"] == "bar":
        a, b = 0.1 * c, sim.MARK_OUT * c
        return [(ax + nx * a + tx * w, ay + ny * a + ty * w), (ax + nx * b + tx * w, ay + ny * b + ty * w),
                (ax + nx * b - tx * w, ay + ny * b - ty * w), (ax + nx * a - tx * w, ay + ny * a - ty * w)]
    r = 0.17 * c
    return (ax + nx * (sim.MARK_OUT * c - r), ay + ny * (sim.MARK_OUT * c - r), r)


@functools.lru_cache(maxsize=16)
def _runs(key):
    return sim.runs(json.loads(key))


def traced(scene):
    return _runs(json.dumps({k: scene[k] for k in ("cols", "rows", "step", "mirrors", "blocks", "pulses")},
                            sort_keys=True))


def state(scene, f: int) -> dict:
    """What frame f draws: heads [(colour, x, y)] px, trails [(colour, [(x, y)...], alpha)] px, lit ports
    [(colour, port, alpha)], stop rings [(colour, x, y, alpha)]."""
    t = f / FPS
    heads, trails, lit, rings = [], [], [], []
    for r in traced(scene):
        a = sim.trail_alpha(r, t)
        if a <= 0:
            continue
        pts = [px(scene, x, y) for tt, x, y in r["pts"] if tt <= t + 1e-9]
        h = sim.head_at(r, t)
        if h is not None:
            hp = px(scene, *h)
            heads.append((r["colour"], hp[0], hp[1]))
            pts.append(hp)
        trails.append((r["colour"], pts, a))
        lit.append((r["colour"], tuple(r["port"]), a))
        e = r["end"]
        if t >= e[-1] - 1e-9:
            if e[0] == "exit":
                lit.append((r["colour"], (e[1], e[2]), a))
            else:
                rings.append((r["colour"],) + px(scene, e[1][0] + 0.5, e[1][1] + 0.5) + (a,))
    return {"heads": heads, "trails": trails, "lit": lit, "rings": rings}


# ------------------------------------------------------------------ drawing
def _dash(d, a, b, col, w, on=10, off=7):
    L = math.dist(a, b)
    if L == 0:
        return
    ux, uy = (b[0] - a[0]) / L, (b[1] - a[1]) / L
    s = 0.0
    while s < L:
        e = min(L, s + on)
        d.line([(a[0] + ux * s, a[1] + uy * s), (a[0] + ux * e, a[1] + uy * e)], fill=col, width=w)
        s += on + off


def _marker(d, scene, port, fill, outline):
    m = marker_poly(scene, port)
    if isinstance(m, tuple):
        cx, cy, r = m
        d.ellipse([cx - r, cy - r, cx + r, cy + r], fill=fill, outline=outline, width=3)
    else:
        d.polygon(m, fill=fill, outline=outline, width=3)


@functools.lru_cache(maxsize=8)
def _static(key: str) -> Image.Image:
    scene = json.loads(key)
    st = scene["style"]
    img = Image.new("RGB", SIZE, tuple(st["bg"]))
    d = ImageDraw.Draw(img)
    x0, y0, c = st["x0"], st["y0"], st["cell"]
    W, H = scene["cols"], scene["rows"]
    board = tuple(st["board"])
    d.rectangle([x0, y0, x0 + W * c, y0 + H * c], fill=board)
    alt = tuple(max(0, min(255, v + (10 if st["theme"] == "dark" else -10))) for v in board)
    for i in range(W):
        for j in range(H):
            if st["board_pat"] == "checker" and (i + j) % 2:
                d.rectangle([x0 + i * c, y0 + j * c, x0 + (i + 1) * c, y0 + (j + 1) * c], fill=alt)
            elif st["board_pat"] == "dots":
                for cx, cy in ((x0 + (i + 0.2) * c, y0 + (j + 0.2) * c), (x0 + (i + 0.8) * c, y0 + (j + 0.8) * c)):
                    d.ellipse([cx - 3, cy - 3, cx + 3, cy + 3], fill=alt)
    g = tuple(st["grid"])
    lines = [((x0 + i * c, y0), (x0 + i * c, y0 + H * c)) for i in range(W + 1)] + \
            [((x0, y0 + j * c), (x0 + W * c, y0 + j * c)) for j in range(H + 1)]
    for a, b in lines:
        if st["grid_style"] == "solid":
            d.line([a, b], fill=g, width=2)
        elif st["grid_style"] == "dashed":
            _dash(d, a, b, g, 2, 12, 8)
        else:
            _dash(d, a, b, g, 3, 3, 7)
    d.rectangle([x0, y0, x0 + W * c, y0 + H * c], outline=g, width=3)
    for port in sim.all_ports(scene):
        _marker(d, scene, port, tuple(st["bg"]), g)
    f = font(sim.LABEL_PX)
    for lab in scene["labels"]:
        d.text((lab["x"], lab["y"]), lab["text"], fill=tuple(st["ink"]), font=f, anchor="mm")
    return img


def _blend(rgb, base, a):
    return tuple(int(round(base[k] + (rgb[k] - base[k]) * a)) for k in range(3))


def _head(d, scene, rgb, x, y):
    st = scene["style"]
    r = st["head_px"] / 2
    ink = tuple(st["ink"])
    if st["head_shape"] == "disc":
        d.ellipse([x - r, y - r, x + r, y + r], fill=rgb, outline=ink, width=2)
    elif st["head_shape"] == "square":
        q = 0.9 * r
        d.rectangle([x - q, y - q, x + q, y + q], fill=rgb, outline=ink, width=2)
    else:
        q = 1.35 * r
        d.polygon([(x, y - q), (x + q, y), (x, y + q), (x - q, y)], fill=rgb, outline=ink, width=2)


def _mirror(d, scene, m):
    st = scene["style"]
    c = st["cell"]
    cx, cy = px(scene, m["cell"][0] + 0.5, m["cell"][1] + 0.5)
    h = 0.36 * c
    a, b = ((cx - h, cy + h), (cx + h, cy - h)) if m["kind"] == "/" else ((cx - h, cy - h), (cx + h, cy + h))
    d.line([a, b], fill=tuple(st["mirror"]), width=st["mirror_w"])


def _clock(d, frame):
    x0, y0, x1, y1 = CLOCK_BOX
    d.rounded_rectangle(CLOCK_BOX, radius=12, fill=(18, 18, 22), outline=(235, 235, 235), width=3)
    d.text(((x0 + x1) / 2, (y0 + y1) / 2), clock_text(frame), fill=(255, 255, 255), font=font(44), anchor="mm")


def frame_at(scene, f: int) -> Image.Image:
    img = _static(json.dumps(scene, sort_keys=True)).copy()
    d = ImageDraw.Draw(img)
    st = scene["style"]
    board, bg = tuple(st["board"]), tuple(st["bg"])
    s = state(scene, f)
    for name, pts, a in s["trails"]:
        if len(pts) >= 2:
            d.line(pts, fill=_blend(colour(scene, name), board, a), width=st["trail_w"], joint="curve")
    c = st["cell"]
    for b in scene["blocks"]:
        m = 0.08 * c
        x, y = px(scene, b[0], b[1])
        d.rectangle([x + m, y + m, x + c - m, y + c - m], fill=tuple(st["block"]), outline=tuple(st["grid"]), width=3)
    for m in scene["mirrors"]:
        if m["shown"]:
            _mirror(d, scene, m)
    for name, port, a in s["lit"]:
        _marker(d, scene, port, _blend(colour(scene, name), bg, a), tuple(st["grid"]))
    for name, x, y, a in s["rings"]:
        r = 0.3 * c
        d.ellipse([x - r, y - r, x + r, y + r], outline=_blend(colour(scene, name), tuple(st["block"]), a), width=5)
    for name, x, y in s["heads"]:
        _head(d, scene, colour(scene, name), x, y)
    _clock(d, f)
    return img


def frame(scene, t: float) -> Image.Image:
    return frame_at(scene, int(round(t * FPS)))
