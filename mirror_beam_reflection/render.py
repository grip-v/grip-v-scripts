"""PIL renderer for mirror_beam_reflection (1280x720, 24 fps).

A board of cells with mirrors on cell diagonals, absorber blocks, emitters
(a wedge or box in the beam colour inside the entry cell, pointing in), and
the pulses: each leaves a persistent trail in its beam colour with a large
round head. One-sided mirrors carry a dark hatched backing on the absorbing
face; half-mirrors are dashed; pivoting mirrors have a hub and visibly turn
over 0.3 s just before their turn time; a pulse that is stopped leaves a
ring where it stopped.

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
TURN_S = 0.3


def duration(scene) -> float:
    return sim.duration(scene)


def n_frames(scene) -> int:
    return sim.n_frames(scene)


def clock_text(frame: int) -> str:
    return f"{sim.clock_tenths(frame) / 10:.1f} s"


def px(scene, x, y):
    st = scene["style"]
    return st["x0"] + x * st["cell"], st["y0"] + y * st["cell"]


@functools.lru_cache(maxsize=16)
def _trace(key):
    return sim.trace(sim.world_of(json.loads(key)))


def traced(scene):
    return _trace(json.dumps(scene, sort_keys=True))


def mirror_angle(m, t):
    """Screen angle (deg) of the mirror line: '/' = -45 (up to the right), '\\' = 45; pivots turn over TURN_S."""
    a0 = -45.0 if m["kind"] == "/" else 45.0
    if m["type"] != "flip" or t <= m["flip_t"] - TURN_S:
        return a0
    u = min(1.0, (t - (m["flip_t"] - TURN_S)) / TURN_S)
    return a0 + 90.0 * u * (1 if m["kind"] == "/" else -1)


def state(scene, f: int) -> dict:
    """What frame f draws: heads [(colour, x, y)] px, trails [(colour, [(x, y)...])] px, stop rings,
    mirror angles (deg, in scene order)."""
    t = f / FPS
    res = traced(scene)
    heads, trails, rings = [], [], []
    for p in res["pulses"]:
        if t < p["pts"][0][0]:
            continue
        pts = [px(scene, x, y) for tt, x, y in p["pts"] if tt <= t]
        h = sim.head_at(p, t)
        if h is not None:
            hp = px(scene, *h)
            heads.append((p["colour"], hp[0], hp[1]))
            pts.append(hp)
        trails.append((p["colour"], pts))
        if p["end"][0] == "absorbed" and t >= p["end"][2]:
            c = p["end"][1]
            rings.append((p["colour"],) + px(scene, c[0] + 0.5, c[1] + 0.5))
    return {"heads": heads, "trails": trails, "rings": rings,
            "angles": [mirror_angle(m, t) for m in scene["mirrors"]]}


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
                cx, cy = x0 + (i + 0.5) * c, y0 + (j + 0.5) * c
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
    for a in scene["absorbers"]:
        m = 0.14 * c
        d.rectangle([x0 + a[0] * c + m, y0 + a[1] * c + m, x0 + (a[0] + 1) * c - m, y0 + (a[1] + 1) * c - m],
                    fill=tuple(st["absorber"]))
    for em in scene["emitters"]:
        cell, dd = sim.edge_entry(scene, em)
        bx, by = x0 + (cell[0] + 0.5 - 0.5 * dd[0]) * c, y0 + (cell[1] + 0.5 - 0.5 * dd[1]) * c
        col = sim.BEAMS[em["colour"]]
        nx, ny = -dd[1], dd[0]
        L, w = 0.3 * c, 0.2 * c
        if st["emitter_shape"] == "wedge":
            pts = [(bx + nx * w, by + ny * w), (bx - nx * w, by - ny * w), (bx + dd[0] * L, by + dd[1] * L)]
        else:
            pts = [(bx + nx * w, by + ny * w), (bx - nx * w, by - ny * w),
                   (bx - nx * w + dd[0] * L, by - ny * w + dd[1] * L), (bx + nx * w + dd[0] * L, by + ny * w + dd[1] * L)]
        d.polygon(pts, fill=col, outline=tuple(st["ink"]))
    f = font(sim.LABEL_PX)
    for lab in scene["labels"]:
        d.text((lab["x"], lab["y"]), lab["text"], fill=tuple(st["ink"]), font=f, anchor="mm")
    return img


def _mirror(d, scene, m, ang):
    st = scene["style"]
    c = st["cell"]
    cx, cy = px(scene, m["cell"][0] + 0.5, m["cell"][1] + 0.5)
    half = st["mirror_len"] * c * math.sqrt(2) / 2
    ux, uy = math.cos(math.radians(ang)), math.sin(math.radians(ang))
    a, b = (cx - ux * half, cy - uy * half), (cx + ux * half, cy + uy * half)
    col, w = tuple(st["mirror"]), st["mirror_w"]
    if m["type"] == "one":
        # backing on the absorbing face: offset away from the silvered face normal
        fx, fy = m["face"]
        n = math.hypot(fx, fy)
        ox, oy = -fx / n * (w / 2 + 3), -fy / n * (w / 2 + 3)
        d.line([(a[0] + ox, a[1] + oy), (b[0] + ox, b[1] + oy)], fill=tuple(st["back"]), width=w + 2)
        for k in range(1, 6):
            u = k / 6
            p = (a[0] + (b[0] - a[0]) * u + ox, a[1] + (b[1] - a[1]) * u + oy)
            d.line([p, (p[0] + ox * 1.2, p[1] + oy * 1.2)], fill=tuple(st["back"]), width=2)
    if m["type"] == "split":
        _dash(d, a, b, col, w, 9, 6)
    else:
        d.line([a, b], fill=col, width=w)
    if m["type"] == "flip":
        r = 7
        d.ellipse([cx - r, cy - r, cx + r, cy + r], fill=tuple(st["ink"]), outline=col, width=2)


def _clock(d, frame):
    x0, y0, x1, y1 = CLOCK_BOX
    d.rounded_rectangle(CLOCK_BOX, radius=12, fill=(18, 18, 22), outline=(235, 235, 235), width=3)
    d.text(((x0 + x1) / 2, (y0 + y1) / 2), clock_text(frame), fill=(255, 255, 255), font=font(44), anchor="mm")


def frame_at(scene, f: int) -> Image.Image:
    img = _static(json.dumps(scene, sort_keys=True)).copy()
    d = ImageDraw.Draw(img)
    st = scene["style"]
    s = state(scene, f)
    for colour, pts in s["trails"]:
        if len(pts) >= 2:
            d.line(pts, fill=sim.BEAMS[colour], width=st["trail_w"], joint="curve")
    for m, ang in zip(scene["mirrors"], s["angles"]):
        _mirror(d, scene, m, ang)
    for colour, x, y in s["rings"]:
        r = st["head_px"] / 2 + 4
        d.ellipse([x - r, y - r, x + r, y + r], outline=sim.BEAMS[colour], width=4)
    for colour, x, y in s["heads"]:
        r = st["head_px"] / 2
        d.ellipse([x - r, y - r, x + r, y + r], fill=sim.BEAMS[colour], outline=tuple(st["ink"]), width=2)
    _clock(d, f)
    return img


def frame(scene, t: float) -> Image.Image:
    return frame_at(scene, int(round(t * FPS)))
