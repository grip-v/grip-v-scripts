"""PIL renderer for fold_punch_unfold: a gridded square sheet seen from
above, with column letters A-H and row numbers 1-8 around its original
position. Each fold first shows the crease as a dashed line, then swings
the moving flap over (its projection shrinks toward the crease, then it
lands mirrored and shows its back face). Then a punch presses down and
leaves a hole through the stack. The clip ends folded; it never unfolds.
The only text is the grid labels and the "Fold k" counter."""
from __future__ import annotations
import functools
import math
import sys
from pathlib import Path

from PIL import Image, ImageDraw
from shapely.geometry import LineString

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from common.video5 import font  # noqa: E402

import sim  # noqa: E402

BG = (236, 234, 228)
INK = (50, 52, 60)
FRONT = (120, 170, 225)
BACK = (246, 236, 205)
EDGE = (60, 80, 110)
X0, Y0, S = 400, 150, 60
INTRO_S, CREASE_S, FOLD_S, PAUSE_S, PUNCH_S, OUTRO_S = 1.0, 0.5, 1.6, 0.6, 1.3, 2.0


def duration(scene) -> float:
    return INTRO_S + len(scene["folds"]) * (CREASE_S + FOLD_S + PAUSE_S) + len(scene["punches"]) * PUNCH_S + OUTRO_S


def _px(x, y):
    return X0 + S * x, Y0 + S * y


def _apply(T, x, y):
    for f in T:
        x, y = sim.reflect(f, x, y)
    return x, y


@functools.lru_cache(maxsize=32)
def _stage(key: str, k: int):
    import json
    folds = json.loads(key)[:k]
    out = []
    for piece, T, moved in sim.facets_tagged(folds):
        grid = []
        for i in range(1, sim.N):
            for line in (LineString([(i, 0), (i, sim.N)]), LineString([(0, i), (sim.N, i)])):
                seg = line.intersection(piece)
                for g in getattr(seg, "geoms", [seg]):
                    if g.geom_type == "LineString" and g.length > 1e-6:
                        grid.append(list(g.coords))
        out.append((list(piece.exterior.coords), T, grid, moved))
    return out


def _fold_pos(f, theta, x, y):
    """Where point (x, y) on the moving side is drawn when the flap has turned by theta."""
    d = sim.side(f, x, y)
    g = {"v": (1, 0), "h": (0, 1), "d1": (1, -1), "d2": (1, 1)}[f["axis"]]
    n2 = g[0] ** 2 + g[1] ** 2
    fx, fy = x - d * g[0] / n2, y - d * g[1] / n2           # foot on the crease
    s = d * math.cos(theta)
    return fx + s * g[0] / n2, fy + s * g[1] / n2


def _shade(c, k):
    return tuple(int(v * k) for v in c)


@functools.lru_cache(maxsize=1)
def _background() -> Image.Image:
    img = Image.new("RGB", (1280, 720), BG)
    d = ImageDraw.Draw(img)
    f = font(22)
    for i, c in enumerate(sim.COLS):
        d.text((X0 + S * (i + 0.5), Y0 - 22), c, fill=INK, font=f, anchor="mm")
    for j in range(sim.N):
        d.text((X0 - 24, Y0 + S * (j + 0.5)), str(j + 1), fill=INK, font=f, anchor="mm")
    d.rectangle([X0, Y0, X0 + S * sim.N, Y0 + S * sim.N], outline=(205, 202, 194), width=1)
    return img


def _timeline(scene, t):
    """('fold', k, u, crease_only) | ('punch', i, u) | ('done',)."""
    t -= INTRO_S
    if t < 0:
        return ("fold", 0, 0.0, False)
    for k in range(len(scene["folds"])):
        if t < CREASE_S:
            return ("fold", k, 0.0, True)
        t -= CREASE_S
        if t < FOLD_S:
            u = t / FOLD_S
            return ("fold", k, u * u * (3 - 2 * u), False)
        t -= FOLD_S
        if t < PAUSE_S:
            return ("fold", k, 1.0, False)
        t -= PAUSE_S
    for i in range(len(scene["punches"])):
        if t < PUNCH_S:
            return ("punch", i, t / PUNCH_S)
        t -= PUNCH_S
    return ("done",)


def frame(scene, t: float) -> Image.Image:
    import json
    img = _background().copy()
    d = ImageDraw.Draw(img)
    folds = scene["folds"]
    key = json.dumps(folds, sort_keys=True)
    st = _timeline(scene, t)
    if st[0] == "fold":
        k, u, crease = st[1], st[2], st[3]
        done = k + 1 if u > 0 else k
    else:
        k, u, crease, done = len(folds), 1.0, False, len(folds)
    facets = _stage(key, done)
    moving = st[0] == "fold" and u > 0
    f = folds[k] if moving else None
    theta = math.pi * u
    draw_list = [(piece, T, grid, moving and moved) for piece, T, grid, moved in facets]
    stay = [x for x in draw_list if not x[3]]
    mov = [x for x in draw_list if x[3]]
    if moving and theta < math.pi / 2:
        mov = mov[::-1]
    for piece, T, grid, mv in stay + mov:
        if mv:
            T0 = T[:-1]
            place = lambda x, y, T0=T0: _fold_pos(f, theta, *_apply(T0, x, y))
            flips = len(T0) + (1 if theta > math.pi / 2 else 0)
            light = 0.78 + 0.22 * abs(math.cos(theta))
        else:
            place = lambda x, y, T=T: _apply(T, x, y)
            flips = len(T)
            light = 1.0
        col = _shade(FRONT if flips % 2 == 0 else BACK, light)
        pts = [_px(*place(x, y)) for x, y in piece]
        d.polygon(pts, fill=col)
        line = _shade(col, 0.82)
        for seg in grid:
            d.line([_px(*place(x, y)) for x, y in seg], fill=line, width=1)
        d.line(pts + [pts[0]], fill=_shade(EDGE, light), width=2, joint="curve")
    if st[0] == "fold" and crease:
        _dashed(d, folds[k])
    # holes punched so far, and the punch tool
    n_done = len(scene["punches"]) if st[0] == "done" else (st[1] if st[0] == "punch" else 0)
    for i, (px, py) in enumerate(scene["punches"]):
        cx, cy = _px(px, py)
        if i < n_done or (st[0] == "punch" and i == st[1] and st[2] >= 0.5):
            d.ellipse([cx - 13, cy - 13, cx + 13, cy + 13], fill=(70, 60, 55), outline=(30, 30, 30), width=2)
        if st[0] == "punch" and i == st[1]:
            v = st[2]
            r = 13 + 40 * abs(v - 0.5) * 2
            a = 0.35 + 0.65 * (1 - abs(v - 0.5) * 2)
            ring = tuple(int(BG[j] * (1 - a) + 40 * a) for j in range(3))
            d.ellipse([cx - r, cy - r, cx + r, cy + r], outline=ring, width=5)
    shown = min(len(folds), k + 1) if st[0] == "fold" and (u > 0 or crease) else (len(folds) if st[0] != "fold"
                                                                                    else 0)
    if shown:
        d.text((60, 60), f"Fold {shown}", fill=INK, font=font(30))
    return img


def _dashed(d, f):
    c = f["c"]
    a, b = {"v": ((c, -0.4), (c, 8.4)), "h": ((-0.4, c), (8.4, c)), "d1": ((c - 0.4, -0.4), (c + 8.4, 8.4)),
            "d2": ((c + 0.4, -0.4), (c - 8.4, 8.4))}[f["axis"]]
    (ax, ay), (bx, by) = _px(*a), _px(*b)
    L = math.hypot(bx - ax, by - ay)
    for s in range(0, int(L), 18):
        e = min(L, s + 10)
        d.line([(ax + (bx - ax) * s / L, ay + (by - ay) * s / L), (ax + (bx - ax) * e / L, ay + (by - ay) * e / L)],
               fill=(200, 60, 60), width=3)
