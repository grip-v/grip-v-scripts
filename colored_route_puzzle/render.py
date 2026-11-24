"""PIL renderer for colored_route_puzzle (1280x720, 24 fps).

The static layer (background colour + pattern, parks) is drawn once per
scene. Every frame then draws, in this order: each route's drawn part (it
grows from its anchor end at a constant rate over route["draw"], and in the
erase scenario shrinks back to the anchor over route["erase"]), dead-end
crossbars, the stop markers (dark shapes with a thin background-coloured
rim), the black ring (traveller scenario) and the clock.

The only text is the clock: a fixed dark panel top-left, "s.t s"
truncated to 0.1 s. Stops are identified by shape, routes by colour.

`track(scene, i)` is the renderer's own per-frame state (the drawn polyline
of every route, the ring centre); answers.py derives every answer from it.
"""
from __future__ import annotations
import functools
import json
import math
import sys
from pathlib import Path

from PIL import Image, ImageDraw, ImageFilter

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE))
from common.video5 import font  # noqa: E402

import sim  # noqa: E402

W, H, FPS = sim.W, sim.H, sim.FPS
TIMER_BG, TIMER_FG, TIMER_EDGE = (26, 28, 34), (255, 255, 255), (200, 200, 205)
SS = 4  # supersampling for sprites


def duration(scene) -> float:
    return scene["n_frames"] / FPS


def timer_text(i: int) -> str:
    """The clock text drawn on frame i (truncated to 0.1 s)."""
    t = (10 * i) // FPS
    return f"{t // 10}.{t % 10} s"


# ---------------------------------------------------------------- geometry

def node_px(scene, n):
    g = scene["grid"]
    return (float(g["x0"] + n[0] * g["step"]), float(g["y0"] + n[1] * g["step"]))


def full_path(scene, r) -> list:
    """Corner polyline of a route in pixels, from its `a` end to its `b` end."""
    nodes = [tuple(n) for n in r["nodes"]]
    out = [nodes[0]]
    for k in range(1, len(nodes) - 1):
        a, b, c = nodes[k - 1], nodes[k], nodes[k + 1]
        if (b[0] - a[0], b[1] - a[1]) != (c[0] - b[0], c[1] - b[1]):
            out.append(b)
    out.append(nodes[-1])
    return [node_px(scene, n) for n in out]


def stop_marks(scene) -> list:
    mh = scene["style"]["mh"]
    return [{"shape": s["shape"], "x": node_px(scene, s["node"])[0], "y": node_px(scene, s["node"])[1], "mh": mh}
            for s in scene["stops"]]


def _cum(P):
    c = [0.0]
    for (x1, y1), (x2, y2) in zip(P, P[1:]):
        c.append(c[-1] + math.hypot(x2 - x1, y2 - y1))
    return c


def cut(P, cum, a, b) -> list:
    """Sub-polyline of P between arc lengths a <= b."""
    if b <= a:
        return []
    out = []

    def at(s):
        for k in range(len(P) - 1):
            if cum[k] <= s <= cum[k + 1] + 1e-9:
                L = cum[k + 1] - cum[k]
                u = 0.0 if L == 0 else (s - cum[k]) / L
                return (P[k][0] + (P[k + 1][0] - P[k][0]) * u, P[k][1] + (P[k + 1][1] - P[k][1]) * u)
        return P[-1]
    out.append(at(a))
    for k in range(1, len(P) - 1):
        if a < cum[k] < b:
            out.append(P[k])
    out.append(at(b))
    return out


def progress(r, i: int) -> float:
    """Drawn fraction of route r on frame i."""
    s, e = r["draw"]
    u = min(1.0, max(0.0, (i - s) / (e - s)))
    if r["erase"] and i > r["erase"][0]:
        g0, g1 = r["erase"]
        u = 1.0 - min(1.0, (i - g0) / (g1 - g0))
    return u


def ease(u: float) -> float:
    return u * u * (3 - 2 * u)


def token_xy(scene, i: int):
    tk = scene["token"]
    if not tk:
        return None
    cur = tk["start"]
    for rd in tk["rides"]:
        t0, t1 = rd["t"]
        if i <= t0:
            break
        if i >= t1:
            cur = rd["to"]
            continue
        P = full_path(scene, scene["routes"][rd["route"]])
        if rd["from"] != scene["routes"][rd["route"]]["a"]:
            P = P[::-1]
        cum = _cum(P)
        s = cum[-1] * ease((i - t0) / (t1 - t0))
        return cut(P, cum, 0.0, max(s, 1e-6))[-1]
    return node_px(scene, scene["stops"][cur]["node"])


def track(scene, i: int) -> dict:
    """The renderer's state on frame i: for every route its drawn polyline
    (from the end it grows from) and the ring centre (None if no ring)."""
    out = []
    for r in scene["routes"]:
        P = full_path(scene, r)
        if r["grow"] == "b":
            P = P[::-1]
        cum = _cum(P)
        u = progress(r, i)
        out.append({"color": r["color"], "rgb": tuple(r["rgb"]), "style": r["style"],
                    "pts": cut(P, cum, 0.0, cum[-1] * u) if u > 0 else []})
    return {"routes": out, "token": token_xy(scene, i)}


# ---------------------------------------------------------------- drawing paths

def rounded(P, radius) -> list:
    """Corner polyline with every interior corner replaced by a quarter arc."""
    if radius <= 0 or len(P) < 3:
        return list(P)
    out = [P[0]]
    for k in range(1, len(P) - 1):
        a, b, c = P[k - 1], P[k], P[k + 1]
        d1 = ((b[0] - a[0]) / max(1e-9, math.hypot(b[0] - a[0], b[1] - a[1])),
              (b[1] - a[1]) / max(1e-9, math.hypot(b[0] - a[0], b[1] - a[1])))
        d2 = ((c[0] - b[0]) / max(1e-9, math.hypot(c[0] - b[0], c[1] - b[1])),
              (c[1] - b[1]) / max(1e-9, math.hypot(c[0] - b[0], c[1] - b[1])))
        r = min(radius, math.hypot(b[0] - a[0], b[1] - a[1]) / 2, math.hypot(c[0] - b[0], c[1] - b[1]) / 2)
        ox, oy = b[0] - d1[0] * r + d2[0] * r, b[1] - d1[1] * r + d2[1] * r
        for j in range(9):
            th = (math.pi / 2) * j / 8
            out.append((ox - d2[0] * r * math.cos(th) + d1[0] * r * math.sin(th),
                        oy - d2[1] * r * math.cos(th) + d1[1] * r * math.sin(th)))
    out.append(P[-1])
    return out


@functools.lru_cache(maxsize=512)
def _draw_path(key: str):
    """(points a->b with rounded corners, cumulative lengths) for a route; key = json of route + grid + style."""
    scene_bits = json.loads(key)
    P = full_path({"grid": scene_bits["grid"]}, scene_bits["route"])
    rad = 0.22 * scene_bits["grid"]["step"] if scene_bits["corner"] == "round" else 0.0
    R = rounded(P, rad)
    return R, _cum(R)


def draw_path(scene, r):
    key = json.dumps({"grid": scene["grid"], "route": {"nodes": r["nodes"]}, "corner": scene["style"]["corner"]},
                     sort_keys=True)
    return _draw_path(key)


def drawn_interval(scene, r, i):
    """Drawn arc-length interval [lo, hi] of the rounded a->b path on frame i (None if nothing)."""
    R, cum = draw_path(scene, r)
    L = cum[-1]
    u = progress(r, i)
    if u <= 0:
        return None
    return (0.0, L * u) if r["grow"] == "a" else (L * (1 - u), L)


def _seg(d, pts, rgb, w):
    if len(pts) < 2:
        return
    d.line([(round(x, 2), round(y, 2)) for x, y in pts], fill=rgb, width=w, joint="curve")


def draw_route(d, scene, r, i):
    iv = drawn_interval(scene, r, i)
    if iv is None:
        return
    st = scene["style"]
    w, rgb = st["w"], tuple(r["rgb"])
    R, cum = draw_path(scene, r)
    lo, hi = iv
    if r["style"] == "solid":
        pts = cut(R, cum, lo, hi)
        _seg(d, pts, rgb, w)
        for x, y in (pts[0], pts[-1]):
            d.ellipse([x - w / 2, y - w / 2, x + w / 2, y + w / 2], fill=rgb)
    elif r["style"] == "dashed":
        on, off = st["dash"]
        k = 0
        while k * (on + off) < hi:
            a, b = max(lo, k * (on + off)), min(hi, k * (on + off) + on)
            if b > a:
                _seg(d, cut(R, cum, a, b), rgb, w)
            k += 1
    else:
        period, rad = st["dot"]
        k = 0
        while k * period + period / 2 <= hi:
            s = k * period + period / 2
            if s >= lo:
                x, y = cut(R, cum, max(0.0, s - 0.01), s)[-1]
                d.ellipse([x - rad, y - rad, x + rad, y + rad], fill=rgb)
            k += 1
    if r["b"] is None and hi >= cum[-1] - 1e-6:
        (x1, y1), (x2, y2) = R[-2], R[-1]
        L = max(1e-9, math.hypot(x2 - x1, y2 - y1))
        nx, ny = -(y2 - y1) / L, (x2 - x1) / L
        half = 1.7 * w
        d.line([(x2 - nx * half, y2 - ny * half), (x2 + nx * half, y2 + ny * half)], fill=rgb, width=max(5, int(0.8 * w)))


# ---------------------------------------------------------------- sprites

def shape_poly(shape: str, cx: float, cy: float, r: float):
    """('ellipse', box) or ('poly', points) of a stop marker of half-size r."""
    if shape == "circle":
        return "ellipse", [cx - r, cy - r, cx + r, cy + r]
    if shape == "square":
        h = 0.86 * r
        return "poly", [(cx - h, cy - h), (cx + h, cy - h), (cx + h, cy + h), (cx - h, cy + h)]
    if shape == "triangle":
        R = 1.15 * r
        return "poly", [(cx + R * math.cos(math.radians(a)), cy + 0.12 * r + R * math.sin(math.radians(a)))
                        for a in (-90, 30, 150)]
    if shape == "diamond":
        R = 1.12 * r
        return "poly", [(cx, cy - R), (cx + R, cy), (cx, cy + R), (cx - R, cy)]
    if shape == "star":
        pts = []
        for k in range(10):
            R = 1.2 * r if k % 2 == 0 else 0.5 * r
            a = math.radians(-90 + 36 * k)
            pts.append((cx + R * math.cos(a), cy + 0.06 * r + R * math.sin(a)))
        return "poly", pts
    if shape == "cross":
        a, b = 0.36 * r, 1.0 * r
        return "poly", [(cx - a, cy - b), (cx + a, cy - b), (cx + a, cy - a), (cx + b, cy - a), (cx + b, cy + a),
                        (cx + a, cy + a), (cx + a, cy + b), (cx - a, cy + b), (cx - a, cy + a), (cx - b, cy + a),
                        (cx - b, cy - a), (cx - a, cy - a)]
    raise ValueError(shape)


def _fill_shape(d, shape, cx, cy, r, fill):
    kind, geo = shape_poly(shape, cx, cy, r)
    if kind == "ellipse":
        d.ellipse(geo, fill=fill)
    else:
        d.polygon(geo, fill=fill)


@functools.lru_cache(maxsize=128)
def marker_sprite(shape: str, mh: int, ink: tuple, rim: tuple) -> Image.Image:
    size = int(2 * (1.3 * mh + 4))
    big = Image.new("RGBA", (size * SS, size * SS), (0, 0, 0, 0))
    d = ImageDraw.Draw(big)
    c = size * SS / 2
    _fill_shape(d, shape, c, c, (mh + 2.5) * SS, rim + (255,))
    _fill_shape(d, shape, c, c, mh * SS, ink + (255,))
    return big.resize((size, size), Image.LANCZOS)


@functools.lru_cache(maxsize=32)
def ring_sprite(R: int, wid: int) -> Image.Image:
    size = 2 * R + 8
    big = Image.new("RGBA", (size * SS, size * SS), (0, 0, 0, 0))
    d = ImageDraw.Draw(big)
    c = size * SS / 2
    halo = 2
    d.ellipse([c - (R + halo) * SS, c - (R + halo) * SS, c + (R + halo) * SS, c + (R + halo) * SS],
              fill=(255, 255, 255, 255))
    d.ellipse([c - R * SS, c - R * SS, c + R * SS, c + R * SS], fill=sim.TOKEN_RGB + (255,))
    d.ellipse([c - (R - wid) * SS, c - (R - wid) * SS, c + (R - wid) * SS, c + (R - wid) * SS],
              fill=(255, 255, 255, 255))
    d.ellipse([c - (R - wid - halo) * SS, c - (R - wid - halo) * SS, c + (R - wid - halo) * SS,
               c + (R - wid - halo) * SS], fill=(0, 0, 0, 0))
    return big.resize((size, size), Image.LANCZOS)


@functools.lru_cache(maxsize=32)
def _shadow(shape: str, mh: int) -> Image.Image:
    size = int(2 * (1.3 * mh + 4))
    im = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    _fill_shape(ImageDraw.Draw(im), shape, size / 2, size / 2, mh, (0, 0, 0, 60))
    return im.filter(ImageFilter.GaussianBlur(2.5))


# ---------------------------------------------------------------- layers

@functools.lru_cache(maxsize=4)
def _static(key: str) -> Image.Image:
    sc = json.loads(key)
    st, g = sc["style"], sc["grid"]
    bg, ink = tuple(st["bg"]), tuple(st["ink"])
    img = Image.new("RGB", (W, H), bg)
    d = ImageDraw.Draw(img)
    faint = sim.mix(bg, ink, 0.09)
    gx0, gy0 = g["x0"], g["y0"]
    gx1, gy1 = gx0 + (g["nx"] - 1) * g["step"], gy0 + (g["ny"] - 1) * g["step"]
    if st["pattern"] == "gradient":
        for y in range(H):
            d.line([(0, y), (W, y)], fill=sim.mix(bg, ink, 0.10 * y / (H - 1)))
    elif st["pattern"] == "grid":
        for i in range(g["nx"]):
            x = gx0 + i * g["step"]
            d.line([(x, gy0), (x, gy1)], fill=faint, width=1)
        for j in range(g["ny"]):
            y = gy0 + j * g["step"]
            d.line([(gx0, y), (gx1, y)], fill=faint, width=1)
    elif st["pattern"] == "dots":
        for i in range(g["nx"]):
            for j in range(g["ny"]):
                x, y = gx0 + i * g["step"], gy0 + j * g["step"]
                d.ellipse([x - 2, y - 2, x + 2, y + 2], fill=faint)
    tint = sim.mix(bg, tuple(st["tint"]), st["tint_u"])
    for o in sc["obstacles"]:
        s = g["step"]
        d.rounded_rectangle([gx0 + o[0] * s - 0.42 * s, gy0 + o[1] * s - 0.42 * s,
                             gx0 + o[2] * s + 0.42 * s, gy0 + o[3] * s + 0.42 * s], radius=int(0.3 * s), fill=tint)
    tb = sc["timer"]["box"]
    d.rectangle([tb[0] - 6, tb[1] - 6, tb[2] + 6, tb[3] + 6], fill=bg)
    return img


def draw_timer(d, scene, i: int):
    tb = scene["timer"]["box"]
    d.rounded_rectangle(tb, radius=10, fill=TIMER_BG, outline=TIMER_EDGE, width=2)
    d.text(((tb[0] + tb[2]) / 2, (tb[1] + tb[3]) / 2), timer_text(i), fill=TIMER_FG,
           font=font(scene["timer"]["size"]), anchor="mm")


def frame_at(scene, i: int) -> Image.Image:
    st = scene["style"]
    key = json.dumps({"style": st, "grid": scene["grid"], "obstacles": scene["obstacles"],
                      "timer": scene["timer"]}, sort_keys=True)
    img = _static(key).copy()
    d = ImageDraw.Draw(img)
    for r in scene["routes"]:
        draw_route(d, scene, r, i)
    mh = st["mh"]
    for m in stop_marks(scene):
        spr = marker_sprite(m["shape"], mh, tuple(st["ink"]), tuple(st["bg"]))
        x, y = int(round(m["x"] - spr.size[0] / 2)), int(round(m["y"] - spr.size[1] / 2))
        if st["shadow"]:
            sh = _shadow(m["shape"], mh)
            img.paste(sh, (x + 2, y + 3), sh)
        img.paste(spr, (x, y), spr)
    tp = token_xy(scene, i)
    if tp is not None:
        spr = ring_sprite(st["token_r"], st["token_w"])
        img.paste(spr, (int(round(tp[0] - spr.size[0] / 2)), int(round(tp[1] - spr.size[1] / 2))), spr)
    draw_timer(ImageDraw.Draw(img), scene, i)
    return img


def frame(scene, t: float) -> Image.Image:
    return frame_at(scene, int(round(t * FPS)))
