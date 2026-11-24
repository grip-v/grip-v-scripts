"""PIL renderer for blinking_networks.

Static layer (cached per scene): themed background texture, links (solid,
twin-rail, or dashed for slow links in marked_speeds), arrowheads on one-way
links, and the node sprites (colour + shape + pattern; a double outline for
nodes that need two pulses). Per frame: moving pulse dots, a brief burst when
a node fires, a persistent halo ring around every lit node, a persistent half
halo on a double-outline node that has received one pulse, and a fading ring
where a pulse fizzles. The clock in the top-right box is the only text.
"""
from __future__ import annotations
import json
import math
import random
import sys
from pathlib import Path

from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from common.video5 import font  # noqa: E402

import sim  # noqa: E402

W, H = sim.W, sim.H
HALO_IN, HALO_OUT = 8, 14          # halo ring radii beyond R
BURST = 0.5                        # s, decorative firing burst
FIZZLE = 0.35                      # s, decorative fizzle ring
PULSE_R = 10                       # pulse dot radius (20 px across)
_CACHE: dict = {}


def duration(scene) -> float:
    return scene["end_t"]


def shape_points(shape, cx, cy, r):
    if shape == "circle":
        return None
    if shape == "square":
        h = 0.88 * r
        return [(cx - h, cy - h), (cx + h, cy - h), (cx + h, cy + h), (cx - h, cy + h)]
    if shape == "diamond":
        h = 1.12 * r
        return [(cx, cy - h), (cx + h, cy), (cx, cy + h), (cx - h, cy)]
    if shape == "triangle":
        h = 1.15 * r
        return [(cx + h * math.cos(math.radians(a)), cy + 0.12 * r + h * math.sin(math.radians(a)))
                for a in (-90, 30, 150)]
    if shape == "star":
        pts = []
        for i in range(10):
            rr = 1.15 * r if i % 2 == 0 else 0.55 * r
            a = math.radians(-90 + 36 * i)
            pts.append((cx + rr * math.cos(a), cy + rr * math.sin(a)))
        return pts
    raise ValueError(shape)


def draw_shape(d, shape, cx, cy, r, **kw):
    pts = shape_points(shape, cx, cy, r)
    if pts is None:
        d.ellipse([cx - r, cy - r, cx + r, cy + r], **kw)
    else:
        d.polygon(pts, **kw)


def shape_mask(shape, r, size):
    m = Image.new("L", (size, size), 0)
    draw_shape(ImageDraw.Draw(m), shape, size / 2, size / 2, r, fill=255)
    return m


def _sprite(node, R, theme):
    size = 4 * R
    c = size / 2
    rgb = tuple(node["rgb"])
    dark = tuple(int(v * 0.5) for v in rgb)
    body = Image.new("RGB", (size, size), rgb)
    pd = ImageDraw.Draw(body)
    pat = node["pattern"]
    if pat == "striped":
        for k in range(-size, 2 * size, 12):
            pd.line([(k, 0), (k + size, size)], fill=dark, width=6)
    elif pat == "dotted":
        for yy in range(int(c) % 12 - 12, size, 12):
            for xx in range(int(c) % 12 - 12, size, 12):
                pd.ellipse([xx - 3, yy - 3, xx + 3, yy + 3], fill=dark)
    elif pat == "ringed":
        draw_shape(pd, node["shape"], c, c, 0.62 * R, fill=dark)
        draw_shape(pd, node["shape"], c, c, 0.36 * R, fill=rgb)
    mask = shape_mask(node["shape"], R, size)
    out = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    out.paste(body, (0, 0), mask)
    od = ImageDraw.Draw(out)
    edge = (14, 14, 18)
    draw_shape(od, node["shape"], c, c, R, outline=edge, width=2)
    if node["threshold"] == 2:
        ring = tuple(theme["halo"])
        draw_shape(od, node["shape"], c, c, R - 1, outline=ring, width=3)
        draw_shape(od, node["shape"], c, c, 0.5 * R, outline=ring, width=2)
    return out


def _dashed(d, p, q, w, col, on=16, off=11):
    L = math.dist(p, q)
    ux, uy = (q[0] - p[0]) / L, (q[1] - p[1]) / L
    s = 0.0
    while s < L:
        e = min(L, s + on)
        d.line([(p[0] + ux * s, p[1] + uy * s), (p[0] + ux * e, p[1] + uy * e)], fill=col, width=w)
        s = e + off


def _background(theme):
    bg = tuple(theme["bg"])
    img = Image.new("RGB", (W, H), bg)
    d = ImageDraw.Draw(img)
    r = random.Random(theme["texture_seed"])
    sgn = 1 if theme["dark"] else -1
    shade = lambda k: tuple(max(0, min(255, v + sgn * k)) for v in bg)  # noqa: E731
    st = theme["style"]
    if st == "gradient":
        for y in range(H):
            d.line([(0, y), (W, y)], fill=shade(int(18 * y / H)))
    elif st == "grid":
        step = r.randint(56, 96)
        for x in range(0, W, step):
            d.line([(x, 0), (x, H)], fill=shade(10), width=1)
        for y in range(0, H, step):
            d.line([(0, y), (W, y)], fill=shade(10), width=1)
    elif st == "dots":
        step = r.randint(28, 48)
        for x in range(step // 2, W, step):
            for y in range(step // 2, H, step):
                d.ellipse([x - 1.5, y - 1.5, x + 1.5, y + 1.5], fill=shade(16))
    elif st == "hatch":
        step = r.randint(22, 40)
        for k in range(-H, W, step):
            d.line([(k, 0), (k + H, H)], fill=shade(7), width=1)
    elif st == "vignette":
        for i in range(24):
            rr = 900 - i * 22
            d.ellipse([W / 2 - rr * 1.25, H / 2 - rr * 0.85, W / 2 + rr * 1.25, H / 2 + rr * 0.85],
                      fill=shade(int(i * 0.8)))
    return img


def _static(scene):
    key = json.dumps([scene["theme"], scene["nodes"], scene["edges"], scene["R"], scene["scenario"]],
                     sort_keys=True)
    if key in _CACHE:
        return _CACHE[key]
    th = scene["theme"]
    img = _background(th)
    d = ImageDraw.Draw(img)
    pos = {n["id"]: (n["x"], n["y"]) for n in scene["nodes"]}
    col, w = tuple(th["link"]), th["link_w"]
    for e in scene["edges"]:
        p, q = pos[e["a"]], pos[e["b"]]
        if scene["scenario"] == "marked_speeds" and e["slow"]:
            _dashed(d, p, q, w, col)
        elif th["rail"]:
            L = math.dist(p, q)
            nx, ny = -(q[1] - p[1]) / L, (q[0] - p[0]) / L
            off = w / 2 + 2.5
            for s in (-1, 1):
                d.line([(p[0] + s * off * nx, p[1] + s * off * ny), (q[0] + s * off * nx, q[1] + s * off * ny)],
                       fill=col, width=max(3, w - 1))
        else:
            d.line([p, q], fill=col, width=w)
        if e["oneway"]:
            L = math.dist(p, q)
            ux, uy = (q[0] - p[0]) / L, (q[1] - p[1]) / L
            mx, my = (p[0] + q[0]) / 2 + ux * 12, (p[1] + q[1]) / 2 + uy * 12
            hw = 13 + w / 2
            d.polygon([(mx, my), (mx - ux * 28 - uy * hw, my - uy * 28 + ux * hw),
                       (mx - ux * 28 + uy * hw, my - uy * 28 - ux * hw)], fill=col)
    R = scene["R"]
    nodes = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    for n in scene["nodes"]:
        sp = _sprite(n, R, th)
        nodes.alpha_composite(sp, (int(round(n["x"] - 2 * R)), int(round(n["y"] - 2 * R))))
    if len(_CACHE) > 6:
        _CACHE.clear()
    _CACHE[key] = (img, nodes)
    return img, nodes


def draw_timer(img, t):
    d = ImageDraw.Draw(img)
    x0, y0, x1, y1 = sim.TIMER_BOX
    d.rounded_rectangle([x0, y0, x1, y1], radius=10, fill=(10, 10, 14), outline=(110, 110, 124), width=2)
    txt = f"{t:.1f} s"
    f = font(42)
    tw = d.textlength(txt, font=f)
    d.text((x1 - 16 - tw, y0 + 10), txt, fill=(255, 255, 255), font=f)


def frame(scene, t: float) -> Image.Image:
    base, nodes = _static(scene)
    img = base.copy()
    d = ImageDraw.Draw(img)
    th, R = scene["theme"], scene["R"]
    halo, pcol = tuple(th["halo"]), tuple(th["pulse"])
    bg = tuple(th["bg"])
    pos = {n["id"]: (n["x"], n["y"]) for n in scene["nodes"]}
    for p in scene["pulses"]:                       # fizzles (decorative, short)
        if p["result"] == "fizzle" and p["t1"] <= t < p["t1"] + FIZZLE:
            a, b = pos[p["from"]], pos[p["to"]]
            L = math.dist(a, b)
            f = (t - p["t1"]) / FIZZLE
            x, y = b[0] - (b[0] - a[0]) / L * (R + 22), b[1] - (b[1] - a[1]) / L * (R + 22)
            rr = 8 + 10 * f
            c = tuple(int(pcol[i] * (1 - f) + bg[i] * f) for i in range(3))
            d.ellipse([x - rr, y - rr, x + rr, y + rr], outline=c, width=3)
    for nid, (x, y) in pos.items():
        tf = scene["fire"].get(nid)
        th_half = scene["half"].get(nid)
        ri, ro = R + HALO_IN, R + HALO_OUT
        if tf is not None and t >= tf:
            f = (t - tf) / BURST
            if f < 1:
                rb = ro + 4 + 22 * f
                c = tuple(int(halo[i] * (1 - f) + bg[i] * f) for i in range(3))
                d.ellipse([x - rb, y - rb, x + rb, y + rb], outline=c, width=4)
            d.ellipse([x - ro, y - ro, x + ro, y + ro], outline=halo, width=ro - ri)
        elif th_half is not None and t >= th_half:
            d.arc([x - ro, y - ro, x + ro, y + ro], 180, 360, fill=halo, width=ro - ri)
    for p in scene["pulses"]:
        if p["t0"] <= t < p["t1"]:
            a, b = pos[p["from"]], pos[p["to"]]
            f = (t - p["t0"]) / (p["t1"] - p["t0"])
            x, y = a[0] + (b[0] - a[0]) * f, a[1] + (b[1] - a[1]) * f
            edge = (10, 10, 14) if th["dark"] else (250, 250, 250)
            d.ellipse([x - PULSE_R - 2, y - PULSE_R - 2, x + PULSE_R + 2, y + PULSE_R + 2], fill=edge)
            d.ellipse([x - PULSE_R, y - PULSE_R, x + PULSE_R, y + PULSE_R], fill=pcol)
    img.paste(nodes, (0, 0), nodes)
    draw_timer(img, t)
    return img
