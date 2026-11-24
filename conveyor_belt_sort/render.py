"""PIL renderer for conveyor_belt_sort (1280x720, 24 fps).

`track(scene)` is what gets drawn: every object's drawn look (colour, shape,
pattern, radius) and its centre on every frame (NaN while it is wholly
inside the tunnel), plus the fixed geometry (tunnel mouth, belt, gate,
chutes, trays and their slots). `frame_at` draws exactly that; answers.py
and frame_check.py read the track, never the rule or the keyframes.

Text on screen: the clock only (fixed panel, top-left).
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
INK = (24, 24, 30)


def duration(scene) -> float:
    return scene["n_frames"] / sim.FPS


@functools.lru_cache(maxsize=8)
def _track(key: str) -> dict:
    scene = json.loads(key)
    n = scene["n_frames"]
    objs = scene["objects"]
    P = np.full((n, len(objs), 2), np.nan)
    d = scene["dir"]
    for j, o in enumerate(objs):
        e = sim.ext(o)
        for f in range(n):
            x, y = sim.pos(o["keys"], f)
            if (x - scene["mouth"]) * d > -e:
                P[f, j] = (x, y)
    return {
        "objects": [{"colour": o["colour"], "rgb": list(sim.COLOURS[o["colour"]]), "shape": o["shape"],
                     "pattern": o["pattern"], "r": o["r"]} for o in objs],
        "pos": P, "n_frames": n, "dir": d, "mouth": scene["mouth"], "tunnel": list(scene["tunnel"]),
        "belt_y": scene["belt_y"], "half": scene["half"], "gate": list(scene["gate"]),
        "bins": {b: {"start": list(v["start"]), "slots": [list(s) for s in v["slots"]], "rect": list(v["rect"])}
                 for b, v in scene["bins"].items()},
        "labels": [dict(lb) for lb in scene["labels"]],
    }


def track(scene) -> dict:
    return _track(json.dumps(scene, sort_keys=True))


def _mix(a, b, u):
    return tuple(int(round(x + (y - x) * u)) for x, y in zip(a, b))


def _poly(shape, r):
    """Outline points of a shape of size r around (0, 0)."""
    if shape == "square":
        h = 0.82 * r
        return [(-h, -h), (h, -h), (h, h), (-h, h)]
    if shape == "triangle":
        R = 1.2 * r
        return [(R * math.sin(a), -R * math.cos(a)) for a in (0, 2 * math.pi / 3, 4 * math.pi / 3)]
    if shape == "star":
        pts = []
        for k in range(10):
            R = 1.2 * r if k % 2 == 0 else 0.55 * r
            a = k * math.pi / 5
            pts.append((R * math.sin(a), -R * math.cos(a)))
        return pts
    return None


@functools.lru_cache(maxsize=64)
def sprite(colour: str, shape: str, pattern: str, r: float, outline: int) -> Image.Image:
    """RGBA sprite at SS scale, centred in a square of side 2 * (EXT * r + 4) * SS."""
    half = int(math.ceil((sim.EXT * r + 4) * SS))
    S = 2 * half
    c = (half, half)
    rgb = sim.COLOURS[colour]
    R = r * SS
    mask = Image.new("L", (S, S), 0)
    dm = ImageDraw.Draw(mask)
    pts = _poly(shape, R)
    if pts is None:
        dm.ellipse([c[0] - R, c[1] - R, c[0] + R, c[1] + R], fill=255)
    else:
        dm.polygon([(c[0] + x, c[1] + y) for x, y in pts], fill=255)
    body = Image.new("RGBA", (S, S), rgb + (255,))
    if pattern != "plain":
        pat = Image.new("L", (S, S), 0)
        dp = ImageDraw.Draw(pat)
        dark = (255, 255, 255) if sum(rgb) < 420 else (40, 40, 46)
        if pattern == "striped":
            w = 0.24 * R
            for off in (-0.62 * R, 0.0, 0.62 * R):
                # a band at 45 degrees through (c + off * (1, 1) / sqrt 2)
                ox, oy = c[0] + off / math.sqrt(2), c[1] + off / math.sqrt(2)
                L = 2 * R
                dp.line([(ox - L, oy + L), (ox + L, oy - L)], fill=255, width=max(2, int(round(w))))
            dark = INK
        else:
            rd = 0.34 * R                        # one round spot in the middle (fits inside every shape)
            dp.ellipse([c[0] - rd, c[1] - rd, c[0] + rd, c[1] + rd], fill=255)
        pm = np.minimum(np.asarray(pat), np.asarray(mask))
        layer = Image.new("RGBA", (S, S), dark + (255,))
        body.paste(layer, (0, 0), Image.fromarray(pm))
    out = Image.new("RGBA", (S, S), (0, 0, 0, 0))
    out.paste(body, (0, 0), mask)
    d = ImageDraw.Draw(out)
    w = outline * SS
    if pts is None:
        d.ellipse([c[0] - R, c[1] - R, c[0] + R, c[1] + R], outline=INK + (255,), width=w)
    else:
        P = [(c[0] + x, c[1] + y) for x, y in pts]
        d.line(P + [P[0], P[1]], fill=INK + (255,), width=w, joint="curve")
    return out


def _bg(scene, d):
    st = scene["style"]
    top, bot = (tuple(c) for c in st["bg"])
    if st["bg_style"] == "gradient":
        for y in range(sim.H):
            d.line([(0, y), (sim.W, y)], fill=_mix(top, bot, y / (sim.H - 1)))
    elif st["bg_style"] == "vignette":
        for k in range(24, 0, -1):
            u = k / 24
            d.ellipse([sim.W / 2 - 900 * u, sim.H / 2 - 620 * u, sim.W / 2 + 900 * u, sim.H / 2 + 620 * u],
                      fill=_mix(top, bot, u))


@functools.lru_cache(maxsize=8)
def _static(key: str) -> Image.Image:
    """Background, chutes, trays, gate and the clock panel (SS scale)."""
    scene = json.loads(key)
    st = scene["style"]
    top = tuple(st["bg"][0])
    img = Image.new("RGB", (sim.W, sim.H), top)
    d = ImageDraw.Draw(img)
    _bg(scene, d)
    img = img.resize((sim.W * SS, sim.H * SS), Image.BILINEAR)
    d = ImageDraw.Draw(img)
    S = lambda *p: [v * SS for v in p]
    metal = tuple(st["metal"])
    belt = tuple(st["belt"])
    h = scene["half"]
    gx, gy = scene["gate"]
    rail = st["rail"] * SS
    floor = _mix(top, metal, st["tray_tint"])
    # chutes: a band from the gate to the tray, rails on both sides
    for b, bn in scene["bins"].items():
        sx, sy = bn["start"]
        L = math.hypot(sx - gx, sy - gy)
        nx, ny = -(sy - gy) / L, (sx - gx) / L
        quad = [(gx + nx * h, gy + ny * h), (sx + nx * h, sy + ny * h), (sx - nx * h, sy - ny * h),
                (gx - nx * h, gy - ny * h)]
        d.polygon([(x * SS, y * SS) for x, y in quad], fill=floor)
        for s in (1, -1):
            d.line(S(gx + s * nx * h, gy + s * ny * h, sx + s * nx * h, sy + s * ny * h), fill=metal, width=rail)
    # trays with slot ticks
    for b, bn in scene["bins"].items():
        x0, y0, x1, y1 = bn["rect"]
        d.rectangle(S(x0, y0, x1, y1), fill=floor)
        d.line(S(x0, y0, x1, y0), fill=metal, width=rail)
        d.line(S(x0, y1, x1, y1), fill=metal, width=rail)
        far = x1 if scene["dir"] == 1 else x0
        d.line(S(far, y0, far, y1), fill=metal, width=rail * 2)
        sl = bn["slots"]
        sp = abs(sl[0][0] - sl[1][0]) if len(sl) > 1 else 2 * h
        for s in sl:
            for xx in (s[0] - sp / 2, s[0] + sp / 2):
                if x0 + 2 < xx < x1 - 2:
                    d.line(S(xx, y0, xx, y0 + 10), fill=metal, width=SS * 3)
                    d.line(S(xx, y1 - 10, xx, y1), fill=metal, width=SS * 3)
    # belt band (cleats are drawn per frame)
    mx = scene["mouth"]
    bx0, bx1 = sorted((mx, gx))
    d.rectangle(S(bx0, gy - h, bx1, gy + h), fill=belt)
    d.line(S(bx0, gy - h, bx1, gy - h), fill=metal, width=rail)
    d.line(S(bx0, gy + h, bx1, gy + h), fill=metal, width=rail)
    d.rounded_rectangle([v * SS for v in sim.TIMER_BOX], radius=10 * SS, fill=TIMER_BG, outline=TIMER_EDGE, width=2 * SS)
    return img


@functools.lru_cache(maxsize=8)
def _tunnel(key: str):
    """The tunnel box (drawn over the objects): (image, top-left corner) at SS scale."""
    scene = json.loads(key)
    st = scene["style"]
    x0, y0, x1, y1 = scene["tunnel"]
    shade = (14, 14, 18) if not st["light"] else (52, 54, 60)
    ox, oy = int(x0 * SS) - 8, int(y0 * SS) - 8
    img = Image.new("RGBA", (int(x1 * SS) - ox + 9, int(y1 * SS) - oy + 9), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    d.rectangle([x0 * SS - ox, y0 * SS - oy, x1 * SS - ox, y1 * SS - oy], fill=shade + (255,),
                outline=tuple(st["metal"]) + (255,), width=4 * SS)
    mx = scene["mouth"]
    d.line([mx * SS - ox, y0 * SS - oy, mx * SS - ox, y1 * SS - oy], fill=tuple(st["metal"]) + (255,), width=6 * SS)
    return img, (ox, oy)


def frame_at(scene, f: int) -> Image.Image:
    st = scene["style"]
    key = json.dumps(scene, sort_keys=True)
    tr = _track(key)
    img = _static(key).copy()
    d = ImageDraw.Draw(img)
    h = scene["half"]
    gx, gy = scene["gate"]
    mx = scene["mouth"]
    bx0, bx1 = sorted((mx, gx))
    v = scene["v2"] / 2
    cl = st["cleat"]
    off = (v * f) % cl
    cc = _mix(tuple(st["belt"]), (255, 255, 255), 0.18)
    k = 0
    while True:
        x = mx + scene["dir"] * (off + k * cl)
        if not bx0 + 3 < x < bx1 - 3:
            if k * cl > abs(gx - mx) + cl:
                break
            k += 1
            continue
        d.line([x * SS, (gy - h + 5) * SS, x * SS, (gy + h - 5) * SS], fill=cc, width=3 * SS)
        k += 1
    d.line([gx * SS, (gy - h) * SS, gx * SS, (gy + h) * SS], fill=tuple(st["metal"]), width=4 * SS)
    for j, o in enumerate(scene["objects"]):
        p = tr["pos"][f, j]
        if np.isnan(p[0]):
            continue
        sp = sprite(o["colour"], o["shape"], o["pattern"], o["r"], st["outline"])
        hs = sp.size[0] // 2
        img.paste(sp, (int(round(p[0] * SS)) - hs, int(round(p[1] * SS)) - hs), sp)
    tun, at = _tunnel(key)
    img.paste(tun, at, tun)
    img = img.reduce(SS)                       # 2x2 box filter back to 1280x720
    d = ImageDraw.Draw(img)
    tb = sim.TIMER_BOX
    d.text(((tb[0] + tb[2]) / 2, (tb[1] + tb[3]) / 2), sim.clock_text(sim.tenths(f)), fill=TIMER_FG,
           font=font(sim.TIMER_FONT), anchor="mm")
    return img


def frame(scene, t: float) -> Image.Image:
    return frame_at(scene, min(scene["n_frames"] - 1, int(round(t * sim.FPS))))
