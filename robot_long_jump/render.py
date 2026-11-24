"""PIL renderer for robot_long_jump (1280x720, 24 fps).

`layout(scene, f)` is the single source of geometry for frame f: for every
robot its contact point on the track, body centre, tilt and state
(stand / hop / jump / landed / tipping / fallen), the footprints it has
left so far and the dent of its sand landing. `frame()` draws exactly what
`layout()` returns; answers.py reads the same per-frame geometry to
re-derive every answer without touching the generator.

The renderer keeps its own replay of the schedule (it sums the hops itself
and never calls sim.py). Besides the timer (top-left box) the only text is
the five ruler numbers 0.0 / 0.5 / 1.0 / 1.5 / 2.0, taken from
scene["labels"]; gridlines every 0.1 run from the ruler up through all lanes.
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

FPS = 24
W, H, SS = 1280, 720, 2
RGB = {"orange": (247, 132, 18), "yellow": (240, 204, 20), "green": (40, 160, 62), "teal": (18, 176, 170),
       "blue": (40, 92, 224), "purple": (134, 62, 204), "pink": (242, 104, 184)}
INK = (30, 30, 34)
FOUL_RGB = (226, 30, 36)
LEG = 10                       # leg length below the body (px)
RULER_BAND = (0, 636, 1280, 720)
FOOT_W, FOOT_H = 22, 9
HALF_W = {"circle": 1.0, "square": 0.86, "triangle": 1.12, "hexagon": 1.0, "diamond": 1.0}   # body half-width / size
DENT_W, DENT_H = 30, 10


# --------------------------------------------------------------------------- timing


def duration(scene) -> float:
    return scene["n_frames"] / FPS


def timer_text(f: int) -> str:
    t = (f * 10) // FPS
    return f"{t // 10}.{t % 10} s"


def x_of(scene, tenths: float) -> float:
    v = scene["view"]
    return v["x0"] + v["ppu"] * tenths / 10


# --------------------------------------------------------------------------- per-frame geometry


def _feet(r):
    out, pos = [], r["start"]
    for h in r["hops"]:
        pos += h
        out.append(pos)
    return out


def _robot_state(scene, r, f):
    """(position in tenths, height above the track px, state, tilt degrees)."""
    tm = scene["timing"]
    ft = _feet(r)
    pos = r["start"]
    for k, h0 in enumerate(r["hop_frames"]):
        if f < h0:
            return pos, 0.0, "stand", 0.0
        if f < h0 + tm["hop_f"]:
            u = (f - h0) / tm["hop_f"]
            return pos + (ft[k] - pos) * u, scene["hop_h"] * 4 * u * (1 - u), "hop", 0.0
        pos = ft[k]
    if r["fouled"]:
        e = r["end_frame"]
        if f < e + tm["tip_f"]:
            return pos, 0.0, "tipping", 90.0 * (f - e) / tm["tip_f"]
        return pos, 0.0, "fallen", 90.0
    if f < r["jump_frame"]:
        return pos, 0.0, "stand", 0.0
    land = pos + r["jump"]
    if f < r["end_frame"]:
        u = (f - r["jump_frame"]) / tm["jump_f"]
        return pos + r["jump"] * u, scene["jump_h"] * 4 * u * (1 - u), "jump", 0.0
    return land, 0.0, "landed", 0.0


def layout(scene, f: int) -> dict:
    """Everything drawn on frame f, in output pixels."""
    v, tm = scene["view"], scene["timing"]
    out = {"robots": {}, "feet": {}, "dents": {}}
    for r in scene["robots"]:
        pos, hgt, state, tilt = _robot_state(scene, r, f)
        ly = v["lane_y"][r["lane"]]
        s = r["size"]
        cx = x_of(scene, pos)
        # body centre: upright robots stand on their legs; a toppling robot pivots on its front foot
        px, py = cx + 0.5 * s, ly
        dx, dy = -0.5 * s, -(LEG + s)
        a = math.radians(tilt)
        bx, by = px + dx * math.cos(a) - dy * math.sin(a), py + dx * math.sin(a) + dy * math.cos(a) - hgt
        # while toppling, lift the body so it ends lying on top of the strip (lowest point 4 px into it)
        by += (tilt / 90.0) * (4 - (HALF_W[r["shape"]] - 0.5) * s)
        out["robots"][r["id"]] = {"contact": cx, "x": bx, "y": by, "state": state, "tilt": tilt,
                                  "lane_y": ly, "size": s}
        ft = _feet(r)
        out["feet"][r["id"]] = [x_of(scene, ft[k]) for k, h0 in enumerate(r["hop_frames"]) if f >= h0 + tm["hop_f"]]
        out["dents"][r["id"]] = (x_of(scene, ft[-1] + r["jump"])
                                 if not r["fouled"] and f >= r["end_frame"] else None)
    return out


# --------------------------------------------------------------------------- drawing helpers


def _mix(c, d, t):
    return tuple(round(a + (b - a) * t) for a, b in zip(c, d))


def _lum(c):
    return 0.299 * c[0] + 0.587 * c[1] + 0.114 * c[2]


def _shape_poly(shape, s, cx, cy):
    if shape == "square":
        h = 0.86 * s
        return [(cx - h, cy - h), (cx + h, cy - h), (cx + h, cy + h), (cx - h, cy + h)]
    if shape == "triangle":
        return [(cx, cy - 1.08 * s), (cx + 1.12 * s, cy + 0.9 * s), (cx - 1.12 * s, cy + 0.9 * s)]
    if shape == "hexagon":
        return [(cx + s * math.cos(math.radians(60 * k)), cy + s * math.sin(math.radians(60 * k))) for k in range(6)]
    if shape == "diamond":
        return [(cx, cy - 1.12 * s), (cx + 1.0 * s, cy), (cx, cy + 1.12 * s), (cx - 1.0 * s, cy)]
    return None


@functools.lru_cache(maxsize=64)
def _sprite(colour, shape, pattern, s, eye):
    """RGBA robot (body + legs + eye) drawn at SS scale, centred on the body centre."""
    S = s * SS
    half = int((s + LEG + 16) * SS)
    img = Image.new("RGBA", (2 * half, 2 * half), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    c = half
    base = RGB[colour]
    # legs
    for lx in (-0.45, 0.45):
        d.line([(c + lx * S, c + 0.6 * S), (c + lx * S, c + S + LEG * SS)], fill=INK, width=4 * SS)
    mask = Image.new("L", img.size, 0)
    md = ImageDraw.Draw(mask)
    poly = _shape_poly(shape, S, c, c)
    if poly:
        md.polygon(poly, fill=255)
    else:
        md.ellipse([c - S, c - S, c + S, c + S], fill=255)
    body = Image.new("RGBA", img.size, base + (255,))
    bd = ImageDraw.Draw(body)
    if pattern == "striped":
        light = _mix(base, (255, 255, 255), 0.45)
        for k in range(-6, 7):
            o = k * 0.62 * S
            bd.line([(c + o - S * 1.4, c + S * 1.4), (c + o + S * 1.4, c - S * 1.4)], fill=light, width=int(0.2 * S))
    elif pattern == "two-tone":
        bd.rectangle([0, c + 0.15 * S, 2 * half, 2 * half], fill=_mix(base, (255, 255, 255), 0.42))
    elif pattern == "ringed":
        dark = _mix(base, (0, 0, 0), 0.42)
        if poly:
            inner = _shape_poly(shape, 0.58 * S, c, c)
            bd.polygon(inner, outline=dark, width=3 * SS)
        else:
            bd.ellipse([c - 0.58 * S, c - 0.58 * S, c + 0.58 * S, c + 0.58 * S], outline=dark, width=3 * SS)
    img.paste(body, (0, 0), mask)
    if poly:
        d.polygon(poly, outline=INK, width=3 * SS)
    else:
        d.ellipse([c - S, c - S, c + S, c + S], outline=INK, width=3 * SS)
    # eye on the front (right) side
    ex, ey = c + 0.32 * S, c - 0.28 * S
    if eye == "visor":
        d.rounded_rectangle([ex - 0.3 * S, ey - 0.13 * S, ex + 0.36 * S, ey + 0.13 * S], radius=0.12 * S,
                            fill=(250, 250, 250), outline=INK, width=2 * SS)
        d.ellipse([ex + 0.06 * S, ey - 0.08 * S, ex + 0.22 * S, ey + 0.08 * S], fill=INK)
    else:
        d.ellipse([ex - 0.2 * S, ey - 0.2 * S, ex + 0.2 * S, ey + 0.2 * S], fill=(250, 250, 250), outline=INK,
                  width=2 * SS)
        d.ellipse([ex - 0.02 * S, ey - 0.09 * S, ex + 0.16 * S, ey + 0.09 * S], fill=INK)
    return img


def _lane_rects(scene, r):
    v = scene["view"]
    ly = v["lane_y"][r["lane"]]
    sh = v["strip_h"]
    run = (x_of(scene, 0) - 36, ly, x_of(scene, scene["foul"]), ly + sh)
    pit = (x_of(scene, scene["foul"]), ly - 4, x_of(scene, 20) + 18, ly + sh + 4)
    return run, pit


@functools.lru_cache(maxsize=8)
def _static(key: str):
    """Background, lanes, gridlines, foul lines, start marks, ruler and its labels (SS scale)."""
    scene = json.loads(key)
    th, v = scene["theme"], scene["view"]
    img = Image.new("RGB", (W * SS, H * SS), tuple(th["bg"]))
    d = ImageDraw.Draw(img)
    Z = lambda box: [b * SS for b in box]
    for r in scene["robots"]:
        run, pit = _lane_rects(scene, r)
        d.rectangle(Z(run), fill=tuple(th["runway"]))
        d.rounded_rectangle(Z(pit), radius=6 * SS, fill=tuple(th["sand"]))
    # gridlines every 0.1, darker every 0.5, from just above the top lane down to the ruler
    top = min(v["lane_y"]) - v["pitch"] * 0.5 - 40
    over = Image.new("RGBA", img.size, (0, 0, 0, 0))
    od = ImageDraw.Draw(over)
    a = th["grid_alpha"]
    for k in range(0, 21):
        x = x_of(scene, k) * SS
        major = k % 5 == 0
        od.line([(x, top * SS), (x, v["ruler_y"] * SS)], fill=(40, 40, 48, round(255 * (a + 0.2 if major else a))),
                width=(4 if major else 3) * SS)
    img = Image.alpha_composite(img.convert("RGBA"), over).convert("RGB")
    d = ImageDraw.Draw(img)
    for r in scene["robots"]:
        run, pit = _lane_rects(scene, r)
        ly = v["lane_y"][r["lane"]]
        fx = x_of(scene, scene["foul"])
        d.rectangle(Z((fx - 3, ly - 14, fx + 3, ly + v["strip_h"] + 8)), fill=FOUL_RGB)
        sx = x_of(scene, r["start"])
        if scene["scenario"] == "staggered":
            d.rectangle(Z((sx - 16, ly - 2, sx + 16, ly + v["strip_h"] + 2)), fill=(250, 250, 250),
                        outline=INK, width=2 * SS)
        else:
            d.rectangle(Z((sx - 2, ly, sx + 2, ly + v["strip_h"])), fill=(250, 250, 250))
    # ruler band
    d.rectangle(Z(RULER_BAND), fill=(250, 250, 248))
    ry = v["ruler_y"]
    d.line(Z((x_of(scene, 0), ry, x_of(scene, 20), ry)), fill=INK, width=3 * SS)
    for k in range(0, 21):
        x = x_of(scene, k)
        major = k % 5 == 0
        d.line(Z((x, ry, x, ry + (18 if major else 10))), fill=INK, width=(4 if major else 3) * SS)
    fx = x_of(scene, scene["foul"])
    d.line(Z((fx, ry - 1, fx, ry + 18)), fill=FOUL_RGB, width=4 * SS)
    for lab in scene["labels"]:
        d.text((lab["x"] * SS, lab["y"] * SS), lab["text"], fill=INK, font=font(lab["size"] * SS), anchor="mm")
    return img


def _draw_timer(d, scene, f):
    x0, y0, x1, y1 = (b * SS for b in scene["timer"]["box"])
    d.rounded_rectangle([x0, y0, x1, y1], radius=10 * SS, fill=(18, 18, 22), outline=(235, 235, 235), width=2 * SS)
    d.text(((x0 + x1) / 2, (y0 + y1) / 2), timer_text(f), fill=(250, 250, 250),
           font=font(scene["timer"]["size"] * SS), anchor="mm")


def _foot_mark(d, x, y, style, colour):
    x, y = x * SS, y * SS
    w, h = FOOT_W * SS / 2, FOOT_H * SS / 2
    if style == "bar":
        d.rounded_rectangle([x - w, y - h, x + w, y + h], radius=3 * SS, fill=colour)
    else:
        d.ellipse([x - w, y - h, x + w, y + h], fill=colour)


def frame(scene, t: float) -> Image.Image:
    f = int(round(t * FPS))
    th, v = scene["theme"], scene["view"]
    img = _static(json.dumps(scene, sort_keys=True)).copy()
    d = ImageDraw.Draw(img)
    lay = layout(scene, f)
    mid = v["strip_h"] / 2
    fx = x_of(scene, scene["foul"])
    for r in scene["robots"]:
        ly = v["lane_y"][r["lane"]]
        for x in lay["feet"][r["id"]]:
            on_sand = x > fx
            _foot_mark(d, x, ly + mid, th["foot"], INK if on_sand else (248, 248, 248))
        dx = lay["dents"][r["id"]]
        if dx is not None:
            d.ellipse([(dx - DENT_W / 2) * SS, (ly + mid - DENT_H / 2) * SS, (dx + DENT_W / 2) * SS,
                       (ly + mid + DENT_H / 2) * SS], fill=_mix(tuple(th["sand"]), (60, 40, 20), 0.55))
    # robots in the air are drawn last so they pass in front of the lanes they cross
    order = sorted(scene["robots"], key=lambda r: lay["robots"][r["id"]]["state"] in ("hop", "jump"))
    for r in order:
        st = lay["robots"][r["id"]]
        sp = _sprite(r["colour"], r["shape"], r["pattern"], r["size"], th["eye"])
        if st["tilt"]:
            sp = sp.rotate(-st["tilt"], resample=Image.BICUBIC)
        img.paste(sp, (round(st["x"] * SS - sp.width / 2), round(st["y"] * SS - sp.height / 2)), sp)
    _draw_timer(d, scene, f)
    return img.resize((W, H), Image.LANCZOS)
