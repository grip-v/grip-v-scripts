"""PIL renderer for balance_scale_weighing (1280x720, 24 fps).

`layout(scene, f)` is the single source of geometry for frame f: the beam
angle, the pivot, the pan / peg positions and, for every weight, its state
(hidden / moving / rest) and where it touches the scale. `frame()` draws
exactly what `layout()` returns; answers.py reads the same per-frame
geometry to re-derive every answer without touching the generator.

The renderer keeps its own replay of the events (which weight is where,
which pan slot it took, the beam's eased angle); it never calls sim.py.
The only text in the frame is the timer in the bottom-left corner.
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
RGB = {"red": (222, 40, 40), "orange": (247, 132, 18), "yellow": (240, 204, 20), "green": (40, 160, 62),
       "teal": (18, 176, 170), "blue": (40, 92, 224), "purple": (134, 62, 204), "pink": (242, 104, 184),
       "brown": (134, 82, 40)}
HALF_BEAM = 250            # pan scenarios: pivot to post, scene units (x camera sc)
POST = 104                 # beam end to pan top
PAN_HALF = 200
SLOT = 96
PEG = 76                   # lever: peg spacing
BEAM_T = 16                # beam thickness
TIMER_BOX = (16, 650, 204, 706)
TRI_W = 1.15
STRIPE, TWO_TONE, RING = 0.45, 0.42, 0.6     # tint / tint / shade used by the patterns
WIDTH = {"circle": 1.0, "square": 0.92, "hexagon": 1.1, "pentagon": 1.07, "triangle": TRI_W}


# --------------------------------------------------------------------------- timing


def duration(scene) -> float:
    return scene["n_frames"] / FPS


def timer_text(f: int) -> str:
    t = (f * 10) // FPS
    return f"{t // 10}.{t % 10} s"


def _smooth(u: float) -> float:
    return u * u * (3 - 2 * u)


# --------------------------------------------------------------------------- replay


def _replay(scene):
    """Change points [(frame, target_deg)], per-object phases and pan slots."""
    lever = scene["scenario"] == "lever"
    mass = {o["id"]: o["mass"] for o in scene["objects"]}
    where = {}                                       # obj -> [side, peg, slot]

    def target():
        tq = sum((1 if s == "right" else -1) * mass[o] * (p if lever else 1) for o, (s, p, _) in where.items())
        a = scene["deg_per_unit"] * tq
        return max(-scene["max_deg"], min(scene["max_deg"], a))

    def free_slot(side):
        used = {sl for s, _, sl in where.values() if s == side}
        k = 0
        while k in used:
            k += 1
        return k

    phases = {o["id"]: [] for o in scene["objects"]}  # (f_from, f_to, kind, a, b) a/b = placements
    for it in scene["initial"]:
        where[it["obj"]] = [it["side"], it["peg"], None if lever else free_slot(it["side"])]
        phases[it["obj"]].append((0, None, "rest", tuple(where[it["obj"]]), None))
    changes = [(0, target())]
    for ev in scene["events"]:
        o = ev["obj"]
        if ev["kind"] in ("drop", "place"):
            where[o] = [ev["side"], ev["peg"], None if lever else free_slot(ev["side"])]
            pl = tuple(where[o])
            phases[o].append((ev["f0"], ev["f1"], "fall", pl, None))
            phases[o].append((ev["f1"], None, "rest", pl, None))
            changes.append((ev["f1"], target()))
        elif ev["kind"] == "lift":
            pl = tuple(where.pop(o))
            _close(phases[o], ev["f0"])
            phases[o].append((ev["f0"], ev["f1"], "lift", pl, None))
            changes.append((ev["f0"], target()))
        else:                                        # hop to the other pan
            src = tuple(where.pop(o))
            _close(phases[o], ev["f0"])
            changes.append((ev["f0"], target()))
            side = "left" if src[0] == "right" else "right"
            where[o] = [side, None, free_slot(side)]
            dst = tuple(where[o])
            phases[o].append((ev["f0"], ev["f1"], "hop", src, dst))
            phases[o].append((ev["f1"], None, "rest", dst, None))
            changes.append((ev["f1"], target()))
    return changes, phases


def _close(ph, f):
    a, _, kind, pl, x = ph[-1]
    ph[-1] = (a, f, kind, pl, x)


@functools.lru_cache(maxsize=64)
def _replay_cached(key: str):
    return _replay(json.loads(key))


def _angle(changes, trans_f, f):
    a_from, a_to, start = changes[0][1], changes[0][1], 0
    for fc, tgt in changes[1:]:
        if fc > f:
            break
        if tgt != a_to:                     # a new target: ease from wherever the beam is now
            a_from, a_to, start = _ease(a_from, a_to, start, trans_f, fc), tgt, fc
    return _ease(a_from, a_to, start, trans_f, f)


def _ease(a_from, a_to, start, trans_f, f):
    u = (f - start) / trans_f
    if u >= 1 or a_from == a_to:
        return a_to
    return a_from + (a_to - a_from) * _smooth(max(0.0, u))


# --------------------------------------------------------------------------- geometry


def _rot(ang, x, y):
    c, s = math.cos(math.radians(ang)), math.sin(math.radians(ang))
    return x * c - y * s, x * s + y * c


def layout(scene, f: int) -> dict:
    """Everything drawn at frame f, in 1280x720 pixel coordinates (y down).

    angle: beam angle in degrees, positive = right end lower.
    pans: {side: {"x", "top"}} pan centre x and top-surface y (pan scenarios).
    pegs: {side: [(x, y), ...]} peg tops, peg 1 nearest the pivot (lever).
    objs: {id: {"state", "x", "y" (bottom-centre contact), "rot", "size"}}.
    """
    key = json.dumps(scene, sort_keys=True)
    changes, phases = _replay_cached(key)
    cam = scene["camera"]
    sc, px, py = cam["sc"], cam["cx"], cam["cy"]
    ang = _angle(changes, scene["trans_f"], f)
    lever = scene["scenario"] == "lever"
    out = {"angle": ang, "pivot": (px, py), "lever": lever, "pans": None, "pegs": None, "objs": {}}

    def pan_at(side, a):
        sgn = 1 if side == "right" else -1
        ex, ey = _rot(a, sgn * HALF_BEAM * sc, 0)
        return px + ex, py + ey - POST * sc

    def peg_at(side, peg, a):
        sgn = 1 if side == "right" else -1
        ex, ey = _rot(a, sgn * peg * PEG * sc, -BEAM_T * sc / 2)
        return px + ex, py + ey

    def contact(pl, a):
        side, peg, slot = pl
        if lever:
            return peg_at(side, peg, a)
        x, top = pan_at(side, a)
        return x + (slot - 1.5) * SLOT * sc, top

    if lever:
        out["pegs"] = {s: [peg_at(s, k, ang) for k in range(1, 5)] for s in ("left", "right")}
    else:
        out["pans"] = {s: dict(zip(("x", "top"), pan_at(s, ang))) for s in ("left", "right")}
    for o in scene["objects"]:
        size = o["size"] * sc
        st = {"state": "hidden", "x": None, "y": None, "rot": 0.0, "size": size}
        for f_from, f_to, kind, pl, dst in phases[o["id"]]:
            if f < f_from or (f_to is not None and f >= f_to):
                continue
            if kind == "rest":
                x, y = contact(pl, ang)
                st.update(state="rest", x=x, y=y, rot=ang if lever else 0.0, side=pl[0], peg=pl[1])
            else:
                u = (f - f_from) / (f_to - f_from)
                if kind == "fall":
                    x, y1 = contact(pl, ang)
                    y = -8 + (y1 + 8) * u * u
                    st.update(state="moving", x=x, y=y, rot=(ang * u) if lever else 0.0)
                elif kind == "lift":
                    x, y0 = contact(pl, ang)
                    st.update(state="moving", x=x, y=y0 + (-8 - y0) * u)      # lifted briskly, at constant speed
                else:
                    x0, y0 = contact(pl, ang)
                    x1, y1 = contact(dst, ang)
                    peak = max(30 + size, min(y0, y1) - 165 * sc)
                    hgt = 4 * (min(y0, y1) - peak) * u * (1 - u)
                    st.update(state="moving", x=x0 + (x1 - x0) * u, y=y0 + (y1 - y0) * u - hgt)
            break
        out["objs"][o["id"]] = st
    return out


# --------------------------------------------------------------------------- drawing


def _mix(c, d, t):
    return tuple(int(round(a + (b - a) * t)) for a, b in zip(c, d))


def _lum(c):
    return 0.299 * c[0] + 0.587 * c[1] + 0.114 * c[2]


def _shape_poly(shape, s):
    """Outline in local coords: bottom-centre at (0, 0), y up is negative. Returns (points, pip centre, spread)."""
    if shape == "circle":
        pts = [(s / 2 * math.cos(2 * math.pi * i / 64), -s / 2 + s / 2 * math.sin(2 * math.pi * i / 64))
               for i in range(64)]
        return pts, (0, -s / 2), 0.22 * s
    if shape == "square":
        h = s * 0.46
        return [(-h, 0), (h, 0), (h, -2 * h), (-h, -2 * h)], (0, -h), 0.22 * s
    if shape == "hexagon":
        r = s * 0.55
        hh = r * math.sqrt(3) / 2
        pts = [(r * math.cos(math.radians(60 * i)), -hh + hh * math.sin(math.radians(60 * i)) / (math.sqrt(3) / 2))
               for i in range(6)]
        return pts, (0, -hh), 0.22 * s
    if shape == "pentagon":
        r = s * 0.56
        cy = -r * math.cos(math.radians(36))
        pts = [(r * math.cos(math.radians(-90 + 72 * i)), cy + r * math.sin(math.radians(-90 + 72 * i)))
               for i in range(5)]
        return pts, (0, cy + 0.05 * s), 0.19 * s
    w = s * TRI_W                                    # triangle, drawn wider to fit the dots
    h = w * 0.88
    return [(-w / 2, 0), (w / 2, 0), (0, -h)], (0, -h * 0.33), 0.14 * w


PIPS = {1: [(0, 0)], 2: [(-1, -1), (1, 1)], 3: [(-1, -1), (0, 0), (1, 1)], 4: [(-1, -1), (1, -1), (-1, 1), (1, 1)],
        5: [(-1, -1), (1, -1), (0, 0), (-1, 1), (1, 1)], 6: [(-1, -1), (-1, 0), (-1, 1), (1, -1), (1, 0), (1, 1)]}


@functools.lru_cache(maxsize=256)
def _sprite(colour, shape, pattern, pips, size_px, outline):
    """RGBA sprite at SS resolution; the object's bottom-centre is the canvas centre."""
    s = size_px * SS
    r = int(math.ceil(s * 0.9)) + 6
    img = Image.new("RGBA", (2 * r, 2 * r), (0, 0, 0, 0))
    pts, (pcx, pcy), spread = _shape_poly(shape, s)
    poly = [(r + x, r + y) for x, y in pts]
    base = RGB[colour]
    fill = Image.new("RGB", img.size, base)
    fd = ImageDraw.Draw(fill)
    if pattern == "striped":
        tint = _mix(base, (255, 255, 255), STRIPE)
        step = 0.26 * s
        k = -2 * r
        while k < 4 * r:
            fd.line([(k, 0), (k + 2 * r, 2 * r)], fill=tint, width=max(2, int(0.09 * s)))
            k += step
    elif pattern == "two-tone":
        top = min(y for _, y in poly)
        fd.rectangle([0, (top + r) / 2, 2 * r, 2 * r], fill=_mix(base, (255, 255, 255), TWO_TONE))
    elif pattern == "ringed":
        cx, cy = r + pcx, r + pcy
        inner = [(cx + (x - cx) * 0.66, cy + (y - cy) * 0.66) for x, y in poly]
        fd.polygon(inner, outline=_mix(base, (0, 0, 0), RING), width=max(3, int(0.07 * s)))
    mask = Image.new("L", img.size, 0)
    ImageDraw.Draw(mask).polygon(poly, fill=255)
    img.paste(fill, (0, 0), mask)
    d = ImageDraw.Draw(img)
    d.polygon(poly, outline=outline, width=max(4, int(0.05 * s)))
    pr = 0.09 * size_px * SS if shape != "triangle" else 0.078 * size_px * SS
    for gx, gy in PIPS.get(pips, []):
        x, y = r + pcx + gx * spread, r + pcy + gy * spread
        d.ellipse([x - pr, y - pr, x + pr, y + pr], fill=(255, 255, 255, 255), outline=(25, 25, 25, 255),
                  width=max(2, int(pr * 0.3)))
    return img


@functools.lru_cache(maxsize=8)
def _background(theme_key: str):
    th = json.loads(theme_key)
    img = Image.new("RGB", (W * SS, H * SS))
    d = ImageDraw.Draw(img)
    fy = th["floor_y"] * SS
    top, bot = th["bg_top"], th["bg_bot"]
    for y in range(0, fy, 4):
        d.rectangle([0, y, W * SS, y + 4], fill=_mix(top, bot, y / fy))
    faint = _mix(top, bot, 0.5)
    faint = _mix(faint, (0, 0, 0) if _lum(faint) > 128 else (255, 255, 255), 0.07)
    pat = th["bg_pattern"]
    if pat == "grid":
        for x in range(0, W * SS, 80 * SS):
            d.line([(x, 0), (x, fy)], fill=faint, width=2 * SS)
        for y in range(0, fy, 80 * SS):
            d.line([(0, y), (W * SS, y)], fill=faint, width=2 * SS)
    elif pat == "stripes":
        for x in range(0, W * SS, 120 * SS):
            d.rectangle([x, 0, x + 60 * SS, fy], fill=faint)
    elif pat == "dots":
        for x in range(30 * SS, W * SS, 90 * SS):
            for y in range(30 * SS, fy, 90 * SS):
                d.ellipse([x - 4 * SS, y - 4 * SS, x + 4 * SS, y + 4 * SS], fill=faint)
    elif pat == "tiles":
        for i, x in enumerate(range(0, W * SS, 160 * SS)):
            for j, y in enumerate(range(0, fy, 160 * SS)):
                if (i + j) % 2:
                    d.rectangle([x, y, x + 160 * SS, y + 160 * SS], fill=faint)
    fl = tuple(th["floor"])
    d.rectangle([0, fy, W * SS, H * SS], fill=fl)
    dark = _mix(fl, (0, 0, 0), 0.2)
    if th["floor_pattern"] == "planks":
        for y in range(fy, H * SS, 34 * SS):
            d.line([(0, y), (W * SS, y)], fill=dark, width=2 * SS)
    elif th["floor_pattern"] == "checker":
        for i, x in enumerate(range(0, W * SS, 60 * SS)):
            for j, y in enumerate(range(fy, H * SS, 60 * SS)):
                if (i + j) % 2:
                    d.rectangle([x, y, x + 60 * SS, y + 60 * SS], fill=dark)
    d.line([(0, fy), (W * SS, fy)], fill=_mix(fl, (0, 0, 0), 0.4), width=3 * SS)
    return img


def _draw_stand(d, scene, lay):
    th, sc = scene["theme"], scene["camera"]["sc"]
    px, py = (v * SS for v in lay["pivot"])
    fy = th["floor_y"] * SS
    col = tuple(th["stand"])
    k = sc * SS
    style = th["stand_style"]
    if scene["scenario"] == "lever":
        d.polygon([(px, py + 6 * k), (px - 70 * k, fy), (px + 70 * k, fy)], fill=col)
    elif style == "pillar":
        d.rectangle([px - 12 * k, py, px + 12 * k, fy - 10 * k], fill=col)
        d.polygon([(px - 80 * k, fy), (px + 80 * k, fy), (px + 50 * k, fy - 22 * k), (px - 50 * k, fy - 22 * k)],
                  fill=col)
    elif style == "a_frame":
        d.polygon([(px - 9 * k, py), (px + 9 * k, py), (px + 90 * k, fy), (px + 66 * k, fy), (px, py + 40 * k),
                   (px - 66 * k, fy), (px - 90 * k, fy)], fill=col)
    else:
        d.rectangle([px - 22 * k, py + 10 * k, px + 22 * k, fy], fill=col)
        d.rectangle([px - 36 * k, py + 4 * k, px + 36 * k, py + 20 * k], fill=col)
        d.rectangle([px - 50 * k, fy - 16 * k, px + 50 * k, fy], fill=col)


def _draw_beam(d, scene, lay):
    th, sc = scene["theme"], scene["camera"]["sc"]
    k = sc * SS
    px, py = lay["pivot"]
    a = lay["angle"]
    col = tuple(th["beam"])
    half = (4.6 * PEG if lay["lever"] else HALF_BEAM + 14) * sc
    t = BEAM_T * sc / 2
    pts = [_rot(a, x, y) for x, y in ((-half, -t), (half, -t), (half, t), (-half, t))]
    d.polygon([((px + x) * SS, (py + y) * SS) for x, y in pts], fill=col)
    if lay["lever"]:
        mark = _mix(col, (255, 255, 255) if _lum(col) < 128 else (0, 0, 0), 0.55)
        for side in ("left", "right"):
            for (x, y) in lay["pegs"][side]:
                # small tick below the beam and a bump on top, so each peg stays visible under a weight
                bx, by = _rot(a, 0, BEAM_T * sc + 7 * sc)
                d.line([(x * SS, y * SS + 2 * k), ((x + bx) * SS, (y + by) * SS)], fill=mark, width=int(4 * k))
                d.ellipse([(x - 5 * sc) * SS, (y - 5 * sc) * SS, (x + 5 * sc) * SS, (y + 5 * sc) * SS], fill=mark)
    else:
        pan = tuple(th["pan"])
        for side in ("left", "right"):
            x, top = lay["pans"][side]["x"], lay["pans"][side]["top"]
            ex, ey = _rot(a, (1 if side == "right" else -1) * HALF_BEAM * sc, 0)
            d.line([((px + ex) * SS, (py + ey) * SS), (x * SS, (top + 10 * sc) * SS)], fill=col, width=int(9 * k))
            d.ellipse([((px + ex) - 8 * sc) * SS, ((py + ey) - 8 * sc) * SS, ((px + ex) + 8 * sc) * SS,
                       ((py + ey) + 8 * sc) * SS], fill=_mix(col, (0, 0, 0), 0.3))
            hw = PAN_HALF * sc
            style = th["pan_style"]
            if style == "tray":
                d.rectangle([(x - hw) * SS, top * SS, (x + hw) * SS, (top + 12 * sc) * SS], fill=pan)
                d.rectangle([(x - hw) * SS, (top - 8 * sc) * SS, (x - hw + 6 * sc) * SS, top * SS], fill=pan)
                d.rectangle([(x + hw - 6 * sc) * SS, (top - 8 * sc) * SS, (x + hw) * SS, top * SS], fill=pan)
            elif style == "bowl":
                d.chord([(x - hw) * SS, (top - 22 * sc) * SS, (x + hw) * SS, (top + 22 * sc) * SS], 0, 180, fill=pan)
            else:
                d.rounded_rectangle([(x - hw) * SS, top * SS, (x + hw) * SS, (top + 9 * sc) * SS],
                                    radius=int(4 * k), fill=pan)
    d.ellipse([(px - 9 * sc) * SS, (py - 9 * sc) * SS, (px + 9 * sc) * SS, (py + 9 * sc) * SS],
              fill=_mix(col, (255, 255, 255), 0.35))


def _draw_timer(d, f):
    x0, y0, x1, y1 = (v * SS for v in TIMER_BOX)
    d.rounded_rectangle([x0, y0, x1, y1], radius=10 * SS, fill=(18, 18, 22), outline=(235, 235, 235),
                        width=2 * SS)
    d.text(((x0 + x1) / 2, (y0 + y1) / 2), timer_text(f), fill=(250, 250, 250), font=font(36 * SS), anchor="mm")


def frame(scene, t: float) -> Image.Image:
    f = int(round(t * FPS))
    th = scene["theme"]
    lay = layout(scene, f)
    img = _background(json.dumps(th, sort_keys=True)).copy()
    d = ImageDraw.Draw(img)
    _draw_stand(d, scene, lay)
    _draw_beam(d, scene, lay)
    bg_l = _lum(_mix(th["bg_top"], th["bg_bot"], 0.5))
    outline = (24, 24, 28, 255) if bg_l > 110 else (240, 240, 240, 255)
    objs = {o["id"]: o for o in scene["objects"]}
    order = sorted(lay["objs"].items(), key=lambda kv: (kv[1]["state"] == "moving", kv[0]))
    for oid, st in order:
        if st["state"] == "hidden":
            continue
        o = objs[oid]
        sp = _sprite(o["colour"], o["shape"], o["pattern"], 0 if o["mystery"] else o["mass"],
                     round(st["size"], 1), outline)
        if st["rot"]:
            sp = sp.rotate(-st["rot"], resample=Image.BICUBIC, center=(sp.width / 2, sp.height / 2))
        img.paste(sp, (int(round(st["x"] * SS - sp.width / 2)), int(round(st["y"] * SS - sp.height / 2))), sp)
    _draw_timer(d, f)
    return img.resize((W, H), Image.LANCZOS)
