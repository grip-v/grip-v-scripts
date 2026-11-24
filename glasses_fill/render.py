"""PIL renderer for glasses_fill (common/video5.py calls frame(scene, t)).

Flat 2D look, as in the original dataset: a row of numbered glasses on a
table with a ml scale (ruler on the left, faint guide lines, a tick every
10 ml on every glass) and patterned jugs on a shelf. Every frame is a pure
function of (scene, t); all logic is finished in sim.py.

On-screen text, all taken from scene["labels"] / scene["clock_xy"]:
  * ruler numbers 0, 20, ..., 100 (the scale the measurement questions read),
  * the glass numbers under the glasses (glasses are otherwise identical),
  * the clock, top-right, seconds to 0.1 s (floor(10 f / 24) / 10).
"""
from __future__ import annotations
import math
import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from common.video5 import font, hex_rgb  # noqa: E402

import sim  # noqa: E402
from sim import FPS, Y_TOP, Y_BOT, TABLE_Y, SHELF_Y, RULER_X, ml_to_y  # noqa: E402

W, H = sim.W, sim.H
POUR_ANGLE = 75.0               # degrees clockwise
SPOUT_GAP = 32                  # spout tip this far above the rim while pouring
RIM_CLEAR = 6                   # the jug never comes closer than this to a glass rim
RULER_HEX = "#3F392E"
GLASS_WALL_HEX = "#7E97AE"
GLASS_TINT_HEX = "#EEF4F8"
TICK_HEX = "#5D7184"
LABEL_HEX = "#2E2A22"
JUG_EDGE_HEX = "#2F3B46"
JUG_PATTERN_HEX = "#3A4A5A"
HOLE_HEX = "#1E1B14"
CLOCK_HEX = "#1F1F1F"

# jug geometry, local coords (origin = sprite centre, y down), drawn 1.3x
JUG_SCALE = 1.3
_s = lambda poly: [(x * JUG_SCALE, y * JUG_SCALE) for x, y in poly]  # noqa: E731
JUG_OUTER = _s([(-40, 55), (40, 55), (31, -48), (52, -64), (27, -58), (-30, -58)])
JUG_INNER = _s([(-36, 51), (36, 51), (28.5, -45), (-28, -45)])
JUG_WINDOW = _s([(-16, -42), (16, -42), (16, 47), (-16, 47)])   # clear panel showing the liquid
JUG_HANDLE = _s([(-33, -38), (-55, -32), (-61, -6), (-52, 18), (-39, 26)])
SPOUT_TIP = (52 * JUG_SCALE, -64 * JUG_SCALE)
JUG_BAND_Y = -28 * JUG_SCALE    # liquid never above this line when upright
SPRITE = 280
SS = 4                          # jugs are drawn at 4x and scaled down -> smooth edges


def duration(scene) -> float:
    return scene["n_frames"] / FPS


def park_pos(j, n_jugs):
    return 640 + (j - (n_jugs - 1) / 2) * 220, SHELF_Y - 55 * JUG_SCALE - 1


def _rot(pt, deg):
    a = math.radians(deg)
    x, y = pt
    return (x * math.cos(a) - y * math.sin(a), x * math.sin(a) + y * math.cos(a))


def _smooth(u):
    u = min(1.0, max(0.0, u))
    return u * u * (3 - 2 * u)


def _hang_below_tip(deg):
    """How far the lowest point of the jug (body or handle) hangs below its spout tip."""
    rt = _rot(SPOUT_TIP, deg)
    return max(_rot(q, deg)[1] - rt[1] for q in JUG_OUTER + JUG_HANDLE)


def jug_pose(scene, j, t):
    """(cx, cy, angle_deg_clockwise) of jug j at time t."""
    px, py = park_pos(j, len(scene["jugs"]))
    for p in scene["pours"]:
        if p["jug"] != j or not (p["t_move"] <= t <= p["t_end"]):
            continue
        gx = scene["glasses"]["cx"][p["glass"]]

        def pivot(deg):
            # turned about the spout tip, which sits just high enough that no part
            # of the jug dips below the rim
            tip = (gx, Y_TOP - max(SPOUT_GAP, RIM_CLEAR + _hang_below_tip(deg)))
            r = _rot(SPOUT_TIP, deg)
            return (tip[0] - r[0], tip[1] - r[1], deg)

        up = pivot(0.0)[:2]
        if t < p["t_tilt"]:
            u = _smooth((t - p["t_move"]) / (p["t_tilt"] - p["t_move"]))
            return (px + (up[0] - px) * u, py + (up[1] - py) * u, 0.0)
        if t < p["t_pour"]:
            return pivot(POUR_ANGLE * _smooth((t - p["t_tilt"]) / (p["t_pour"] - p["t_tilt"])))
        if t < p["t_untilt"]:
            return pivot(POUR_ANGLE)
        if t < p["t_back"]:
            return pivot(POUR_ANGLE * (1 - _smooth((t - p["t_untilt"]) / (p["t_back"] - p["t_untilt"]))))
        u = _smooth((t - p["t_back"]) / (p["t_end"] - p["t_back"]))
        return (up[0] + (px - up[0]) * u, up[1] + (py - up[1]) * u, 0.0)
    return (px, py, 0.0)


def active_pour(scene, t):
    for p in scene["pours"]:
        if p["t_move"] <= t <= p["t_end"]:
            return p
    return None


def label_box(scene, lab):
    """Pixel box (x0, y0, x1, y1) of one label as drawn."""
    d = ImageDraw.Draw(Image.new("L", (1, 1)))
    return d.textbbox((lab["x"], lab["y"]), lab["text"], font=font(scene["label_px"]), anchor=lab["anchor"])


def clock_text(frame: int) -> str:
    return f"{sim.clock_tenths(frame) / 10:.1f} s"


def clock_box(scene):
    d = ImageDraw.Draw(Image.new("L", (1, 1)))
    x, y = scene["clock_xy"]
    return d.textbbox((x, y), "10.0 s", font=font(scene["label_px"]), anchor="ra")


# ------------------------------------------------------------------ static layer
def _static_layer(scene):
    pal = scene["palette"]
    img = Image.new("RGB", (W, H), pal["bg"])
    d = ImageDraw.Draw(img)
    for ml in range(0, sim.CAP + 1, sim.STEP):
        y = ml_to_y(ml)
        d.line([(RULER_X, y), (sim.GLASS_X1, y)], fill=pal["guide"], width=1)
        tl = 18 if ml % sim.SCALE_LABEL_EVERY == 0 else 10
        d.line([(RULER_X - tl, y), (RULER_X, y)], fill=RULER_HEX, width=3)
    d.line([(RULER_X, Y_TOP), (RULER_X, Y_BOT)], fill=RULER_HEX, width=3)
    d.rectangle([sim.GLASS_X0 - 40, TABLE_Y, W, TABLE_Y + 12], fill=pal["table"])
    d.line([(230, SHELF_Y + 2), (1050, SHELF_Y + 2)], fill=pal["table"], width=6)
    f = font(scene["label_px"])
    for lab in scene["labels"]:
        d.text((lab["x"], lab["y"]), lab["text"], font=f, fill=LABEL_HEX, anchor=lab["anchor"])
    return img


# ------------------------------------------------------------------ glasses
def glass_x(scene, g):
    cx, w = scene["glasses"]["cx"][g], scene["glasses"]["w"]
    return cx - w / 2, cx + w / 2


def _draw_glass(d, scene, g, t):
    x0, x1 = glass_x(scene, g)
    w = x1 - x0
    d.rectangle([x0, Y_TOP, x1, Y_BOT], fill=GLASS_TINT_HEX)
    h = 0.0
    for col, ml in sim.glass_layers(scene, g, t):
        d.rectangle([x0, ml_to_y(h + ml), x1, ml_to_y(h)], fill=sim.LIQUIDS[col])
        h += ml
    for ml in range(sim.STEP, sim.CAP, sim.STEP):
        y = ml_to_y(ml)
        tl = 14 if ml == 50 else 8
        d.line([(x0 + 2, y), (x0 + 2 + tl, y)], fill=TICK_HEX, width=3)
    wall = hex_rgb(GLASS_WALL_HEX)
    # shine stripe (kept off the centre column, where levels are measured)
    d.line([(x1 - w * 0.2, Y_TOP + 14), (x1 - w * 0.2, Y_BOT - 14)], fill="#FFFFFF", width=3)
    d.line([(x0 - 2, Y_TOP - 6), (x0 - 2, Y_BOT)], fill=wall, width=4)
    d.line([(x1 + 2, Y_TOP - 6), (x1 + 2, Y_BOT)], fill=wall, width=4)
    d.line([(x0 - 6, Y_TOP - 6), (x0, Y_TOP - 6)], fill=wall, width=3)
    d.line([(x1, Y_TOP - 6), (x1 + 6, Y_TOP - 6)], fill=wall, width=3)
    d.rounded_rectangle([x0 - 4, Y_BOT, x1 + 4, TABLE_Y], radius=6, fill="#B9CAD8", outline=wall, width=2)

    slot = (sim.GLASS_X1 - sim.GLASS_X0) / scene["glasses"]["n"]
    for lk in scene["leaks"]:
        if lk["glass"] != g or t < lk["t_crack"]:
            continue
        hx, hy = x1 + 2, ml_to_y(lk["hole_ml"]) - (5 if lk["hole_ml"] == 0 else 0)
        for dx, dy in ((-7, -8), (6, -9), (-8, 6), (5, 8)):              # crack star
            d.line([(hx, hy), (hx + dx, hy + dy)], fill=HOLE_HEX, width=3)
        d.ellipse([hx - 5, hy - 5, hx + 5, hy + 5], fill=HOLE_HEX)
        gap = slot - w
        land_x = hx + max(12.0, min(30.0, gap * 0.4))
        drained = sim.drained_at(lk, sim.peak_ml(scene, g), t)
        first_col = sim.leak_flowing(scene, g, lk["t_crack"] + 1e-3)
        if drained > 0 and first_col:
            rw = min(gap * 0.42, 6 + 2.6 * math.sqrt(drained))
            d.ellipse([land_x - rw, TABLE_Y - 3, land_x + rw, TABLE_Y + 5], fill=sim.LIQUIDS[first_col])
        col = sim.leak_flowing(scene, g, t)
        if col:
            pts = [(hx + (land_x - hx) * k / 12, hy + (TABLE_Y - hy) * (k / 12) ** 2) for k in range(13)]
            d.line(pts, fill=sim.LIQUIDS[col], width=5)


# ------------------------------------------------------------------ jugs
def _pattern_tile(pattern, size):
    tile = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    d = ImageDraw.Draw(tile)
    ink = hex_rgb(JUG_PATTERN_HEX) + (255,)
    if pattern == "striped":            # bold vertical bars
        for x in range(2, size, 14):
            d.rectangle([x, 0, x + 6, size], fill=ink)
    elif pattern == "banded":           # bold horizontal bars
        for y in range(2, size, 14):
            d.rectangle([0, y, size, y + 6], fill=ink)
    elif pattern == "dotted":           # big polka dots
        for y in range(6, size, 14):
            for x in range(6 + (y // 14 % 2) * 7, size, 14):
                d.ellipse([x - 4.5, y - 4.5, x + 4.5, y + 4.5], fill=ink)
    elif pattern == "checkered":        # 10 px checks
        for y in range(0, size, 10):
            for x in range(0, size, 10):
                if (x // 10 + y // 10) % 2 == 0:
                    d.rectangle([x, y, x + 9, y + 9], fill=ink)
    elif pattern == "zigzag":           # thick horizontal zigzags
        for y in range(6, size, 13):
            pts = [(x, y + (5 if (x // 7) % 2 else -2)) for x in range(0, size + 7, 7)]
            d.line(pts, fill=ink, width=3)
    elif pattern == "wavy":             # smooth vertical waves
        for x in range(4, size, 16):
            pts = [(x + 4 * math.sin(y / 5.0), y) for y in range(0, size + 1, 2)]
            d.line(pts, fill=ink, width=4)
    elif pattern == "diagonal":         # thick 45-degree hatching
        for k in range(-size, size, 13):
            d.line([(k, size), (k + size, 0)], fill=ink, width=5)
    elif pattern == "crosshatch":       # thin hatching both ways
        for k in range(-size, size, 20):
            d.line([(k, size), (k + size, 0)], fill=ink, width=3)
            d.line([(k, 0), (k + size, size)], fill=ink, width=3)
    return tile


def _round_corners(pts, radii, steps=8):
    out = []
    n = len(pts)
    for i in range(n):
        p0, p1, p2 = np.array(pts[i - 1], float), np.array(pts[i], float), np.array(pts[(i + 1) % n], float)
        r = radii[i]
        a = p1 + (p0 - p1) / max(np.linalg.norm(p0 - p1), 1e-9) * min(r, np.linalg.norm(p0 - p1) / 2)
        b = p1 + (p2 - p1) / max(np.linalg.norm(p2 - p1), 1e-9) * min(r, np.linalg.norm(p2 - p1) / 2)
        for k in range(steps + 1):
            u = k / steps
            out.append(tuple((1 - u) ** 2 * a + 2 * u * (1 - u) * p1 + u * u * b))
    return out


def _bezier(p0, p1, p2, p3, n=40):
    p0, p1, p2, p3 = (np.array(p, float) for p in (p0, p1, p2, p3))
    return [tuple((1 - u) ** 3 * p0 + 3 * u * (1 - u) ** 2 * p1 + 3 * u * u * (1 - u) * p2 + u ** 3 * p3)
            for u in np.linspace(0, 1, n)]


class _JugArt:
    """Hi-res jug artwork for one pattern. Liquid is drawn in the jug's own frame
    with its surface horizontal in the WORLD, then the jug is rotated and scaled down."""

    def __init__(self, pattern, body_hex):
        size = SPRITE * SS
        o = size / 2
        sc = lambda pts: [(x * SS + o, y * SS + o) for x, y in pts]  # noqa: E731
        edge = hex_rgb(JUG_EDGE_HEX) + (255,)
        sp = Image.new("RGBA", (size, size), (0, 0, 0, 0))
        d = ImageDraw.Draw(sp)
        h = JUG_HANDLE
        curve = sc(_bezier(h[0], (h[0][0] - 34, h[0][1] - 6), (h[-1][0] - 30, h[-1][1] + 8), h[-1]))
        hw = 11 * SS
        d.line(curve, fill=edge, width=hw, joint="curve")
        for x, y in (curve[0], curve[-1]):
            d.ellipse([x - hw / 2, y - hw / 2, x + hw / 2, y + hw / 2], fill=edge)
        s = JUG_SCALE
        outer = sc(_round_corners(JUG_OUTER, [9 * s, 9 * s, 4 * s, 2 * s, 3 * s, 5 * s]))
        d.polygon(outer, fill=hex_rgb(body_hex) + (255,))
        body_mask = Image.new("L", (size, size), 0)
        md = ImageDraw.Draw(body_mask)
        md.polygon(outer, fill=255)
        (wx0, wy0), (wx1, wy1) = JUG_WINDOW[0], JUG_WINDOW[2]
        wbox = [wx0 * SS + o, wy0 * SS + o, wx1 * SS + o, wy1 * SS + o]
        md.rounded_rectangle(wbox, radius=6 * SS, fill=0)
        tile = _pattern_tile(pattern, SPRITE).resize((size, size), Image.NEAREST)
        sp.paste(tile, (0, 0), Image.composite(tile, Image.new("RGBA", tile.size), body_mask).split()[3])
        d.rounded_rectangle(wbox, radius=6 * SS, fill=hex_rgb(GLASS_TINT_HEX) + (255,), outline=edge, width=2 * SS)
        d.line(outer + [outer[0]], fill=edge, width=3 * SS, joint="curve")
        self.base = np.array(sp)
        inner = np.zeros((size, size), np.uint8)
        cv2.fillPoly(inner, [np.round(np.array(sc(JUG_INNER))).astype(np.int32)], 1)
        win = np.zeros((size, size), np.uint8)
        cv2.rectangle(win, (int(wbox[0]), int(wbox[1])), (int(wbox[2]), int(wbox[3])), 1, -1)
        win = cv2.erode(win, np.ones((4 * SS + 1, 4 * SS + 1), np.uint8))
        self.iy, self.ix = np.nonzero(inner)
        self.X = (self.ix - o) / SS
        self.Y = (self.iy - o) / SS
        self.visible = win[self.iy, self.ix].astype(bool)
        self.cap_px = int((self.Y >= JUG_BAND_Y).sum())   # upright area below the band = JUG_CAP
        self.cache = {}

    def image(self, ml, colour, deg):
        key = (round(ml, 3), colour, round(deg, 3))
        if key in self.cache:
            return self.cache[key]
        art = self.base.copy()
        k = min(int(round(ml / sim.JUG_CAP * self.cap_px)), len(self.X))
        if k > 0:
            a = math.radians(deg)
            height = -(self.X * math.sin(a) + self.Y * math.cos(a))   # world "up" of each inside pixel
            thr = np.partition(height, k - 1)[k - 1]
            sel = (height <= thr) & self.visible
            art[self.iy[sel], self.ix[sel]] = hex_rgb(sim.LIQUIDS[colour]) + (255,)
        im = Image.fromarray(art)
        if deg:
            im = im.rotate(-deg, resample=Image.BICUBIC, expand=True)
        w, h = im.size
        im = Image.fromarray(cv2.resize(np.array(im), (round(w / SS), round(h / SS)), interpolation=cv2.INTER_AREA))
        if not deg:                                    # parked jugs repeat a lot: keep them
            self.cache[key] = im
        return im


def jug_sprite(scene, j, t):
    """(RGBA image, top-left x, top-left y) of jug j at time t."""
    art = _base(scene)["jugs"][j]
    cx, cy, deg = jug_pose(scene, j, t)
    im = art.image(sim.jug_ml(scene, j, t), scene["jugs"][j]["colour"], deg)
    return im, int(round(cx - im.width / 2)), int(round(cy - im.height / 2))


# ------------------------------------------------------------------ frames
_BASE = {"scene": None}


def _base(scene):
    if _BASE["scene"] is not scene:
        _BASE.update(scene=scene, static=_static_layer(scene),
                     jugs=[_JugArt(jg["pattern"], scene["palette"]["jug_body"]) for jg in scene["jugs"]])
    return _BASE


def stream_on(scene, t):
    """The active pour if liquid is falling from its spout at time t."""
    p = active_pour(scene, t)
    if p and p["got_ml"] > 0 and p["t_pour"] < t < p["t_flow_end"]:
        return p
    return None


def frame(scene, t: float) -> Image.Image:
    base = _base(scene)
    img = base["static"].copy()
    d = ImageDraw.Draw(img)
    for g in range(scene["glasses"]["n"]):
        _draw_glass(d, scene, g, t)
    act = active_pour(scene, t)
    order = [j for j in range(len(scene["jugs"])) if not act or j != act["jug"]] + ([act["jug"]] if act else [])
    for j in order:
        im, x, y = jug_sprite(scene, j, t)
        img.paste(im, (x, y), im)
    p = stream_on(scene, t)
    if p:
        d = ImageDraw.Draw(img)
        gx = scene["glasses"]["cx"][p["glass"]]
        surf = ml_to_y(sum(ml for _, ml in sim.glass_layers(scene, p["glass"], t)))
        d.line([(gx, Y_TOP - SPOUT_GAP + 2), (gx, surf)], fill=sim.LIQUIDS[scene["jugs"][p["jug"]]["colour"]], width=7)
    x, y = scene["clock_xy"]
    d.text((x, y), clock_text(int(round(t * FPS))), font=font(scene["label_px"]), fill=CLOCK_HEX, anchor="ra")
    return img
