"""PIL renderer for cipher_card_trace (v3).

The wheel: the letter ring (outer) and the symbol ring (inner, red/black
card ranks), both drawn from `scene["labels"]`; one of them turns when a
joker (or, in `trigger`, a card) has landed. The pile(s): every card slides
in from off screen and lands exactly on the pile, covering what is there.
The only text: the ring letters and ranks, the cards' own corner ranks and
the clock (top-left). No decoded letter is ever drawn.

`track(scene, i)` is the renderer's own per-frame state (ring rotations in
degrees, every card drawn and where); `frame_at` draws exactly that.
answers.py reads only `track` and the scene's static description.
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

import sim  # noqa: E402

W, H, FPS = 1280, 720, 24
STEP = 360 / 26
RED, BLACK = (200, 20, 35), (20, 20, 24)
INK = (30, 28, 24)
TIMER_BG, TIMER_FG, TIMER_EDGE = (24, 26, 32), (255, 255, 255), (200, 200, 205)
LINE = (120, 112, 96)


def _smooth(u: float) -> float:
    return u * u * (3 - 2 * u)


def _ease_out(u: float) -> float:
    return 1 - (1 - u) ** 3


def ring_rotation(scene, i: int) -> dict:
    """Rotation of each ring at frame i, in degrees clockwise."""
    rot = {"inner": scene["start"]["inner"] * STEP, "outer": scene["start"]["outer"] * STEP}
    mv = scene["moving"]
    for d in scene["deal"]:
        if not d["turn"] or i < d["ts"]:
            continue
        u = 1.0 if i >= d["te"] else _smooth((i - d["ts"]) / (d["te"] - d["ts"]))
        rot[mv] += d["turn"] * STEP * u
    return rot


def track(scene, i: int) -> dict:
    """What frame i shows: ring rotations and the cards drawn, bottom to top
    as (deal index, centre x, centre y). Only the top card of each pile is
    drawn (the cards below are exactly covered), then any card sliding in."""
    g = scene["geometry"]
    tops = {}
    sliding = []
    for k, d in enumerate(scene["deal"]):
        if i >= d["land"]:
            tops[d["pile"]] = k
        elif i >= d["s"]:
            u = _ease_out((i - d["s"]) / (d["land"] - d["s"]))
            pile = g["piles"][d["pile"]]
            ex, ey = sim.entry_point(g, pile)
            sliding.append((k, ex + (pile[0] - ex) * u, ey + (pile[1] - ey) * u))
    drawn = [(k, *g["piles"][p]) for p, k in sorted(tops.items())] + sliding
    return {"rot": ring_rotation(scene, i), "cards": [(k, float(x), float(y)) for k, x, y in drawn]}


# ---------------------------------------------------------------- drawing helpers

def _mix(a, b, t):
    return tuple(int(round(a[k] + (b[k] - a[k]) * t)) for k in range(3))


def _ring_poly(cx, cy, a0, a1, r_in, r_out, n=6):
    pts = [(cx + r_out * math.cos(math.radians(a0 + (a1 - a0) * k / n)),
            cy + r_out * math.sin(math.radians(a0 + (a1 - a0) * k / n))) for k in range(n + 1)]
    pts += [(cx + r_in * math.cos(math.radians(a1 - (a1 - a0) * k / n)),
             cy + r_in * math.sin(math.radians(a1 - (a1 - a0) * k / n))) for k in range(n + 1)]
    return pts


def _ink(lb):
    return {"dark": INK, "red": RED, "black": BLACK}[lb["ink"]]


def draw_ring(d, scene, which: str, rot_deg: float):
    g, st = scene["geometry"], scene["style"]
    cx, cy = g["cx"], g["cy"]
    r_in, r_out = (g["r2"], g["r3"]) if which == "outer" else (g["r1"], g["r2"])
    fa, fb = (st["outer_a"], st["outer_b"]) if which == "outer" else (st["inner_a"], st["inner_b"])
    for lb in scene["labels"]:
        if lb["ring"] != which:
            continue
        a = -90 + lb["index"] * STEP + rot_deg
        d.polygon(_ring_poly(cx, cy, a - STEP / 2, a + STEP / 2, r_in, r_out),
                  fill=tuple(fa if lb["index"] % 2 else fb))
        b = math.radians(a - STEP / 2)
        d.line([(cx + r_in * math.cos(b), cy + r_in * math.sin(b)),
                (cx + r_out * math.cos(b), cy + r_out * math.sin(b))], fill=LINE, width=3)
    for lb in scene["labels"]:
        if lb["ring"] != which:
            continue
        a = math.radians(-90 + lb["index"] * STEP + rot_deg)
        d.text((cx + lb["r"] * math.cos(a), cy + lb["r"] * math.sin(a)), lb["text"], fill=_ink(lb),
               font=font(lb["size"]), anchor="mm")
    d.ellipse([cx - r_out, cy - r_out, cx + r_out, cy + r_out], outline=LINE, width=3)


def _heart(cx, cy, s):
    pts = []
    for k in range(48):
        a = 2 * math.pi * k / 48
        x = 16 * math.sin(a) ** 3
        y = -(13 * math.cos(a) - 5 * math.cos(2 * a) - 2 * math.cos(3 * a) - math.cos(4 * a))
        pts.append((cx + x * s / 17, cy + y * s / 17 + s * 0.1))
    return pts


def pip(d, suit, cx, cy, s):
    col = RED if suit in ("hearts", "diamonds") else BLACK
    if suit == "hearts":
        d.polygon(_heart(cx, cy, s), fill=col)
    elif suit == "diamonds":
        d.polygon([(cx, cy - s), (cx + s * 0.72, cy), (cx, cy + s), (cx - s * 0.72, cy)], fill=col)
    elif suit == "spades":
        d.polygon([(x, 2 * cy - y - s * 0.2) for x, y in _heart(cx, cy, s * 0.95)], fill=col)
        d.polygon([(cx, cy + s * 0.2), (cx + s * 0.38, cy + s), (cx - s * 0.38, cy + s)], fill=col)
    else:
        r = s * 0.42
        for dx, dy in ((0, -0.45), (-0.48, 0.12), (0.48, 0.12)):
            d.ellipse([cx + dx * s - r, cy + dy * s - r, cx + dx * s + r, cy + dy * s + r], fill=col)
        d.polygon([(cx, cy), (cx + s * 0.38, cy + s), (cx - s * 0.38, cy + s)], fill=col)


def _star(cx, cy, r_out, r_in, n=5):
    return [(cx + (r_out if k % 2 == 0 else r_in) * math.cos(math.radians(-90 + 180 * k / n)),
             cy + (r_out if k % 2 == 0 else r_in) * math.sin(math.radians(-90 + 180 * k / n))) for k in range(2 * n)]


def corner_font(scene) -> int:
    return max(36, int(round(scene["geometry"]["cw"] * 0.22)))


def draw_card(d, scene, c: dict, x: float, y: float):
    g, st = scene["geometry"], scene["style"]
    cw, ch = g["cw"], g["ch"]
    x0, y0, x1, y1 = x - cw / 2, y - ch / 2, x + cw / 2, y + ch / 2
    rad = int(cw * 0.08)
    if st["shadow"]:
        d.rounded_rectangle([x0 + 5, y0 + 7, x1 + 5, y1 + 7], rad, fill=_mix(tuple(st["bg"]), (0, 0, 0), 0.45))
    if c["kind"] == "joker":
        jc = tuple(st["joker_rgb"])
        d.rounded_rectangle([x0, y0, x1, y1], rad, fill=(255, 255, 255), outline=jc, width=7)
        d.polygon(_star(x, y, cw * 0.34, cw * 0.14), fill=jc)
        for sx, sy in ((x0 + cw * 0.18, y0 + cw * 0.18), (x1 - cw * 0.18, y1 - cw * 0.18)):
            d.polygon(_star(sx, sy, cw * 0.09, cw * 0.04), fill=jc)
        return
    col = RED if c["color"] == "red" else BLACK
    d.rounded_rectangle([x0, y0, x1, y1], rad, fill=(255, 255, 255), outline=tuple(st["card_edge"]), width=3)
    f = font(corner_font(scene))
    cs = cw * 0.17
    d.text((x0 + cs, y0 + cs * 1.05), c["rank"], fill=col, font=f, anchor="mm")
    pip(d, c["suit"], x0 + cs, y0 + cs * 2.35, cw * 0.075)
    d.text((x1 - cs, y1 - cs * 1.05), c["rank"], fill=col, font=f, anchor="mm")
    pip(d, c["suit"], x1 - cs, y1 - cs * 2.35, cw * 0.075)
    pip(d, c["suit"], x, y + cw * 0.02, cw * st["pip_scale"])


@functools.lru_cache(maxsize=8)
def _static(key: str) -> Image.Image:
    scene = json.loads(key)
    g, st = scene["geometry"], scene["style"]
    bg = tuple(st["bg"])
    img = Image.new("RGB", (W, H), bg)
    d = ImageDraw.Draw(img)
    light = _mix(bg, (255, 255, 255), 0.07)
    if st["bg_pattern"] == "gradient":
        for y in range(H):
            d.line([(0, y), (W, y)], fill=_mix(bg, (0, 0, 0), 0.25 * y / (H - 1)))
    elif st["bg_pattern"] == "dots":
        for x in range(14, W, 34):
            for y in range(14, H, 34):
                d.ellipse([x - 2, y - 2, x + 2, y + 2], fill=light)
    elif st["bg_pattern"] == "weave":
        for x in range(-H, W, 28):
            d.line([(x, H), (x + H, 0)], fill=light, width=2)
            d.line([(x, 0), (x + H, H)], fill=light, width=2)
    tb = scene["timer"]["box"]
    d.rectangle([tb[0] - 6, tb[1] - 6, tb[2] + 6, tb[3] + 6], fill=bg)
    for px, py in g["piles"]:
        x0, y0 = px - g["cw"] / 2, py - g["ch"] / 2
        d.rounded_rectangle([x0, y0, x0 + g["cw"], y0 + g["ch"]], int(g["cw"] * 0.08),
                            outline=_mix(bg, (255, 255, 255), 0.3), width=3)
    if scene["moving"] != "outer":
        draw_ring(d, scene, "outer", scene["start"]["outer"] * STEP)
    return img


def draw_timer(d, scene, i: int):
    tb = scene["timer"]["box"]
    d.rounded_rectangle(tb, radius=10, fill=TIMER_BG, outline=TIMER_EDGE, width=2)
    d.text(((tb[0] + tb[2]) / 2, (tb[1] + tb[3]) / 2), sim.clock_text(sim.timer_tenths(i)),
           fill=TIMER_FG, font=font(scene["timer"]["size"]), anchor="mm")


def frame_at(scene, i: int) -> Image.Image:
    g, st = scene["geometry"], scene["style"]
    img = _static(json.dumps(scene, sort_keys=True)).copy()
    d = ImageDraw.Draw(img)
    tr = track(scene, i)
    if scene["moving"] == "outer":
        draw_ring(d, scene, "outer", tr["rot"]["outer"])
    draw_ring(d, scene, "inner", tr["rot"]["inner"])
    cx, cy, r1 = g["cx"], g["cy"], g["r1"]
    d.ellipse([cx - r1, cy - r1, cx + r1, cy + r1], fill=tuple(st["hub"]), outline=LINE, width=3)
    d.ellipse([cx - 14, cy - 14, cx + 14, cy + 14], fill=_mix(tuple(st["hub"]), (255, 255, 255), 0.6))
    for k, x, y in tr["cards"]:
        draw_card(d, scene, scene["deal"][k], x, y)
    draw_timer(d, scene, i)
    return img


def frame(scene, t: float) -> Image.Image:
    return frame_at(scene, int(round(t * FPS)))


def duration(scene) -> float:
    return scene["n_frames"] / FPS
