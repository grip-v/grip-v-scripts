"""PIL renderer for tangled_braid.

The braid comes out of a loom bar and scrolls away from it (up, down, left
or right, by layout); everything farther than the viewing window from the
bar fades into the background, so old crossings leave the frame. At each
crossing the over-strand is drawn last on top of a background halo, which
cuts a visible gap in the under-strand. Line-style phases (dashes, dots,
bands, beads) follow the braid's length coordinate, so they move with the
braid. The only text is the clock (top-left).

`track(scene, i)` is the renderer's own per-frame state: where every strand
leaves the bar (its across-coordinate u) and every crossing on screen with
the strand drawn under and the one drawn over. `frame_at` draws exactly
that; answers.py reads only `track`.
"""
from __future__ import annotations
import bisect
import functools
import json
import math
import sys
from pathlib import Path

from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from common.video5 import font, hex_rgb  # noqa: E402

import sim  # noqa: E402

W, H, FPS = sim.W, sim.H, sim.FPS
INK = (40, 40, 46)
TIMER_BG, TIMER_FG, TIMER_EDGE = (24, 26, 32), (255, 255, 255), (200, 200, 205)
SAMPLE = 3.0                       # px between polyline samples along the braid
EXT = 12.0                         # braid drawn this far past the bar line, under the bar (no end-cut at the bar)


def _mix(a, b, t):
    return tuple(int(round(a[k] + (b[k] - a[k]) * t)) for k in range(3))


def tint(c):
    return _mix(c, (255, 255, 255), 0.55)


def dark(c):
    return _mix(c, (0, 0, 0), 0.55)


def ease(curve: str, t: float) -> float:
    t = min(1.0, max(0.0, t))
    if curve == "cosine":
        return 0.5 - 0.5 * math.cos(math.pi * t)
    if curve == "smoother":
        return t * t * t * (t * (6 * t - 15) + 10)
    return t * t * (3 - 2 * t)


# ---------------------------------------------------------------- static geometry (own replay)

@functools.lru_cache(maxsize=16)
def _geom(key: str) -> dict:
    scene = json.loads(key)
    order = list(scene["start"])
    orders = [list(order)]
    for st in scene["steps"]:
        for mv in st["moves"]:
            order[mv["i"]], order[mv["i"] + 1] = order[mv["i"] + 1], order[mv["i"]]
        orders.append(list(order))
    S = [st["S"] for st in scene["steps"]]
    lay = scene["layout"]
    cols = [lay["u0"] + lay["gap"] * p for p in range(len(order))]
    crosses = []                   # per step: list of (under, over, i)
    for k, st in enumerate(scene["steps"]):
        o = orders[k]
        row = []
        for mv in st["moves"]:
            lo, hi = o[mv["i"]], o[mv["i"] + 1]
            over = lo if mv["over"] == "low" else hi
            row.append((hi if over == lo else lo, over, mv["i"]))
        crosses.append(row)
    return {"orders": orders, "S": S, "L": scene["timing"]["L"], "cols": cols, "crosses": crosses,
            "curve": scene["style"]["curve"]}


def _key(scene) -> str:
    return json.dumps(scene, sort_keys=True)


def u_at(G, name: str, s: float) -> float:
    """Across-coordinate of a strand at braid length s."""
    S, L = G["S"], G["L"]
    k = bisect.bisect_right(S, s) - 1          # last step starting at or before s
    if k < 0:
        return G["cols"][G["orders"][0].index(name)]
    if s >= S[k] + L:
        return G["cols"][G["orders"][k + 1].index(name)]
    p0 = G["orders"][k].index(name)
    p1 = G["orders"][k + 1].index(name)
    if p0 == p1:
        return G["cols"][p0]
    e = ease(G["curve"], (s - S[k]) / L)
    return G["cols"][p0] + (G["cols"][p1] - G["cols"][p0]) * e


def to_screen(scene, s_head: float, s: float, u: float):
    lay = scene["layout"]
    along = lay["loom"] - lay["grow"] * (s_head - s)
    return (u, along) if lay["orient"] == "v" else (along, u)


# ---------------------------------------------------------------- per-frame state

def track(scene, i: int) -> dict:
    """What frame i shows: u of every strand at the bar, and every crossing
    on screen as (step, under, over, s of the meeting point)."""
    G = _geom(_key(scene))
    s_head = sim.head_s(scene, i)
    Wn = sim.window(scene)
    head = {n: u_at(G, n, s_head) for n in scene["start"]}
    cr = []
    for k, row in enumerate(G["crosses"]):
        S = G["S"][k]
        if S < s_head and S + G["L"] > s_head - Wn:
            for under, over, _ in row:
                cr.append({"step": k, "under": under, "over": over, "s_mid": S + G["L"] / 2})
    return {"s": s_head, "head": head, "crossings": cr}


# ---------------------------------------------------------------- drawing

def draw_strand(d, pts, ss, st: dict, color, bg):
    w = sim.WIDTHS[st["thickness"]]
    style = st["style"]
    if len(pts) < 2:
        return
    if style == "outlined":
        d.line(pts, fill=INK, width=w + 4, joint="curve")
        d.line(pts, fill=color, width=w - 2, joint="curve")
    elif style == "double":
        d.line(pts, fill=color, width=w + 4, joint="curve")
        d.line(pts, fill=bg, width=max(2, w // 2 - 1), joint="curve")
    elif style == "beaded":
        d.line(pts, fill=color, width=max(4, w // 2 + 1), joint="curve")
    elif style == "dashed":
        d.line(pts, fill=color, width=max(4, w // 3 + 1), joint="curve")
    else:
        d.line(pts, fill=color, width=w, joint="curve")
    if style in ("dashed", "banded"):
        period, on = (28, 16) if style == "dashed" else (20, 5)
        fill = color if style == "dashed" else dark(color)
        for (a, va), (b, _) in zip(zip(pts, ss), zip(pts[1:], ss[1:])):
            if (va % period) < on:
                d.line([a, b], fill=fill, width=w)
    if style in ("dotted", "beaded"):
        period = 18 if style == "dotted" else 24
        rr = w * 0.22 if style == "dotted" else w * 0.75
        fill = (255, 255, 255) if style == "dotted" else color
        for (a, va), (b, vb) in zip(zip(pts, ss), zip(pts[1:], ss[1:])):
            if int(va // period) != int(vb // period):
                x, y = b
                d.ellipse([x - rr, y - rr, x + rr, y + rr], fill=fill)


@functools.lru_cache(maxsize=8)
def _static(key: str):
    """Background image (colour + pattern) and the window mask."""
    scene = json.loads(key)
    stl, lay = scene["style"], scene["layout"]
    bg = hex_rgb(stl["bg_hex"])
    img = Image.new("RGB", (W, H), bg)
    d = ImageDraw.Draw(img)
    step = stl["pattern_step"]
    faint = _mix(bg, (0, 0, 0), 0.06)
    if stl["pattern"] == "grid":
        for x in range(0, W, step):
            d.line([(x, 0), (x, H)], fill=faint, width=1)
        for y in range(0, H, step):
            d.line([(0, y), (W, y)], fill=faint, width=1)
    elif stl["pattern"] == "dots":
        for x in range(step // 2, W, step):
            for y in range(step // 2, H, step):
                d.ellipse([x - 2, y - 2, x + 2, y + 2], fill=faint)
    elif stl["pattern"] == "gradient":
        for y in range(H):
            d.line([(0, y), (W, y)], fill=_mix(bg, (0, 0, 0), 0.08 * y / H))
    elif stl["pattern"] == "hatch":
        for x in range(-H, W, step):
            d.line([(x, 0), (x + H, H)], fill=faint, width=1)
    mask = Image.new("L", (W, H), 0)
    md = ImageDraw.Draw(mask)
    loom, far, g = lay["loom"], lay["far"], lay["grow"]
    Wn = abs(loom - far)
    for j in range(int(Wn) + 1):
        dist = j                                   # distance from the bar
        a = 255 if dist <= Wn - sim.FADE else int(255 * (Wn - dist) / sim.FADE)
        c = loom - g * j
        if lay["orient"] == "v":
            md.line([(0, c), (W, c)], fill=a)
        else:
            md.line([(c, 0), (c, H)], fill=a)
    return img, mask


def _bar(d, scene):
    lay = scene["layout"]
    cols = sim.columns(scene)
    col = hex_rgb(sim.LOOMS[scene["style"]["loom"]])
    g = lay["grow"]
    a0, a1 = lay["loom"] + g * 3, lay["loom"] + g * 25
    lo, hi = cols[0] - 42, cols[-1] + 42
    if lay["orient"] == "v":
        d.rounded_rectangle([lo, min(a0, a1), hi, max(a0, a1)], radius=7, fill=col)
        for u in cols:
            c = (a0 + a1) / 2
            d.ellipse([u - 5, c - 5, u + 5, c + 5], fill=_mix(col, (255, 255, 255), 0.35))
    else:
        d.rounded_rectangle([min(a0, a1), lo, max(a0, a1), hi], radius=7, fill=col)
        for u in cols:
            c = (a0 + a1) / 2
            d.ellipse([c - 5, u - 5, c + 5, u + 5], fill=_mix(col, (255, 255, 255), 0.35))


def draw_timer(d, scene, i: int):
    tb = scene["timer"]["box"]
    d.rounded_rectangle(tb, radius=10, fill=TIMER_BG, outline=TIMER_EDGE, width=2)
    d.text(((tb[0] + tb[2]) / 2, (tb[1] + tb[3]) / 2), sim.clock_text(sim.timer_tenths(i)),
           fill=TIMER_FG, font=font(scene["timer"]["size"]), anchor="mm")


def _samples(a: float, b: float) -> list:
    """a, every multiple of SAMPLE strictly inside (a, b), b: grid-aligned so
    pattern phases don't jitter from frame to frame."""
    k0, k1 = math.floor(a / SAMPLE) + 1, math.ceil(b / SAMPLE) - 1
    return [a] + [k * SAMPLE for k in range(k0, k1 + 1)] + [b]


def frame_at(scene, i: int) -> Image.Image:
    key = _key(scene)
    G = _geom(key)
    bgimg, mask = _static(key)
    bg = hex_rgb(scene["style"]["bg_hex"])
    tr = track(scene, i)
    s_head = tr["s"]
    Wn = sim.window(scene)
    lo_s = s_head - Wn - 4
    sty = {s["name"]: s for s in scene["strands"]}
    col = {s["name"]: hex_rgb(s["color"]) for s in scene["strands"]}
    layer = bgimg.copy()
    d = ImageDraw.Draw(layer)
    ss = _samples(lo_s, s_head + EXT)
    for n in scene["start"]:
        pts = [to_screen(scene, s_head, s, u_at(G, n, s)) for s in ss]
        draw_strand(d, pts, ss, sty[n], col[n], bg)
    # halos: cut the background (with its pattern) back in along each over-strand's crossing
    hm = Image.new("L", (W, H), 0)
    hd = ImageDraw.Draw(hm)
    overs = []
    for c in tr["crossings"]:
        S, L, o = G["S"][c["step"]], G["L"], c["over"]
        h0, h1 = max(lo_s, S + 0.22 * L), min(s_head + EXT, S + 0.78 * L)
        if h1 > h0:
            hs = _samples(h0, h1)
            hd.line([to_screen(scene, s_head, s, u_at(G, o, s)) for s in hs], fill=255,
                    width=sim.WIDTHS[sty[o]["thickness"]] + 2 * scene["style"]["halo"] + 4, joint="curve")
        a0, a1 = max(lo_s, S), min(s_head + EXT, S + L)
        if a1 > a0:
            overs.append((o, _samples(a0, a1)))
    if overs:
        layer.paste(bgimg, (0, 0), hm)
    for o, os_ in overs:
        draw_strand(d, [to_screen(scene, s_head, s, u_at(G, o, s)) for s in os_], os_, sty[o], col[o], bg)
    img = Image.composite(layer, bgimg, mask)
    d = ImageDraw.Draw(img)
    _bar(d, scene)
    draw_timer(d, scene, i)
    return img


def frame(scene, t: float) -> Image.Image:
    return frame_at(scene, int(round(t * FPS)))


def duration(scene) -> float:
    return scene["n_frames"] / FPS
