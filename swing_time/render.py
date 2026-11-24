"""PIL renderer for swing_time.

A board of squares with coloured posts (circle / square / triangle /
diamond) at grid corners, a tinted start square, a checkered goal square
and, in the pits scenario, hatched pit squares. The particle hooks a rope
on a post and swings to its next square: the rope, the arc it is tracing
and a ring around the pivot post exist ONLY while the swing is in progress.
Every landing leaves a persistent outline marker on that square, so the
number of swings can be read from any later frame.

The ONLY text in the frame is the clock in the top-left box (seconds to
0.1 s, truncated, from the frame number); the board is always drawn right
of x = 240 px so nothing covers it. Pure function of (scene, frame).
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

FPS = sim.FPS
SIZE = (1280, 720)
CLOCK_BOX = sim.CLOCK_BOX


def duration(scene) -> float:
    return sim.n_frames(scene) / FPS


def n_frames(scene) -> int:
    return sim.n_frames(scene)


def to_px(scene, x, y):
    """World (x right, y up, squares 1 unit) -> screen pixels."""
    st = scene["style"]
    return st["x0"] + x * st["cell"], st["y0"] + (scene["h"] - y) * st["cell"]


def clock_text(frame: int) -> str:
    return f"{sim.clock_tenths(frame) / 10:.1f} s"


def _smooth(u):
    return u * u * (3 - 2 * u)


def state(scene, f: int) -> dict:
    """What frame f shows: the particle's world position, the pivot post's corner
    (or None between swings), the signed angle turned so far on the current
    swing and the squares it has landed on so far."""
    ev = sim.events(scene)
    here = tuple(scene["start"])
    landed = []
    by_colour = {p["colour"]: tuple(p["xy"]) for p in scene["posts"]}
    for (s0, s1), sw in zip(ev, scene["swings"]):
        if f < s0:
            break
        post = by_colour[sw["post"]]
        if f < s1:
            c = sim.centre(here)
            r = math.dist(c, post)
            a0 = math.atan2(c[1] - post[1], c[0] - post[0])
            turned = sim.sweep_of(here, tuple(sw["land"]), post, sw["dir"]) * _smooth((f - s0) / (s1 - s0))
            a = a0 + turned
            return {"pos": (post[0] + r * math.cos(a), post[1] + r * math.sin(a)), "pivot": post,
                    "from": here, "turned": turned, "landed": landed}
        here = tuple(sw["land"])
        landed.append(here)
    return {"pos": sim.centre(here), "pivot": None, "from": None, "turned": 0.0, "landed": landed}


def _poly_post(d, shape, cx, cy, s, fill, outline):
    r = s / 2
    if shape == "circle":
        d.ellipse([cx - r, cy - r, cx + r, cy + r], fill=fill, outline=outline, width=3)
    elif shape == "square":
        r *= 0.9
        d.rectangle([cx - r, cy - r, cx + r, cy + r], fill=fill, outline=outline, width=3)
    elif shape == "triangle":
        r *= 1.15
        pts = [(cx + r * math.cos(math.radians(a)), cy - r * math.sin(math.radians(a))) for a in (90, 210, 330)]
        d.polygon(pts, fill=fill, outline=outline, width=3)
    else:
        r *= 1.12
        d.polygon([(cx, cy - r), (cx + r, cy), (cx, cy + r), (cx - r, cy)], fill=fill, outline=outline, width=3)


def _dash(d, a, b, col, w, on=12, off=8):
    L = math.dist(a, b)
    if L < 1e-6:
        return
    k = 0.0
    while k < L:
        e = min(L, k + on)
        d.line([(a[0] + (b[0] - a[0]) * k / L, a[1] + (b[1] - a[1]) * k / L),
                (a[0] + (b[0] - a[0]) * e / L, a[1] + (b[1] - a[1]) * e / L)], fill=col, width=w)
        k += on + off


@functools.lru_cache(maxsize=32)
def _static(key):
    sc = json.loads(key)
    st = sc["style"]
    img = Image.new("RGB", SIZE, tuple(st["bg"]))
    d = ImageDraw.Draw(img)
    cell, w, h = st["cell"], sc["w"], sc["h"]
    pits = {tuple(p) for p in sc["pits"]}
    for x in range(w):
        for y in range(h):
            (ax, ay), (bx, by) = to_px(sc, x, y + 1), to_px(sc, x + 1, y)
            col = st["board"][(x + y) % 2] if st["board_pat"] == "checker" else st["board"][0]
            d.rectangle([ax, ay, bx, by], fill=tuple(col))
            if [x, y] == sc["start"]:
                d.rectangle([ax, ay, bx, by], fill=tuple(st["start"]))
            elif [x, y] == sc["goal"]:
                n = 4
                for i in range(n):
                    for j in range(n):
                        d.rectangle([ax + i * cell / n, ay + j * cell / n, ax + (i + 1) * cell / n,
                                     ay + (j + 1) * cell / n], fill=tuple(st["goal"][(i + j) % 2]))
            elif (x, y) in pits:
                d.rectangle([ax, ay, bx, by], fill=tuple(st["pit"][0]))
                for k in range(-3, 4):
                    d.line(_hatch(ax, ay, cell, k * cell / 4), fill=tuple(st["pit"][1]), width=3)
    gcol = tuple(st["grid"])
    segs = [((to_px(sc, 0, y)), to_px(sc, w, y)) for y in range(h + 1)] + \
           [((to_px(sc, x, 0)), to_px(sc, x, h)) for x in range(w + 1)]
    for a, b in segs:
        if st["grid_style"] == "solid":
            d.line([a, b], fill=gcol, width=3)
        elif st["grid_style"] == "dashed":
            _dash(d, a, b, gcol, 3, 14, 8)
        else:
            _dash(d, a, b, gcol, 3, 4, 7)
    return img


def _hatch(ax, ay, cell, o):
    """One diagonal hatch line (offset o) clipped to the square [ax, ax+cell] x [ay, ay+cell]."""
    # line x - ax + (y - ay) = cell/2 + o  (anti-diagonal family)
    c = cell / 2 + o + cell / 2
    pts = []
    for x in (0.0, cell):
        y = c - x
        if 0 <= y <= cell:
            pts.append((ax + x, ay + y))
    for y in (0.0, cell):
        x = c - y
        if 0 <= x <= cell:
            pts.append((ax + x, ay + y))
    pts = sorted(set(pts))
    return [pts[0], pts[-1]] if len(pts) >= 2 else [(ax, ay), (ax, ay)]


def _marker(d, st, cx, cy):
    m = st["marker"]
    r = m["px"] / 2
    col = tuple(m["col"])
    if m["kind"] == "ring":
        d.ellipse([cx - r, cy - r, cx + r, cy + r], outline=col, width=4)
    elif m["kind"] == "box":
        r *= 0.92
        d.rectangle([cx - r, cy - r, cx + r, cy + r], outline=col, width=4)
    else:
        r *= 1.15
        d.polygon([(cx, cy - r), (cx + r, cy), (cx, cy + r), (cx - r, cy)], outline=col, width=4)


def _under(st):
    """Under-stroke colour for the rope and pivot ring: white on light boards, black on dark ones."""
    return (255, 255, 255) if st["theme"] == "light" else (0, 0, 0)


def _clock(d, frame):
    d.rounded_rectangle(CLOCK_BOX, radius=12, fill=(10, 10, 14), outline=(235, 235, 235), width=3)
    d.text(((CLOCK_BOX[0] + CLOCK_BOX[2]) / 2, (CLOCK_BOX[1] + CLOCK_BOX[3]) / 2),
           clock_text(frame), fill=(255, 255, 255), font=font(44), anchor="mm")


def frame(scene, t: float) -> Image.Image:
    f = int(round(t * FPS))
    st = scene["style"]
    key = json.dumps({k: scene[k] for k in ("style", "w", "h", "pits", "start", "goal")}, sort_keys=True)
    img = _static(key).copy()
    d = ImageDraw.Draw(img)
    s = state(scene, f)

    for sq in s["landed"]:
        _marker(d, st, *to_px(scene, *sim.centre(sq)))

    if s["pivot"] is not None:
        post, here = s["pivot"], s["from"]
        c = sim.centre(here)
        r = math.dist(c, post)
        a0 = math.atan2(c[1] - post[1], c[0] - post[0])
        total = s["turned"]
        n = max(2, int(abs(total) * r * 24) + 2)
        pts = [to_px(scene, post[0] + r * math.cos(a0 + total * j / (n - 1)),
                     post[1] + r * math.sin(a0 + total * j / (n - 1))) for j in range(n)]
        tc, tw = tuple(st["trail"]["col"]), st["trail"]["w"]
        if st["trail"]["style"] == "solid":
            d.line(pts, fill=tc, width=tw, joint="curve")
        else:
            acc, last = 0.0, pts[0]
            for p in pts[1:]:
                acc += math.dist(last, p)
                last = p
                if acc >= 9:
                    acc = 0.0
                    d.ellipse([p[0] - tw / 2 - 1, p[1] - tw / 2 - 1, p[0] + tw / 2 + 1, p[1] + tw / 2 + 1], fill=tc)
        a, b = to_px(scene, *post), to_px(scene, *s["pos"])
        rc, rw = tuple(st["rope"]["col"]), st["rope"]["w"]
        d.line([a, b], fill=_under(st), width=rw + 4)          # contrast under-stroke: readable over pits too
        if st["rope"]["style"] == "solid":
            d.line([a, b], fill=rc, width=rw)
        elif st["rope"]["style"] == "dashed":
            _dash(d, a, b, rc, rw, 12, 7)
        else:
            d.line([a, b], fill=rc, width=3)
            L = math.dist(a, b)
            for k in range(1, int(L // 16) + 1):
                u = k * 16 / L
                bx, by = a[0] + (b[0] - a[0]) * u, a[1] + (b[1] - a[1]) * u
                d.ellipse([bx - 4, by - 4, bx + 4, by + 4], fill=rc)

    out = tuple(st["outline"])
    for p in scene["posts"]:
        cx, cy = to_px(scene, *p["xy"])
        if s["pivot"] is not None and tuple(p["xy"]) == tuple(s["pivot"]):
            R = st["post_px"] / 2 + 10
            d.ellipse([cx - R - 2, cy - R - 2, cx + R + 2, cy + R + 2], outline=_under(st), width=8)
            d.ellipse([cx - R, cy - R, cx + R, cy + R], outline=tuple(st["rope"]["col"]), width=4)
        _poly_post(d, p["shape"], cx, cy, st["post_px"], sim.POST_COLOURS[p["colour"]], out)

    cx, cy = to_px(scene, *s["pos"])
    pp = st["particle"]
    R = pp["px"] / 2
    fill, rim = tuple(pp["fill"]), tuple(pp["rim"])
    d.ellipse([cx - R, cy - R, cx + R, cy + R], fill=fill, outline=rim, width=3)
    if pp["pattern"] == "bullseye":
        d.ellipse([cx - R * 0.45, cy - R * 0.45, cx + R * 0.45, cy + R * 0.45], fill=rim)
    elif pp["pattern"] == "ring":
        d.ellipse([cx - R * 0.6, cy - R * 0.6, cx + R * 0.6, cy + R * 0.6], outline=rim, width=3)
    _clock(d, f)
    return img
