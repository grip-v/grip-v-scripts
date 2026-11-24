"""PIL renderer for rolling_races (1280x720, 24 fps).

Every track is drawn from the shared start ledge to the shared finish hole in
its ball's colour, any gap left open with a short take-off lip at its near
edge. A ball that reaches the finish drops into the hole and fills its own
slot (a ring in its colour) in the finish rack beside the hole; a ball that
falls through a gap comes to rest on the hatched pit floor. The rack's slot
order is fixed and random, so it never shows the finishing order.

A ball in the air (jumping a gap or falling) is drawn behind the other
colours' tracks: each ball rides only its own track, so a ball flying or
falling across another colour's track passes behind it. A rolling ball is
drawn on top. A short bar (the start gate) stands just downhill of the start until the
release frame. Text: the clock (a box in the top corner opposite the start), nothing else.
Balls are named by colour. Pure function of (scene, frame).
"""
from __future__ import annotations
import functools
import json
import math
import sys
from pathlib import Path

from PIL import Image, ImageChops, ImageDraw

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from common.video5 import font  # noqa: E402
import sim  # noqa: E402

FPS = sim.FPS
SIZE = sim.SIZE
CLOCK_PX = 44
LIP_PX = 16


def duration(scene) -> float:
    return sim.duration(scene)


def n_frames(scene) -> int:
    return sim.n_frames(scene)


def clock_text(frame: int) -> str:
    return f"{sim.clock_tenths(frame) / 10:.1f} s"


def colour(scene, name):
    return sim.BALLS[scene["style"]["theme"]][name]


def state(scene, f: int) -> dict:
    """What frame f draws: balls {colour: (x, y)} (centres, px; balls resting on the pit floor included,
    balls in the finish hole left out), rack [colours whose slot is filled], gate (True while the start
    gate is closed: every frame before the release frame) and air [colours flying or falling]."""
    balls, rack, air = {}, [], []
    for b in scene["balls"]:
        p = sim.ball_centre(scene, b, f)
        if p is None:
            rack.append(b["colour"])
        else:
            balls[b["colour"]] = (p[0], p[1])
            if p[2]:
                air.append(b["colour"])
    return {"balls": balls, "rack": sorted(rack), "gate": f < sim.release_frame(scene), "air": sorted(air)}


# ------------------------------------------------------------------ drawing
def _shade(rgb, k):
    return tuple(max(0, min(255, int(round(v * k)))) for v in rgb)


def track_runs(scene, b):
    """Screen polylines of a ball's track: the drawn stretches between its gaps."""
    sh = sim.shape(scene, b)
    pts = sim.polyline(sh)
    gp = sim.gaps(b)
    cuts = [0.0] + [v for g in gp for v in (g[0], g[1])] + [scene["W"]]
    runs = []
    for xa, xb in zip(cuts[::2], cuts[1::2]):
        ia, ib = sim.seg_at(sh, xa), sim.seg_at(sh, xb)
        inner = [pts[j] for j in range(ia + 1, ib + 1, 3) if xa < pts[j][0] < xb]
        line = [(xa, sim.depth_at(sh, xa))] + inner + [(xb, sim.depth_at(sh, xb))]
        runs.append([sim.px(scene, x, y) for x, y in line])
    return runs


def lip_line(scene, b, g):
    """The take-off lip: from the gap's near edge along the launch direction (screen px)."""
    sh = sim.shape(scene, b)
    pts = sim.polyline(sh)
    i = sim.seg_at(sh, g["x0"] - 1e-9)
    phi = math.atan2(pts[i + 1][1] - pts[i][1], pts[i + 1][0] - pts[i][0]) - math.radians(g["lip"])
    x0, y0 = sim.px(scene, g["x0"], sim.depth_at(sh, g["x0"]))
    dx = math.cos(phi) * (-1 if scene["style"]["mirror"] else 1)
    return [(x0 - dx * 4, y0 - math.sin(phi) * 4), (x0 + dx * LIP_PX, y0 + math.sin(phi) * LIP_PX)]


def _sleepers(d, line, col, w, step, half):
    acc = 0.0
    for (xa, ya), (xb, yb) in zip(line, line[1:]):
        L = math.dist((xa, ya), (xb, yb))
        if L == 0:
            continue
        ux, uy = (xb - xa) / L, (yb - ya) / L
        s = step - acc
        while s <= L:
            cx, cy = xa + ux * s, ya + uy * s
            d.line([(cx - uy * half, cy + ux * half), (cx + uy * half, cy - ux * half)], fill=col, width=3)
            s += step
        acc = (acc + L) % step


def _draw_track(d, scene, b, ink=None):
    """A ball's track (casing / sleepers, line, lips). With `ink`, everything in that one value (a mask)."""
    st = scene["style"]
    w = st["track_w"]
    col = colour(scene, b["colour"]) if ink is None else ink
    runs = track_runs(scene, b)
    for line in runs:
        if st["track_style"] == "cased":
            d.line(line, fill=_shade(col, 0.55) if ink is None else ink, width=w + 4, joint="curve")
        elif st["track_style"] == "sleepers":
            _sleepers(d, line, col, w, st["sleeper_step"], w / 2 + 4)
    for line in runs:
        d.line(line, fill=col, width=w, joint="curve")
    for g in b.get("gaps", ()):
        d.line(lip_line(scene, b, g), fill=col, width=w + 1)


@functools.lru_cache(maxsize=8)
def other_tracks(key: str) -> dict:
    """colour -> L mask of every OTHER ball's track. A ball in the air is drawn behind the other colours'
    tracks (each ball rides only its own track), so a ball flying or falling across another track visibly
    passes behind it instead of looking as if it should land on it."""
    scene = json.loads(key)
    out = {}
    for b in scene["balls"]:
        m = Image.new("L", SIZE, 0)
        d = ImageDraw.Draw(m)
        for o in scene["balls"]:
            if o["colour"] != b["colour"]:
                _draw_track(d, scene, o, ink=255)
        out[b["colour"]] = m
    return out


@functools.lru_cache(maxsize=8)
def _static(key: str) -> Image.Image:
    scene = json.loads(key)
    st = scene["style"]
    bg, ink = tuple(st["bg"]), tuple(st["ink"])
    img = Image.new("RGB", SIZE, bg)
    d = ImageDraw.Draw(img)
    faint = tuple(int(round(v + (c - v) * 0.12)) for v, c in zip(bg, ink))
    step = st["bg_step"]
    if st["bg_pat"] == "grid":
        for x in range(step // 2, SIZE[0], step):
            d.line([(x, 0), (x, SIZE[1])], fill=faint, width=1)
        for y in range(step // 2, SIZE[1], step):
            d.line([(0, y), (SIZE[0], y)], fill=faint, width=1)
    elif st["bg_pat"] == "dots":
        for x in range(step // 2, SIZE[0], step):
            for y in range(step // 2, SIZE[1], step):
                d.ellipse([x - 2, y - 2, x + 2, y + 2], fill=faint)
    # pit floor
    fy = sim.floor_px(scene)
    pit = tuple(st["pit"])
    d.rectangle([0, fy, SIZE[0], SIZE[1]], fill=pit)
    hatch = _shade(pit, 0.82 if st["theme"] == "light" else 1.35)
    for x in range(-40, SIZE[0] + 40, 22):
        d.line([(x, SIZE[1]), (x + 40, fy + 3)], fill=hatch, width=2)
    d.line([(0, fy), (SIZE[0], fy)], fill=ink, width=3)
    # start ledge
    sx, sy = sim.px(scene, 0, 0)
    sgn = -1 if st["mirror"] else 1
    d.line([(sx - sgn * 64, sy), (sx, sy)], fill=ink, width=6)
    d.line([(sx - sgn * 64, sy - 34), (sx - sgn * 64, sy + 3)], fill=ink, width=6)
    # finish rack (behind) and slots
    xs = [sim.slot_px(scene, c) for c in scene["rack"]]
    r = st["R"] + 6
    d.rounded_rectangle([xs[0][0] - r - 4, xs[0][1] - r - 4, xs[0][0] + r + 4, xs[-1][1] + r + 4], radius=r,
                        fill=tuple(st["rack"]), outline=ink, width=2)
    for c, (x, y) in zip(scene["rack"], xs):
        rr = st["R"] + 1
        d.ellipse([x - rr, y - rr, x + rr, y + rr], outline=colour(scene, c), width=3)
    # tracks
    for b in scene["balls"]:
        _draw_track(d, scene, b)
    # finish hole
    hx, hy = sim.hole_px(scene)
    d.ellipse([hx - 22, hy - 7, hx + 22, hy + 13], fill=tuple(st["hole"]), outline=ink, width=2)
    return img


def _ball(d, scene, b, x, y):
    st = scene["style"]
    R = st["R"]
    rgb = colour(scene, b["colour"])
    ink = tuple(st["ink"]) if st["theme"] == "light" else (20, 20, 24)
    d.ellipse([x - R, y - R, x + R, y + R], fill=rgb, outline=ink, width=2)
    lum = 0.3 * rgb[0] + 0.59 * rgb[1] + 0.11 * rgb[2]
    det = (30, 30, 34) if lum > 165 else (255, 255, 255)
    p = b["pattern"]
    if p == "dot":
        q = 0.34 * R
        d.ellipse([x - q, y - q, x + q, y + q], fill=det)
    elif p == "ring":
        q = 0.6 * R
        d.ellipse([x - q, y - q, x + q, y + q], outline=det, width=2)
    elif p == "cross":
        q = 0.62 * R
        d.line([(x - q, y), (x + q, y)], fill=det, width=2)
        d.line([(x, y - q), (x, y + q)], fill=det, width=2)


def gate_box(scene):
    """The start gate: a bar just downhill of the start, above the track line."""
    st = scene["style"]
    sx, sy = sim.px(scene, 0, 0)
    gx = sx + (-1 if st["mirror"] else 1) * (st["R"] + 6)
    return [gx - 3, sy - 2 * st["R"] - 12, gx + 3, sy - 5]


def _clock(d, scene, frame):
    box = sim.clock_box(scene)
    d.rounded_rectangle(box, radius=12, fill=(18, 18, 22), outline=(235, 235, 235), width=3)
    d.text(((box[0] + box[2]) / 2, (box[1] + box[3]) / 2), clock_text(frame), fill=(255, 255, 255),
           font=font(CLOCK_PX), anchor="mm")


def frame_at(scene, f: int) -> Image.Image:
    img = _static(json.dumps(scene, sort_keys=True)).copy()
    d = ImageDraw.Draw(img)
    s = state(scene, f)
    by = {b["colour"]: b for b in scene["balls"]}
    for c in s["rack"]:
        x, y = sim.slot_px(scene, c)
        _ball(d, scene, by[c], x, y)
    key = json.dumps(scene, sort_keys=True)
    static, others = _static(key), other_tracks(key)
    R = scene["style"]["R"]
    for b in reversed(scene["balls"]):
        if b["colour"] in s["balls"]:
            x, y = s["balls"][b["colour"]]
            _ball(d, scene, b, x, y)
            if b["colour"] not in s["air"]:
                continue
            # a flying or falling ball passes behind the other colours' tracks (inside its disc only)
            box = (int(x - R - 2), int(y - R - 2), int(x + R + 3), int(y + R + 3))
            disc = Image.new("L", (box[2] - box[0], box[3] - box[1]), 0)
            ImageDraw.Draw(disc).ellipse([x - R - box[0] - 1, y - R - box[1] - 1, x + R - box[0] + 1,
                                          y + R - box[1] + 1], fill=255)
            m = ImageChops.multiply(others[b["colour"]].crop(box), disc)
            img.paste(static.crop(box), box, m)
    if s["gate"]:
        d.rectangle(gate_box(scene), fill=tuple(scene["style"]["ink"]))
    _clock(d, scene, f)
    return img


def frame(scene, t: float) -> Image.Image:
    return frame_at(scene, int(round(t * FPS)))
