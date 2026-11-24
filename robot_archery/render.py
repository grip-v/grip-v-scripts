"""PIL renderer for robot_archery (1280x720, 24 fps).

Target (ten rings, ring numbers 2-10 on the ray straight down from the
centre) on one side, robots on the other: a queue along the bottom, the
shooting spot next to the target, a bench at the top for eliminated
robots. Each turn the front robot steps onto the spot (unless it is
already there), an arrow flies to the target and lands; the arrow stays
there as a marker in the shooter's colour and shape. A new record moves
the black-and-white dashed record circle to the new arrow. A robot that
loses its last pip turns pale with closed eyes and walks to the bench;
otherwise it walks to the back of the queue (stay_on: a record setter
stays on the spot). In two_lives every robot has two pips over its head;
a lost pip turns hollow.

Text: the clock (top-left box) and the ring numbers (sim.labels), nothing
else. `state(scene, f)` is the per-frame geometry the drawing uses;
answers.py re-derives every answer from it. Pure function of (scene, frame).
"""
from __future__ import annotations
import functools
import json
import math
import sys
from pathlib import Path

from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from common.video5 import font  # noqa: E402
import sim  # noqa: E402

FPS = sim.FPS
SIZE = sim.SIZE
CLOCK_BOX = sim.CLOCK_BOX
CLOCK_PX = 44
INK = (24, 24, 28)
STEP, FLY0, WALK0, WALK1 = 0.25, 0.3, 0.6, 0.95       # phases of a turn (fractions of T)
PIP_R = 10


def duration(scene) -> float:
    return sim.duration(scene)


def n_frames(scene) -> int:
    return scene["frames"]


def clock_text(frame: int) -> str:
    return f"{sim.clock_tenths(frame) / 10:.1f} s"


def colour(scene, i):
    return sim.PALETTE[scene["robots"][i]["colour"]]


# ------------------------------------------------------------------ game states
@functools.lru_cache(maxsize=64)
def _states(key: str):
    """Settled state before every turn and after the last one: queue, robot on the spot, bench, pips, record."""
    sc = json.loads(key)
    g = sim.play(sc["order"], sc["draws"], sc["lives"], sc["stay_on"])
    queue, spot, bench = list(sc["order"]), None, []
    lives = {r: sc["lives"] for r in sc["order"]}
    record = None
    out = []
    for t in g["turns"]:
        out.append({"queue": list(queue), "spot": spot, "bench": list(bench), "lives": dict(lives), "record": record})
        s = t["shooter"]
        if spot != s:
            queue.remove(s)
        spot = None
        lives[s] = t["lives_after"]
        if t["record"]:
            record = t["d"]
            if sc["stay_on"]:
                spot = s
            else:
                queue.append(s)
        elif t["out"]:
            bench.append(s)
        else:
            queue.append(s)
    out.append({"queue": list(queue), "spot": spot, "bench": list(bench), "lives": dict(lives), "record": record})
    return out, g["turns"]


def _key(scene):
    return json.dumps({k: scene[k] for k in ("order", "draws", "lives", "stay_on")}, sort_keys=True)


def _places(scene, st, shooter_on_spot=None):
    lay = scene["layout"]
    pos = {}
    q = [r for r in st["queue"] if r != shooter_on_spot]
    for i, r in enumerate(q):
        pos[r] = tuple(lay["slots"][i])
    for j, r in enumerate(st["bench"]):
        pos[r] = tuple(lay["bench"][j])
    if st["spot"] is not None:
        pos[st["spot"]] = tuple(lay["spot"])
    if shooter_on_spot is not None:
        pos[shooter_on_spot] = tuple(lay["spot"])
    return pos


def _ease(u):
    u = min(1.0, max(0.0, u))
    return u * u * (3 - 2 * u)


def _lerp(a, b, u):
    return (a[0] + (b[0] - a[0]) * u, a[1] + (b[1] - a[1]) * u)


def state(scene, f: int) -> dict:
    """Frame f: robots {xy, out, lives}, landed arrows [{robot, xy}] in landing order, the arrow in flight
    ({robot, xy, dir} or None) and the record circle radius in px (None before the first record)."""
    t = f / FPS
    states, turns = _states(_key(scene))
    T = scene["T"]
    k = sum(1 for lt in scene["land"] if lt <= t)                 # arrows landed so far
    cur = None                                                    # turn in progress (started, maybe landed)
    for j in range(len(turns)):
        if sim.turn_start(scene, j) <= t < sim.turn_start(scene, j) + WALK1 * T + 1e-9:
            cur = j
    n = len(scene["robots"])
    before = states[cur] if cur is not None else states[min(k, len(turns))]
    out_now = set(states[k]["bench"]) if k <= len(turns) else set()
    lives = states[k]["lives"]
    if cur is None:
        pos = _places(scene, states[k])
    else:
        ts = sim.turn_start(scene, cur)
        sh = turns[cur]["shooter"]
        p0 = _places(scene, before)
        p1 = _places(scene, before, shooter_on_spot=sh)
        p2 = _places(scene, states[cur + 1])
        u = (t - ts) / T
        pos = {}
        for r in range(n):
            if u < WALK0:
                pos[r] = _lerp(p0[r], p1[r], _ease(u / STEP))
            else:                                                 # L-shaped walk: across first, then up / down
                e = _ease((u - WALK0) / (WALK1 - WALK0))
                pos[r] = (p1[r][0] + (p2[r][0] - p1[r][0]) * min(1.0, 2 * e),
                          p1[r][1] + (p2[r][1] - p1[r][1]) * max(0.0, 2 * e - 1))
    robots = [{"xy": pos[r], "out": r in out_now, "lives": lives[r]} for r in range(n)]
    arrows = [{"robot": turns[j]["shooter"], "xy": sim.target_xy(scene, scene["draws"][j], scene["angles"][j])}
              for j in range(k)]
    flight = None
    if cur is not None and cur >= k:
        ts = sim.turn_start(scene, cur)
        t0, t1 = ts + FLY0 * T, scene["land"][cur]
        if t >= t0:
            a = _bow(scene)
            b = sim.target_xy(scene, scene["draws"][cur], scene["angles"][cur])
            u = (t - t0) / (t1 - t0)
            flight = {"robot": turns[cur]["shooter"], "xy": _lerp(a, b, u), "dir": (b[0] - a[0], b[1] - a[1])}
    rec = states[k]["record"]
    return {"robots": robots, "arrows": arrows, "flight": flight,
            "record_px": None if rec is None else rec * scene["target"]["unit"]}


def _bow(scene):
    lay = scene["layout"]
    return (lay["spot"][0], lay["spot"][1] - lay["body"] * 0.2)


# ------------------------------------------------------------------ drawing helpers
def _shade(c, d):
    return tuple(max(0, min(255, v + d)) for v in c)


def _mix(a, b, u):
    return tuple(round(x + (y - x) * u) for x, y in zip(a, b))


def shape_pts(shape, cx, cy, r):
    if shape == "circle":
        return None
    n, rot = {"square": (4, 45), "triangle": (3, 90), "diamond": (4, 90), "hexagon": (6, 0), "pentagon": (5, 90)}[shape]
    if shape == "square":
        r *= 1.25
    if shape == "triangle":
        r *= 1.25
        cy += r * 0.18
    return [(cx + r * math.cos(math.radians(rot + 360 * i / n)), cy - r * math.sin(math.radians(rot + 360 * i / n)))
            for i in range(n)]


def draw_shape(d, shape, cx, cy, r, fill, outline=INK, width=2):
    pts = shape_pts(shape, cx, cy, r)
    if pts is None:
        d.ellipse([cx - r, cy - r, cx + r, cy + r], fill=fill, outline=outline, width=width)
    else:
        d.polygon(pts, fill=fill, outline=outline, width=width)


@functools.lru_cache(maxsize=16)
def _static(key: str) -> Image.Image:
    sc = json.loads(key)
    st, tg = sc["style"], sc["target"]
    bg = tuple(st["bg"])
    img = Image.new("RGB", SIZE, bg)
    d = ImageDraw.Draw(img)
    step = st["pat_step"]
    if st["bg_pat"] == "gradient":
        for y in range(SIZE[1]):
            d.line([(0, y), (SIZE[0], y)], fill=_shade(bg, -round(18 * y / SIZE[1])))
    elif st["bg_pat"] == "tiles":
        for x in range(0, SIZE[0], step * 2):
            d.line([(x, 0), (x, SIZE[1])], fill=_shade(bg, -12))
        for y in range(0, SIZE[1], step * 2):
            d.line([(0, y), (SIZE[0], y)], fill=_shade(bg, -12))
    elif st["bg_pat"] == "grass":
        for y in range(step // 2, SIZE[1], step):
            for x in range((y // step) % 2 * step // 2, SIZE[0], step):
                d.line([(x, y), (x + 4, y - 9)], fill=_shade(bg, -16), width=2)
                d.line([(x + 6, y), (x + 8, y - 7)], fill=_shade(bg, -16), width=2)
    # ground line under the queue
    lay = sc["layout"]
    gy = lay["slots"][0][1] + lay["body"] / 2 + 6
    xs = [p[0] for p in lay["slots"]]
    d.line([(min(xs) - lay["body"], gy), (max(xs) + lay["body"], gy)], fill=_shade(bg, -40), width=4)
    by = lay["bench"][0][1] + lay["body"] / 2 + 6
    bx = [p[0] for p in lay["bench"]]
    d.rectangle([min(bx) - lay["body"] * 0.8, by, max(bx) + lay["body"] * 0.8, by + 10], fill=(150, 112, 70),
                outline=INK, width=2)
    sx, sy = lay["spot"]
    d.ellipse([sx - lay["body"] * 0.8, sy + lay["body"] / 2 - 2, sx + lay["body"] * 0.8, sy + lay["body"] / 2 + 12],
              fill=_shade(bg, -35))
    # target
    cx, cy, u = tg["cx"], tg["cy"], tg["unit"]
    cols = sim.TARGETS[st["rings"]]
    d.ellipse([cx - 10 * u - 6, cy - 10 * u - 6, cx + 10 * u + 6, cy + 10 * u + 6], fill=(72, 72, 78), outline=INK,
              width=2)
    for ring in range(10, 0, -1):
        d.ellipse([cx - ring * u, cy - ring * u, cx + ring * u, cy + ring * u], fill=cols[ring - 1],
                  outline=(90, 90, 96), width=2)
    d.ellipse([cx - 3, cy - 3, cx + 3, cy + 3], fill=INK)
    for lab in sc["labels"]:
        b = sim.label_box(lab)
        d.rounded_rectangle([b[0] - 4, b[1] - 3, b[2] + 4, b[3] + 3], radius=5, fill=(255, 255, 255),
                            outline=INK, width=1)
        d.text((lab["x"], lab["y"]), lab["text"], fill=INK, font=font(lab["size"]), anchor=lab["anchor"])
    return img


def label_tag(lab):
    """The white tag drawn behind a ring number."""
    b = sim.label_box(lab)
    return (b[0] - 4, b[1] - 3, b[2] + 4, b[3] + 3)


def _dashed_circle(d, cx, cy, r, gaps=(), width=4):
    """Black-and-white dashed circle; dashes whose span touches a box in `gaps` (the ring-number tags) are left
    out so the circle never runs under a label."""
    n = max(24, int(2 * math.pi * r / 14))
    for i in range(n):
        a0, a1 = 360 * i / n, 360 * (i + 1) / n
        pts = [(cx + r * math.cos(math.radians(a)), cy + r * math.sin(math.radians(a)))
               for a in (a0, (a0 + a1) / 2, a1)]
        if any(g[0] - 6 <= x <= g[2] + 6 and g[1] - 6 <= y <= g[3] + 6 for g in gaps for x, y in pts):
            continue
        d.arc([cx - r, cy - r, cx + r, cy + r], a0, a1, fill=INK if i % 2 == 0 else (255, 255, 255), width=width)


def _robot(d, scene, i, xy, out, lives):
    rb = scene["robots"][i]
    b = scene["layout"]["body"]
    col = colour(scene, i)
    fill = _mix(col, (205, 205, 205), 0.25) if out else col
    x, y = xy
    r = b / 2
    # legs
    d.rectangle([x - r * 0.55, y + r * 0.7, x - r * 0.25, y + r + 6], fill=INK)
    d.rectangle([x + r * 0.25, y + r * 0.7, x + r * 0.55, y + r + 6], fill=INK)
    draw_shape(d, rb["shape"], x, y, r * 0.95, fill, INK, 3)
    dark = _shade(fill, -55)
    pat = rb["pattern"]
    # the patch around (x, y + 0.12 r) is always plain body colour (frame_check reads it)
    if pat == "stripes":
        for dx in (-0.35, 0, 0.35):
            d.line([(x + dx * r, y + r * 0.34), (x + dx * r, y + r * 0.68)], fill=dark, width=4)
    elif pat == "dots":
        for dx in (-0.35, 0, 0.35):
            d.ellipse([x + dx * r - 4, y + r * 0.5 - 4, x + dx * r + 4, y + r * 0.5 + 4], fill=dark)
    elif pat == "band":
        d.rectangle([x - r * 0.45, y + r * 0.36, x + r * 0.45, y + r * 0.56], fill=dark)
    # eyes
    for dx in (-0.3, 0.3):
        ex, ey = x + dx * r, y - r * 0.34
        if out:
            d.line([(ex - 6, ey), (ex + 6, ey)], fill=INK, width=3)
        else:
            d.ellipse([ex - 6, ey - 6, ex + 6, ey + 6], fill=(255, 255, 255), outline=INK, width=2)
            d.ellipse([ex - 2.5, ey - 2.5, ex + 2.5, ey + 2.5], fill=INK)
    if scene["lives"] > 1:
        for j in range(scene["lives"]):
            px = x + (j - (scene["lives"] - 1) / 2) * (2 * PIP_R + 6)
            py = y - r - PIP_R - 8
            if j < lives:
                d.ellipse([px - PIP_R, py - PIP_R, px + PIP_R, py + PIP_R], fill=(225, 30, 60), outline=INK, width=2)
            else:
                d.ellipse([px - PIP_R, py - PIP_R, px + PIP_R, py + PIP_R], fill=None, outline=INK, width=3)


def _clock(d, f):
    d.rounded_rectangle(CLOCK_BOX, radius=10, fill=(24, 24, 30))
    x0, y0, x1, y1 = CLOCK_BOX
    d.text(((x0 + x1) / 2, (y0 + y1) / 2), clock_text(f), fill=(255, 255, 255), font=font(CLOCK_PX), anchor="mm")


def static_key(scene):
    return json.dumps({k: scene[k] for k in ("style", "target", "layout", "labels")}, sort_keys=True)


def frame_at(scene, f: int) -> Image.Image:
    img = _static(static_key(scene)).copy()
    d = ImageDraw.Draw(img)
    s = state(scene, f)
    tg = scene["target"]
    if s["record_px"] is not None:
        _dashed_circle(d, tg["cx"], tg["cy"], s["record_px"], [label_tag(lab) for lab in scene["labels"]])
    for a in s["arrows"]:
        x, y = a["xy"]
        draw_shape(d, scene["robots"][a["robot"]]["shape"], x, y, sim.MARK_R, colour(scene, a["robot"]), INK, 2)
        d.ellipse([x - 2, y - 2, x + 2, y + 2], fill=INK)
    order = sorted(range(len(s["robots"])), key=lambda r: s["robots"][r]["xy"][1])
    for r in order:
        rs = s["robots"][r]
        _robot(d, scene, r, rs["xy"], rs["out"], rs["lives"])
    fl = s["flight"]
    if fl is not None:
        x, y = fl["xy"]
        L = math.hypot(*fl["dir"]) or 1
        ux, uy = fl["dir"][0] / L, fl["dir"][1] / L
        d.line([(x - 40 * ux, y - 40 * uy), (x, y)], fill=INK, width=4)
        d.polygon([(x + 10 * ux, y + 10 * uy), (x - 6 * ux - 7 * uy, y - 6 * uy + 7 * ux),
                   (x - 6 * ux + 7 * uy, y - 6 * uy - 7 * ux)], fill=colour(scene, fl["robot"]), outline=INK)
        d.line([(x - 40 * ux - 6 * uy, y - 40 * uy + 6 * ux), (x - 32 * ux, y - 32 * uy)],
               fill=colour(scene, fl["robot"]), width=4)
    _clock(d, f)
    return img


def frame(scene, t: float) -> Image.Image:
    return frame_at(scene, int(round(t * FPS)))
