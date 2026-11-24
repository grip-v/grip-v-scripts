"""Scene generation for turntable_voxels (from the image benchmark's
cube_structure + orthographic categories, made temporal).

A structure of unit cubes stands on a round turntable with a GX x GY grid
(3-5 squares each way). Every square holds a stack 0-5 cubes tall (no
overhangs). 1-3 cubes are painted a strong colour (red, yellow, ...); the
rest share one pale base colour. The camera looks at the turntable from the
front, 22-40 degrees above it, with an orthographic projection.

The clip starts with the turntable standing still (the START VIEW, seen
straight from the front so only top and front faces show), then it turns.
How it turns depends on the scenario (`scene["scenario"]`):

    spin     one smooth turn of 1-1.75 turns, then it stands still
    stops    2-3 turns of 1/4-1 turn each, standing still 1-2 s in between
    reverse  1-1.5 turns one way, then 1/4-3/4 turn back the other way
    build    one smooth turn while 2-4 cubes drop onto the structure
    remove   one smooth turn while 2-3 top cubes rise up and vanish

Every turn ends on a multiple of a quarter turn, so every still view shows
one side of the structure square on. The clip always ends standing still
(the END VIEW).

Angles are in degrees, counter-clockwise seen from above (so a positive
turn moves the turntable's near edge to the right on screen). Grid
coordinates: x = column left to right in the start view, y = row from the
front (0) to the back, z = height.

Visibility in a still view comes from an ID-buffer render (pixels per cube).
A scene is kept only if, in every still view, every cube is either clearly
visible (>= max(0.15 S^2, 14 S) pixels, i.e. a band >= 14 px deep across a
face) or not visible at all.
"""
from __future__ import annotations
import functools
import hashlib
import json
import math
import random

import numpy as np
from PIL import Image, ImageDraw

W, H = 1280, 720
FPS = 24
TIMER_BOX = (1078, 12, 1266, 78)
TIMER_KEEP = 18                  # px between the structure and the clock box
SCENARIOS = ("spin", "stops", "reverse", "build", "remove")
EVENT_S = 0.6                    # a drop / lift takes this long
LIFT_H = 2.0                     # a dropped cube falls (a lifted cube rises) this many cube heights
GROW = 0.35                      # fraction of the drop / lift spent growing / shrinking
VIS_FRAC = 0.15                  # visible means >= max(VIS_FRAC * S^2, VIS_BAND * S) pixels in a still view
VIS_BAND = 14                    # (a visible part is a band across the face at least 14 px deep)
SEEN_FRAC = 0.10                 # "seen during the clip": >= SEEN_FRAC * S^2 pixels at some sampled moment
MAX_AVG = 100.0                  # deg/s average over a leg; the smoothstep peak is 1.5x this
MIN_AVG = 50.0
THICK = 0.3                      # turntable thickness (cube units)
MAX_DUR = 10.0

COLOURS = {"red": (214, 46, 42), "yellow": (240, 198, 28), "green": (32, 150, 62), "purple": (124, 62, 184),
           "orange": (242, 122, 22), "blue": (34, 82, 206), "pink": (234, 92, 170)}
BASES = {"white": (228, 228, 222), "grey": (196, 198, 204), "sand": (226, 208, 166), "sky": (186, 210, 238),
         "mint": (188, 226, 202), "lilac": (212, 198, 234)}
CONFUSABLE = {"white": (), "grey": (), "sand": ("yellow", "orange"), "sky": ("blue",), "mint": ("green",),
              "lilac": ("purple", "pink")}
BACKGROUNDS = {"cream": (238, 234, 224), "paper": (245, 245, 241), "mist": (222, 230, 238), "sage": (224, 234, 222),
               "stone": (212, 208, 202), "charcoal": (34, 36, 42), "navy": (24, 32, 54), "plum": (46, 30, 48),
               "forest": (22, 40, 34)}
BG_STYLES = ("plain", "vgrad", "vignette", "floor")
DISCS = {"wood": (180, 144, 102), "steel": (150, 154, 162), "slate": (82, 92, 110), "ivory": (234, 228, 212),
         "cork": (198, 166, 122), "felt": (58, 112, 84), "maroon": (122, 48, 54)}
LAYOUTS = ("random", "pyramid", "stairs", "towers", "terrace", "ring")
OUTLINE = (38, 42, 50)


# ------------------------------------------------------------------ basics
def cs(deg: float) -> tuple[float, float]:
    """(cos, sin) of an angle in degrees, exact on multiples of 90."""
    d = deg % 360.0
    if abs(d - round(d)) < 1e-12 and round(d) % 90 == 0:
        return {0: (1.0, 0.0), 90: (0.0, 1.0), 180: (-1.0, 0.0), 270: (0.0, -1.0)}[int(round(d)) % 360]
    r = math.radians(deg)
    return math.cos(r), math.sin(r)


def luma(c) -> float:
    return 0.299 * c[0] + 0.587 * c[1] + 0.114 * c[2]


def ease(u: float) -> float:
    u = min(1.0, max(0.0, u))
    return u * u * (3 - 2 * u)


def angle(scene, t: float) -> float:
    """Turntable angle (degrees, CCW from above) at time t."""
    a = 0.0
    for lg in scene["legs"]:
        if t >= lg["t1"]:
            a = lg["a1"]
        elif t > lg["t0"]:
            return lg["a0"] + (lg["a1"] - lg["a0"]) * ease((t - lg["t0"]) / (lg["t1"] - lg["t0"]))
        else:
            break
    return float(a)


def peak_speed(scene) -> float:
    """Largest angular speed (deg/s) anywhere in the clip: 1.5x the fastest leg's average."""
    return max(1.5 * abs(lg["a1"] - lg["a0"]) / (lg["t1"] - lg["t0"]) for lg in scene["legs"])


def holds(scene) -> list[tuple[float, float, float]]:
    """Still intervals (t0, t1, angle): the start view, any stops, the end view."""
    out, prev, a = [], 0.0, 0.0
    for lg in scene["legs"]:
        if lg["t0"] - prev > 1e-6:
            out.append((prev, lg["t0"], a))
        prev, a = lg["t1"], lg["a1"]
    out.append((prev, scene["duration"], a))
    return out


def start_cubes(scene) -> list[tuple[int, int, int]]:
    h = scene["heights"]
    return [(x, y, z) for y in range(scene["gy"]) for x in range(scene["gx"]) for z in range(h[y][x])]


def marked_map(scene) -> dict:
    return {(m["x"], m["y"], m["z"]): m["colour"] for m in scene["marked"]}


def heights_after(scene, t: float) -> list[list[int]]:
    """Stack heights once every event that has finished by time t is applied."""
    h = [list(r) for r in scene["heights"]]
    for ev in scene["events"]:
        if ev["t1"] <= t:
            h[ev["y"]][ev["x"]] += 1 if ev["kind"] == "drop" else -1
    return h


def end_heights(scene):
    return heights_after(scene, scene["duration"] + 1)


def cubes_at(scene, t: float) -> list[tuple]:
    """Everything drawn at time t: (x, y, z, scale, colour_key, moving). colour_key is 'base' or a colour."""
    mk = marked_map(scene)
    h = [list(r) for r in scene["heights"]]
    gone = set()
    moving = []
    for ev in scene["events"]:
        u = (t - ev["t0"]) / (ev["t1"] - ev["t0"])
        if ev["kind"] == "drop":
            if t >= ev["t1"]:
                h[ev["y"]][ev["x"]] += 1
            elif t > ev["t0"]:
                moving.append((ev["x"], ev["y"], ev["z"] + LIFT_H * (1 - u * u), min(1.0, u / GROW), "base", True))
        else:
            if t >= ev["t1"]:
                h[ev["y"]][ev["x"]] -= 1
            elif t > ev["t0"]:
                gone.add((ev["x"], ev["y"], ev["z"]))
                moving.append((ev["x"], ev["y"], ev["z"] + LIFT_H * u * u, min(1.0, (1 - u) / GROW),
                               mk.get((ev["x"], ev["y"], ev["z"]), "base"), True))
    out = []
    for y in range(scene["gy"]):
        for x in range(scene["gx"]):
            for z in range(h[y][x]):
                if (x, y, z) not in gone:
                    out.append((x, y, z, 1.0, mk.get((x, y, z), "base"), False))
    return out + moving


def static_cubes(heights, scene) -> list[tuple]:
    mk = marked_map(scene)
    return [(x, y, z, 1.0, mk.get((x, y, z), "base"), False)
            for y in range(len(heights)) for x in range(len(heights[0])) for z in range(heights[y][x])]


# ------------------------------------------------------------- projection
def project(scene, phi: float, p) -> tuple[float, float, float]:
    """World point -> (screen x, screen y, depth) with the turntable turned by phi degrees."""
    c, s = cs(phi)
    ce, se = cs(scene["elev"])
    x, y, z = p[0] - scene["gx"] / 2, p[1] - scene["gy"] / 2, p[2]
    xr = x * c - y * s
    yr = x * s + y * c
    S = scene["S"]
    return scene["cx"] + S * xr, scene["cy"] - S * (yr * se + z * ce), yr * ce - z * se


FACES = {  # normal: corner offsets
    (0, 0, 1): [(0, 0, 1), (1, 0, 1), (1, 1, 1), (0, 1, 1)],
    (0, -1, 0): [(0, 0, 0), (1, 0, 0), (1, 0, 1), (0, 0, 1)],
    (0, 1, 0): [(0, 1, 0), (1, 1, 0), (1, 1, 1), (0, 1, 1)],
    (-1, 0, 0): [(0, 0, 0), (0, 1, 0), (0, 1, 1), (0, 0, 1)],
    (1, 0, 0): [(1, 0, 0), (1, 1, 0), (1, 1, 1), (1, 0, 1)],
}


def rot_normal(n, phi):
    c, s = cs(phi)
    return n[0] * c - n[1] * s, n[0] * s + n[1] * c


def faces(scene, phi: float, cubes) -> list[tuple]:
    """Drawable faces back to front: (cube index, normal, rotated normal (nx, ny), screen polygon)."""
    occ = {(int(c[0]), int(c[1]), int(c[2])) for c in cubes if not c[5]}
    ce, se = cs(scene["elev"])
    out = []
    for k, (x, y, z, sc, _col, mv) in enumerate(cubes):
        cen = project(scene, phi, (x + 0.5, y + 0.5, z + 0.5))[2]
        for n, corners in FACES.items():
            if not mv and (x + n[0], y + n[1], z + n[2]) in occ:
                continue
            nx, ny = rot_normal(n, phi)
            if -ny * ce + n[2] * se <= 1e-9:          # facing away from the camera (or edge-on)
                continue
            poly = [project(scene, phi, (x + 0.5 + (a - 0.5) * sc, y + 0.5 + (b - 0.5) * sc,
                                         z + 0.5 + (c - 0.5) * sc))[:2] for a, b, c in corners]
            out.append((cen, k, n, (nx, ny), poly))
    out.sort(key=lambda f: -f[0])
    return [f[1:] for f in out]


def disc_radius(scene) -> float:
    return 0.5 * math.hypot(scene["gx"], scene["gy"]) + 0.75


def id_image(scene, phi: float, cubes, scale: float = 1.0):
    """ID-buffer render: array of (cube index + 1) of the front-most surface, 0 elsewhere, and its origin."""
    fs = faces(scene, phi, cubes)
    if not fs:
        return np.zeros((1, 1), dtype=np.int64), (0, 0)
    xs = [p[0] for f in fs for p in f[3]]
    ys = [p[1] for f in fs for p in f[3]]
    x0, y0 = math.floor(min(xs)) - 2, math.floor(min(ys)) - 2
    w = int((math.ceil(max(xs)) + 2 - x0) * scale) + 1
    h = int((math.ceil(max(ys)) + 2 - y0) * scale) + 1
    img = Image.new("I", (w, h), 0)
    d = ImageDraw.Draw(img)
    for k, _n, _nr, poly in fs:
        d.polygon([((px - x0) * scale, (py - y0) * scale) for px, py in poly], fill=k + 1)
    return np.asarray(img, dtype=np.int64), (x0, y0)


def id_counts(scene, phi: float, cubes, scale: float = 1.0) -> np.ndarray:
    """Number of pixels where each cube is the front-most surface."""
    arr, _ = id_image(scene, phi, cubes, scale)
    return np.bincount(arr.ravel(), minlength=len(cubes) + 1)[1:len(cubes) + 1]


def event_samples(ev) -> list[float]:
    """5 frame times spread over a drop / lift."""
    return [round((ev["t0"] + (j + 0.5) / 5 * (ev["t1"] - ev["t0"])) * FPS) / FPS for j in range(5)]


def event_visible(scene, ev) -> int:
    """At how many of 5 moments during a drop / lift the moving cube is clearly visible."""
    good = 0
    for t in event_samples(ev):
        cubes = cubes_at(scene, t)
        idx = [i for i, c in enumerate(cubes) if c[5]]
        cnt = id_counts(scene, angle(scene, t), cubes)
        good += bool(idx) and cnt[idx[0]] >= vis_thr(scene)
    return good


@functools.lru_cache(maxsize=512)
def _view_counts(key: str, phi: float, hkey: str) -> tuple:
    scene = json.loads(key)
    cubes = static_cubes(json.loads(hkey), scene)
    return tuple(int(v) for v in id_counts(scene, phi, cubes))


def _skey(scene) -> str:
    keep = {k: scene[k] for k in ("gx", "gy", "S", "cx", "cy", "elev")}
    keep["marked"] = scene["marked"]
    return json.dumps(keep, sort_keys=True)


def view(scene, phi: float, heights) -> dict:
    """Still view of a static structure: {(x, y, z): pixel count}."""
    counts = _view_counts(_skey(scene), float(phi), json.dumps(heights))
    cubes = static_cubes(heights, scene)
    return {(c[0], c[1], c[2]): counts[i] for i, c in enumerate(cubes)}


def vis_thr(scene) -> float:
    return max(VIS_FRAC * scene["S"] ** 2, VIS_BAND * scene["S"])


def visible(scene, phi, heights) -> set:
    thr = vis_thr(scene)
    return {k for k, v in view(scene, phi, heights).items() if v >= thr}


def ambiguous(scene, phi, heights) -> bool:
    thr = vis_thr(scene)
    return any(0 < v < thr for v in view(scene, phi, heights).values())


def front_view(heights) -> list[int]:
    """Silhouette from the front, straight on: tallest stack in each column, left to right."""
    return [max(heights[y][x] for y in range(len(heights))) for x in range(len(heights[0]))]


def rotate_heights(heights, k: int):
    """Heights as laid out after k quarter turns counter-clockwise (seen from above):
    the grid the camera then sees square on, x = left to right, y = front to back."""
    h = [list(r) for r in heights]
    for _ in range(k % 4):
        gy, gx = len(h), len(h[0])
        # (x, y) -> (x', y') = (gy - 1 - y, x) on a gx' = gy, gy' = gx grid
        n = [[0] * gy for _ in range(gx)]
        for y in range(gy):
            for x in range(gx):
                n[x][gy - 1 - y] = h[y][x]
        h = n
    return h


def quarter(a: float) -> int:
    return int(round(a / 90.0)) % 4


# ---------------------------------------------------------------- seen set
def sample_times(scene) -> list[float]:
    n = int(scene["duration"] / 0.2)
    return [round(i * 0.2, 4) for i in range(n + 1)]


@functools.lru_cache(maxsize=64)
def _seen(key: str) -> tuple:
    scene = json.loads(key)
    thr = SEEN_FRAC * (scene["S"] * 0.5) ** 2
    seen = set()
    for t in sample_times(scene):
        cubes = cubes_at(scene, t)
        cnt = id_counts(scene, angle(scene, t), cubes, 0.5)
        for c, v in zip(cubes, cnt):
            if v >= thr and not c[5]:
                seen.add((c[0], c[1], c[2]))
    return tuple(sorted(seen))


def seen_cubes(scene) -> set:
    return set(_seen(json.dumps(scene, sort_keys=True)))


# ------------------------------------------------------------------ check
def bbox_points(scene) -> list:
    """Screen points that bound everything drawn over the clip."""
    R = disc_radius(scene)
    pts = []
    for k in range(72):
        c, s = cs(k * 5)
        p = project(scene, 0.0, (scene["gx"] / 2 + R * c, scene["gy"] / 2 + R * s, 0))
        pts.append(p[:2])
        pts.append((p[0], p[1] + scene["S"] * THICK * cs(scene["elev"])[0]))
    top = top_height(scene)
    for t in sample_times(scene):
        a = angle(scene, t)
        for gx, gy in ((0, 0), (scene["gx"], 0), (0, scene["gy"]), (scene["gx"], scene["gy"])):
            pts.append(project(scene, a, (gx, gy, top))[:2])
    return pts


def _near_int(t, tol=0.13):
    return abs(t - round(t)) <= tol


def key_times(scene) -> list[tuple[str, float]]:
    """Moments a question may ask about (they sit within 0.13 s of a whole second)."""
    out = [("start", scene["legs"][0]["t0"])]
    sc = scene["scenario"]
    if sc == "stops":
        hs = holds(scene)
        out.append(("stop", hs[1][0]))
        out.append(("stop_len", hs[1][1] - hs[1][0]))
    if sc == "reverse":
        out.append(("reverse", scene["legs"][0]["t1"]))
    for i, ev in enumerate(scene["events"]):
        out.append((f"event{i}", ev["t1"] if ev["kind"] == "drop" else ev["t0"]))
    return out


def check(scene) -> list[str]:
    errs = []
    gx, gy, h = scene["gx"], scene["gy"], scene["heights"]
    if scene["scenario"] not in SCENARIOS:
        errs.append("unknown scenario")
    if not (3 <= gx <= 5 and 3 <= gy <= 5) or len(h) != gy or any(len(r) != gx for r in h):
        errs.append("bad grid")
        return errs
    n = len(start_cubes(scene))
    if not 6 <= n <= 36:
        errs.append(f"{n} cubes")
    if sum(1 for r in h for v in r if v) < 4:
        errs.append("fewer than 4 stacks")
    if scene["duration"] > MAX_DUR + 1e-9:
        errs.append("clip longer than 10 s")
    # turning programme
    legs = scene["legs"]
    prev = 0.0
    for lg in legs:
        if lg["t0"] < prev - 1e-9 or lg["t1"] <= lg["t0"]:
            errs.append("legs out of order")
        if abs(lg["a1"] - lg["a0"]) < 89 or abs(lg["a1"] % 90) > 1e-9:
            errs.append("a turn that does not end on a quarter turn")
        avg = abs(lg["a1"] - lg["a0"]) / (lg["t1"] - lg["t0"])
        if not MIN_AVG - 1e-6 <= avg <= MAX_AVG + 1e-6:
            errs.append(f"turn speed {avg:.0f} deg/s out of range")
        prev = lg["t1"]
    hs = holds(scene)
    if hs[0][1] - hs[0][0] < 0.9 or hs[-1][1] - hs[-1][0] < 1.0:
        errs.append("start or end view too short")
    for a, b, _ in hs[1:-1]:
        if b - a < 0.85:
            errs.append("a stop is too short")
    for name, t in key_times(scene):
        if not _near_int(t):
            errs.append(f"{name} time {t:.2f} too close to a rounding boundary")
        if name != "stop_len" and (t < 0.5 or t > scene["duration"] - 0.5):
            errs.append(f"{name} time too close to the clip ends")
    # events: inside a turn, apart from each other, on top of the stack at that time
    hh = [list(r) for r in h]
    last = -9.0
    for ev in scene["events"]:
        if not any(lg["t0"] + 0.4 <= ev["t0"] and ev["t1"] <= lg["t1"] - 0.4 for lg in legs):
            errs.append("an event outside a turn")
        if ev["t0"] - last < 0.15:
            errs.append("events too close together")
        last = ev["t1"]
        top = hh[ev["y"]][ev["x"]]
        if ev["kind"] == "drop":
            if ev["z"] != top or top >= 6:
                errs.append("bad drop")
            hh[ev["y"]][ev["x"]] += 1
        else:
            if ev["z"] != top - 1 or top < 1:
                errs.append("bad lift")
            hh[ev["y"]][ev["x"]] -= 1
    # marked cubes
    cols = [m["colour"] for m in scene["marked"]]
    if not 1 <= len(cols) <= 3 or len(set(cols)) != len(cols):
        errs.append("marked cubes")
    occ = set(start_cubes(scene))
    for m in scene["marked"]:
        if (m["x"], m["y"], m["z"]) not in occ or m["colour"] in CONFUSABLE[scene["theme"]["base"]]:
            errs.append("bad marked cube")
    # framing: inside the frame and clear of the clock
    pts = bbox_points(scene)
    xs, ys = [p[0] for p in pts], [p[1] for p in pts]
    if min(xs) < 24 or max(xs) > W - 24 or min(ys) < 24 or max(ys) > H - 16:
        errs.append("structure leaves the frame")
    x0, y0, x1, y1 = TIMER_BOX
    if any(p[0] > x0 - TIMER_KEEP and p[1] < y1 + TIMER_KEEP for p in pts):
        errs.append("structure near the clock")
    if scene["S"] < 40:
        errs.append("cubes too small")
    if errs:
        return errs
    # still views: no partly visible cube
    for t0, t1, a in hs:
        if ambiguous(scene, a, heights_after(scene, t0 + 1e-6)):
            errs.append(f"a cube is only partly visible in the still view at {t0:.2f}-{t1:.2f} s")
    for i, ev in enumerate(scene["events"]):
        if event_visible(scene, ev) < 3:
            errs.append(f"the moving cube of event {i} is hidden for most of its motion")
    hid = n - len(visible(scene, 0.0, h))
    if hid < 1:
        errs.append("no hidden cube at the start")
    # everything a question may need is seen at some point of the clip
    seen = seen_cubes(scene)
    evcols = {(ev["x"], ev["y"]) for ev in scene["events"]}
    for y in range(gy):
        for x in range(gx):
            if h[y][x] and (x, y, h[y][x] - 1) not in seen and (x, y) not in evcols:
                errs.append(f"the top of stack ({x}, {y}) is never seen")
    for m in scene["marked"]:
        if (m["x"], m["y"], m["z"]) not in seen:
            errs.append(f"the {m['colour']} cube is never seen")
    return errs


def signature(scene) -> str:
    keep = {k: scene[k] for k in ("scenario", "gx", "gy", "heights", "marked", "theme")}
    keep["elev"] = round(scene["elev"], 1)
    keep["S"] = round(scene["S"], 0)
    keep["legs"] = [[round(lg["t0"], 2), round(lg["t1"], 2), lg["a0"], lg["a1"]] for lg in scene["legs"]]
    keep["events"] = [[e["kind"], e["x"], e["y"], round(e["t0"], 2)] for e in scene["events"]]
    return json.dumps(keep, sort_keys=True)


def sig_hash(scene) -> str:
    return hashlib.sha256(signature(scene).encode()).hexdigest()


# ------------------------------------------------------------------ build
def _heights(rng: random.Random, gx: int, gy: int) -> tuple[str, list]:
    style = rng.choice(LAYOUTS)
    hmax = rng.choice([3, 4, 4, 5, 5])
    h = [[0] * gx for _ in range(gy)]
    if style == "random":
        p = rng.uniform(0.55, 0.95)
        for y in range(gy):
            for x in range(gx):
                if rng.random() < p:
                    h[y][x] = rng.randint(1, hmax)
    elif style == "pyramid":
        cx, cy = rng.uniform(0.5, gx - 1.5), rng.uniform(0.5, gy - 1.5)
        for y in range(gy):
            for x in range(gx):
                v = hmax - abs(x - cx) - abs(y - cy) + rng.uniform(-0.8, 0.8)
                h[y][x] = max(0, min(hmax, int(round(v))))
    elif style == "stairs":
        axis = rng.choice(["back", "left", "right", "front"])
        for y in range(gy):
            for x in range(gx):
                f = {"back": y / (gy - 1), "front": 1 - y / (gy - 1), "left": 1 - x / (gx - 1),
                     "right": x / (gx - 1)}[axis]
                v = 1 + f * (hmax - 1) + rng.uniform(-0.9, 0.9)
                h[y][x] = max(0, min(hmax, int(round(v)))) if rng.random() < 0.9 else 0
    elif style == "towers":
        for y in range(gy):
            for x in range(gx):
                h[y][x] = 1 if rng.random() < 0.7 else 0
        for _ in range(rng.randint(2, 4)):
            h[rng.randrange(gy)][rng.randrange(gx)] = rng.randint(max(2, hmax - 1), hmax)
    elif style == "terrace":
        for _ in range(rng.randint(3, 6)):
            x0, y0 = rng.randrange(gx), rng.randrange(gy)
            x1, y1 = rng.randint(x0, gx - 1), rng.randint(y0, gy - 1)
            for y in range(y0, y1 + 1):
                for x in range(x0, x1 + 1):
                    h[y][x] = min(hmax, h[y][x] + 1)
    else:  # ring: a tall rim around a low middle
        for y in range(gy):
            for x in range(gx):
                edge = x in (0, gx - 1) or y in (0, gy - 1)
                h[y][x] = rng.randint(max(1, hmax - 2), hmax) if edge and rng.random() < 0.75 else \
                    (rng.randint(0, 2) if not edge else 0)
    return style, h


def _leg(rng, t0, A, near_int: bool):
    """A turn by A degrees starting at t0; its end time sits near a whole second when near_int."""
    lo, hi = t0 + abs(A) / MAX_AVG, t0 + abs(A) / MIN_AVG
    if near_int:
        ks = [k for k in range(math.ceil(lo - 0.12), math.floor(hi + 0.12) + 1)]
        ks = [k for k in ks if lo <= k + 0.12 and k - 0.12 <= hi]
        if not ks:
            return None
        k = rng.choice(ks)
        t1 = rng.uniform(max(lo, k - 0.12), min(hi, k + 0.12))
    else:
        t1 = rng.uniform(lo, hi)
    return t1


def _program(rng, sc):
    """Legs (and room for events) for one scenario, or None if it doesn't fit 10 s."""
    d = rng.choice([-1, 1])
    t = rng.choice([1, 1, 2]) + rng.uniform(-0.1, 0.12)
    legs = []
    a = 0
    if sc in ("spin", "build", "remove"):
        A = rng.choice([360, 450, 540, 630] if sc == "spin" else [360, 450, 540])
        t1 = _leg(rng, t, A, False)
        legs.append((t, t1, a, a + d * A))
    elif sc == "stops":
        nst = rng.choice([1, 1, 2])
        while True:
            parts = [rng.choice([90, 180, 270, 360]) for _ in range(nst + 1)]
            if 360 <= sum(parts) <= 720:
                break
        for i, A in enumerate(parts):
            last = i == len(parts) - 1
            t1 = _leg(rng, t, A, not last)
            if t1 is None:
                return None
            legs.append((t, t1, a, a + d * A))
            a += d * A
            if not last:
                t = t1 + rng.choice([1, 1, 2]) + rng.uniform(-0.1, 0.1)
    else:  # reverse
        A1, A2 = rng.choice([360, 450, 540]), rng.choice([90, 180, 270])
        t1 = _leg(rng, t, A1, True)
        if t1 is None:
            return None
        legs.append((t, t1, 0, d * A1))
        t2 = _leg(rng, t1, A2, False)
        legs.append((t1, t2, d * A1, d * (A1 - A2)))
    end = legs[-1][1] + rng.uniform(1.0, 1.8)
    if end > MAX_DUR:
        return None
    return [{"t0": round(a0, 4), "t1": round(a1, 4), "a0": b0, "a1": b1} for a0, a1, b0, b1 in legs], round(end, 4)


def _events(rng, sc, legs, h):
    if sc not in ("build", "remove"):
        return []
    lg = legs[0]
    k = rng.choice([2, 3, 3, 4] if sc == "build" else [2, 2, 3])
    if sc == "build":       # landing moments near whole seconds
        slots = [s for s in range(1, 11) if lg["t0"] + 0.4 + EVENT_S <= s - 0.12 and s + 0.12 <= lg["t1"] - 0.4]
    else:                   # lift-off moments near whole seconds
        slots = [s for s in range(1, 11) if lg["t0"] + 0.4 <= s - 0.12 and s + 0.12 + EVENT_S <= lg["t1"] - 0.4]
    if len(slots) < 2:
        return None
    slots = sorted(rng.sample(slots, min(k, len(slots))))
    hh = [list(r) for r in h]
    gx, gy = len(h[0]), len(h)
    out = []
    for s in slots:
        tk = s + rng.uniform(-0.1, 0.1)
        t0, t1 = (tk - EVENT_S, tk) if sc == "build" else (tk, tk + EVENT_S)
        if sc == "build":
            cells = [(x, y) for y in range(gy) for x in range(gx) if hh[y][x] < 6]
            x, y = rng.choice(cells)
            out.append({"kind": "drop", "x": x, "y": y, "z": hh[y][x], "t0": round(t0, 4), "t1": round(t1, 4)})
            hh[y][x] += 1
        else:
            cells = [(x, y) for y in range(gy) for x in range(gx) if hh[y][x] >= 1]
            x, y = rng.choice(cells)
            out.append({"kind": "lift", "x": x, "y": y, "z": hh[y][x] - 1, "t0": round(t0, 4), "t1": round(t1, 4)})
            hh[y][x] -= 1
    return out


def _theme(rng):
    while True:
        base = rng.choice(sorted(BASES))
        bg = rng.choice(sorted(BACKGROUNDS))
        disc = rng.choice(sorted(DISCS))
        b, g, dc = BASES[base], BACKGROUNDS[bg], DISCS[disc]
        if abs(luma(dc) - luma(g)) >= 35 and math.dist(b, dc) >= 70:
            break
    dark = luma(BACKGROUNDS[bg]) < 100
    return {"base": base, "base_rgb": list(b), "bg": bg, "bg_rgb": list(g), "dark": dark,
            "bg_style": rng.choice(BG_STYLES), "disc": disc, "disc_rgb": list(dc),
            "marker_rgb": [250, 250, 250] if luma(dc) < 150 else [22, 22, 26],
            "outline_w": 3, "grid_w": 3}


def top_height(scene) -> float:
    """Highest point any cube reaches during the clip (cube units above the turntable)."""
    top = max(max(r) for r in scene["heights"])
    hh = [list(r) for r in scene["heights"]]
    for ev in scene["events"]:
        top = max(top, ev["z"] + LIFT_H + 1)
        hh[ev["y"]][ev["x"]] += 1 if ev["kind"] == "drop" else -1
        top = max(top, max(max(r) for r in hh))
    return top


def _framing(rng, scene) -> bool:
    """Cube size S and the turntable centre (cx, cy): everything stays inside the frame and clear of the clock."""
    gx, gy = scene["gx"], scene["gy"]
    R = disc_radius(scene)
    ce, se = cs(scene["elev"])
    up = 0.5 * math.hypot(gx, gy) * se + top_height(scene) * ce      # above the disc centre
    down = R * se + THICK * ce                                        # below the disc centre
    S = min((W - 60) / (2 * R), (H - 30 - 22) / (up + down)) * rng.uniform(0.84, 1.0)
    clock_x = TIMER_BOX[0] - TIMER_KEEP - 6
    clock_y = TIMER_BOX[3] + TIMER_KEEP + 6
    while S >= 40:
        opts = []
        x_lo, x_hi = 30 + S * R, W - 30 - S * R
        y_lo, y_hi = 30 + S * up, H - 22 - S * down
        if min(x_hi, clock_x - S * R) >= x_lo and y_hi >= y_lo:
            opts.append((x_lo, min(x_hi, clock_x - S * R), y_lo, y_hi))       # left of the clock
        if x_hi >= x_lo and y_hi >= clock_y + S * up:
            opts.append((x_lo, x_hi, clock_y + S * up, y_hi))                 # below the clock
        if opts:
            xa, xb, ya, yb = rng.choice(opts)
            scene["S"] = round(S, 3)
            scene["cx"] = round(rng.uniform(xa, xb), 2)
            scene["cy"] = round(rng.uniform(ya, yb), 2)
            return True
        S *= 0.94
    return False


def build(rng: random.Random) -> dict:
    sc = rng.choice(SCENARIOS)
    tries = 0
    while True:
        tries += 1
        gx, gy = rng.choice([3, 4, 4, 5]), rng.choice([3, 4, 4, 5])
        layout, h = _heights(rng, gx, gy)
        n = sum(map(sum, h))
        if not 6 <= n <= 36 or sum(1 for r in h for v in r if v) < 4:
            continue
        for _ in range(30):              # the turning programme is cheap: redraw it before the structure
            prog = _program(rng, sc)
            if prog is not None:
                break
        if prog is None:
            continue
        legs, dur = prog
        evs = _events(rng, sc, legs, h)
        if evs is None:
            continue
        theme = _theme(rng)
        allowed = [c for c in sorted(COLOURS) if c not in CONFUSABLE[theme["base"]]]
        cols = rng.sample(allowed, rng.choice([1, 2, 2, 3, 3]))
        scene = {"scenario": sc, "layout": layout, "gx": gx, "gy": gy, "heights": h,
                 "elev": round(rng.uniform(22.0, 40.0), 2), "legs": legs, "duration": dur, "events": evs,
                 "theme": theme, "marked": []}
        if not _framing(rng, scene):
            continue
        cubes = start_cubes(scene)
        hid = [c for c in cubes if c not in visible(scene, 0.0, h)]
        if not hid:
            continue
        pick = []
        for col in cols:
            pool = [c for c in cubes if c not in pick]
            if hid and rng.random() < 0.45:
                pool = [c for c in hid if c not in pick] or pool
            c = rng.choice(pool)
            pick.append(c)
        scene["marked"] = [{"colour": col, "x": c[0], "y": c[1], "z": c[2]} for col, c in zip(cols, pick)]
        if sc == "remove" and rng.random() < 0.35:       # sometimes a coloured top cube is the one lifted
            tops = [m for m in scene["marked"] if m["z"] == h[m["y"]][m["x"]] - 1]
            if tops:
                m = rng.choice(tops)
                hh = heights_after(scene, scene["events"][0]["t0"] - 1e-6)
                if hh[m["y"]][m["x"]] == m["z"] + 1:
                    ev = scene["events"][0]
                    ev["x"], ev["y"], ev["z"] = m["x"], m["y"], m["z"]
                    _redo_events(scene)
        if check(scene):
            continue
        import questions          # local import: questions imports sim
        if not questions.pool_ok(scene):
            continue
        scene["tries"] = tries
        return scene


def _redo_events(scene):
    """Recompute z of later events after one was moved (stacks shift)."""
    hh = [list(r) for r in scene["heights"]]
    for ev in scene["events"]:
        if ev["kind"] == "drop":
            ev["z"] = hh[ev["y"]][ev["x"]]
            hh[ev["y"]][ev["x"]] += 1
        else:
            if hh[ev["y"]][ev["x"]] < 1:
                ev["z"] = -1          # rejected by check
            else:
                ev["z"] = hh[ev["y"]][ev["x"]] - 1
                hh[ev["y"]][ev["x"]] -= 1
