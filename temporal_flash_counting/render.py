"""PIL renderer for temporal_flash_counting (1280x720, 24 fps).

Every shape is drawn every frame in one of three looks:
  rest   dim fill (background mixed with the shape's colour) + coloured outline
  lit    full-colour fill, light outline and a soft halo
  white  near-white fill (a "white flash"), coloured outline and a pale halo
A decoy pulse only scales the shape (up to +30 %). Drift shapes move along
their lanes (sim.position). The background (colour + texture, grid plates,
lamp panel, drift rails) is fixed for the clip.

The only text is the timer: top-left, on a dark plate, seconds to 0.1 s,
40 px bold. sim.AREA (where every shape stays) starts below the plate.
`state(scene, f)` is the per-frame drawing state (centre, radius, fill) that
answers.py and frame_check.py read instead of the flash list.
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
from common.shapes import outline, unit_shape  # noqa: E402
from common.video5 import font  # noqa: E402

import sim  # noqa: E402

TIMER_XY = (30, 22)
TIMER_SIZE = 40
TIMER_PLATE = (14, 12, 196, 80)
TIMER_INK = (238, 238, 242)
TIMER_BG = (6, 6, 8)


def duration(scene) -> float:
    return scene["n_frames"] / sim.FPS


def _mix(a, b, u):
    return tuple(int(round(a[k] + (b[k] - a[k]) * u)) for k in range(3))


def bg_rgb(scene):
    return sim.BGS[scene["style"]["bg_name"]]


def palette(scene, i) -> dict:
    """The three fill colours shape i can show."""
    st = scene["style"]
    c = sim.COLORS[scene["shapes"][i]["color"]]
    bg = bg_rgb(scene)
    return {"rest": _mix(bg, c, st["rest_mix"]), "lit": c, "white": _mix(c, (255, 255, 255), st["white_mix"])}


@functools.lru_cache(maxsize=16)
def _index(key: str):
    scene = json.loads(key)
    n, nf = len(scene["shapes"]), scene["n_frames"]
    mode = [["rest"] * nf for _ in range(n)]
    scale = [[1.0] * nf for _ in range(n)]
    for f in scene["flashes"]:
        for k in range(f["f0"], f["f1"]):
            mode[f["i"]][k] = "white" if f["white"] else "lit"
    for p in scene["pulses"]:
        d = p["f1"] - p["f0"]
        for k in range(p["f0"], p["f1"]):
            scale[p["i"]][k] = 1 + sim.PULSE_SCALE * math.sin(math.pi * (k - p["f0"] + 0.5) / d)
    return mode, scale


def _key(scene) -> str:
    return json.dumps({k: scene[k] for k in ("shapes", "flashes", "pulses", "n_frames")}, sort_keys=True)


def state(scene, f: int) -> list[dict]:
    """What is drawn for each shape at frame f."""
    mode, scale = _index(_key(scene))
    out = []
    for i, s in enumerate(scene["shapes"]):
        x, y = sim.position(scene, i, f)
        m = mode[i][f] if 0 <= f < scene["n_frames"] else "rest"
        sc = scale[i][f] if 0 <= f < scene["n_frames"] else 1.0
        out.append({"x": x, "y": y, "r": s["r"] * sc, "fill": palette(scene, i)[m], "glow": m != "rest"})
    return out


def _poly(scene, i, x, y, r):
    s = scene["shapes"][i]
    return outline({"type": s["type"], "x": x, "y": y, "size": r, "rot": s["rot"], "aspect": s["aspect"]})


@functools.lru_cache(maxsize=None)
def _inner_unit(kind: str, aspect: float):
    from shapely.geometry import Polygon
    from shapely.ops import polylabel
    p = polylabel(Polygon(unit_shape(kind, aspect)), tolerance=0.01)
    return p.x, p.y


def inner_point(scene, i, x, y, r) -> tuple[int, int]:
    """A point well inside shape i (pole of inaccessibility) in pixels."""
    s = scene["shapes"][i]
    ux, uy = _inner_unit(s["type"], s["aspect"])
    th = math.radians(s["rot"])
    c, sn = math.cos(th), math.sin(th)
    return int(round(x + r * (ux * c - uy * sn))), int(round(y + r * (ux * sn + uy * c)))


_BG_CACHE: dict = {}


def _background(scene) -> Image.Image:
    st = scene["style"]
    key = json.dumps([st, scene.get("rows"), scene.get("cols"), scene.get("cell"),
                      [(s["x"], s["y"], s["r"], s.get("motion")) for s in scene["shapes"]]], sort_keys=True)
    if key in _BG_CACHE:
        return _BG_CACHE[key]
    bg = bg_rgb(scene)
    a = np.zeros((sim.H, sim.W, 3), dtype=np.float32) + np.array(bg, dtype=np.float32)
    amp, step = st["tex_amp"], st["tex_step"]
    yy, xx = np.mgrid[0:sim.H, 0:sim.W]
    tex = st["texture"]
    if tex == "vignette":
        d = np.hypot((xx - 640) / 640, (yy - 400) / 400)
        a *= (1.15 - 0.35 * np.clip(d, 0, 1.3))[..., None]
    elif tex == "dots":
        m = ((xx % step) - step / 2) ** 2 + ((yy % step) - step / 2) ** 2 <= 9
        a[m] += amp
    elif tex == "stripes":
        a[((xx + yy) % step) < 3] += amp
    elif tex == "grid":
        a[(xx % step < 2) | (yy % step < 2)] += amp
    elif tex == "noise":
        rnd = np.sin(xx * 12.9898 + yy * 78.233) * 43758.5453
        a += ((rnd - np.floor(rnd)) - 0.5)[..., None] * amp * 1.4
    img = Image.fromarray(np.clip(a, 0, 255).astype(np.uint8))
    d = ImageDraw.Draw(img)
    plate = _mix(bg, (255, 255, 255), 0.07)
    if st["plate"] != "none" and scene.get("cell"):
        h = scene["cell"] * 0.46
        for s in scene["shapes"]:
            box = [s["x"] - h, s["y"] - h, s["x"] + h, s["y"] + h]
            if st["plate"] == "rounded":
                d.rounded_rectangle(box, radius=h * 0.18, fill=plate)
            else:
                d.ellipse(box, fill=plate)
    if st.get("panel"):
        d.rounded_rectangle(st["panel"], radius=30, fill=_mix(bg, (255, 255, 255), 0.1),
                            outline=_mix(bg, (255, 255, 255), 0.22), width=4)
        for s in scene["shapes"]:
            R = s["r"] * 1.16
            d.ellipse([s["x"] - R, s["y"] - R, s["x"] + R, s["y"] + R], fill=_mix(bg, (0, 0, 0), 0.5),
                      outline=_mix(bg, (255, 255, 255), 0.35), width=3)
    if st.get("rails"):
        for s in scene["shapes"]:
            m = s["motion"]
            d.line([(m["xa"], s["y"]), (m["xb"], s["y"])], fill=_mix(bg, (255, 255, 255), 0.12), width=3)
    if len(_BG_CACHE) > 8:
        _BG_CACHE.clear()
    _BG_CACHE[key] = img
    return img


def draw_timer(d: ImageDraw.ImageDraw, f: int):
    d.rounded_rectangle(TIMER_PLATE, radius=10, fill=TIMER_BG)
    d.text(TIMER_XY, sim.timer_text(f), fill=TIMER_INK, font=font(TIMER_SIZE))


def frame_at(scene, f: int) -> Image.Image:
    img = _background(scene).copy()
    d = ImageDraw.Draw(img)
    bg = bg_rgb(scene)
    st = scene["style"]
    for i, s in enumerate(state(scene, f)):
        c = sim.COLORS[scene["shapes"][i]["color"]]
        pts = _poly(scene, i, s["x"], s["y"], s["r"])
        if s["glow"]:
            halo = _mix(bg, s["fill"], st["glow_mix"])
            hp = _poly(scene, i, s["x"], s["y"], s["r"] * sim.GLOW)
            d.polygon(hp, fill=halo)
            edge = _mix(c, (255, 255, 255), 0.45) if s["fill"] == c else c
        else:
            edge = _mix(bg, c, 0.9)
        d.polygon(pts, fill=s["fill"])
        d.line(pts + [pts[0]], fill=edge, width=st["outline"], joint="curve")
    draw_timer(d, f)
    return img


def frame(scene, t: float) -> Image.Image:
    return frame_at(scene, int(round(t * sim.FPS)))
