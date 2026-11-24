"""PIL renderer for rainbow_bagel: the whole bagel next to an unnumbered unit
ruler, the cut, then every slice lifts off, turns to the same upright arch,
glides, passes completely behind the opaque screen and settles in its tray.
All pieces of a ring look alike; one turns gold mid-flight. The clock in the
top-left corner is the only text."""
from __future__ import annotations
import math
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from common.video5 import font  # noqa: E402

import sim  # noqa: E402

FPS = sim.FPS


def duration(scene) -> float:
    return scene["n_frames"] / FPS


def _dark(c, k=0.6):
    return tuple(int(v * k) for v in c)


def _light(c, k=0.55):
    return tuple(int(v + (255 - v) * k) for v in c)


def piece_poly(scene, i: int, f: int) -> np.ndarray:
    """The outline drawn for piece i on frame f (pixel coordinates)."""
    return sim.poly(scene, i, f)


def fill_of(scene, i: int, f: int) -> tuple:
    """The fill colour drawn for piece i on frame f."""
    if sim.is_gold(scene, i, f):
        return sim.GOLD
    return tuple(scene["style"]["ring_colours"][scene["pieces"][i]["ring"] - 1])


def layout(scene) -> dict:
    """Fixed furniture of the clip, as drawn."""
    return {"screen": sim.SCR, "trays": {t: (x0, sim.TRAY_Y[0], x1, sim.TRAY_Y[1])
                                         for t, (x0, x1) in sim.TRAY_X[scene["n_trays"]].items()},
            "ruler": dict(scene["ruler"]), "clock_box": tuple(scene["text"]["timer"]["box"]),
            "bagel_centre": (scene["bagel"]["cx"], scene["bagel"]["cy"])}


def _pattern(d, scene, i, f, fill):
    st = scene["style"]
    p = scene["pieces"][i]
    pat = st["pattern"]
    if pat == "plain":
        return
    x, y, rot = sim.state(scene, i, f)
    seeds = st["seeds"][p["ring"] - 1]
    pts = []
    for rf, af, ang in seeds:
        rho = p["r"] + rf * (p["R"] - p["r"])
        a = math.pi / 2 + (af - 0.5) * 2 * p["half"]
        pts.append((rho * math.cos(a), p["d"] - rho * math.sin(a), ang))
    if pat == "band":
        rho = (p["r"] + p["R"]) / 2
        a = math.pi / 2 + np.linspace(-0.8, 0.8, 20) * p["half"]
        loc = np.stack([rho * np.cos(a), p["d"] - rho * np.sin(a)], 1)
        line = sim.place(loc, x, y, rot)
        d.line([tuple(q) for q in line], fill=_light(fill, 0.45), width=4)
        return
    for lx, ly, ang in pts:
        c = sim.place(np.array([[lx, ly]]), x, y, rot)[0]
        if pat == "sesame":
            th = math.radians(ang + rot)
            ell = [(c[0] + 4.5 * math.cos(th) * math.cos(s) - 2.4 * math.sin(th) * math.sin(s),
                    c[1] - 4.5 * math.sin(th) * math.cos(s) - 2.4 * math.cos(th) * math.sin(s))
                   for s in np.linspace(0, 2 * math.pi, 10, endpoint=False)]
            d.polygon(ell, fill=(250, 238, 206))
        else:   # speckle
            d.ellipse([c[0] - 2.5, c[1] - 2.5, c[0] + 2.5, c[1] + 2.5], fill=_dark(fill, 0.45))


def _screen(d, scene):
    st = scene["style"]
    base = sim.SCREENS[st["screen"]]
    x0, y0, x1, y1 = sim.SCR
    d.rectangle(sim.SCR, fill=base)
    stripe = _dark(base, 0.82)
    pat = st["screen_pattern"]
    if pat == "hstripe":
        for y in range(y0 + 10, y1, 16):
            d.line([x0, y, x1, y], fill=stripe, width=4)
    elif pat == "vstripe":
        for x in range(x0 + 12, x1, 24):
            d.line([x, y0, x, y1], fill=stripe, width=4)
    elif pat == "diag":
        for x in range(x0 - (y1 - y0), x1, 26):
            d.line([max(x, x0), y0 + max(0, x0 - x), min(x + (y1 - y0), x1), y1 - max(0, x + (y1 - y0) - x1)],
                   fill=stripe, width=4)
    elif pat == "brick":
        for j, y in enumerate(range(y0, y1, 24)):
            d.line([x0, y, x1, y], fill=stripe, width=3)
            for x in range(x0 + (24 if j % 2 else 0), x1, 48):
                d.line([x, y, x, min(y + 24, y1)], fill=stripe, width=3)
    d.rectangle(sim.SCR, outline=_dark(base, 0.6), width=4)


def _ruler(d, scene, ink):
    ru = scene["ruler"]
    x0, y, s, n = ru["x0"], ru["y"], ru["step"], ru["n"]
    d.line([x0, y, x0 + n * s, y], fill=ink, width=4)
    for j in range(n + 1):
        x = x0 + j * s
        hh = 22 if j in (0, n) else 14
        d.line([x, y - hh, x, y], fill=ink, width=3)


def _whole_bagel(d, scene, bg):
    b = scene["bagel"]
    cx, cy, u, r0, k = b["cx"], b["cy"], b["u"], scene["hole"], scene["rings"]
    cols = scene["style"]["ring_colours"]
    for ring in range(k, 0, -1):
        R = (r0 + ring) * u
        d.ellipse([cx - R, cy - R, cx + R, cy + R], fill=tuple(cols[ring - 1]),
                  outline=_dark(cols[ring - 1]), width=3)
    r = r0 * u
    d.ellipse([cx - r, cy - r, cx + r, cy + r], fill=bg, outline=_dark(cols[0]), width=3)


def frame(scene, t: float) -> Image.Image:
    f = int(round(t * FPS))
    th = sim.theme(scene)
    img = Image.new("RGB", (sim.W, sim.H), th["bg"])
    d = ImageDraw.Draw(img)
    for x0, y0, x1, y1 in layout(scene)["trays"].values():
        d.rounded_rectangle([x0, y0, x1, y1], radius=12, fill=th["tray"], outline=th["ink"], width=3)
    _ruler(d, scene, th["ink"])
    pcs = scene["pieces"]
    if f < sim.CUT0:
        _whole_bagel(d, scene, th["bg"])
        for i in range(len(pcs)):
            _pattern(d, scene, i, f, fill_of(scene, i, f))
    else:
        def stage(i):
            p = pcs[i]
            if f < p["lift"]:
                return (0, 0, i)
            if f >= p["land"]:
                return (1, 0, i)
            return (2, p["lift"], i)
        for i in sorted(range(len(pcs)), key=stage):
            fill = fill_of(scene, i, f)
            edge = sim.GOLD_EDGE if fill == sim.GOLD else _dark(fill)
            d.polygon([tuple(q) for q in piece_poly(scene, i, f)], fill=fill, outline=edge,
                      width=scene["style"]["outline"])
            _pattern(d, scene, i, f, fill)
    _screen(d, scene)
    bx = scene["text"]["timer"]["box"]
    d.rounded_rectangle(bx, radius=8, fill=th["panel"], outline=th["ink"], width=2)
    d.text(((bx[0] + bx[2]) / 2, (bx[1] + bx[3]) / 2), sim.clock_text(f), fill=th["clock"],
           font=font(scene["text"]["timer"]["font_px"]), anchor="mm")
    return img


def track(scene) -> list[list[dict]]:
    """Per frame, per piece: what is drawn (outline points and fill colour)."""
    out = []
    for f in range(scene["n_frames"]):
        out.append([{"poly": piece_poly(scene, i, f), "fill": fill_of(scene, i, f)}
                    for i in range(len(scene["pieces"]))])
    return out
