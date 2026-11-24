"""PIL renderer for inertia_race (1280x720, 24 fps).

`layout(scene, f)` is the single source of geometry for frame f: for every racer
its contact point on the lane, its body centre, the shape as drawn, the rotation
angle it is drawn at and its state (held / rolling / finished). `frame()` draws
exactly what `layout()` returns; answers.py reads the same per-frame geometry to
re-derive every answer without touching the generator.

The renderer keeps its own replay of the kinematics (it evaluates the closed form
itself from the stored scene and never calls sim.py), so a divergence between
generator and renderer shows up as a test failure instead of hiding.

Besides the timer (top-left box) the only text is the five ruler numbers
0.00 / 0.25 / 0.50 / 0.75 / 1.00, taken from scene["labels"]; gridlines every
0.05 of the ramp cross every lane and meet the ruler below them.
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

RGB = {"orange": (247, 132, 18), "yellow": (240, 204, 20), "green": (40, 160, 62),
       "teal": (18, 176, 170), "blue": (40, 92, 224), "purple": (134, 62, 204),
       "pink": (242, 104, 184)}
INK = (28, 28, 32)
SPOKE = (250, 250, 250)

TAU = {"slider": 1.0, "solid_sphere": math.sqrt(1.4), "solid_disc": math.sqrt(1.5),
       "hollow_sphere": math.sqrt(5.0 / 3.0), "hoop": math.sqrt(2.0)}
SLIP_GRIP = 0.55
INNER = {"hollow_sphere": 0.60, "hoop": 0.80}      # annulus inner radius, as a fraction
SPOKES = {"solid_sphere": 1, "solid_disc": 3, "hollow_sphere": 2, "hoop": 2}
SKIRT = 7.0                                        # shaded under-edge of each lane slab


# --------------------------------------------------------------------------- timing


def duration(scene) -> float:
    return scene["n_frames"] / FPS


def timer_text(f: int) -> str:
    t = (f * 10) // FPS
    return f"{t // 10}.{t % 10} s"


# --------------------------------------------------------------------------- ramp frame


def point(view: dict, s: float, off: float) -> tuple[float, float]:
    """Pixel point at ramp fraction `s`, displaced `off` px along the surface normal."""
    d = s * view["run"]
    return (view["x0"] + view["ux"] * d + view["nx"] * off,
            view["y0"] + view["uy"] * d + view["ny"] * off)


def to_fraction(scene: dict, pt) -> float:
    """Project a pixel point back onto the ramp axis, using the ruler labels only.

    The normal component drops out of the projection, so a body's contact point
    converts to its ramp fraction regardless of which lane it is in.
    """
    lab = {lb["text"]: (lb["x"], lb["y"]) for lb in scene["labels"]}
    (ax, ay), (bx, by) = lab["0.00"], lab["1.00"]
    dx, dy = bx - ax, by - ay
    return ((pt[0] - ax) * dx + (pt[1] - ay) * dy) / (dx * dx + dy * dy)


# --------------------------------------------------------------------------- per-frame geometry


def _travel(r: dict, t0: float, f: int) -> float:
    """Ramp fraction covered by `r` at frame `f`, from the closed form."""
    if f <= r["release_f"]:
        return r["start"]
    t = (f - r["release_f"]) / FPS
    a = 2.0 / (TAU[r["k_shape"]] * t0) ** 2
    return min(1.0, r["start"] + 0.5 * a * t * t)


def _turns(r: dict, t0: float, f: int) -> float:
    if r["shape"] == "slider":
        return 0.0
    grip = SLIP_GRIP if r["violates"] == "slip" else 1.0
    return grip * (_travel(r, t0, f) - r["start"]) / (2.0 * math.pi * r["radius"])


def layout(scene, f: int) -> dict:
    """Everything drawn on frame f, in output pixels."""
    v, t0 = scene["view"], scene["t0"]
    out = {"racers": {}}
    for r in scene["racers"]:
        s = _travel(r, t0, f)
        surf = v["lane_off"][r["lane"]]
        contact = point(v, s, surf)
        centre = point(v, s, surf + r["r_px"])
        if f <= r["release_f"]:
            state = "held"
        elif s >= 1.0:
            state = "finished"
        else:
            state = "rolling"
        turns = _turns(r, t0, f)
        out["racers"][r["id"]] = {
            "contact_x": contact[0], "contact_y": contact[1],
            "cx": centre[0], "cy": centre[1],
            "body": r["shape"], "phi": (turns * 360.0) % 360.0, "turns": turns,
            "state": state, "r_px": r["r_px"], "lane_surface": surf,
        }
    out["finish"] = list(point(v, 1.0, v["ruler_off"]))
    return out


# --------------------------------------------------------------------------- drawing helpers


def _mix(c, d, t):
    return tuple(round(a + (b - a) * t) for a, b in zip(c, d))


def _quad(view, s0, s1, o0, o1):
    """The four corners of a band between two ramp fractions and two normal offsets."""
    return [point(view, s0, o0), point(view, s1, o0), point(view, s1, o1), point(view, s0, o1)]


@functools.lru_cache(maxsize=96)
def _sprite(shape, colour, pattern, rp):
    """RGBA body at SS scale, centred, with a spoke pattern that makes rotation readable."""
    R = rp * SS
    half = int(R + 6 * SS)
    img = Image.new("RGBA", (2 * half, 2 * half), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    c = half
    base = RGB[colour]
    box = [c - R, c - R, c + R, c + R]

    if shape == "slider":
        # a block: it translates and never spins, so it carries no spoke
        h = 0.84 * R
        d.rounded_rectangle([c - h, c - h, c + h, c + h], radius=0.18 * R,
                            fill=base, outline=INK, width=max(2, int(0.07 * R)))
        d.line([(c - h * 0.55, c + h * 0.45), (c + h * 0.55, c + h * 0.45)],
               fill=_mix(base, (0, 0, 0), 0.30), width=max(2, int(0.06 * R)))
        return img

    inner = INNER.get(shape)
    d.ellipse(box, fill=base, outline=INK, width=max(2, int(0.08 * R)))
    if shape == "solid_sphere":
        for k, t in ((0.62, 0.28), (0.40, 0.44), (0.22, 0.58)):
            o = 0.26 * R
            d.ellipse([c - o - k * R, c - o - k * R, c - o + k * R, c - o + k * R],
                      fill=_mix(base, (255, 255, 255), t))
    elif inner is not None:
        ri = inner * R
        if shape == "hoop":
            d.ellipse([c - ri, c - ri, c + ri, c + ri], fill=(0, 0, 0, 0))
        else:
            d.ellipse([c - ri, c - ri, c + ri, c + ri], fill=_mix(base, (255, 255, 255), 0.55))
        d.ellipse([c - ri, c - ri, c + ri, c + ri], outline=INK, width=max(2, int(0.06 * R)))

    if pattern == "striped":
        light = _mix(base, (255, 255, 255), 0.40) + (190,)
        over = Image.new("RGBA", img.size, (0, 0, 0, 0))
        od = ImageDraw.Draw(over)
        for k in range(-5, 6):
            o = k * 0.46 * R
            od.line([(c + o - R, c + R), (c + o + R, c - R)], fill=light,
                    width=max(2, int(0.13 * R)))
        ring = Image.new("L", img.size, 0)
        rd = ImageDraw.Draw(ring)
        rd.ellipse([c - R * 0.94, c - R * 0.94, c + R * 0.94, c + R * 0.94], fill=255)
        if inner is not None:
            ri = inner * R
            rd.ellipse([c - ri, c - ri, c + ri, c + ri], fill=0)
        img.paste(Image.alpha_composite(img, over), (0, 0), ring)

    # radial spokes plus a rim pip: what a viewer counts to get the turn count
    d = ImageDraw.Draw(img)
    n = SPOKES[shape]
    r_in = INNER.get(shape, 0.0) * R
    for k in range(n):
        a = math.radians(180.0 * k / n)
        dx, dy = math.cos(a), math.sin(a)
        for sgn in (1, -1):
            d.line([(c + sgn * dx * r_in, c + sgn * dy * r_in),
                    (c + sgn * dx * R * 0.95, c + sgn * dy * R * 0.95)],
                   fill=SPOKE, width=max(2, int(0.11 * R)))
    pip = 0.18 * R
    px, py = c + R * 0.78, c
    d.ellipse([px - pip, py - pip, px + pip, py + pip], fill=INK, outline=SPOKE,
              width=max(1, int(0.05 * R)))
    return img


@functools.lru_cache(maxsize=8)
def _static(key: str):
    """Lanes, gridlines, start marks, finish line, ruler and its labels (SS scale)."""
    scene = json.loads(key)
    th, v = scene["theme"], scene["view"]
    n = len(scene["racers"])
    img = Image.new("RGB", (W * SS, H * SS), tuple(th["bg"]))
    d = ImageDraw.Draw(img)
    Z = lambda pts: [(x * SS, y * SS) for x, y in pts]
    lw = lambda w: max(1, int(w * SS))
    top = v["lane_off"][0]
    lane_t = v["lane_t"]

    # one ramp slab per lane, with background between them, so the lanes read as
    # parallel ramps rather than as bands painted on a single block
    for i in range(n):
        o = v["lane_off"][i]
        d.polygon(Z(_quad(v, 0.0, 1.0, o - lane_t, o - lane_t - SKIRT)),
                  fill=_mix(tuple(th["ramp"]), (0, 0, 0), 0.32))
        d.polygon(Z(_quad(v, 0.0, 1.0, o, o - lane_t)), fill=tuple(th["ramp"]))
        d.polygon(Z(_quad(v, 0.0, 1.0, o, o - 3.0)), fill=tuple(th["lane"]))
        d.line(Z([point(v, 0.0, o), point(v, 1.0, o)]), fill=INK, width=lw(2))
        d.line(Z([point(v, 0.0, o - lane_t - SKIRT), point(v, 1.0, o - lane_t - SKIRT)]),
               fill=INK, width=lw(1))

    # gridlines every 0.05, darker every 0.25: faint across the whole stack so
    # lanes can be compared, solid where they cross a lane slab
    over = Image.new("RGBA", img.size, (0, 0, 0, 0))
    od = ImageDraw.Draw(over)
    a = th["grid_alpha"]
    for k in range(v["ticks"] + 1):
        s = k / v["ticks"]
        major = k % v["tick_major"] == 0
        al = round(255 * (a + 0.22 if major else a))
        wd = lw(3 if major else 2)
        od.line(Z([point(v, s, top), point(v, s, v["ruler_off"])]),
                fill=(40, 40, 48, round(al * 0.40)), width=wd)
        for i in range(n):
            o = v["lane_off"][i]
            od.line(Z([point(v, s, o), point(v, s, o - lane_t)]), fill=(40, 40, 48, al), width=wd)
    img = Image.alpha_composite(img.convert("RGBA"), over).convert("RGB")
    d = ImageDraw.Draw(img)

    # start marks: only where racers are held at different places along the ramp
    if scene["scenario"] in ("handicap", "pursuit"):
        for r in scene["racers"]:
            o = v["lane_off"][r["lane"]]
            d.polygon(Z(_quad(v, max(0.0, r["start"] - 0.005), r["start"] + 0.005,
                              o + 2.0, o - lane_t - 2.0)),
                      fill=(252, 252, 252), outline=INK, width=lw(1))

    # finish line, across every lane and down to the ruler
    d.line(Z([point(v, 1.0, top + 8.0), point(v, 1.0, v["ruler_off"])]),
           fill=tuple(th["finish"]), width=lw(5))

    # ruler: axis, ticks and the five numbers
    ro = v["ruler_off"]
    d.line(Z([point(v, 0.0, ro), point(v, 1.0, ro)]), fill=INK, width=lw(3))
    for k in range(v["ticks"] + 1):
        s = k / v["ticks"]
        major = k % v["tick_major"] == 0
        d.line(Z([point(v, s, ro), point(v, s, ro - (16.0 if major else 9.0))]),
               fill=INK, width=lw(3 if major else 2))
    for lab in scene["labels"]:
        d.text((lab["x"] * SS, lab["y"] * SS), lab["text"], fill=INK,
               font=font(lab["size"] * SS), anchor="mm")
    return img


def _draw_timer(d, scene, f):
    x0, y0, x1, y1 = (b * SS for b in scene["timer"]["box"])
    d.rounded_rectangle([x0, y0, x1, y1], radius=10 * SS, fill=(18, 18, 22),
                        outline=(235, 235, 235), width=2 * SS)
    d.text(((x0 + x1) / 2, (y0 + y1) / 2), timer_text(f), fill=(250, 250, 250),
           font=font(scene["timer"]["size"] * SS), anchor="mm")


def frame(scene, t: float) -> Image.Image:
    f = int(round(t * FPS))
    img = _static(json.dumps(scene, sort_keys=True)).copy()
    d = ImageDraw.Draw(img)
    lay = layout(scene, f)
    # lower lanes last, so a body never covers the lane drawn in front of it
    for r in sorted(scene["racers"], key=lambda r: r["lane"]):
        st = lay["racers"][r["id"]]
        sp = _sprite(r["shape"], r["colour"], r["pattern"], r["r_px"])
        if st["phi"]:
            sp = sp.rotate(-st["phi"], resample=Image.BICUBIC)
        img.paste(sp, (round(st["cx"] * SS - sp.width / 2),
                       round(st["cy"] * SS - sp.height / 2)), sp)
    _draw_timer(d, scene, f)
    return img.resize((W, H), Image.LANCZOS)