"""PIL renderer for gear_train.

Each gear is an involute-profile spur gear (20 degree pressure angle, a little
backlash) drawn once per scene as an anti-aliased RGBA sprite (3x supersampled):
coloured body with a darker outline, an optional face pattern (holes, spokes
or a ring), a hub (grey axle; the driver's hub is large and black) and a white
marker dot with a black rim that turns with the gear. Per frame every sprite
is rotated to its exact angle and placed at its exact (sub-pixel) centre with
one affine resample, so meshing teeth interlock without overlapping. The
brake block (brake scenario) is drawn on top. The clock in the top-right box
is the only text.

Coordinates: continuous pixel space (pixel i spans [i, i+1]); gear angles are
counter-clockwise positive on screen.
"""
from __future__ import annotations
import json
import math
import random
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from common.video5 import font  # noqa: E402

import sim  # noqa: E402

W, H = sim.W, sim.H
SS = 3                     # supersampling for sprites
ALPHA = math.radians(20)   # pressure angle
BACKLASH = 0.07            # modules of tooth-thickness taken off at the pitch circle
MARK_R = 0.72              # marker centre, fraction of the root radius
_CACHE: dict = {}


def duration(scene) -> float:
    return scene["duration"]


def _inv(a):
    return math.tan(a) - a


def tooth_profile(z: int, m: float) -> list:
    """Closed outline of a spur gear, maths coords (x right, y up), tooth 0 centred on angle 0."""
    r = m * z / 2
    rb = r * math.cos(ALPHA)
    ra = r + m
    rf = r - 1.25 * m
    psi_p = (math.pi * m / 2 - BACKLASH * m) / (2 * r)

    def psi(rho):
        rho = max(rho, rb)
        return max(0.012, psi_p + _inv(ALPHA) - _inv(math.acos(min(1.0, rb / rho))))
    lo = max(rf, rb)
    radii = [rf] + ([] if lo <= rf else [lo]) + [lo + (ra - lo) * k / 6 for k in range(1, 7)]
    pts = []
    for k in range(z):
        c = 2 * math.pi * k / z
        flank = [(rho, c - psi(rho)) for rho in radii]
        tip = [(ra, c - psi(ra) + 2 * psi(ra) * j / 4) for j in range(1, 4)]
        back = [(rho, c + psi(rho)) for rho in reversed(radii)]
        c2 = 2 * math.pi * (k + 1) / z
        a0, a1 = c + psi(rf), c2 - psi(rf)
        root = [(rf, a0 + (a1 - a0) * j / 4) for j in range(1, 4)]
        for rho, a in flank + tip + back + root:
            pts.append((rho * math.cos(a), rho * math.sin(a)))
    return pts


def count_teeth(z: int, m: float) -> int:
    """Teeth counted off the drawn outline: runs of vertices out at the tip circle."""
    ra = m * z / 2 + m
    out = [math.hypot(x, y) > ra - 0.05 * m for x, y in tooth_profile(z, m)]
    return sum(1 for i in range(len(out)) if out[i] and not out[i - 1])


def shade(rgb, f):
    return tuple(int(max(0, min(255, c * f))) for c in rgb)


def _sprite(scene, g):
    m = scene["m"]
    z = g["z"]
    ra = m * z / 2 + m
    rf = m * z / 2 - 1.25 * m
    S = 2 * int(math.ceil(ra + 4))
    big = S * SS
    c = big / 2

    def P(x, y):                       # maths coords (px) -> supersampled draw coords
        return (c + x * SS - 0.5, c - y * SS - 0.5)

    def circ(d, x, y, r, **kw):
        cx, cy = P(x, y)
        d.ellipse([cx - r * SS, cy - r * SS, cx + r * SS, cy + r * SS], **kw)

    rgb = tuple(g["rgb"])
    dark = shade(rgb, 0.55)
    body = Image.new("RGBA", (big, big), (0, 0, 0, 0))
    d = ImageDraw.Draw(body)
    outline = [P(x, y) for x, y in tooth_profile(z, m)]
    ow = scene["theme"]["outline_w"]
    d.polygon(outline, fill=rgb + (255,))
    d.line(outline + [outline[0]], fill=dark + (255,), width=ow * SS, joint="curve")
    pat, n = g["pattern"], g["pattern_n"]
    rm = max(10.5, 0.1 * rf)
    if pat == "holes" and rf >= 50:
        for k in range(n):
            a = math.pi / n + 2 * math.pi * k / n
            circ(d, 0.38 * rf * math.cos(a), 0.38 * rf * math.sin(a), 0.11 * rf, fill=shade(rgb, 0.4) + (255,),
                 outline=dark + (255,), width=2 * SS)
    elif pat == "spokes":
        wsp = max(3.0, 0.07 * rf)
        for k in range(n):
            a = math.pi / n + 2 * math.pi * k / n
            d.line([P(0, 0), P(0.5 * rf * math.cos(a), 0.5 * rf * math.sin(a))], fill=dark + (255,),
                   width=int(wsp * SS))
    elif pat == "ring":
        wr = max(3.0, 0.06 * rf)
        cx, cy = P(0, 0)
        rr = 0.48 * rf * SS
        d.ellipse([cx - rr, cy - rr, cx + rr, cy + rr], outline=dark + (255,), width=int(wr * SS))
    if g["id"] == scene["driver"]:
        circ(d, 0, 0, max(13.0, 0.26 * rf), fill=(12, 12, 14, 255), outline=(235, 235, 235, 255), width=2 * SS)
    else:
        circ(d, 0, 0, max(6.0, 0.12 * rf), fill=(190, 192, 198, 255), outline=(40, 40, 46, 255), width=2 * SS)
    circ(d, MARK_R * rf, 0, rm, fill=(255, 255, 255, 255), outline=(10, 10, 10, 255), width=int(2.5 * SS))
    return body.reduce(SS)


def _background(theme):
    bg = tuple(theme["bg"])
    img = Image.new("RGB", (W, H), bg)
    d = ImageDraw.Draw(img)
    r = random.Random(theme["texture_seed"])
    sgn = 1 if theme["dark"] else -1
    tone = lambda k: tuple(int(max(0, min(255, c + sgn * k))) for c in bg)  # noqa: E731
    st = theme["style"]
    if st == "gradient":
        top, bot = tone(18), tone(-10)
        for y in range(H):
            f = y / (H - 1)
            d.line([(0, y), (W, y)], fill=tuple(int(top[i] * (1 - f) + bot[i] * f) for i in range(3)))
    elif st == "grid":
        step = r.choice([40, 48, 60])
        for x in range(0, W, step):
            d.line([(x, 0), (x, H)], fill=tone(12), width=1)
        for y in range(0, H, step):
            d.line([(0, y), (W, y)], fill=tone(12), width=1)
    elif st == "dots":
        step = r.choice([28, 36, 44])
        for y in range(step // 2, H, step):
            for x in range(step // 2, W, step):
                d.ellipse([x - 2, y - 2, x + 2, y + 2], fill=tone(16))
    elif st == "pegboard":
        step = r.choice([44, 52])
        for y in range(step // 2, H, step):
            for x in range(step // 2, W, step):
                d.ellipse([x - 5, y - 5, x + 5, y + 5], fill=tone(-14 if theme["dark"] else 22))
    elif st == "hatch":
        for k in range(-H, W, r.choice([22, 30])):
            d.line([(k, 0), (k + H, H)], fill=tone(9), width=2)
    elif st == "vignette":
        yy, xx = np.mgrid[0:H, 0:W]
        v = ((xx - W / 2) / (W / 2)) ** 2 + ((yy - H / 2) / (H / 2)) ** 2
        arr = np.asarray(img).astype(np.float32) * (1 - 0.09 * np.clip(v, 0, 1.6))[..., None]
        img = Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))
    return img


def _static(scene):
    key = json.dumps([scene["gears"], scene["m"], scene["theme"], scene["driver"]], sort_keys=True)
    if key in _CACHE:
        return _CACHE[key]
    bg = _background(scene["theme"])
    sprites = {g["id"]: _sprite(scene, g) for g in scene["gears"]}
    if len(_CACHE) > 6:
        _CACHE.clear()
    _CACHE[key] = (bg, sprites)
    return bg, sprites


def _place(img, sprite, cx, cy, th):
    S = sprite.size[0]
    half = S / 2
    reach = half * 1.0 + 2
    X0, Y0 = int(math.floor(cx - reach)), int(math.floor(cy - reach))
    size = int(math.ceil(2 * reach)) + 2
    c, s = math.cos(th), math.sin(th)
    a, b, e = c, -s, c
    dd = s
    cc = half + c * (X0 - cx) - s * (Y0 - cy)
    f = half + s * (X0 - cx) + c * (Y0 - cy)
    tile = sprite.transform((size, size), Image.AFFINE, (a, b, cc, dd, e, f), resample=Image.BICUBIC)
    img.alpha_composite(tile, (X0, Y0)) if X0 >= 0 and Y0 >= 0 and X0 + size <= W and Y0 + size <= H else \
        _composite_clipped(img, tile, X0, Y0)


def _composite_clipped(img, tile, X0, Y0):
    x0, y0 = max(0, X0), max(0, Y0)
    x1, y1 = min(W, X0 + tile.size[0]), min(H, Y0 + tile.size[1])
    if x1 <= x0 or y1 <= y0:
        return
    img.alpha_composite(tile.crop((x0 - X0, y0 - Y0, x1 - X0, y1 - Y0)), (x0, y0))


def draw_brake(img, scene, t):
    if not scene["program"].get("brake"):
        return
    d = ImageDraw.Draw(img)
    polys = sim.pad_polys(scene, sim.pad_inner(scene, t))
    edge = (230, 230, 236) if scene["theme"]["dark"] else (24, 24, 28)
    for poly in reversed(polys):
        pts = [(x - 0.5, y - 0.5) for x, y in poly]
        d.polygon(pts, fill=sim.BRAKE_RGB, outline=edge, width=3)


def draw_timer(img, t):
    d = ImageDraw.Draw(img)
    x0, y0, x1, y1 = sim.TIMER_BOX
    d.rounded_rectangle([x0, y0, x1, y1], radius=10, fill=(10, 10, 14), outline=(110, 110, 124), width=2)
    txt = f"{t:.1f} s"
    f = font(42)
    tw = d.textlength(txt, font=f)
    d.text((x1 - 16 - tw, y0 + 10), txt, fill=(255, 255, 255), font=f)


def frame(scene, t: float) -> Image.Image:
    bg, sprites = _static(scene)
    img = bg.convert("RGBA")
    for g in scene["gears"]:
        cx, cy = sim.centre(scene, g["id"], t)
        _place(img, sprites[g["id"]], cx, cy, sim.theta(scene, g["id"], t))
    draw_brake(img, scene, t)
    img = img.convert("RGB")
    draw_timer(img, t)
    return img
