"""PIL renderer for shell_game_tracking_v2: 3-8 identical cups on a numbered shelf. During the reveal the cups
that hide balls are raised and every ball is drawn on the shelf; after `reveal_end` the cups are down and balls
are never rendered again, so no later frame leaks a ball's location."""
from __future__ import annotations
import math
import sys
from pathlib import Path

from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from common.video5 import hex_rgb, font  # noqa: E402

W, H, SS = 1280, 720, 2
BG = (20, 20, 20)
MARGIN = 70
LOWER_S = 0.3


def duration(scene) -> float:
    return scene["duration"]


def layout(n):
    sp = (W - 2 * MARGIN) / n
    hw = min(95.0, 0.36 * sp)
    return sp, hw, hw * 1.14


def _cup_y():
    return H * 0.50


def _positions(scene, t):
    """cup id (= starting slot) -> (x, y)."""
    n = scene["n_cups"]
    sp, hw, hh = layout(n)
    xs = [MARGIN + sp * (k + 0.5) for k in range(n)]
    y0, amp = _cup_y(), hh * 2.1      # one cup of the pair hops clear above the row, the other slides level
    row = list(range(n))
    for s in scene["swaps"]:
        if t >= s["end"]:
            row[s["i"]], row[s["j"]] = row[s["j"]], row[s["i"]]
        elif t >= s["start"]:
            f = (t - s["start"]) / (s["end"] - s["start"])
            pos = {row[k]: (xs[k], y0) for k in range(n) if k not in (s["i"], s["j"])}
            arc = -amp * math.sin(math.pi * f)
            xi, xj = xs[s["i"]], xs[s["j"]]
            pos[row[s["i"]]] = (xi + (xj - xi) * f, y0 + arc)
            pos[row[s["j"]]] = (xj + (xi - xj) * f, y0)
            return pos
    return {row[k]: (xs[k], y0) for k in range(n)}


def _star(cx, cy, r):
    pts = []
    for k in range(10):
        a = -math.pi / 2 + k * math.pi / 5
        rr = r if k % 2 == 0 else r * 0.42
        pts.append((cx + rr * math.cos(a), cy + rr * math.sin(a)))
    return pts


def _triangle(cx, cy, r):
    return [(cx + r * math.cos(-math.pi / 2 + k * 2 * math.pi / 3),
             cy + r * math.sin(-math.pi / 2 + k * 2 * math.pi / 3)) for k in range(3)]


def _draw_ball(d, b, x, bottom, hw):
    """Ball of given kind centred on x, resting with its lowest point at `bottom` (all in supersampled px)."""
    col = hex_rgb(b["hex"])
    k = b["kind"]
    r = {"small": 0.30, "big": 0.50, "ring": 0.45, "triangle": 0.48, "star": 0.50}[k] * hw * SS
    cy = bottom - r
    box = [x - r, cy - r, x + r, cy + r]
    if k == "ring":
        d.ellipse(box, fill=col)
        ri = r * 0.50
        d.ellipse([x - ri, cy - ri, x + ri, cy + ri], fill=BG)
        return
    d.ellipse(box, fill=col)
    if k in ("triangle", "star"):
        lum = 0.3 * col[0] + 0.59 * col[1] + 0.11 * col[2]
        ink = (25, 25, 25) if lum > 140 else (255, 255, 255)
        pts = _triangle(x, cy + r * 0.05, r * 0.62) if k == "triangle" else _star(x, cy, r * 0.68)
        d.polygon(pts, fill=ink)


def _label_font():
    return font(30)


def frame(scene, t: float) -> Image.Image:
    n = scene["n_cups"]
    sp, hw, hh = layout(n)
    y0 = _cup_y()
    cup = hex_rgb(scene["cup_hex"])
    img = Image.new("RGB", (W * SS, H * SS), BG)
    d = ImageDraw.Draw(img)
    gy = y0 + hh + 16
    d.line([(40 * SS, gy * SS), ((W - 40) * SS, gy * SS)], fill=(70, 70, 75), width=5 * SS)
    small = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    sd = ImageDraw.Draw(small)
    f = _label_font()
    for k in range(n):                                   # fixed slot numbers under the shelf
        txt = str(k + 1)
        tw = sd.textlength(txt, font=f)
        sd.text((MARGIN + sp * (k + 0.5) - tw / 2, gy + 14), txt, font=f, fill=(190, 190, 195))
    reveal = t <= scene["reveal_end"]
    lift_of = {}
    if reveal:
        lift = 1.25 * hh * min(1.0, (scene["reveal_end"] - t) / LOWER_S)
        for b in scene["balls"]:                         # balls sit on the shelf line under their raised cup
            lift_of[b["slot"]] = lift
            _draw_ball(d, b, (MARGIN + sp * (b["slot"] + 0.5)) * SS, (y0 + hh) * SS, hw)
    for c, (x, y) in _positions(scene, t).items():
        y -= lift_of.get(c, 0.0)
        x, y, w_, h_ = x * SS, y * SS, hw * SS, hh * SS
        d.polygon([(x - w_, y + h_), (x + w_, y + h_), (x + w_ * 0.68, y - h_), (x - w_ * 0.68, y - h_)],
                  fill=cup, outline=(255, 255, 255), width=4 * SS)
    out = img.resize((W, H), Image.LANCZOS).convert("RGBA")
    out.alpha_composite(small)
    return out.convert("RGB")
