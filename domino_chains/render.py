"""PIL renderer for domino_chains: a top-down board. Standing dominoes are
dark bars across their lane; a toppling domino swings flat along the
direction it was hit (FALL_S seconds) and turns light. Lettered junction
dominoes are blue and larger. Faint guide lines show each lane."""
from __future__ import annotations
import sys
from pathlib import Path

from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from common.video5 import font  # noqa: E402

import sim  # noqa: E402

BG = (226, 232, 220)
INK = (40, 42, 48)
MUTED = (100, 106, 100)
X0, Y0, S = 80, 130, 70
FALL_S = 0.3


def duration(scene) -> float:
    return scene["end_t"]


def _p(x, y):
    return X0 + S * x, Y0 + S * y


def _rect(cx, cy, dx, dy, half_along, half_perp):
    px, py = -dy, dx
    return [_p(cx + sa * half_along * dx + sp * half_perp * px, cy + sa * half_along * dy + sp * half_perp * py)
            for sa, sp in ((-1, -1), (1, -1), (1, 1), (-1, 1))]


def _mix(a, b, f):
    return tuple(int(a[i] + (b[i] - a[i]) * f) for i in range(3))


def frame(scene, t: float) -> Image.Image:
    img = Image.new("RGB", (1280, 720), BG)
    d = ImageDraw.Draw(img)
    d.text((60, 30), "Domino chains", fill=INK, font=font(30))
    d.text((60, 72), "one push knocks over A; forks, merges and gaps decide what falls", fill=MUTED, font=font(19))
    d.text((1010, 30), f"t = {t:5.1f} s", fill=INK, font=font(30))

    pos = {q["id"]: (q["x"], q["y"]) for q in scene["dominoes"]}
    for lane in scene["lanes"]:
        d.line([_p(*pos[lane["u"]]), _p(*pos[lane["v"]])], fill=(200, 208, 196), width=3)
    dirs = sim.fall_dir(scene)
    lane_dir = {}
    for lane in scene["lanes"]:
        for i in lane["ids"]:
            lane_dir[i] = lane["dir"]
    fall = scene["fall"]
    for q in sorted(scene["dominoes"], key=lambda q: q["label"] is not None):
        big = q["label"] is not None
        ha, hp = (0.07, 0.26) if big else (0.05, 0.18)
        stand_col, flat_col = ((40, 70, 150), (150, 180, 235)) if big else ((52, 50, 56), (214, 200, 170))
        tf = fall.get(q["id"])
        if tf is not None and t >= tf:
            f = min(1.0, (t - tf) / FALL_S)
            dx, dy = dirs[q["id"]]
            length = 2 * ha + f * (0.5 if big else 0.4)
            bx, by = q["x"] - ha * dx, q["y"] - ha * dy
            poly = _rect(bx + length / 2 * dx, by + length / 2 * dy, dx, dy, length / 2, hp)
            d.polygon(poly, fill=_mix(stand_col, flat_col, f), outline=INK)
        else:
            dx, dy = lane_dir.get(q["id"], (1.0, 0.0))
            d.polygon(_rect(q["x"], q["y"], dx, dy, ha, hp), fill=stand_col, outline=INK)
        if big:
            lx, ly = _p(q["x"] - 0.05, q["y"] - 0.62)
            d.ellipse([lx - 15, ly - 15, lx + 15, ly + 15], fill=(255, 255, 255), outline=(40, 70, 150), width=2)
            d.text((lx - 7, ly - 12), q["label"], fill=(40, 70, 150), font=font(20))

    ax, ay = pos["A"]
    if sim.PUSH_T - 0.8 <= t <= sim.PUSH_T + 0.3:
        k = min(1.0, max(0.0, (t - (sim.PUSH_T - 0.8)) / 0.8))
        tip = ax - 0.15 - (1 - k) * 0.6
        x1, y1 = _p(tip, ay)
        x0, _ = _p(tip - 0.7, ay)
        d.line([x0, y1, x1 - 12, y1], fill=(200, 60, 50), width=6)
        d.polygon([(x1, y1), (x1 - 16, y1 - 11), (x1 - 16, y1 + 11)], fill=(200, 60, 50))
    return img
