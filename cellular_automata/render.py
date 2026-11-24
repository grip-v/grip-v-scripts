"""PIL renderer for cellular_automata (1280x720, 24 fps).

2-D scenarios draw the grid in place: generation 0 holds for start_f frames,
then every generation fades in over fade_f frames and holds. The elementary
scenario draws a space-time diagram: each new generation fades in as a new
row under the previous ones (rows not reached yet are not drawn).

Marked cells carry a coloured outline (ring / square frame / diamond frame);
in the elementary scenario the marked columns carry a coloured triangle /
dot / square above the top row. The torus grid has a dashed border. The only
text is the timer in the top-left corner (seconds to 0.1 s), which never
overlaps the grid area (sim.AREA).
"""
from __future__ import annotations
import sys
from pathlib import Path

from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from common.video5 import font  # noqa: E402

import sim  # noqa: E402

TIMER_XY = (36, 34)
TIMER_SIZE = 40


def duration(scene) -> float:
    return sim.n_frames(scene) / sim.FPS


def gen_at_frame(scene, f: int) -> tuple[int, float]:
    """(generation index shown, fade progress 0..1 from generation k-1)."""
    t = scene["timing"]
    if f < t["start_f"]:
        return 0, 1.0
    k = min(scene["gens"], 1 + (f - t["start_f"]) // t["gen_f"])
    into = f - sim.change_frame(scene, k)
    return k, min(1.0, (into + 1) / t["fade_f"])


def timer_text(f: int) -> str:
    return f"{sim.fmt_t(sim.tenths(f))} s"


def _colour(scene, v: int):
    st = scene["style"]
    return tuple(st["dead"] if v == 0 else st["live"] if v == 1 else st["dying"])


def _mix(a, b, u):
    return tuple(int(round(a[i] + (b[i] - a[i]) * u)) for i in range(3))


def cell_box(scene, r: int, c: int) -> tuple[int, int, int, int]:
    """Pixel box of the drawn cell (display row r, column c)."""
    st = scene["style"]
    p, g = st["cell_px"], st["gap"]
    x, y = st["x0"] + c * p, st["y0"] + r * p
    return x + g // 2, y + g // 2, x + p - (g - g // 2) - 1, y + p - (g - g // 2) - 1


def cell_centre(scene, r: int, c: int) -> tuple[int, int]:
    x0, y0, x1, y1 = cell_box(scene, r, c)
    return (x0 + x1) // 2, (y0 + y1) // 2


def _cell(d: ImageDraw.ImageDraw, scene, r, c, col):
    box = cell_box(scene, r, c)
    shape = scene["style"]["cell_shape"]
    if shape == "square":
        d.rectangle(box, fill=col)
    elif shape == "rounded":
        d.rounded_rectangle(box, radius=max(3, (box[2] - box[0]) // 4), fill=col)
    else:
        d.ellipse(box, fill=col)


def _marker(d: ImageDraw.ImageDraw, scene, m):
    col = tuple(m["rgb"])
    st = scene["style"]
    r, c = m["cell"]
    if scene["scenario"] == "elementary":
        cx = st["x0"] + c * st["cell_px"] + st["cell_px"] // 2
        s = max(20, min(28, st["cell_px"]))
        yb = st["y0"] - 8
        if m["shape"] == "triangle":
            d.polygon([(cx - s // 2, yb - s), (cx + s // 2, yb - s), (cx, yb)], fill=col)
        elif m["shape"] == "dot":
            d.ellipse([cx - s // 2, yb - s, cx + s // 2, yb], fill=col)
        else:
            d.rectangle([cx - s // 2 + 2, yb - s + 2, cx + s // 2 - 2, yb - 2], fill=col)
        return
    x0, y0, x1, y1 = cell_box(scene, r, c)
    x0, y0, x1, y1 = x0 - 1, y0 - 1, x1 + 1, y1 + 1
    if m["shape"] == "ring":
        d.ellipse([x0, y0, x1, y1], outline=col, width=4)
    elif m["shape"] == "square frame":
        d.rectangle([x0, y0, x1, y1], outline=col, width=4)
    else:
        cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
        d.polygon([(cx, y0), (x1, cy), (cx, y1), (x0, cy)], outline=col, width=4)


def _dashed_border(d: ImageDraw.ImageDraw, scene):
    st = scene["style"]
    x0, y0 = st["x0"] - 7, st["y0"] - 7
    x1, y1 = st["x0"] + scene["cols"] * st["cell_px"] + 6, st["y0"] + scene["rows"] * st["cell_px"] + 6
    ink = tuple(st["ink"])
    for a, b, horiz, fixed in ((x0, x1, True, y0), (x0, x1, True, y1), (y0, y1, False, x0), (y0, y1, False, x1)):
        p = a
        while p < b:
            q = min(b, p + 14)
            d.line([(p, fixed), (q, fixed)] if horiz else [(fixed, p), (fixed, q)], fill=ink, width=4)
            p += 22


def frame_at(scene, f: int) -> Image.Image:
    st = scene["style"]
    img = Image.new("RGB", (1280, 720), tuple(st["bg"]))
    d = ImageDraw.Draw(img)
    h = scene["history"]
    k, u = gen_at_frame(scene, f)
    if scene["scenario"] == "elementary":
        for row in range(k + 1):
            for c in range(scene["cols"]):
                col = _colour(scene, h[row][0][c])
                if row == k and u < 1:
                    col = _mix(tuple(st["bg"]), col, u)
                _cell(d, scene, row, c, col)
    else:
        cur, prev = h[k], h[max(0, k - 1)]
        for r in range(scene["rows"]):
            for c in range(scene["cols"]):
                a, b = prev[r][c], cur[r][c]
                col = _colour(scene, b) if (a == b or u >= 1) else _mix(_colour(scene, a), _colour(scene, b), u)
                _cell(d, scene, r, c, col)
        if scene["wrap"]:
            _dashed_border(d, scene)
    for m in scene["markers"]:
        _marker(d, scene, m)
    d.text(TIMER_XY, timer_text(f), fill=tuple(st["ink"]), font=font(TIMER_SIZE))
    return img


def frame(scene, t: float) -> Image.Image:
    return frame_at(scene, int(round(t * sim.FPS)))
