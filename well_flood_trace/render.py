"""PIL renderer for well_flood_trace: frame(scene, t) and duration(scene).

The original look, kept: an off-white square board with a header strip above
an N x N grid; every section shows its depth value; water rises from the
bottom of a section as a blue overlay, light blue while the section is still
filling and dark blue once it is full (and passing water on); the entry
section is outlined in red. Changes from the original, for readability and
the clock rule only:
  * the header strip shows the clock (seconds to 0.1 s, floor(10 f / 24)
    tenths) instead of "simulated minute"; it is the only text besides the
    depth values;
  * grid lines are 3 px and the depth digits are larger, dark with a light
    halo, and white on a full (dark blue) section;
  * a second entry is outlined in purple from the moment it starts pouring; a
    pour that stops turns its outline grey, a pour that doubles gets a double-
    thickness outline; walls blocking a spill are thick dark bars.
A section with water always shows at least a 4 px strip (6 px above an entry
outline or a wall bar covering the bottom of the cell), so its first wet frame
is visible.
"""
from __future__ import annotations

from PIL import Image, ImageDraw

import sim
from common.video5 import font

FPS = sim.FPS
W, H = 1280, 720

# original config.py colours, plus two close variants
PALETTE = [
    {"bg": (247, 245, 239), "dry": (239, 233, 220), "line": (185, 178, 160), "digit": (90, 83, 70),
     "rising": (111, 168, 220), "full": (44, 95, 138), "head": (240, 236, 223), "head_text": (30, 27, 20)},
    {"bg": (244, 241, 232), "dry": (236, 230, 214), "line": (180, 172, 152), "digit": (86, 78, 64),
     "rising": (116, 179, 224), "full": (37, 88, 138), "head": (237, 232, 218), "head_text": (30, 27, 20)},
    {"bg": (246, 246, 242), "dry": (238, 235, 227), "line": (176, 172, 160), "digit": (82, 80, 72),
     "rising": (104, 160, 214), "full": (31, 79, 126), "head": (238, 236, 228), "head_text": (28, 28, 26)},
]
ENTRY = {"red": (201, 66, 58), "purple": (123, 63, 176)}
STOPPED = (154, 149, 138)
WALL = (59, 53, 43)
LINE_W = 3
OUTLINE_W = 5

_BASE = {"scene": None, "img": None}


def duration(scene) -> float:
    return scene["n_frames"] / FPS


def cell_rect(scene, i):
    lay = scene["layout"]
    r, c = divmod(i, scene["C"])
    x, y = lay["x0"] + c * lay["cell"], lay["y0"] + r * lay["cell"]
    return x, y, x + lay["cell"], y + lay["cell"]


def clock_box(scene):
    lay = scene["layout"]
    return lay["bx"], lay["by"], lay["bx"] + lay["S"], lay["by"] + lay["head"]


def level_at(scene, i, t):
    """Water in section i at time t, or None while dry."""
    st = scene["track"]["start"][i]
    if st is None or t < st:
        return None
    cur = None
    for sg in scene["track"]["level"][i]:
        if sg[0] <= t:
            cur = sg
        else:
            break
    return min(float(scene["depth"][i]), cur[1] + cur[2] * (t - cur[0]))


def is_full(scene, i, t) -> bool:
    f = scene["track"]["full"][i]
    return f is not None and f <= t


def wall_bar(scene) -> int:
    return max(9, scene["layout"]["cell"] // 7)


def covers(scene, i):
    """(top, bottom): rows of section i's interior hidden under an entry outline or a wall
    bar on its top / bottom edge. The first water is drawn above the bottom cover."""
    x0, y0, x1, y1 = cell_rect(scene, i)
    top = bottom = 0
    if i in [tp["cell"] for tp in scene["taps"]]:
        top = bottom = 2 + 2 * OUTLINE_W - LINE_W      # room for a doubled outline
    half = wall_bar(scene) / 2 - LINE_W + 1
    C = scene["C"]
    for a, b in scene["walls"]:
        if {a, b} == {i, i - C}:
            top = max(top, half)
        if {a, b} == {i, i + C}:
            bottom = max(bottom, half)
    return top, bottom


def water_top(scene, i, lv):
    """Pixel row of the water surface in section i for volume lv: at least 4 px of water,
    or 6 px above an entry outline / wall bar covering the bottom of the cell."""
    x0, y0, x1, y1 = cell_rect(scene, i)
    inner = (y1 - LINE_W) - (y0 + LINE_W)
    cov = covers(scene, i)[1]
    return y1 - LINE_W - max(cov + 6.0 if cov else 4.0, lv / scene["depth"][i] * inner)


def pour_rate(tap, t):
    for f0, f1, rate in tap["segs"]:
        if sim.tap_time(f0) <= t and (f1 is None or t < sim.tap_time(f1)):
            return rate
    return 0.0


def _base(scene) -> Image.Image:
    pal = PALETTE[scene["palette"]]
    lay = scene["layout"]
    img = Image.new("RGB", (W, H), pal["bg"])
    d = ImageDraw.Draw(img)
    d.rectangle(clock_box(scene), fill=pal["head"])
    N, cell = scene["R"], lay["cell"]
    x0, y0 = lay["x0"], lay["y0"]
    d.rectangle([x0, y0, x0 + N * cell, y0 + N * cell], fill=pal["dry"])
    return img


def _digits(d, scene, pal, t):
    """Depth values: dark with a light halo, white on a full (dark blue) section."""
    f = font(int(scene["layout"]["cell"] * 0.38))
    for i, v in enumerate(scene["depth"]):
        x0, y0, x1, y1 = cell_rect(scene, i)
        if is_full(scene, i, t):
            d.text(((x0 + x1) / 2, (y0 + y1) / 2), str(v), font=f, fill=(255, 255, 255), anchor="mm")
        else:
            d.text(((x0 + x1) / 2, (y0 + y1) / 2), str(v), font=f, fill=pal["digit"], anchor="mm",
                   stroke_width=3, stroke_fill=pal["dry"])


def _grid(d, scene, pal):
    lay = scene["layout"]
    N, cell, x0, y0 = scene["R"], lay["cell"], lay["x0"], lay["y0"]
    for k in range(N + 1):
        d.line([(x0 + k * cell, y0), (x0 + k * cell, y0 + N * cell)], fill=pal["line"], width=LINE_W)
        d.line([(x0, y0 + k * cell), (x0 + N * cell, y0 + k * cell)], fill=pal["line"], width=LINE_W)
    bar = wall_bar(scene)
    for a, b in scene["walls"]:
        lo, hi = min(a, b), max(a, b)
        hx0, hy0, hx1, hy1 = cell_rect(scene, hi)
        if hi - lo == 1:
            d.rectangle([hx0 - bar / 2, hy0 + 2, hx0 + bar / 2, hy1 - 2], fill=WALL)
        else:
            d.rectangle([hx0 + 2, hy0 - bar / 2, hx1 - 2, hy0 + bar / 2], fill=WALL)


def _entries(d, scene, t):
    for k, tap in enumerate(scene["taps"]):
        on = sim.tap_time(tap["segs"][0][0])
        if k > 0 and t < on:
            continue                      # a second entry shows once it starts pouring
        x0, y0, x1, y1 = cell_rect(scene, tap["cell"])
        rate = pour_rate(tap, t)
        col = ENTRY[tap["colour"]]
        width = OUTLINE_W
        if tap["change"] == "stop" and t >= sim.tap_time(tap["segs"][0][1]):
            col = STOPPED
        elif tap["change"] == "double" and rate > tap["segs"][0][2]:
            width = OUTLINE_W * 2
        d.rectangle([x0 + 2, y0 + 2, x1 - 2, y1 - 2], outline=col, width=width)


def _clock(d, scene, fr, pal):
    x0, y0, x1, y1 = clock_box(scene)
    d.text(((x0 + x1) / 2, (y0 + y1) / 2), f"{sim.clock_tenths(fr) / 10:.1f} s", fill=pal["head_text"],
           font=font(int((y1 - y0) * 0.58)), anchor="mm")


def frame(scene, t: float) -> Image.Image:
    if _BASE["scene"] is not scene:
        _BASE["scene"], _BASE["img"] = scene, _base(scene)
    img = _BASE["img"].copy()
    d = ImageDraw.Draw(img)
    pal = PALETTE[scene["palette"]]
    for i in range(scene["R"] * scene["C"]):
        lv = level_at(scene, i, t)
        if lv is None:
            continue
        x0, y0, x1, y1 = cell_rect(scene, i)
        col = pal["full"] if is_full(scene, i, t) else pal["rising"]
        d.rectangle([x0 + LINE_W, water_top(scene, i, lv), x1 - LINE_W, y1 - LINE_W], fill=col)
    _grid(d, scene, pal)
    _digits(d, scene, pal, t)
    _entries(d, scene, t)
    _clock(d, scene, int(round(t * FPS)), pal)
    return img
