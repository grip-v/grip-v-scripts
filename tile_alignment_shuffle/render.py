"""PIL renderer for tile_alignment_shuffle (v3).

The grid sits somewhere in the frame (position and size vary per clip),
with row numbers on its left and column numbers above it; a clock in a dark
box sits at the top-left. Those numbers and the clock are the only text.
Each tile is a bordered card with one solid shape or one symbol.

Motion per scenario:
  * swap / neighbour / double: the trading tiles lift (grow a little, cast
    a shadow) and pass each other on bowed paths that stay inside the board;
  * cycle: the three tiles lift and glide straight to their new cells;
  * slide: the row or column glides one cell; the end tile slides under
    the board's edge and comes back in at the other end (clipped to the
    board, so nothing moves over the numbers).

Timeline (frames): scene["timing"] intro, then per move `move` + `pause`,
then `hold`. Frame f shows time f / 24 s and the clock reads it to 0.1 s
(rounded down)."""
from __future__ import annotations
import functools
import json
import math
import sys
from pathlib import Path

from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from common.video5 import font  # noqa: E402
from common.shapes import unit_shape  # noqa: E402

import sim

FPS = sim.FPS
SS = 3                      # supersampling for the tile sprites


def duration(scene) -> float:
    return scene["n_frames"] / FPS


def ease(u: float) -> float:
    return u * u * (3 - 2 * u)


def geom(scene):
    lay = scene["layout"]
    return lay["span"] / scene["n"], lay["x0"], lay["y0"]


def to_px(scene, rc):
    cell, x0, y0 = geom(scene)
    return (x0 + cell * (rc[1] + 0.5), y0 + cell * (rc[0] + 0.5))


def tile_size(scene) -> int:
    cell = geom(scene)[0]
    return int(round(cell * (1 - scene["style"]["gap"])))


@functools.lru_cache(maxsize=16)
def _boards(key: str):
    return sim.grids(json.loads(key))


def boards(scene):
    return _boards(json.dumps({"start": scene["start"], "moves": scene["moves"]}, sort_keys=True))


def state(scene, f: int) -> dict:
    """Everything drawn at frame f except the static background:
    board  - the resting board (tile ids) underneath,
    pos    - pixel centre of every tile (a wrapping slide tile: its larger visible copy),
    draw   - [(tid, x, y, lifted)] for the moving tiles, every visible copy,
    moving - ids of the tiles in motion (empty when all tiles rest),
    u      - eased progress of the move in motion."""
    S = len(scene["moves"])
    gs = boards(scene)
    tm = scene["timing"]
    k = None
    for j in range(S):
        s = sim.move_start(scene, j)
        if s < f < s + tm["move"]:
            k = j
            break
    done = sum(1 for j in range(S) if sim.move_end(scene, j) <= f)
    g = gs[k] if k is not None else gs[done]
    n = scene["n"]
    pos = {g[r][c]: to_px(scene, (r, c)) for r in range(n) for c in range(n)}
    out = {"board": g, "pos": pos, "draw": [], "moving": [], "u": 0.0, "kind": None}
    if k is None:
        return out
    m = scene["moves"][k]
    e = ease((f - sim.move_start(scene, k)) / tm["move"])
    out.update(u=e, kind=m["kind"])
    lifted = m["kind"] != "slide"
    for i, ((r, c), _) in enumerate(m["pairs"]):
        tid = g[r][c]
        pr, pc = sim.path_pos(n, m, i, e)
        x, y = to_px(scene, (pr, pc))
        out["moving"].append(tid)
        out["draw"].append((tid, x, y, lifted))
        if m["kind"] == "slide" and not (0 <= pr <= n - 1 and 0 <= pc <= n - 1):
            dr, dc = m["shift"]
            x2, y2 = to_px(scene, (pr - dr * n, pc - dc * n))
            out["draw"].append((tid, x2, y2, False))
            if e >= 0.5:
                x, y = x2, y2
        pos[tid] = (x, y)
    return out


# --------------------------------------------------------------------------- sprites and background


@functools.lru_cache(maxsize=256)
def sprite(kind: str, color: str, size: int, card: str, radius: float, border: int, scale: float) -> Image.Image:
    big = size * SS
    img = Image.new("RGBA", (big, big), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    fill, edge = sim.CARDS[card]
    m = border * SS / 2
    d.rounded_rectangle([m, m, big - m, big - m], radius * big, fill=fill, outline=edge, width=border * SS)
    col = sim.COLORS[color]
    if kind in sim.SHAPES:
        r = big * scale
        pts = [(big / 2 + r * x, big / 2 + r * y) for x, y in unit_shape(kind, 1.0)]
        dark = tuple(int(v * 0.55) for v in col)
        d.polygon(pts, fill=col, outline=dark, width=2 * SS)
    else:
        d.text((big / 2, big / 2), kind, fill=col, font=font(int(big * scale * 1.85)), anchor="mm")
    return img.resize((size, size), Image.LANCZOS)


def _sprite_for(scene, tid, size):
    d = scene["designs"][scene["tiles"][tid]]
    st = scene["style"]
    return sprite(d["kind"], d["color"], size, scene["card"], st["radius"], st["border"], st["scale"])


@functools.lru_cache(maxsize=16)
def _background(key: str) -> Image.Image:
    sc = json.loads(key)
    th = sim.THEMES[sc["theme"]]
    img = Image.new("RGB", (1280, 720), th["bg"])
    d = ImageDraw.Draw(img)
    lay, n = sc["layout"], sc["n"]
    x0, y0, span, pad = lay["x0"], lay["y0"], lay["span"], lay["pad"]
    cell = span / n
    d.rounded_rectangle([x0 - pad, y0 - pad, x0 + span + pad, y0 + span + pad], 14, fill=th["slot"])
    for r in range(n):
        for c in range(n):
            d.rounded_rectangle([x0 + c * cell + 3, y0 + r * cell + 3, x0 + (c + 1) * cell - 3,
                                 y0 + (r + 1) * cell - 3], 8, outline=th["cell"], width=3)
    for lab in sc["labels"]:
        d.text((lab["x"], lab["y"]), lab["text"], fill=th["ink"], font=font(lab["size"]), anchor="mm")
    return img


def background(scene) -> Image.Image:
    key = json.dumps({k: scene[k] for k in ("theme", "layout", "labels", "n")}, sort_keys=True)
    return _background(key)


def draw_timer(img, scene, f):
    d = ImageDraw.Draw(img)
    x0, y0, x1, y1 = scene["timer"]["box"]
    d.rounded_rectangle([x0, y0, x1, y1], radius=10, fill=(18, 18, 22), outline=(235, 235, 235), width=2)
    d.text(((x0 + x1) / 2, (y0 + y1) / 2), sim.timer_text(f), fill=(250, 250, 250),
           font=font(scene["timer"]["size"]), anchor="mm")


def _paste(img, spr, x, y):
    w = spr.size[0]
    img.paste(spr, (int(round(x - w / 2)), int(round(y - w / 2))), spr)


def frame(scene, t: float) -> Image.Image:
    f = int(round(t * FPS))
    st = state(scene, f)
    img = background(scene).copy()
    size = tile_size(scene)
    moving = set(st["moving"])
    for tid, (x, y) in st["pos"].items():
        if tid not in moving:
            _paste(img, _sprite_for(scene, tid, size), x, y)
    if moving:
        lay = scene["layout"]
        pad = lay["pad"]
        ox, oy = lay["x0"] - pad, lay["y0"] - pad
        side = lay["span"] + 2 * pad
        layer = Image.new("RGBA", (side, side), (0, 0, 0, 0))
        lift = math.sin(math.pi * st["u"])
        big = int(round(size * (1 + scene["style"]["lift"] * lift)))
        ld = ImageDraw.Draw(layer)
        sh = int(round(3 + 8 * lift))
        for tid, x, y, lifted in st["draw"]:
            if lifted:
                ld.rounded_rectangle([x - ox - big / 2 + sh, y - oy - big / 2 + sh, x - ox + big / 2 + sh,
                                      y - oy + big / 2 + sh], scene["style"]["radius"] * big, fill=(0, 0, 0, 70))
        for tid, x, y, lifted in st["draw"]:
            _paste(layer, _sprite_for(scene, tid, big if lifted else size), x - ox, y - oy)
        mask = layer.getchannel("A")
        if st["kind"] == "slide":                     # slide under the board edge, not over the pad
            clip = Image.new("L", (side, side), 0)
            ImageDraw.Draw(clip).rectangle([pad, pad, pad + lay["span"] - 1, pad + lay["span"] - 1], fill=255)
            mask = Image.composite(mask, clip, clip)
        img.paste(layer, (ox, oy), mask)
    draw_timer(img, scene, f)
    return img
