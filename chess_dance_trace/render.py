"""PIL renderer for chess_dance_trace (1280x720, 24 fps).

The static layer (background, board, holes, coordinate labels) is drawn once
per scene; every frame pastes the pieces at the positions given by
`track(scene, i)` and draws the clock. The only text is the clock (fixed
panel, top-left, "s.t s", truncated to 0.1 s) and the board coordinates
from scene["labels"] (file letters under the board, rank numbers beside
it), drawn outside the board where no piece ever goes.

Pieces are python-chess artwork recoloured with the piece's own colour.
Within a round every mover glides from its square to its target over
timing["move_f"] frames (smoothstep easing); knights hop in an arc. A
captured piece disappears on the frame its capturer lands.
"""
from __future__ import annotations
import functools
import io
import json
import sys
from pathlib import Path

import cairosvg
import chess
import chess.svg
from PIL import Image, ImageDraw, ImageFilter

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE))
from common.video5 import font  # noqa: E402

import sim  # noqa: E402

W, H = sim.W, sim.H
EDGE = (52, 46, 40)
HOLE = (54, 54, 60)
HOLE_HATCH = (92, 92, 100)
TIMER_BG, TIMER_FG, TIMER_EDGE = (26, 28, 34), (255, 255, 255), (200, 200, 205)


def duration(scene) -> float:
    return scene["n_frames"] / sim.FPS


def ease(u: float) -> float:
    return u * u * (3 - 2 * u)


def square_centre(scene, f: float, r: float):
    st = scene["style"]
    c = st["cell"]
    return st["x0"] + (f + 0.5) * c, st["y0"] + (scene["N"] - 1 - r + 0.5) * c


def track(scene, i: int) -> list:
    """What frame i shows for each piece: (centre x, centre y, visible)."""
    st = scene["style"]
    out = []
    for p_idx, p in enumerate(scene["pieces"]):
        cur = sim.sq(p["start"])
        x, y = square_centre(scene, *cur)
        visible = True
        for r, mv in enumerate(scene["moves"], 1):
            s, land = sim.round_frames(scene, r)
            for m in mv:
                if m.get("capture") == p_idx and i >= land:
                    visible = False
                if m["p"] != p_idx or i <= s:
                    continue
                a, b = sim.sq(m["from"]), sim.sq(m["to"])
                if i >= land:
                    cur = b
                    x, y = square_centre(scene, *b)
                else:
                    u = ease((i - s) / (land - s))
                    x, y = square_centre(scene, a[0] + (b[0] - a[0]) * u, a[1] + (b[1] - a[1]) * u)
                    if p["type"] == "N":
                        y -= st["arc"] * st["cell"] * 4 * u * (1 - u)
        out.append((x, y, visible))
    return out


def moving_now(scene, i: int) -> set:
    out = set()
    for r, mv in enumerate(scene["moves"], 1):
        s, land = sim.round_frames(scene, r)
        if s < i < land:
            out |= {m["p"] for m in mv}
    return out


def _light(rgb) -> bool:
    return 0.299 * rgb[0] + 0.587 * rgb[1] + 0.114 * rgb[2] > 150


@functools.lru_cache(maxsize=256)
def _artwork(t: str, fill: str, size: int, black: bool) -> Image.Image:
    svg = chess.svg.piece(chess.Piece(sim.PT[t], chess.BLACK if black else chess.WHITE), size=size)
    if fill:
        svg = svg.replace('fill="#fff"', f'fill="{fill}"').replace("fill:#ffffff", f"fill:{fill}")
    png = cairosvg.svg2png(bytestring=svg.encode(), output_width=size, output_height=size)
    return Image.open(io.BytesIO(png)).convert("RGBA")


@functools.lru_cache(maxsize=256)
def icon(t: str, rgb: tuple, size: int, style: str) -> Image.Image:
    """The sprite of one piece, size x size px. 'figure': the artwork filled
    with the piece colour; 'disc' / 'tile': a token in the piece colour
    with the artwork drawn on it in black or white."""
    if style == "figure":
        return _artwork(t, "#%02x%02x%02x" % rgb, size, False)
    ss = 4
    big = Image.new("RGBA", (size * ss, size * ss), (0, 0, 0, 0))
    d = ImageDraw.Draw(big)
    box = [ss, ss, size * ss - ss - 1, size * ss - ss - 1]
    if style == "disc":
        d.ellipse(box, fill=rgb + (255,), outline=(25, 25, 30, 255), width=3 * ss)
    else:
        d.rounded_rectangle(box, radius=size * ss // 5, fill=rgb + (255,), outline=(25, 25, 30, 255), width=3 * ss)
    im = big.resize((size, size), Image.LANCZOS)
    g = int(round(size * 0.8))
    art = _artwork(t, "", g, _light(rgb))
    im.alpha_composite(art, ((size - g) // 2, (size - g) // 2 - 1))
    return im


@functools.lru_cache(maxsize=64)
def shadow(size: int) -> Image.Image:
    im = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    d = ImageDraw.Draw(im)
    d.ellipse([size * 0.18, size * 0.72, size * 0.82, size * 0.95], fill=(0, 0, 0, 70))
    return im.filter(ImageFilter.GaussianBlur(2))


def _mix(a, b, u):
    return tuple(int(round(a[k] + (b[k] - a[k]) * u)) for k in range(3))


@functools.lru_cache(maxsize=4)
def _static(key: str) -> Image.Image:
    scene = json.loads(key)
    st = scene["style"]
    bg, ink = tuple(st["bg"]), tuple(st["ink"])
    img = Image.new("RGB", (W, H), bg)
    d = ImageDraw.Draw(img)
    faint = _mix(bg, ink, 0.07)
    if st["bg_pattern"] == "gradient":
        for y in range(H):
            d.line([(0, y), (W, y)], fill=_mix(bg, ink, 0.09 * y / (H - 1)))
    elif st["bg_pattern"] == "dots":
        for x in range(12, W, 36):
            for y in range(12, H, 36):
                d.ellipse([x - 2, y - 2, x + 2, y + 2], fill=faint)
    elif st["bg_pattern"] == "stripes":
        for x in range(-H, W, 30):
            d.line([(x, H), (x + H, 0)], fill=faint, width=2)
    # the timer panel area is kept plain so the clock reads cleanly
    tb = scene["timer"]["box"]
    d.rectangle([tb[0] - 6, tb[1] - 6, tb[2] + 6, tb[3] + 6], fill=bg)
    N, c, x0, y0 = scene["N"], st["cell"], st["x0"], st["y0"]
    holes = {sim.sq(h) for h in scene["holes"]}
    e = st["edge_w"]
    d.rectangle([x0 - e, y0 - e, x0 + N * c - 1 + e, y0 + N * c - 1 + e], fill=EDGE)
    for f in range(N):
        for r in range(N):
            x, y = x0 + f * c, y0 + (N - 1 - r) * c
            if (f, r) in holes:
                tile = Image.new("RGB", (c, c), HOLE)
                td = ImageDraw.Draw(tile)
                for k in range(0, 2 * c, 14):
                    td.line([(k, 0), (k - c, c)], fill=HOLE_HATCH, width=3)
                img.paste(tile, (x, y))
                continue
            d.rectangle([x, y, x + c - 1, y + c - 1], fill=tuple(st["light"] if (f + r) % 2 else st["dark"]))
    for lb in scene["labels"]:
        d.text((lb["x"], lb["y"]), lb["text"], fill=ink, font=font(lb["size"]), anchor="mm")
    return img


def draw_timer(d, scene, i: int):
    tb = scene["timer"]["box"]
    d.rounded_rectangle(tb, radius=10, fill=TIMER_BG, outline=TIMER_EDGE, width=2)
    d.text(((tb[0] + tb[2]) / 2, (tb[1] + tb[3]) / 2), sim.clock_text(sim.timer_tenths(i)),
           fill=TIMER_FG, font=font(scene["timer"]["size"]), anchor="mm")


def frame_at(scene, i: int) -> Image.Image:
    st = scene["style"]
    img = _static(json.dumps(scene, sort_keys=True)).copy()
    size = int(round(st["piece_scale"] * st["cell"]))
    moving = moving_now(scene, i)
    tr = track(scene, i)
    order = [k for k in range(len(tr)) if k not in moving] + sorted(moving)
    for k in order:
        x, y, vis = tr[k]
        if not vis:
            continue
        p = scene["pieces"][k]
        px, py = int(round(x - size / 2)), int(round(y - size / 2))
        if st["shadow"]:
            sh = shadow(size)
            img.paste(sh, (px, py + 2), sh)
        im = icon(p["type"], tuple(p["rgb"]), size, st["piece_style"])
        img.paste(im, (px, py), im)
    draw_timer(ImageDraw.Draw(img), scene, i)
    return img


def frame(scene, t: float) -> Image.Image:
    return frame_at(scene, int(round(t * sim.FPS)))
