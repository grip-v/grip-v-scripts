"""PIL renderer for shell_game_tracking (1280x720, 24 fps).

`track(scene)` is what gets drawn: on every frame the bottom-centre of every
cup as an UNORDERED list (sorted by x, then y: the cups all look alike, so a
frame tells nothing about which cup is which), the tokens that can be seen
on that frame (colour, shape, position), the mats the cups stand on, and the
cup and token sizes. `frame_at` draws exactly that; answers.py and
frame_check.py read the track, never the moves.

Text on screen: the clock only (fixed panel, top-left).
"""
from __future__ import annotations
import functools
import json
import math
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFilter

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from common.video5 import font  # noqa: E402

import sim  # noqa: E402

SS = 2
TIMER_BG = (16, 18, 24)
TIMER_EDGE = (120, 126, 140)
TIMER_FG = (245, 246, 250)
INK = (24, 24, 30)


def duration(scene) -> float:
    return scene["n_frames"] / sim.FPS


def _mix(a, b, u):
    return tuple(int(round(x + (y - x) * u)) for x, y in zip(a, b))


def edge(st):
    return INK if st["light"] else (236, 238, 244)


@functools.lru_cache(maxsize=8)
def _track(key: str) -> dict:
    sc = json.loads(key)
    P = sim.poses(sc)
    cups = []
    toks = []
    for f in range(sc["n_frames"]):
        pts = sorted((round(float(x), 3), round(float(y), 3)) for x, y in P[f])
        cups.append([list(p) for p in pts])
        toks.append([{"colour": sc["tokens"][i]["colour"], "shape": sc["tokens"][i]["shape"], "x": x, "y": y}
                     for i, x, y in sim.token_visible(sc, f)])
    return {"cups": cups, "tokens": toks, "slots": [list(s) for s in sc["slots"]], "n_frames": sc["n_frames"],
            "cup": dict(sc["cup"]), "tok_size": sc["tok_size"], "labels": [dict(l) for l in sc["labels"]]}


def track(scene) -> dict:
    return _track(json.dumps(scene, sort_keys=True))


@functools.lru_cache(maxsize=32)
def cup_sprite(shape, pattern, rgb, w, h, outline, edge_rgb):
    W_, H_ = int(round(w * SS)), int(round(h * SS))
    img = Image.new("RGBA", (W_, H_), (0, 0, 0, 0))
    mask = Image.new("L", (W_, H_), 0)
    md = ImageDraw.Draw(mask)
    if shape == "cup":
        md.polygon([(0, H_ - 1), (W_ - 1, H_ - 1), (W_ * 0.82, 0), (W_ * 0.18, 0)], fill=255)
    elif shape == "dome":
        md.pieslice([0, 0, W_ - 1, 2 * H_ - 2], 180, 360, fill=255)
    elif shape == "box":
        md.rectangle([W_ * 0.06, H_ * 0.08, W_ * 0.94, H_ - 1], fill=255)
        md.rectangle([0, 0, W_ - 1, H_ * 0.2], fill=255)
    else:  # bell
        md.ellipse([W_ * 0.2, 0, W_ * 0.8, H_ * 0.6], fill=255)
        md.polygon([(0, H_ - 1), (W_ - 1, H_ - 1), (W_ * 0.78, H_ * 0.3), (W_ * 0.22, H_ * 0.3)], fill=255)
    fill = Image.new("RGBA", (W_, H_), tuple(rgb) + (255,))
    fd = ImageDraw.Draw(fill)
    mark = _mix(rgb, (255, 255, 255), 0.45) if sum(rgb) < 360 else _mix(rgb, INK, 0.45)
    if pattern == "bands":
        for k in (0.35, 0.62):
            fd.rectangle([0, H_ * k, W_, H_ * k + 6 * SS], fill=mark)
    elif pattern == "dots":
        r = 4 * SS
        for i in range(3):
            for j in range(2):
                cx, cy = W_ * (0.32 + 0.18 * i), H_ * (0.45 + 0.25 * j)
                fd.ellipse([cx - r, cy - r, cx + r, cy + r], fill=mark)
    elif pattern == "stripe":
        fd.rectangle([W_ * 0.44, 0, W_ * 0.56, H_], fill=mark)
    img.paste(fill, (0, 0), mask)
    # outline: the band between the mask and the mask eroded by the outline width
    pad = outline * SS + 2
    big = Image.new("L", (W_ + 2 * pad, H_ + 2 * pad), 0)
    big.paste(mask, (pad, pad))
    er = big.filter(ImageFilter.MinFilter(2 * outline * SS + 1)).crop((pad, pad, pad + W_, pad + H_))
    ring = Image.fromarray((np.asarray(mask, dtype=np.int16) - np.asarray(er, dtype=np.int16)).clip(0, 255)
                           .astype(np.uint8))
    img.paste(Image.new("RGBA", (W_, H_), tuple(edge_rgb) + (255,)), (0, 0), ring)
    return img


@functools.lru_cache(maxsize=32)
def token_sprite(shape, rgb, size, edge_rgb):
    d = int(round(size * SS))
    img = Image.new("RGBA", (d, d), (0, 0, 0, 0))
    dr = ImageDraw.Draw(img)
    c, e, w = tuple(rgb) + (255,), tuple(edge_rgb) + (255,), 3 * SS
    if shape == "ball":
        dr.ellipse([0, 0, d - 1, d - 1], fill=c, outline=e, width=w)
    elif shape == "cube":
        dr.rectangle([d * 0.06, d * 0.06, d * 0.94, d - 1], fill=c, outline=e, width=w)
    elif shape == "diamond":
        dr.polygon([(d / 2, 0), (d - 1, d / 2), (d / 2, d - 1), (0, d / 2)], fill=c, outline=e, width=w)
    elif shape == "triangle":
        dr.polygon([(d / 2, 0), (d - 1, d - 1), (0, d - 1)], fill=c, outline=e, width=w)
    else:  # star
        pts = []
        for k in range(10):
            r = d / 2 if k % 2 == 0 else d * 0.22
            a = -math.pi / 2 + k * math.pi / 5
            pts.append((d / 2 + r * math.cos(a), d / 2 + r * math.sin(a)))
        dr.polygon(pts, fill=c, outline=e, width=w)
    return img


@functools.lru_cache(maxsize=8)
def _static(key: str) -> Image.Image:
    sc = json.loads(key)
    st = sc["style"]
    top, bot = (tuple(c) for c in st["bg"])
    img = Image.new("RGB", (sim.W, sim.H), top)
    d = ImageDraw.Draw(img)
    if st["bg_style"] == "gradient":
        for y in range(sim.H):
            d.line([(0, y), (sim.W, y)], fill=_mix(top, bot, y / (sim.H - 1)))
    elif st["bg_style"] == "vignette":
        for k in range(24, 0, -1):
            u = k / 24
            d.ellipse([sim.W / 2 - 900 * u, sim.H / 2 - 620 * u, sim.W / 2 + 900 * u, sim.H / 2 + 620 * u],
                      fill=_mix(top, bot, u))
    img = img.resize((sim.W * SS, sim.H * SS), Image.BILINEAR)
    d = ImageDraw.Draw(img)
    tab = tuple(st["table"])
    cw = sc["cup"]["w"]
    xs = [p[0] for p in sc["slots"]]
    ys = [p[1] for p in sc["slots"]]
    rim = _mix(tab, INK, 0.35)
    d.rounded_rectangle([(min(xs) - cw * 0.9) * SS, (min(ys) - 18) * SS, (max(xs) + cw * 0.9) * SS,
                         (max(ys) + 26) * SS], radius=14 * SS, fill=tab, outline=rim, width=3 * SS)
    mat = _mix(tab, INK, 0.18)
    for x, y in sc["slots"]:
        if st["mat"] == "ellipse":
            d.ellipse([(x - cw * 0.62) * SS, (y - 12) * SS, (x + cw * 0.62) * SS, (y + 12) * SS], fill=mat)
        elif st["mat"] == "square":
            d.rectangle([(x - cw * 0.6) * SS, (y - 10) * SS, (x + cw * 0.6) * SS, (y + 10) * SS], fill=mat)
        else:
            d.ellipse([(x - cw * 0.62) * SS, (y - 12) * SS, (x + cw * 0.62) * SS, (y + 12) * SS], outline=mat,
                      width=4 * SS)
    tb = sim.TIMER_BOX
    d.rounded_rectangle([v * SS for v in tb], radius=10 * SS, fill=TIMER_BG, outline=TIMER_EDGE, width=2 * SS)
    return img


def frame_at(scene, f: int) -> Image.Image:
    st = scene["style"]
    key = json.dumps(scene, sort_keys=True)
    tr = track(scene)
    img = _static(key).copy()
    ts = scene["tok_size"]
    for t in tr["tokens"][f]:
        sp = token_sprite(t["shape"], sim.TOKENS[t["colour"]], ts, edge(st))
        img.paste(sp, (int(round(t["x"] * SS - sp.size[0] / 2)), int(round((t["y"] - 4) * SS - sp.size[1]))), sp)
    cw, ch = scene["cup"]["w"], scene["cup"]["h"]
    sp = cup_sprite(st["cup_shape"], st["cup_pattern"], sim.CUP_COLOURS[st["cup_colour"]], cw, ch, st["outline"],
                    edge(st))
    for x, y in sorted(tr["cups"][f], key=lambda p: p[1]):          # nearer (lower) cups drawn last
        img.paste(sp, (int(round((x - cw / 2) * SS)), int(round((y - ch) * SS))), sp)
    img = img.reduce(SS)
    d = ImageDraw.Draw(img)
    tb = sim.TIMER_BOX
    d.text(((tb[0] + tb[2]) / 2, (tb[1] + tb[3]) / 2), sim.clock_text(sim.tenths(f)), fill=TIMER_FG,
           font=font(sim.TIMER_FONT), anchor="mm")
    return img


def frame(scene, t: float) -> Image.Image:
    return frame_at(scene, min(scene["n_frames"] - 1, int(round(t * sim.FPS))))
