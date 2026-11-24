"""PIL renderer for multi_agent_chase (v3): patterned balls in a walled box.
Catchers look like any other ball. A caught ball simply vanishes and its
catcher grows smoothly; patterns scale with the ball. Scenery: grey pillars
(pillars), a hatched shaded circle (safe_zone). The only text is the clock
(seconds to 0.1 s) in a fixed box at the top right, above the arena.
Frame f shows sim.track(scene)[f] and the clock reads floor(f * 10 / 24)."""
from __future__ import annotations
import functools
import json
import math
import sys
from pathlib import Path

from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from common.video5 import font  # noqa: E402

import sim  # noqa: E402

SIZE = sim.SIZE
CLOCK_BOX = sim.CLOCK_BOX
CLOCK_PX = 34
PILLAR = (120, 124, 132)
ZONE = (120, 190, 120)
SS = 3


def duration(scene) -> float:
    return sim.duration(scene)


def n_frames(scene) -> int:
    return scene["frames"]


def clock_tenths(f: int) -> int:
    return f * 10 // sim.FPS


def clock_text(f: int) -> str:
    return f"{clock_tenths(f) / 10:.1f} s"


def _mix(c, w, f):
    return tuple(int(a + (b - a) * f) for a, b in zip(c, w))


@functools.lru_cache(maxsize=1024)
def sprite(color: str, pat: str, r: int) -> Image.Image:
    size = (2 * r + 6) * SS
    c = size / 2
    R = r * SS
    col = sim.COLORS[color] + (255,)
    pale = _mix(sim.COLORS[color], (255, 255, 255), 0.55) + (255,)
    tex = Image.new("RGBA", (size, size), pale)
    d = ImageDraw.Draw(tex)
    q = 2 * R / 5                                   # pattern period scales with the ball
    if pat in ("solid", "starred"):
        d.rectangle([0, 0, size, size], fill=col)
    elif pat == "striped":
        y = c - R
        while y < c + R:
            d.rectangle([0, y, size, y + q / 2], fill=col)
            y += q
    elif pat == "dotted":
        s = 2 * R / 4
        for i in range(-1, 6):
            for j in range(-1, 6):
                x, y = c - R + (i + 0.5 * (j % 2)) * s, c - R + j * s
                d.ellipse([x - s * 0.22, y - s * 0.22, x + s * 0.22, y + s * 0.22], fill=col)
    elif pat == "checkered":
        s = 2 * R / 4
        for i in range(4):
            for j in range(4):
                if (i + j) % 2 == 0:
                    d.rectangle([c - R + i * s, c - R + j * s, c - R + (i + 1) * s, c - R + (j + 1) * s], fill=col)
    elif pat == "ringed":
        for k, rr in enumerate([R, R * 0.66, R * 0.33]):
            d.ellipse([c - rr, c - rr, c + rr, c + rr], fill=col if k % 2 == 0 else pale)
    elif pat == "two-tone":
        d.rectangle([0, 0, size, c], fill=col)
    elif pat == "quartered":
        d.rectangle([0, 0, c, c], fill=col)
        d.rectangle([c, c, size, size], fill=col)
    elif pat == "grid":
        s = 2 * R / 4
        w = max(SS * 2, R * 0.12)
        for k in range(1, 4):
            d.rectangle([c - R + k * s - w / 2, 0, c - R + k * s + w / 2, size], fill=col)
            d.rectangle([0, c - R + k * s - w / 2, size, c - R + k * s + w / 2], fill=col)
    if pat == "starred":
        pts = [(c + (R * 0.62 if i % 2 == 0 else R * 0.26) * math.cos(-math.pi / 2 + i * math.pi / 5),
                c + (R * 0.62 if i % 2 == 0 else R * 0.26) * math.sin(-math.pi / 2 + i * math.pi / 5))
               for i in range(10)]
        d.polygon(pts, fill=(255, 255, 255, 255))
    mask = Image.new("L", (size, size), 0)
    ImageDraw.Draw(mask).ellipse([c - R, c - R, c + R, c + R], fill=255)
    out = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    out.paste(tex, (0, 0), mask)
    ImageDraw.Draw(out).ellipse([c - R, c - R, c + R, c + R], outline=_mix(sim.COLORS[color], (0, 0, 0), 0.45) + (255,),
                                width=2 * SS)
    return out.resize((size // SS, size // SS), Image.LANCZOS)


def _static_key(scene):
    return json.dumps({k: scene.get(k) for k in ("arena", "style", "pillars", "zone")}, sort_keys=True)


@functools.lru_cache(maxsize=8)
def _static(key: str) -> Image.Image:
    sc = json.loads(key)
    bg, floor, wall = sim.STYLES[sc["style"]["palette"]]
    img = Image.new("RGB", SIZE, bg)
    d = ImageDraw.Draw(img)
    x0, y0, x1, y1 = sc["arena"]
    ws = sc["style"]["wall"]
    t = 8 if ws == "thick" else 5
    d.rectangle([x0 - t, y0 - t, x1 + t, y1 + t], fill=wall)
    if ws == "double":
        d.rectangle([x0 - 2, y0 - 2, x1 + 2, y1 + 2], fill=_mix(wall, (255, 255, 255), 0.6))
    d.rectangle([x0, y0, x1, y1], fill=floor)
    faint = _mix(floor, (0, 0, 0), 0.07)
    if sc["style"]["floor"] == "grid":
        for x in range(x0 + 60, x1, 60):
            d.line([x, y0, x, y1], fill=faint, width=1)
        for y in range(y0 + 60, y1, 60):
            d.line([x0, y, x1, y], fill=faint, width=1)
    elif sc["style"]["floor"] == "dots":
        for x in range(x0 + 40, x1, 40):
            for y in range(y0 + 40, y1, 40):
                d.ellipse([x - 1.5, y - 1.5, x + 1.5, y + 1.5], fill=faint)
    z = sc.get("zone")
    if z:
        zx, zy, zr = z["x"], z["y"], z["r"]
        lay = Image.new("RGBA", SIZE, (0, 0, 0, 0))
        ld = ImageDraw.Draw(lay)
        ld.ellipse([zx - zr, zy - zr, zx + zr, zy + zr], fill=ZONE + (80,))
        for k in range(-zr * 2, zr * 2, 18):
            ld.line([zx + k - zr, zy + zr, zx + k + zr, zy - zr], fill=ZONE + (110,), width=3)
        m = Image.new("L", SIZE, 0)
        ImageDraw.Draw(m).ellipse([zx - zr, zy - zr, zx + zr, zy + zr], fill=255)
        img.paste(lay, (0, 0), Image.composite(lay, Image.new("RGBA", SIZE, (0, 0, 0, 0)), m))
        d = ImageDraw.Draw(img)
        d.ellipse([zx - zr, zy - zr, zx + zr, zy + zr], outline=_mix(ZONE, (0, 0, 0), 0.35), width=4)
    for p in sc.get("pillars") or []:
        px, py, pr = p["x"], p["y"], p["r"]
        d.ellipse([px - pr, py - pr, px + pr, py + pr], fill=PILLAR, outline=(70, 72, 80), width=4)
        d.ellipse([px - pr * 0.55, py - pr * 0.55, px + pr * 0.55, py + pr * 0.55], outline=(150, 154, 162), width=3)
    return img


def _clock(d, f):
    d.rounded_rectangle(CLOCK_BOX, radius=10, fill=(24, 24, 30))
    x0, y0, x1, y1 = CLOCK_BOX
    d.text(((x0 + x1) / 2, (y0 + y1) / 2), clock_text(f), fill=(255, 255, 255), font=font(CLOCK_PX), anchor="mm")


def state(scene, f: int):
    return sim.state(scene, f)


def frame_at(scene, f: int) -> Image.Image:
    img = _static(_static_key(scene)).copy()
    d = ImageDraw.Draw(img)
    _clock(d, f)
    row = state(scene, f)
    for i in sorted(range(len(row)), key=lambda i: (row[i][2], i)):
        x, y, r, alive = row[i]
        if not alive:
            continue
        b = scene["balls"][i]
        sp = sprite(b["color"], b["pattern"], int(round(r)))
        img.paste(sp, (round(x - sp.width / 2), round(y - sp.height / 2)), sp)
    return img


def frame(scene, t: float) -> Image.Image:
    return frame_at(scene, min(n_frames(scene) - 1, int(round(t * sim.FPS))))
