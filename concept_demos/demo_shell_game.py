"""Shell game / token tracking -- closed-form permutation tracking through
a sequence of scheduled cup swaps. The real dataset would render all cups
identically (that's the point); here cups are colored distinctly so the
demo is visually self-verifying. A gold dot marks the token's true cup at
all times (in the real dataset this would be hidden after t=0 and the
question would ask the model to track it)."""
from __future__ import annotations
import math
from pathlib import Path

from PIL import Image, ImageDraw

from render_util import render_clip, save_key_frames, hex_to_rgb

CANVAS = (440, 300)
CUP_HOME_X = [110, 220, 330]
CUP_Y = 190
CUP_COLORS = ["#3F8FE4", "#4FBF6B", "#E4553F"]  # cup_id -> color (demo only)
SWAPS = [(1.0, 1.6, 0, 1), (2.0, 2.6, 1, 2), (3.0, 3.6, 0, 2)]  # start,end,slot_i,slot_j
TOKEN_CUP_ID = 0
DURATION_S = 4.2
FPS = 20


def perm_before(t):
    perm = [0, 1, 2]
    for (s, e, i, j) in SWAPS:
        if t >= e:
            perm[i], perm[j] = perm[j], perm[i]
    return perm


def active_swap(t):
    for sw in SWAPS:
        s, e, i, j = sw
        if s <= t <= e:
            return sw
    return None


def cup_positions(t):
    sw = active_swap(t)
    if sw is None:
        perm = perm_before(t)
        return {perm[k]: (CUP_HOME_X[k], CUP_Y) for k in range(3)}
    s, e, i, j = sw
    perm = perm_before(s)
    cup_i, cup_j = perm[i], perm[j]
    frac = (t - s) / (e - s)
    xi, xj = CUP_HOME_X[i], CUP_HOME_X[j]
    x_a = xi + (xj - xi) * frac
    x_b = xj + (xi - xj) * frac
    arc = -22 * math.sin(math.pi * frac)
    positions = {perm[k]: (CUP_HOME_X[k], CUP_Y) for k in range(3) if k not in (i, j)}
    positions[cup_i] = (x_a, CUP_Y + arc)
    positions[cup_j] = (x_b, CUP_Y - arc)
    return positions


def token_cup_slot(t):
    """Which SLOT (home x index) currently holds the token, at time t."""
    perm = perm_before(max(t, 0.0)) if active_swap(t) is None else perm_before(active_swap(t)[0])
    return perm.index(TOKEN_CUP_ID)


def render_frame(t):
    img = Image.new("RGB", CANVAS, hex_to_rgb("#141414"))
    draw = ImageDraw.Draw(img, "RGBA")
    positions = cup_positions(t)
    # token dot (drawn first, "under" the cup)
    tx, ty = positions[TOKEN_CUP_ID]
    draw.ellipse([tx - 10, ty + 15, tx + 10, ty + 35], fill=(230, 200, 50, 255))
    for cup_id, (x, y) in positions.items():
        color = hex_to_rgb(CUP_COLORS[cup_id])
        draw.polygon([(x - 38, y + 40), (x + 38, y + 40), (x + 24, y - 40), (x - 24, y - 40)],
                     fill=color + (255,), outline=(255, 255, 255, 255), width=2)
    draw.text((10, 10), f"t={t:.2f}s", fill=(255, 255, 255, 255))
    swap = active_swap(t)
    if swap:
        draw.text((10, 30), "swapping...", fill=(255, 220, 100, 255))
    return img


if __name__ == "__main__":
    out_dir = Path(__file__).resolve().parent / "out" / "shell_game"
    render_clip(render_frame, DURATION_S, FPS, out_dir / "shell_game.mp4")
    key_times = [0.0, 1.3, 1.8, 2.3, 2.9, 3.3, 4.0]
    paths = save_key_frames(render_frame, key_times, out_dir / "frames", "shell_game")
    for t in key_times:
        slot = token_cup_slot(t) if active_swap(t) is None else None
        print(f"t={t:.2f}  token_home_slot={'mid-swap' if slot is None else slot}")
    print("wrote", out_dir / "shell_game.mp4")
