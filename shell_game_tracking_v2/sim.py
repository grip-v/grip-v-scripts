"""Shell game v2: 3-8 identical cups in a row (one cup colour per clip), 1-4 distinct balls (colour + kind:
small ball, big ball, ring, ball with a triangle / star inside) shown under raised cups at the start, then
hidden for good while the cups swap places, one swap at a time.

A swap exchanges the cups in slots i and j; slots are numbered 0..N-1 from the left (shown to viewers as
'slot 1'..'slot N'). Tiers: easy 3-4 cups / 4-6 swaps, medium 5-6 / 7-10, hard 7-8 / 11-16."""
from __future__ import annotations
import random

TIERS = {
    "easy":   {"cups": (3, 4), "swaps": (4, 6),   "dur": (0.75, 0.95), "gap": (0.30, 0.50)},
    "medium": {"cups": (5, 6), "swaps": (7, 10),  "dur": (0.55, 0.75), "gap": (0.25, 0.40)},
    "hard":   {"cups": (7, 8), "swaps": (11, 16), "dur": (0.42, 0.55), "gap": (0.15, 0.30)},
}
CUP_COLORS = {"slate": "#8FA0B8", "teal": "#3FA796", "terracotta": "#C8694B",
              "sand": "#C9B27C", "mauve": "#B08AA8", "gray": "#9AA0A6"}
BALL_COLORS = {"red": "#E03C3C", "green": "#3DBE5A", "yellow": "#F2C62E", "purple": "#8A4FD6",
               "orange": "#F28C28", "blue": "#3C7BE0", "pink": "#F06FB0"}
KINDS = {"small": "small ball", "big": "big ball", "ring": "ring",
         "triangle": "ball with a triangle", "star": "ball with a star"}
FIRST_SWAP_DELAY = (0.45, 0.7)   # pause after the cups come down
HOLD_END = 1.0                   # last swap ends this long before the clip ends (>= 0.8 required)


def track(slot: int, swaps: list[dict]) -> list[int]:
    """Slot of a cup before any swap and after each swap: len(swaps) + 1 entries."""
    out = [slot]
    for s in swaps:
        cur = out[-1]
        out.append(s["j"] if cur == s["i"] else s["i"] if cur == s["j"] else cur)
    return out


def build(rng: random.Random) -> dict:
    tier = list(TIERS)[rng.randrange(3)]
    cfg = TIERS[tier]
    n = rng.randint(*cfg["cups"])
    nb = rng.randint(1, min(4, n - 1))
    cup_color = rng.choice(list(CUP_COLORS))
    colours = rng.sample(list(BALL_COLORS), nb)
    slots = rng.sample(range(n), nb)
    balls = []
    for c, s in zip(colours, slots):
        kind = rng.choice(list(KINDS))
        balls.append({"colour": c, "hex": BALL_COLORS[c], "kind": kind, "slot": s,
                      "name": f"the {c} {KINDS[kind]}"})
    reveal_end = round(1.5 + 0.25 * (nb - 1) + rng.uniform(0.0, 0.3), 2)
    t = reveal_end + rng.uniform(*FIRST_SWAP_DELAY)
    cur = [b["slot"] for b in balls]
    swaps = []
    for _ in range(rng.randint(*cfg["swaps"])):
        dur = round(rng.uniform(*cfg["dur"]), 2)
        start = round(t, 2)
        end = round(start + dur, 2)
        if rng.random() < 0.6:                      # keep the ball cups involved so they actually move
            i = rng.choice(cur)
            j = rng.choice([x for x in range(n) if x != i])
        else:
            i, j = rng.sample(range(n), 2)
        swaps.append({"start": start, "end": end, "i": i, "j": j})
        cur = [j if x == i else i if x == j else x for x in cur]
        t = end + rng.uniform(*cfg["gap"])
    return {"tier": tier, "n_cups": n, "cup_color": cup_color, "cup_hex": CUP_COLORS[cup_color],
            "balls": balls, "reveal_end": reveal_end, "swaps": swaps,
            "duration": round(swaps[-1]["end"] + HOLD_END, 2)}


def check(scene: dict) -> list[str]:
    errs = []
    n, balls, swaps = scene["n_cups"], scene["balls"], scene["swaps"]
    cfg = TIERS[scene["tier"]]
    if not cfg["cups"][0] <= n <= cfg["cups"][1]:
        errs.append("cup count outside tier range")
    if not cfg["swaps"][0] <= len(swaps) <= cfg["swaps"][1]:
        errs.append("swap count outside tier range")
    if not 1 <= len(balls) <= min(4, n - 1):
        errs.append("bad ball count")
    if len({b["slot"] for b in balls}) != len(balls) or len({b["colour"] for b in balls}) != len(balls):
        errs.append("balls share a cup or a colour")
    if any(a["end"] > b["start"] for a, b in zip(swaps, swaps[1:])) or swaps[0]["start"] < scene["reveal_end"]:
        errs.append("swaps overlap or start before the reveal ends")
    if scene["duration"] - swaps[-1]["end"] < 0.8:
        errs.append("last swap ends too close to the clip end")
    return errs
