"""PIL renderer for robot_swimming_trials: the full bid table is visible
from the start. A gold pointer then steps through bids from highest to
lowest; a winning bid turns green and its race header shows the winner, a
crossed-out bid flashes red with its reason and stays struck through."""
from __future__ import annotations
import sys
from pathlib import Path

from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from common.video5 import font, hex_rgb  # noqa: E402

BG = (245, 244, 239)
INK = (44, 44, 50)
GREEN, RED, GOLD = (52, 150, 80), (200, 60, 50), (214, 164, 30)
INTRO_S, STEP_S, OUTRO_S = 1.4, 1.0, 1.8


def duration(scene) -> float:
    return INTRO_S + len(scene["log"]) * STEP_S + OUTRO_S


def frame(scene, t: float) -> Image.Image:
    img = Image.new("RGB", (1280, 720), BG)
    d = ImageDraw.Draw(img)
    n_r, robots, log = scene["n_races"], scene["robots"], scene["log"]
    col_w = 230 if n_r == 3 else 300
    x_name, x0 = 150, 360
    row_h = min(56, 470 / len(robots))
    y0 = 190
    d.text((60, 30), "Robot Swimming Trials", fill=INK, font=font(30))
    d.text((60, 72), "bids are resolved from the highest down; each robot can win only one race",
           fill=(110, 110, 118), font=font(20))

    k = int((t - INTRO_S) // STEP_S) if t >= INTRO_S else -1
    shown = log[:max(0, min(k + 1, len(log)))]
    u = (t - INTRO_S) - k * STEP_S if 0 <= k < len(log) else 1.0
    cell_state = {}
    for i, e in enumerate(shown):
        live = i == k and k < len(log)
        if live and u < 0.35:
            cell_state[(e["robot"], e["race"])] = ("pointer", None)
        else:
            cell_state[(e["robot"], e["race"])] = ("win" if e["outcome"] == "wins" else "struck",
                                                   e["outcome"] if live else None)
    winners = {e["race"]: e["robot"] for e in shown
               if e["outcome"] == "wins" and cell_state[(e["robot"], e["race"])][0] == "win"}

    for j in range(n_r):
        cx = x0 + j * col_w
        d.rounded_rectangle([cx, y0 - 70, cx + col_w - 20, y0 - 14], radius=8, fill=(230, 232, 238))
        d.text((cx + 14, y0 - 64), f"Race {j + 1}", fill=INK, font=font(22))
        if j + 1 in winners:
            d.text((cx + 14, y0 - 38), f"winner: {winners[j + 1]}", fill=GREEN, font=font(18))

    qualified = set(winners.values())
    for i, r in enumerate(robots):
        y = y0 + i * row_h
        c = hex_rgb(r["color"])
        d.ellipse([x_name - 40, y + 8, x_name - 14, y + 34], fill=c, outline=INK, width=2)
        d.text((x_name, y + 8), r["name"], fill=GREEN if r["name"] in qualified else INK, font=font(22))
        for j, b in enumerate(scene["bids"][r["name"]]):
            cx = x0 + j * col_w
            box = [cx, y + 4, cx + col_w - 20, y + row_h - 6]
            state, why = cell_state.get((r["name"], j + 1), (None, None))
            fill = {"win": (206, 236, 212), "struck": (238, 232, 230), "pointer": (255, 240, 196)}.get(state, (255, 255, 255))
            d.rounded_rectangle(box, radius=6, fill=fill, outline=GOLD if state == "pointer" else (205, 205, 210),
                                width=4 if state == "pointer" else 1)
            bar = (col_w - 110) * b / 100
            d.rectangle([cx + 70, y + 16, cx + 70 + bar, y + row_h - 18], fill=c)
            d.text((cx + 10, y + 10), f"{b / 100:.2f}", fill=(150, 150, 156) if state == "struck" else INK,
                   font=font(19))
            if state == "struck":
                d.line([cx + 6, (box[1] + box[3]) / 2, cx + col_w - 26, (box[1] + box[3]) / 2], fill=RED, width=3)
    if 0 <= k < len(log):
        e = log[k]
        msg = {"wins": f"{e['robot']} wins Race {e['race']} with {e['bid'] / 100:.2f}",
               "race already won": f"{e['robot']}'s {e['bid'] / 100:.2f}: Race {e['race']} already has a winner",
               "robot already qualified": f"{e['robot']}'s {e['bid'] / 100:.2f}: {e['robot']} already qualified"}[e["outcome"]]
        if u >= 0.35:
            d.text((60, 670), msg, fill=GREEN if e["outcome"] == "wins" else RED, font=font(22))
        else:
            d.text((60, 670), f"checking {e['robot']}'s bid for Race {e['race']}: {e['bid'] / 100:.2f}",
                   fill=INK, font=font(22))
    elif k >= len(log):
        d.text((60, 670), "every race has a winner", fill=GREEN, font=font(22))
    return img
