"""Frame-level check of rendered clips against the scene data.

    python frame_check.py --dataset-dir <dir> [--report frame_check.txt]

For every clip it decodes the MP4 (OpenCV) and:
  * at the middle of every stretch a card or joker lies uncovered on top
    of its pile, identifies it by template-matching the pile against all
    52 card faces and the joker; the sequence read per pile must be the
    dealing order (so the card count and order come from the pixels);
  * two frames after every landing reads the whole wheel: each of the 26
    slots of the letter ring is template-matched against the 26 letters,
    each slot of the symbol ring against the 13 ranks (+ red/black from
    the ink colour). Every read label must equal the scene's label at that
    slot (label text = scene data), and the letter in the slot of the
    landed card's symbol must be the card's letter in the stored code
    (a decode read straight off the pixels);
  * reads the clock (template match against all readings 0.0-9.9 s) on
    every landing frame, every frame showing a clock moment a question
    names, and the landing frame of every clock answer;
  * at every frame of a `top_at` / `under_at` moment re-reads the pile or
    the wheel and compares it with the stored answer;
  * repeats the label layout checks of sim.check (sizes, bands, overlaps,
    clear of timer and cards).
Prints a summary and exits 1 on any failure.
"""
from __future__ import annotations
import argparse
import functools
import json
import math
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))
import render  # noqa: E402
import sim  # noqa: E402
from common.video5 import font  # noqa: E402

GX, GY = 21, 15          # half size of a glyph crop


def frames_of(path: Path) -> list:
    import cv2
    cap = cv2.VideoCapture(str(path))
    out = []
    while True:
        ok, fr = cap.read()
        if not ok:
            break
        out.append(fr[:, :, ::-1].astype(np.int16))
    cap.release()
    return out


# ---------------------------------------------------------------- clock

_TIMER = {}


def _timer_templates(scene):
    key = (tuple(scene["timer"]["box"]), scene["timer"]["size"])
    if key not in _TIMER:
        tb = scene["timer"]["box"]
        out = {}
        for k in range(100):
            img = Image.new("RGB", (sim.W, sim.H), (0, 0, 0))
            d = ImageDraw.Draw(img)
            d.rounded_rectangle(tb, radius=10, fill=render.TIMER_BG, outline=render.TIMER_EDGE, width=2)
            d.text(((tb[0] + tb[2]) / 2, (tb[1] + tb[3]) / 2), sim.clock_text(k), fill=render.TIMER_FG,
                   font=font(scene["timer"]["size"]), anchor="mm")
            out[k] = np.asarray(img, dtype=np.int16)[tb[1] + 4:tb[3] - 4, tb[0] + 4:tb[2] - 4]
        _TIMER[key] = out
    return _TIMER[key]


def read_timer(scene, fr) -> int:
    tb = scene["timer"]["box"]
    crop = fr[tb[1] + 4:tb[3] - 4, tb[0] + 4:tb[2] - 4]
    tm = _timer_templates(scene)
    return min(tm, key=lambda k: np.abs(crop - tm[k]).mean())


# ---------------------------------------------------------------- cards

ALL_CARDS = [{"kind": "card", "color": c, "rank": r, "suit": s, "turn": 0}
             for c, suits in sim.SUITS.items() for s in suits for r in sim.RANKS] + [{"kind": "joker", "turn": 0}]


def _cid(c):
    return "joker" if c["kind"] == "joker" else f"{c['rank']}{c['suit'][0]}"


@functools.lru_cache(maxsize=8)
def _card_templates(key):
    scene = json.loads(key)
    g = scene["geometry"]
    out = {}
    for c in ALL_CARDS:
        img = Image.new("RGB", (g["cw"] + 40, g["ch"] + 40), tuple(scene["style"]["bg"]))
        render.draw_card(ImageDraw.Draw(img), scene, c, 20 + g["cw"] / 2, 20 + g["ch"] / 2)
        a = np.asarray(img, dtype=np.int16)
        out[_cid(c)] = a[32:20 + g["ch"] - 12, 32:20 + g["cw"] - 12]
    return out


def read_card(scene, fr, pile) -> str:
    g = scene["geometry"]
    px, py = pile
    x0, y0 = int(round(px - g["cw"] / 2)), int(round(py - g["ch"] / 2))
    crop = fr[y0 + 12:y0 + g["ch"] - 12, x0 + 12:x0 + g["cw"] - 12]
    key = json.dumps({k: scene[k] for k in ("geometry", "style")}, sort_keys=True)
    tm = _card_templates(key)
    h, w = min(crop.shape[0], next(iter(tm.values())).shape[0]), min(crop.shape[1], next(iter(tm.values())).shape[1])
    return min(tm, key=lambda k: np.abs(crop[:h, :w] - tm[k][:h, :w]).mean())


# ---------------------------------------------------------------- wheel

def _mask(a):
    """Ink pixels: clearly darker than the light ring fills."""
    return a.sum(axis=2) < 330


def _glyph(text, ink, x, y, x0, y0, fill, size):
    img = Image.new("RGB", (2 * GX, 2 * GY), fill)
    ImageDraw.Draw(img).text((x - x0, y - y0), text, fill=ink, font=font(size), anchor="mm")
    return np.asarray(img, dtype=np.int16)


def read_wheel(scene, fr) -> dict:
    """Glyphs read in each of the 26 slots of both rings (at rest)."""
    g = scene["geometry"]
    out = {"outer": [], "inner": []}
    for ring, r, cands in (("outer", (g["r2"] + g["r3"]) / 2, [(ch, "dark") for ch in sim.ABC]),
                           ("inner", (g["r1"] + g["r2"]) / 2, [(rk, "black") for rk in sim.RANKS])):
        for s in range(26):
            a = math.radians(-90 + s * sim.STEP_DEG)
            x, y = g["cx"] + r * math.cos(a), g["cy"] + r * math.sin(a)
            x0, y0 = int(round(x)) - GX, int(round(y)) - GY
            crop = fr[y0:y0 + 2 * GY, x0:x0 + 2 * GX]
            got = _mask(crop)
            best, bd = None, 1e9
            for text, ink in cands:
                ref = _mask(_glyph(text, render._ink({"ink": ink}), x, y, x0, y0, (250, 250, 250), sim.LABEL_FONT))
                dd = (ref ^ got).mean()
                if dd < bd:
                    best, bd = text, dd
            if ring == "inner":
                px = crop[got]
                red = len(px) and float((px[:, 0] - px[:, 1]).mean()) > 60
                best = f"{'red' if red else 'black'}:{best}"
            out[ring].append(best)
    return out


def expected_wheel(scene, i) -> dict:
    rot = render.ring_rotation(scene, i)
    oi, oo = round(rot["inner"] / sim.STEP_DEG), round(rot["outer"] / sim.STEP_DEG)
    lab = {(lb["ring"], lb["index"]): lb for lb in scene["labels"]}
    return {"outer": [lab[("outer", (s - oo) % 26)]["text"] for s in range(26)],
            "inner": [f"{lab[('inner', (s - oi) % 26)]['ink']}:{lab[('inner', (s - oi) % 26)]['text']}"
                      for s in range(26)]}


# ---------------------------------------------------------------- clip

def check_clip(scene, qs, frames) -> list:
    errs = []
    nf, deal, g = scene["n_frames"], scene["deal"], scene["geometry"]
    if len(frames) != nf:
        return [f"decoded {len(frames)} frames, expected {nf}"]
    # top cards: the sequence read per pile is the dealing order
    seq = {p: [] for p in range(len(g["piles"]))}
    for k in range(len(deal)):
        a, b = sim.top_span(scene, k)
        p = deal[k]["pile"]
        seq[p].append(read_card(scene, frames[(a + b) // 2], g["piles"][p]))
    for p in seq:
        want = [_cid(d) for d in deal if d["pile"] == p]
        if seq[p] != want:
            errs.append(f"pile {p}: read {seq[p]} != dealt {want}")
    # wheel at every landing: labels equal scene data, decode off the pixels
    C = sim.cards(scene)
    letters = dict(zip(C, scene["code"]))
    for k, d in enumerate(deal):
        f = d["land"] + 2
        got, want = read_wheel(scene, frames[f]), expected_wheel(scene, f)
        if got != want:
            bad = sum(x != y for r in ("outer", "inner") for x, y in zip(got[r], want[r]))
            errs.append(f"frame {f}: {bad} wheel labels misread")
        if read_timer(scene, frames[d["land"]]) != sim.timer_tenths(d["land"]):
            errs.append(f"frame {d['land']}: clock misread")
        if d["kind"] == "card":
            slot = got["inner"].index(sim.symbol(d)) if sim.symbol(d) in got["inner"] else None
            if slot is None or got["outer"][slot] != letters[k]:
                errs.append(f"card {k}: letter read off the wheel != {letters[k]}")
    # moments named in questions, clock answers
    for q in qs:
        key, p = q["derivation_key"], q["params"]
        if key in ("top_at", "under_at"):
            lo, hi = sim.frames_showing(p["t"])
            for i in range(lo, hi + 1):
                if read_timer(scene, frames[i]) != p["t"]:
                    errs.append(f"{q['qid']}: frame {i} clock != {p['t']}")
                if key == "top_at":
                    k = next(k for k in sim.cards(scene) if sim.card_name(deal[k]) == q["groundtruth"])
                    if read_card(scene, frames[i], g["piles"][p["pile"]]) != _cid(deal[k]):
                        errs.append(f"{q['qid']}: frame {i} pile shows another card")
                else:
                    w = read_wheel(scene, frames[i])
                    s = w["inner"].index(p["sym"]) if p["sym"] in w["inner"] else None
                    if s is None or w["outer"][s] != q["groundtruth"]:
                        errs.append(f"{q['qid']}: frame {i} wheel reads another letter")
        if key == "land_time":
            f = deal[p["x"]]["land"]
            if sim.clock_text(read_timer(scene, frames[f])) != q["groundtruth"]:
                errs.append(f"{q['qid']}: clock at the landing frame != {q['groundtruth']}")
    errs += sim.label_errors(scene) + sim.geometry_errors(g)
    return errs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset-dir", required=True)
    ap.add_argument("--report", default="frame_check.txt")
    args = ap.parse_args()
    root = Path(args.dataset_dir)
    lines, total, n = [], 0, 0
    for line in open(root / "annotations.jsonl", encoding="utf8"):
        s = json.loads(line)
        n += 1
        errs = check_clip(s["params"], s["questions"], frames_of(root / s["media_path"]))
        total += len(errs)
        lines += [f"{s['sample_id']}: {e}" for e in errs]
        print(f"{s['sample_id']}: {len(errs)} failures", flush=True)
    lines.append(f"RESULT: {n} clips, {total} frame-check failures")
    (root / args.report).write_text("\n".join(lines) + "\n")
    print(lines[-1])
    for e in lines[:-1][:20]:
        print("  -", e)
    sys.exit(1 if total else 0)


if __name__ == "__main__":
    main()
