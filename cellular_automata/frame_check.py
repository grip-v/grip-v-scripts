"""Frame-level check of rendered clips against an independent replay.

    python frame_check.py --dataset-dir <dir>

For every clip it decodes the MP4 and, using answers.py's sparse replay
(not the stored history):
  * classifies every cell centre (nearest of empty / filled / fading colour)
    in a still frame of every generation and compares the grid and the
    filled-cell count with the replay;
  * checks the frame before each change still shows the old generation and
    the frame after the fade shows the new one;
  * for every moment a question names ("3.4 s"), checks that every frame
    whose timer reads it shows that one generation, and that the timer
    region of those frames matches the timer text (pixel template vs. the
    neighbouring readings);
  * for time answers, checks the timer at the event frame shows the answer;
  * checks each marker is drawn in its colour at its cell / column.
Prints a summary and exits 1 on any failure.
"""
from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))
import answers  # noqa: E402
import render  # noqa: E402
import sim  # noqa: E402
from common.video5 import font  # noqa: E402

TIMER_BOX = (24, 26, 230, 86)


def _frames(path: Path) -> list[np.ndarray]:
    import cv2
    cap = cv2.VideoCapture(str(path))
    out = []
    while True:
        ok, fr = cap.read()
        if not ok:
            break
        out.append(fr[:, :, ::-1].astype(int))
    cap.release()
    return out


def _timer_ref(scene, text: str) -> np.ndarray:
    st = scene["style"]
    img = Image.new("RGB", (1280, 720), tuple(st["bg"]))
    ImageDraw.Draw(img).text(render.TIMER_XY, text, fill=tuple(st["ink"]), font=font(render.TIMER_SIZE))
    x0, y0, x1, y1 = TIMER_BOX
    return np.asarray(img, dtype=int)[y0:y1, x0:x1]


def _timer_ok(scene, fr: np.ndarray, text: str) -> bool:
    x0, y0, x1, y1 = TIMER_BOX
    crop = fr[y0:y1, x0:x1]
    tn = int(round(float(text.split()[0]) * 10))
    d_true = np.abs(crop - _timer_ref(scene, text)).mean()
    d_other = min(np.abs(crop - _timer_ref(scene, f"{sim.fmt_t(t)} s")).mean() for t in (tn - 1, tn + 1) if t >= 0)
    return d_true < 4 and d_true < 0.7 * d_other


def _read_grid(scene, fr: np.ndarray, k: int) -> dict:
    st = scene["style"]
    pal = {0: st["dead"], 1: st["live"]}
    if st["dying"]:
        pal[2] = st["dying"]
    out = {}
    one_d = scene["scenario"] == "elementary"
    for r in range(scene["rows"]):
        for c in range(scene["cols"]):
            x, y = render.cell_centre(scene, k if one_d else r, c)
            px = fr[y - 2:y + 3, x - 2:x + 3].reshape(-1, 3).mean(axis=0)
            v = min(pal, key=lambda s: np.abs(px - np.array(pal[s])).sum())
            if v:
                out[(r, c)] = v
    return out


def _grid_at(scene, fr, k):
    """2-D: the whole grid; elementary: row k of the diagram."""
    return _read_grid(scene, fr, k)


def check_clip(s: dict, root: Path) -> list[str]:
    scene, sid = s["params"], s["sample_id"]
    errs = []
    frs = _frames(root / s["media_path"])
    n = sim.n_frames(scene)
    if len(frs) != n:
        return [f"{sid}: {len(frs)} frames decoded, expected {n}"]
    h = answers._replay(scene)
    G = scene["gens"]
    for k in range(G + 1):
        lo, hi = sim.stable_window(scene, k)
        g = _grid_at(scene, frs[(lo + hi) // 2], k)
        if g != h[k]:
            errs.append(f"{sid}: generation {k} pixels differ from the replay "
                        f"({len(set(g.items()) ^ set(h[k].items()))} cells)")
        if sum(v == 1 for v in g.values()) != sum(v == 1 for v in h[k].values()):
            errs.append(f"{sid}: generation {k} filled-cell count differs")
        if k >= 1:
            cf = sim.change_frame(scene, k)
            if scene["scenario"] != "elementary" and _grid_at(scene, frs[cf - 1], k) != h[k - 1]:
                errs.append(f"{sid}: frame before change {k} doesn't show generation {k - 1}")
            if _grid_at(scene, frs[cf + scene["timing"]["fade_f"] - 1], k) != h[k]:
                errs.append(f"{sid}: frame after fade {k} doesn't show generation {k}")
    toks = set()
    for q in s["questions"]:
        p = q["params"]
        toks.update(t for t in [p.get("t"), p.get("ta"), p.get("tb")] + list(p.get("ts", [])) if t)
        if q["derivation_key"] in ("first_change_time", "biggest_change_time", "empty_time") and \
                not q["answer_text"].startswith("never"):
            f = next(f for f in range(n) if render.timer_text(f) == q["answer_text"])
            fs = [f for f in range(n) if render.timer_text(f) == q["answer_text"]]
            ev = [sim.change_frame(scene, k) for k in range(1, G + 1)]
            if not any(e in fs for e in ev):
                errs.append(f"{sid}/{q['qid']}: answer time {q['answer_text']} is not shown at any change frame")
            if not _timer_ok(scene, frs[f], q["answer_text"]):
                errs.append(f"{sid}/{q['qid']}: timer pixels don't read {q['answer_text']}")
    for t in sorted(toks - {"start", "end"}):
        fs = [f for f in range(n) if render.timer_text(f) == f"{t} s"]
        k = answers._gen_of(scene, t)
        for f in fs:
            if k is None or _grid_at(scene, frs[f], k) != h[k]:
                errs.append(f"{sid}: frame {f} (timer {t} s) doesn't show one still generation")
            if not _timer_ok(scene, frs[f], f"{t} s"):
                errs.append(f"{sid}: timer pixels at frame {f} don't read {t} s")
    fr = frs[n // 2]
    for m in scene["markers"]:
        r, c = m["cell"]
        if scene["scenario"] == "elementary":
            x = scene["style"]["x0"] + c * scene["style"]["cell_px"] + scene["style"]["cell_px"] // 2
            s_ = max(20, min(28, scene["style"]["cell_px"]))
            y = scene["style"]["y0"] - 8 - s_ // 2
        else:
            x0, y0, x1, y1 = render.cell_box(scene, r, c)
            x, y = x0 + 1, (y0 + y1) // 2
        px = fr[y - 1:y + 2, x - 1:x + 2].reshape(-1, 3).mean(axis=0)
        if np.abs(px - np.array(m["rgb"])).sum() > 90:
            errs.append(f"{sid}: marker {m['colour']} {m['shape']} not visible at its cell")
    return errs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset-dir", required=True)
    args = ap.parse_args()
    root = Path(args.dataset_dir)
    errs, n = [], 0
    with open(root / "annotations.jsonl", encoding="utf8") as f:
        for line in f:
            s = json.loads(line)
            n += 1
            errs += check_clip(s, root)
    print(f"frame check: {n} clips, {len(errs)} failures")
    for e in errs[:30]:
        print("  -", e)
    sys.exit(1 if errs else 0)


if __name__ == "__main__":
    main()
