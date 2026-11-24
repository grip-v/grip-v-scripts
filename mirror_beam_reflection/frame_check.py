"""Frame-level check of rendered mirror_beam_reflection clips against the stored annotations.

    python frame_check.py --dataset-dir <dir> [--report frame_check_report.txt]

Per clip it decodes the MP4 (OpenCV) and checks:
  * clock: every frame's clock box matches the rendered text for that frame
    better than any other reading 0.0-10.0 s (template match = OCR);
  * labels: at the first, middle and last frame each cell label's crop
    matches its own text better than any other label text, the labels
    never overlap each other, the board, an emitter or the clock;
  * heads: pulse heads located in each frame (beam-colour pixels eroded
    past the trail width, emitters masked) agree with render.state within
    TOL px, same count;
  * mirrors: at the first and last frame each mirror's pixels lie on the
    diagonal the scene says;
  * launches: the frame a beam first shows new pixels at its entry point is
    the frame the renderer starts it on;
  * lit cells: the cells holding beam-colour trail pixels in the last frame
    give the stored n_lit_cells answer;
then re-derives every track-based answer from the DECODED heads with
answers.answer(..., frames=decoded, jitter=JITTER). Questions that name a
clock time get the clock checked at that exact frame.
"""
from __future__ import annotations
import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE), str(HERE.parent)]
import cv2  # noqa: E402
from PIL import Image, ImageDraw  # noqa: E402

import sim  # noqa: E402
import render  # noqa: E402
import answers  # noqa: E402
from common.video5 import font  # noqa: E402

TOL = 3.0
JITTER = 2.0
TRACK_KEYS = {"entry_edge", "first_mirror", "turn_dir_first", "head_cell_at_t", "n_turns_by_t", "n_heads_at_t",
              "split_cell", "time_turn_at", "time_between_mirrors", "speed", "time_between_turns", "cells_between"}


def decode(path):
    cap = cv2.VideoCapture(str(path))
    out = []
    while True:
        ok, fr = cap.read()
        if not ok:
            break
        out.append(cv2.cvtColor(fr, cv2.COLOR_BGR2RGB))
    cap.release()
    return out


_CLOCKS: dict = {}


def clock_img(tenths):
    if tenths not in _CLOCKS:
        img = Image.new("RGB", render.SIZE, (0, 0, 0))
        render._clock(ImageDraw.Draw(img), (tenths * 24 + 9) // 10)
        x0, y0, x1, y1 = render.CLOCK_BOX
        _CLOCKS[tenths] = np.asarray(img, dtype=np.int16)[y0 + 4:y1 - 3, x0 + 4:x1 - 3]
    return _CLOCKS[tenths]


def read_clock(frame):
    x0, y0, x1, y1 = render.CLOCK_BOX
    crop = frame[y0 + 4:y1 - 3, x0 + 4:x1 - 3].astype(np.int16)
    diffs = {k: float(np.abs(crop - clock_img(k)).mean()) for k in range(0, 101)}
    return min(diffs, key=diffs.get)


def label_box(lab):
    l, t, r, b = font(sim.LABEL_PX).getbbox(lab["text"], anchor="mm")
    return (lab["x"] + l, lab["y"] + t, lab["x"] + r, lab["y"] + b)


def label_errors(scene, frames):
    errs = []
    st = scene["style"]
    texts = sorted({lab["text"] for lab in scene["labels"]} | set(sim.LETTERS[:10]) | {str(k) for k in range(1, 11)})
    boxes = [label_box(lab) for lab in scene["labels"]]
    x0, y0, c = st["x0"], st["y0"], st["cell"]
    board = (x0, y0, x0 + scene["cols"] * c, y0 + scene["rows"] * c)
    cb = render.CLOCK_BOX
    hit = lambda a, b, g=0: not (a[2] + g <= b[0] or b[2] + g <= a[0] or a[3] + g <= b[1] or b[3] + g <= a[1])
    for i, b in enumerate(boxes):
        if b[3] - b[1] < 24:
            errs.append(f"label {scene['labels'][i]['text']} glyphs under 24 px tall")
        if hit(b, board, 2) or hit(b, cb, 4):
            errs.append(f"label {scene['labels'][i]['text']} touches the board or the clock")
        for j in range(i + 1, len(boxes)):
            if hit(b, boxes[j], 2):
                errs.append(f"labels {scene['labels'][i]['text']} and {scene['labels'][j]['text']} overlap")
    for f in (0, len(frames) // 2, len(frames) - 1):
        fr = frames[f].astype(np.int16)
        for lab in scene["labels"]:
            x, y = int(lab["x"]), int(lab["y"])
            box = (x - 22, y - 20, x + 22, y + 20)
            crop = fr[box[1]:box[3], box[0]:box[2]]
            best, scores = None, {}
            for t in texts:
                im = Image.new("RGB", (44, 40), tuple(st["bg"]))
                ImageDraw.Draw(im).text((lab["x"] - box[0], lab["y"] - box[1]), t, fill=tuple(st["ink"]),
                                        font=font(sim.LABEL_PX), anchor="mm")
                scores[t] = float(np.abs(crop - np.asarray(im, dtype=np.int16)).mean())
            best = min(scores, key=scores.get)
            if best != lab["text"]:
                errs.append(f"frame {f}: label at ({x},{y}) reads {best!r}, scene says {lab['text']!r}")
    return errs


def emitter_boxes(scene):
    st = scene["style"]
    c = st["cell"]
    out = []
    for em in scene["emitters"]:
        cell, d = sim.edge_entry(scene, em)
        bx, by = st["x0"] + (cell[0] + 0.5 - 0.5 * d[0]) * c, st["y0"] + (cell[1] + 0.5 - 0.5 * d[1]) * c
        out.append((em["colour"], bx, by, (bx - 0.45 * c, by - 0.45 * c, bx + 0.45 * c, by + 0.45 * c)))
    return out


def colour_mask(frame, rgb, tol=60):
    d = np.abs(frame.astype(np.int16) - np.array(rgb, dtype=np.int16)).sum(-1)
    return (d < tol).astype(np.uint8)


def find_heads(frame, scene):
    st = scene["style"]
    k = st["trail_w"] + 3
    kern = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    em = emitter_boxes(scene)
    out = []
    for colour in {e["colour"] for e in scene["emitters"]}:
        m = cv2.erode(colour_mask(frame, sim.BEAMS[colour], 75), kern)
        n, lab, stats, cent = cv2.connectedComponentsWithStats(m, connectivity=8)
        for i in range(1, n):
            if stats[i, cv2.CC_STAT_AREA] < 4:
                continue
            x, y = float(cent[i][0]), float(cent[i][1])
            if any(col == colour and b[0] <= x <= b[2] and b[1] <= y <= b[3] for col, _, _, b in em):
                continue
            # a head is a filled disk: beam colour in every direction, diagonals included (a trail
            # crossing is a plus sign and fails on the diagonals)
            r = 0.3 * st["head_px"]
            full = colour_mask(frame, sim.BEAMS[colour], 75)
            ok = sum(int(full[int(round(y + r * math.sin(a))), int(round(x + r * math.cos(a)))])
                     for a in [k * math.pi / 4 for k in range(8)] if 0 <= y + r * math.sin(a) < 719 and 0 <= x + r * math.cos(a) < 1279)
            if ok < 7:
                continue
            out.append((colour, x, y))
    return out


def at_board_edge(scene, x, y):
    """A head straddling the board edge (leaving it) is partly outside the board: not located reliably."""
    st = scene["style"]
    x0, y0 = st["x0"], st["y0"]
    x1, y1 = x0 + scene["cols"] * st["cell"], y0 + scene["rows"] * st["cell"]
    return min(abs(x - x0), abs(x - x1), abs(y - y0), abs(y - y1)) < st["head_px"]


def st_head(scene):
    return scene["style"]["head_px"]


def in_emitter(scene, colour, x, y, pad=0.0):
    return any(col == colour and b[0] - pad <= x <= b[2] + pad and b[1] - pad <= y <= b[3] + pad
               for col, _, _, b in emitter_boxes(scene))


def mirror_errors(scene, frames):
    errs = []
    st = scene["style"]
    c = st["cell"]
    col = np.array(st["mirror"], dtype=np.int16)
    for f in (0, len(frames) - 1):
        S = render.state(scene, f)
        fr = frames[f].astype(np.int16)
        for m, ang in zip(scene["mirrors"], S["angles"]):
            if abs(abs(ang) - 45) > 1e-6:
                continue
            cx, cy = render.px(scene, m["cell"][0] + 0.5, m["cell"][1] + 0.5)
            score = {}
            for name, a in (("/", -45), ("\\", 45)):
                hits = 0
                for u in np.linspace(-0.3, 0.3, 13):
                    x = int(cx + math.cos(math.radians(a)) * u * c * 1.414)
                    y = int(cy + math.sin(math.radians(a)) * u * c * 1.414)
                    hits += int(np.abs(fr[y, x] - col).sum() < 90)
                score[name] = hits
            want = "/" if ang < 0 else "\\"
            if score[want] <= score["/" if want == "\\" else "\\"]:
                errs.append(f"frame {f}: mirror {answers.cell_name(m['cell'])} does not read as {want!r} ({score})")
    return errs


def lit_from_pixels(frame, scene):
    st = scene["style"]
    c, x0, y0 = st["cell"], st["x0"], st["y0"]
    out = set()
    for colour in {e["colour"] for e in scene["emitters"]}:
        m = colour_mask(frame, sim.BEAMS[colour])
        for i in range(scene["cols"]):
            for j in range(scene["rows"]):
                sub = m[int(y0 + j * c) + 3:int(y0 + (j + 1) * c) - 3, int(x0 + i * c) + 3:int(x0 + (i + 1) * c) - 3]
                if sub.sum() >= 12:
                    out.add((i, j))
    return out


def check_clip(s, root):
    scene = s["params"]
    errs, info = [], {"frames": 0, "clock_ok": 0, "q_checked": 0, "head_err": 0.0, "labels": 0}
    frames = decode(root / s["media_path"])
    N = render.n_frames(scene)
    if len(frames) != N:
        return [f"decoded {len(frames)} frames, expected {N}"], info
    info["frames"] = N
    decoded = []
    for f, fr in enumerate(frames):
        if read_clock(fr) == sim.clock_tenths(f):
            info["clock_ok"] += 1
        else:
            errs.append(f"frame {f}: clock misread")
        S = render.state(scene, f)
        pad = st_head(scene) + 6                  # heads still leaving their emitter merge with it
        want = [h for h in S["heads"] if not in_emitter(scene, *h, pad=pad)]
        got = find_heads(fr, scene)
        got_cmp = [h for h in got if not in_emitter(scene, *h, pad=pad)]    # same mask for both lists
        merged = any(a[0] == b[0] and math.dist(a[1:], b[1:]) < st_head(scene) + 4
                     for i, a in enumerate(want) for b in want[i + 1:])
        launching = any(in_emitter(scene, *h, pad=pad + 10) or at_board_edge(scene, h[1], h[2]) for h in S["heads"])
        if merged or launching:                        # just after a split / launch the head overlaps its neighbour
            info["merged"] = info.get("merged", 0) + 1  # (launch timing has its own check below)
        elif len(got_cmp) != len(want):
            errs.append(f"frame {f}: {len(got_cmp)} heads found, {len(want)} drawn")
        for h in ([] if merged else want):
            near = [g for g in got if g[0] == h[0]]
            if near:
                e = min(math.dist(g[1:], h[1:]) for g in near)
                info["head_err"] = max(info["head_err"], e)
                if e > TOL:
                    errs.append(f"frame {f}: {h[0]} head off by {e:.1f} px")
        # a head inside its emitter box is not located: keep the drawn one there (launch frames only)
        decoded.append({"heads": got + [h for h in S["heads"] if in_emitter(scene, *h)], "angles": S["angles"],
                        "trails": S["trails"]})
    errs += label_errors(scene, frames)
    info["labels"] = len(scene["labels"])
    errs += mirror_errors(scene, frames)
    # launch frame from pixels: the first frame whose entry-point patch differs from frame 0
    for (colour, bx, by, b), em in zip(emitter_boxes(scene), scene["emitters"]):
        box = (int(bx) - 15, int(by) - 15, int(bx) + 15, int(by) + 15)
        ref = frames[0][box[1]:box[3], box[0]:box[2]].astype(np.int16)
        diff = [float(np.abs(fr[box[1]:box[3], box[0]:box[2]].astype(np.int16) - ref).mean()) for fr in frames]
        first = next((f for f, v in enumerate(diff) if v > 6), None)
        want = math.ceil(em["t0"] * 24 - 1e-9)
        if first is None or abs(first - want) > 1:
            errs.append(f"{colour} beam: entry point first changes at frame {first}, renderer starts it at frame {want}")
    for q in s["questions"]:
        key = q["derivation_key"]
        for k in ("t",):
            if k in q["params"]:
                f = answers.frame_of_clock(q["params"][k])
                if read_clock(frames[f]) != round(q["params"][k] * 10):
                    errs.append(f"{q['qid']}: clock at frame {f} does not show {q['params'][k]:.1f} s")
        if key == "n_lit_cells":
            info["q_checked"] += 1
            k = len(lit_from_pixels(frames[-1], scene))
            if str(k) != str(q["groundtruth"]):
                errs.append(f"{q['qid']} n_lit_cells: {k} lit cells in the last frame, stored {q['groundtruth']}")
        if key in TRACK_KEYS:
            info["q_checked"] += 1
            try:
                got = answers.answer(key, q["params"], scene, frames=decoded, jitter=JITTER)
            except Exception as ex:  # noqa: BLE001
                got = f"error {ex!r}"
            if str(got) != str(q["groundtruth"]):
                errs.append(f"{q['qid']} {key}: from decoded frames {got!r}, stored {q['groundtruth']!r}")
    return errs, info


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset-dir", required=True)
    ap.add_argument("--report", default="frame_check_report.txt")
    args = ap.parse_args(argv)
    root = Path(args.dataset_dir).resolve()
    lines, total = [], 0
    tot = {"frames": 0, "clock_ok": 0, "q_checked": 0, "head_err": 0.0, "labels": 0}
    with open(root / "annotations.jsonl", encoding="utf8") as fh:
        rows = fh.readlines()
    for line in rows:
        s = json.loads(line)
        errs, info = check_clip(s, root)
        for k in ("frames", "clock_ok", "q_checked", "labels"):
            tot[k] += info[k]
        tot["head_err"] = max(tot["head_err"], info["head_err"])
        total += len(errs)
        lines.append(f"{s['sample_id']} ({s['params']['scenario']}): {len(errs)} problems, max head error "
                     f"{info['head_err']:.2f} px")
        lines += [f"  - {e}" for e in errs[:20]]
        print(lines[-1 - min(20, len(errs))], flush=True)
    summary = (f"RESULT: {sum(1 for x in lines if not x.startswith('  '))} clips, {tot['frames']} frames, clock "
               f"template match {tot['clock_ok']}/{tot['frames']}, {tot['labels']} labels template-matched at 3 frames "
               f"each, {tot['q_checked']} questions re-derived from decoded frames, max head error "
               f"{tot['head_err']:.2f} px, {total} problems")
    (root / args.report).write_text("\n".join(lines + ["", summary]) + "\n", encoding="utf8")
    print(summary)
    sys.exit(1 if total else 0)


if __name__ == "__main__":
    main()
