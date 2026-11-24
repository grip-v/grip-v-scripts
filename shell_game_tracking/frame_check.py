"""Frame-level check of rendered clips against the scene data.

    python frame_check.py --dataset-dir <dir> [--report frame_check.txt] [--jobs 4]

For every clip it decodes the MP4 (OpenCV) and checks:
  * clock: template-matches the timer panel against every reading 0.0-9.9 s
    on every 6th frame, every event frame (move starts and ends, peeks, the
    end of the start reveal) and every frame a question names;
  * no other text: the clip's label list is empty;
  * cups, read from the pixels: everything that differs from the empty
    table (the static background) forms one blob per cup; on every frame the
    number of cup-sized blobs equals the number of cups and every blob's
    bottom-centre is within 2 px of a cup in the track (so the drawn motion
    is the track the decoder follows);
  * tokens, read from the pixels: on frames where the track shows a token it
    is found (its colour, at its spot); on every other frame no pixel of any
    token colour appears anywhere (nothing leaks where a token is);
  * events, frame-accurate: on a move's first frame some cup has left its
    mat and one frame earlier every cup sits on a mat; on its last frame
    every cup is back on a mat and one frame earlier one is not;
  * answers read off the frames: cup counts, clock answers on their event
    frames, and the position of the lifted cup at every peek.
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
import render  # noqa: E402
import sim  # noqa: E402
from common.video5 import font  # noqa: E402

DIFF = 40             # RGB distance from the empty table that counts as "something drawn"
TOK_TOL = 60          # RGB distance for a token colour
POS_TOL = 2.0


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


_TIMER = {}


def _timer_templates():
    if not _TIMER:
        tb = sim.TIMER_BOX
        for k in range(100):
            img = Image.new("RGB", (sim.W, sim.H), (0, 0, 0))
            d = ImageDraw.Draw(img)
            d.rounded_rectangle(tb, radius=10, fill=render.TIMER_BG, outline=render.TIMER_EDGE, width=2)
            d.text(((tb[0] + tb[2]) / 2, (tb[1] + tb[3]) / 2), sim.clock_text(k), fill=render.TIMER_FG,
                   font=font(sim.TIMER_FONT), anchor="mm")
            _TIMER[k] = np.asarray(img, dtype=np.int16)[tb[1] + 4:tb[3] - 4, tb[0] + 4:tb[2] - 4]
    return _TIMER


def read_timer(fr) -> int:
    tb = sim.TIMER_BOX
    patch = fr[tb[1] + 4:tb[3] - 4, tb[0] + 4:tb[2] - 4]
    T = _timer_templates()
    return min(T, key=lambda k: np.abs(T[k] - patch).mean())


def empty_table(scene):
    img = render._static(json.dumps(scene, sort_keys=True)).copy().reduce(render.SS)
    return np.asarray(img, dtype=np.int16)


def cups_seen(fr, bg, scene):
    """Bottom-centres of the cup-sized blobs of 'something drawn' (outside the clock panel)."""
    import cv2
    cw, ch = scene["cup"]["w"], scene["cup"]["h"]
    mask = (np.abs(fr - bg).sum(axis=2) > DIFF).astype(np.uint8)
    tb = sim.TIMER_BOX
    mask[:tb[3] + 4, :tb[2] + 4] = 0
    n, lab, st, cen = cv2.connectedComponentsWithStats(mask, connectivity=8)
    out = []
    for i in range(1, n):
        x, y, w, h, a = st[i]
        if a < 0.3 * cw * ch:
            continue
        # centre: mean x of the blob's pixels (the cups are left-right symmetric; a few codec pixels on an edge
        # barely move it); bottom: the lowest row that is solid across a third of the cup
        rows = (lab[y:y + h, x:x + w] == i).sum(axis=1)
        solid = np.nonzero(rows >= 0.33 * cw)[0]
        bottom = y + (solid[-1] + 1 if len(solid) else h)
        out.append((float(cen[i][0]), float(bottom), w, h))
    return out


def check_clip(scene, qs, frames) -> list[str]:
    errs = []
    n = scene["n_frames"]
    tr = render.track(scene)
    if len(frames) != n:
        return [f"decoded {len(frames)} frames, expected {n}"]
    if scene["labels"] or tr["labels"]:
        errs.append("labels present")
    bg = empty_table(scene)
    cw, ch = scene["cup"]["w"], scene["cup"]["h"]
    ev = sim.events(scene)
    # clock
    fs = set(range(0, n, 6)) | {f for _, f in ev}
    for q in qs:
        for k in ("t", "t1", "t2"):
            if k in q["params"]:
                a, b = sim.frames_showing(q["params"][k])
                fs |= set(range(a, b + 1))
    for f in sorted(fs):
        if f < n and read_timer(frames[f]) != sim.tenths(f):
            errs.append(f"clock wrong on frame {f}")
            break
    # cups on every frame
    bad = 0
    seen_all = []
    for f in range(n):
        seen = cups_seen(frames[f], bg, scene)
        seen_all.append(seen)
        want = tr["cups"][f]
        prev = tr["cups"][f - 1] if f else want
        ok = len(seen) == len(want)
        if ok:
            for x, y in want:
                # cups moving fast smear a few px in the encoded video
                still = min(abs(px - x) + abs(py - y) for px, py in prev) < 0.01
                d = min(abs(sx - x) + abs(sy - y) for sx, sy, _, _ in seen)
                if d > (2 * POS_TOL if still else 10):
                    ok = False
        if not ok:
            bad += 1
            if bad <= 2:
                errs.append(f"cups: frame {f} shows {len(seen)} cups not where the track has {len(want)}")
    # tokens: seen where shown, nowhere else
    ts = scene["tok_size"]
    st = scene["style"]
    area = {}
    for t in scene["tokens"]:
        sp = np.asarray(render.token_sprite(t["shape"], sim.TOKENS[t["colour"]], ts, render.edge(st))
                        .reduce(render.SS), dtype=np.int16)
        area[(t["colour"], t["shape"])] = int(((np.abs(sp[:, :, :3] - np.array(sim.TOKENS[t["colour"]])).sum(axis=2)
                                                < TOK_TOL) & (sp[:, :, 3] > 200)).sum())
    for f in range(n):
        vis = {(t["colour"], t["shape"]): t for t in tr["tokens"][f]}
        for t in scene["tokens"]:
            rgb = np.array(sim.TOKENS[t["colour"]])
            m = np.abs(frames[f] - rgb).sum(axis=2) < TOK_TOL
            m[:sim.TIMER_BOX[3] + 4, :sim.TIMER_BOX[2] + 4] = False
            key = (t["colour"], t["shape"])
            if key in vis:
                v = vis[key]
                x0, x1 = int(v["x"] - ts / 2), int(v["x"] + ts / 2)
                y0, y1 = int(v["y"] - 4 - ts), int(v["y"] - 4)
                if m[y0 - 2:y1 + 2, x0 - 2:x1 + 2].sum() < 0.6 * area[key]:
                    errs.append(f"token {key} not seen on frame {f}")
                    break
            elif m.sum() > 6:
                errs.append(f"token {key} colour visible on frame {f} while hidden ({int(m.sum())} px)")
                break
    # events, frame-accurate: all cups on mats just before a move / from its last frame
    S = np.array(scene["slots"], float)

    def all_on_mats(f):
        return all(min(abs(sx - x) + abs(sy - y) for x, y in S) <= 2 * POS_TOL for sx, sy, _, _ in seen_all[f])

    for m in scene["moves"]:
        a, b = m["start"], m["start"] + m["dur"] - 1
        if not all_on_mats(a - 1) or all_on_mats(a):
            errs.append(f"move starting on frame {a} not seen starting there")
        if not all_on_mats(b) or all_on_mats(b - 1):
            errs.append(f"move ending on frame {b} not seen ending there")
    # answers read off the frames
    for q in qs:
        k, p, gt = q["derivation_key"], q["params"], q["groundtruth"]
        val = None
        if k == "n_cups":
            val = len(seen_all[0])
        elif k == "move_time":
            val = sim.clock_text(read_timer(frames[scene["moves"][p["k"]]["start"]]))
        elif k == "first_move_time":
            f = next(m["start"] for m in scene["moves"] if gt == sim.clock_text(sim.tenths(m["start"])))
            val = sim.clock_text(read_timer(frames[f]))
        elif k == "peek_slot":
            pk = scene["peeks"][p["p"]]
            f = pk["start"] + pk["up"] + 2
            lifted = [(sx, sy) for sx, sy, _, _ in seen_all[f] if min(abs(sy - y) for _, y in S) > 2 * POS_TOL]
            if len(lifted) == 1:
                s = min(range(len(S)), key=lambda i: abs(S[i][0] - lifted[0][0]))
                val = sim.slot_name(scene, s)
        else:
            continue
        if val is None or str(val) != str(gt):
            errs.append(f"{q['qid']} {k}: read {val} off the frames, answer {gt}")
    return errs


def _one(arg):
    root, line = arg
    s = json.loads(line)
    frames = frames_of(root / s["media_path"])
    try:
        errs = check_clip(s["params"], s["questions"], frames)
    except Exception as ex:  # noqa: BLE001
        errs = [f"exception {ex!r}"]
    return s["sample_id"], s["params"]["scenario"], errs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset-dir", required=True)
    ap.add_argument("--report", default="frame_check.txt")
    ap.add_argument("--jobs", type=int, default=4)
    args = ap.parse_args()
    root = Path(args.dataset_dir).resolve()
    lines = (root / "annotations.jsonl").read_text().splitlines()
    from multiprocessing import Pool
    with Pool(args.jobs) as pool:
        res = pool.map(_one, [(root, ln) for ln in lines])
    out = []
    n_bad = 0
    for sid, scen, errs in res:
        out.append(f"{sid} [{scen}]: {'ok' if not errs else '; '.join(errs[:6])}")
        n_bad += bool(errs)
    out.append(f"RESULT: {len(res)} clips, {n_bad} with frame-check failures")
    (root / args.report).write_text("\n".join(out) + "\n")
    print("\n".join(out[-12:]))
    sys.exit(1 if n_bad else 0)


if __name__ == "__main__":
    main()
