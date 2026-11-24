"""Frame-level check of rendered clips against the stored annotations.

    python frame_check.py --dataset-dir <dir> [--report frame_check_report.txt]

For every clip it decodes the MP4 (OpenCV) and, per frame:
  * locates each mass by its colour (nearest-palette pixels inside the mass's
    footprint, largest connected blob) and compares the blob centre with the
    centre the renderer meant to draw (render.state), within TOL px;
  * checks the stop post is drawn exactly while the mass is held;
  * reads the clock: the decoded clock box must match the rendered clock
    text for that frame better than the text for the tenth before or after
    (template match = OCR over the 101 possible readings);
  * checks no mass blob touches the clock box and no two mass blobs come
    within 15 px of each other.
Then it rebuilds a track from the DECODED centres and stop posts and
re-derives every answer with answers.answer(..., tr=decoded, h=JITTER).
Every question except shape_of / pattern_of (sprite looks, checked by eye)
must give the stored answer. Questions that name a clock time also get the
clock template check at that exact frame (the time in the question is what
the clock shows there).
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

TOL = 3.0            # px, decoded blob centre vs drawn centre
JITTER = 2.0         # px hysteresis for crossings / turns on decoded tracks
STATIC = {"shape_of", "pattern_of"}
PAL = np.array(list(sim.COLOURS.values()), dtype=np.int32)
NAMES = list(sim.COLOURS)


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
    """Best-matching clock reading (tenths) for a decoded frame, by template match."""
    x0, y0, x1, y1 = render.CLOCK_BOX
    crop = frame[y0 + 4:y1 - 3, x0 + 4:x1 - 3].astype(np.int16)
    diffs = {k: float(np.abs(crop - clock_img(k)).mean()) for k in range(0, 101)}
    return min(diffs, key=diffs.get), diffs


def locate(frame, scene, o):
    """Centre (x, y) and box of the mass's blob, or None."""
    x0, y0, x1, y1 = [int(v) for v in sim.footprint(scene, o)]
    x0, y0 = max(0, x0 - 4), max(0, y0 - 4)
    x1, y1 = min(1280, x1 + 4), min(720, y1 + 4)
    roi = frame[y0:y1, x0:x1].astype(np.int32)
    d = ((roi[:, :, None, :] - PAL[None, None, :, :]) ** 2).sum(-1)
    near = d.argmin(-1)
    mask = ((near == NAMES.index(o["colour"])) & (d.min(-1) < 45 ** 2)).astype(np.uint8)
    n, lab, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    if n < 2:
        return None
    k = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    bx, by, bw, bh = stats[k, :4]
    return (x0 + bx + (bw - 1) / 2, y0 + by + (bh - 1) / 2), (x0 + bx, y0 + by, x0 + bx + bw, y0 + by + bh)


def count_blobs(frame):
    """Palette-coloured blobs of at least 250 px anywhere in the frame (the object count)."""
    img = frame[::2, ::2].astype(np.int32)
    d = ((img[:, :, None, :] - PAL[None, None, :, :]) ** 2).sum(-1)
    near, ok = d.argmin(-1), d.min(-1) < 45 ** 2
    kern = np.ones((5, 5), np.uint8)
    boxes = []
    for k in range(len(NAMES)):
        m = cv2.morphologyEx(((near == k) & ok).astype(np.uint8), cv2.MORPH_CLOSE, kern)
        cnt, _, stats, _ = cv2.connectedComponentsWithStats(m, connectivity=8)
        boxes += [tuple(r[:5]) for r in stats[1:] if r[4] >= 250 / 4]
    # a blob nested inside a bigger blob's box is a blurred pattern (e.g. a white ring on red), not a mass
    inside = lambda a, b: b[4] > a[4] and b[0] <= a[0] and b[1] <= a[1] and a[0] + a[2] <= b[0] + b[2] + 1 \
        and a[1] + a[3] <= b[1] + b[3] + 1
    return sum(1 for a in boxes if not any(inside(a, b) for b in boxes))


def post_drawn(frame, scene, o, centre):
    """Is the stop post drawn next to the mass (sampled at the post's middle)?"""
    cx, cy = centre
    h = o["size"] / 2
    if scene["orient"] == "horizontal":
        px = cx - h - 6 if o["s"] > 0 else cx + h + 6
        py = (cy + 13 + o["ruler"]) / 2
    else:
        py = cy - h - 6 if o["s"] > 0 else cy + h + 6
        px = (cx + 13 + o["ruler"]) / 2
    col = np.array(render._shade(tuple(scene["style"]["fixture"]), -30))
    patch = frame[int(py) - 1:int(py) + 2, int(px) - 1:int(px) + 2].reshape(-1, 3).astype(int)
    return float(np.abs(patch - col).sum(1).min()) < 60


def check_clip(s, root):
    scene = s["params"]
    errs, info = [], {"frames": 0, "centre_err_max": 0.0, "clock_ok": 0, "q_checked": 0}
    frames = decode(root / s["media_path"])
    N = render.n_frames(scene)
    if len(frames) != N:
        return [f"decoded {len(frames)} frames, expected {N}"], info
    info["frames"] = N
    xy = {o["colour"]: [] for o in scene["objects"]}
    held = {o["colour"]: [] for o in scene["objects"]}
    cbox = render.CLOCK_BOX
    for f, fr in enumerate(frames):
        k, diffs = read_clock(fr)
        if k != sim.clock_tenths(f):
            errs.append(f"frame {f}: clock reads {k / 10:.1f}, expected {sim.clock_tenths(f) / 10:.1f}")
        else:
            info["clock_ok"] += 1
        st = {x["colour"]: x for x in render.state(scene, f)}
        boxes = []
        for o in scene["objects"]:
            got = locate(fr, scene, o)
            if got is None:
                errs.append(f"frame {f}: {o['colour']} mass not found")
                xy[o["colour"]].append(st[o["colour"]]["centre"])
                held[o["colour"]].append(False)
                continue
            (cx, cy), box = got
            want = st[o["colour"]]["centre"]
            e = math.dist((cx, cy), want)
            info["centre_err_max"] = max(info["centre_err_max"], e)
            if e > TOL:
                errs.append(f"frame {f}: {o['colour']} blob at ({cx:.1f},{cy:.1f}), drawn at "
                            f"({want[0]:.1f},{want[1]:.1f})")
            xy[o["colour"]].append((float(cx), float(cy)))
            hd = o["mode"] == "held" and post_drawn(fr, scene, o, (cx, cy))
            held[o["colour"]].append(hd)
            if hd != st[o["colour"]]["held"]:
                errs.append(f"frame {f}: {o['colour']} stop post {'seen' if hd else 'missing'}")
            if not (box[2] < cbox[0] or box[0] > cbox[2] or box[3] < cbox[1] or box[1] > cbox[3]):
                errs.append(f"frame {f}: {o['colour']} touches the clock")
            boxes.append((o["colour"], box))
        for i in range(len(boxes)):
            for j in range(i + 1, len(boxes)):
                a, b = boxes[i][1], boxes[j][1]
                if not (a[2] + 15 <= b[0] or b[2] + 15 <= a[0] or a[3] + 15 <= b[1] or b[3] + 15 <= a[1]):
                    errs.append(f"frame {f}: {boxes[i][0]} and {boxes[j][0]} within 15 px")
    blobs = count_blobs(frames[N // 2])
    if blobs != len(scene["objects"]):
        errs.append(f"mid-clip frame shows {blobs} coloured masses, scene has {len(scene['objects'])}")
    tr = {"xy": xy, "held": held}
    for q in s["questions"]:
        if q["derivation_key"] in STATIC:
            continue
        info["q_checked"] += 1
        try:
            got = answers.answer(q["derivation_key"], q["params"], scene, tr=tr, h=JITTER)
        except Exception as ex:  # noqa: BLE001
            got = f"error {ex!r}"
        if str(got) != str(q["groundtruth"]):
            errs.append(f"{q['qid']} {q['derivation_key']}: from decoded frames {got!r}, stored {q['groundtruth']!r}")
        for k in ("t", "t0", "t1"):
            if k in q["params"]:
                f = answers.frame_of_clock(q["params"][k])
                if read_clock(frames[f])[0] != round(q["params"][k] * 10):
                    errs.append(f"{q['qid']}: clock at frame {f} does not show {q['params'][k]:.1f} s")
    return errs, info


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset-dir", required=True)
    ap.add_argument("--report", default="frame_check_report.txt")
    args = ap.parse_args(argv)
    root = Path(args.dataset_dir).resolve()
    lines, total, tot = [], 0, {"frames": 0, "clock_ok": 0, "q_checked": 0, "centre_err_max": 0.0}
    for line in open(root / "annotations.jsonl", encoding="utf8"):
        s = json.loads(line)
        errs, info = check_clip(s, root)
        for k in ("frames", "clock_ok", "q_checked"):
            tot[k] += info[k]
        tot["centre_err_max"] = max(tot["centre_err_max"], info["centre_err_max"])
        total += len(errs)
        lines.append(f"{s['sample_id']} ({s['params']['scenario']}): {len(errs)} problems, "
                     f"max centre error {info['centre_err_max']:.2f} px")
        lines += [f"  - {e}" for e in errs[:20]]
        print(lines[-1 - min(20, len(errs))], flush=True)
    summary = (f"RESULT: {sum(1 for x in lines if not x.startswith('  '))} clips, "
               f"{tot['frames']} frames, clock template match {tot['clock_ok']}/{tot['frames']}, "
               f"{tot['q_checked']} questions re-derived from decoded frames, "
               f"max blob-centre error {tot['centre_err_max']:.2f} px, {total} problems")
    (root / args.report).write_text("\n".join(lines + ["", summary]) + "\n", encoding="utf8")
    print(summary)
    sys.exit(1 if total else 0)


if __name__ == "__main__":
    main()
