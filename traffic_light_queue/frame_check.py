"""Frame-level check of rendered traffic_light_queue clips against the stored annotations.

    python frame_check.py --dataset-dir <dir> [--report frame_check_report.txt]

Per clip it decodes the MP4 (OpenCV) and, per frame:
  * reads the clock: the clock box must match the rendered text for that frame
    better than any other reading 0.0-10.0 s (template match);
  * reads every lamp: mean colour of the lamp disc, red or green, which must
    agree with the drawn light;
  * locates every vehicle by its colour inside the middle band of its lane
    (pixels near the palette colour; the extent along the lane, plus the 3 px
    outline, gives the box): it must be found where it is drawn (the edge that gives
    the front within TOL px) and nowhere when it is not drawn (a sliver under 20 px at the gate or
    the far end is allowed either way);
then it rebuilds the tracks from the DECODED boxes and lamp colours and
re-derives every answer with answers.answer(..., frames=decoded,
jitter=JITTER). Questions that name a clock time get the clock checked at that
exact frame.
"""
from __future__ import annotations
import argparse
import json
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

TOL = 4.0
JITTER = 0.75
OUTLINE = 3
RED, GREEN = np.array((230, 40, 40)), np.array((40, 210, 80))


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


def read_lamp(frame, scene, li):
    x, y = render.lamp_xy(scene, li)
    r = max(3, scene["style"]["lamp_px"] // 3)
    m = frame[int(y) - r:int(y) + r + 1, int(x) - r:int(x) + r + 1].reshape(-1, 3).mean(0)
    return bool(np.abs(m - RED).sum() < np.abs(m - GREEN).sum())


def find_car(frame, scene, car):
    """Box (x0, y0, x1, y1) of the vehicle's colour in the middle band of its lane, or None."""
    lane = scene["lanes"][car["lane"]]
    W = scene["style"]["lane_w"]
    band = int(W * 0.28)                               # the whole body width, inside the lane marks
    col = np.array(sim.COLOURS[car["colour"]], dtype=np.int32)
    (ax, ay), (ex, ey) = lane["a"], lane["b"]
    if lane["dir"] == "h":
        c = int(round(ay))
        strip = frame[c - band:c + band + 1, int(ax):int(ex) + 1].astype(np.int32)
        axis, origin = 1, int(ax)
    else:
        c = int(round(ax))
        strip = frame[int(ay):int(ey) + 1, c - band:c + band + 1].astype(np.int32)
        axis, origin = 0, int(ay)
    hit = (((strip - col) ** 2).sum(-1) < 30 ** 2)
    prof = hit.sum(axis=1 - axis) if axis == 1 else hit.sum(axis=1)
    idx = np.nonzero(prof >= 4)[0]
    if len(idx) < 4:
        return None
    runs = np.split(idx, np.nonzero(np.diff(idx) > 16)[0] + 1)   # the body (gaps of windows / the truck cab are short)
    idx = max(runs, key=len)                           # stray edge pixels of other vehicles are tiny runs
    if len(idx) < 4:
        return None
    lo, hi = float(origin + idx[0] - OUTLINE), float(origin + idx[-1] + 1 + OUTLINE)
    lo = max(lo, ax if lane["dir"] == "h" else ay)
    end = ex if lane["dir"] == "h" else ey
    hi = end if hi >= end - 8 else hi                  # touching the far end: the front is cut off (back + length rules)
    w = W * 0.31
    return (lo, ay - w, hi, ay + w) if lane["dir"] == "h" else (ax - w, lo, ax + w, hi)


def check_clip(s, root):
    scene = s["params"]
    errs = []
    frames = decode(root / s["media_path"])
    N = render.n_frames(scene)
    info = {"frames": len(frames), "clock_ok": 0, "q_checked": 0, "err": 0.0}
    if len(frames) != N:
        return [f"decoded {len(frames)} frames, expected {N}"], info
    decoded = []
    for f, fr in enumerate(frames):
        if read_clock(fr) == answers.clock_tenths(f):
            info["clock_ok"] += 1
        else:
            errs.append(f"frame {f}: clock reads wrong")
        st = render.state(scene, f)
        lights = []
        for li, Ld in enumerate(st["lights"]):
            red = read_lamp(fr, scene, li)
            if red != Ld["red"]:
                errs.append(f"frame {f}: lamp {li} reads {'red' if red else 'green'}")
            lights.append({"red": red})
        drawn = {c["id"]: c["box"] for c in st["cars"]}
        cars = []
        for car in scene["cars"]:
            b = find_car(fr, scene, car)
            k = 0 if scene["lanes"][car["lane"]]["dir"] == "h" else 1
            d = drawn.get(car["id"])
            small = d is not None and d[k + 2] - d[k] < 20
            if (b is None) != (d is None) and not small:
                errs.append(f"frame {f}: {car['colour']} {'missing' if b is None else 'found where none is drawn'}")
            if b is not None and d is not None:
                end = scene["lanes"][car["lane"]]["b"][k]
                e = abs(b[k] - d[k]) if b[k + 2] >= end - 1 else abs(b[k + 2] - d[k + 2])   # the edge that sets the front
                info["err"] = max(info["err"], e)
                if e > TOL and not small:
                    errs.append(f"frame {f}: {car['colour']} box {e:.1f} px off")
            if b is not None:
                cars.append({"id": car["id"], "box": b})
        decoded.append({"cars": cars, "lights": lights})
    for q in s["questions"]:
        key = q["derivation_key"]
        if "t" in q["params"]:
            f = answers.frame_of_clock(q["params"]["t"])
            if read_clock(frames[f]) != round(q["params"]["t"] * 10):
                errs.append(f"{q['qid']}: clock at frame {f} does not show {q['params']['t']:.1f} s")
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
    tot = {"frames": 0, "clock_ok": 0, "q_checked": 0, "err": 0.0}
    with open(root / "annotations.jsonl", encoding="utf8") as fh:
        rows = fh.readlines()
    for line in rows:
        s = json.loads(line)
        errs, info = check_clip(s, root)
        for k in ("frames", "clock_ok", "q_checked"):
            tot[k] += info[k]
        tot["err"] = max(tot["err"], info["err"])
        total += len(errs)
        lines.append(f"{s['sample_id']} ({s['params']['scenario']}): {len(errs)} problems, max box error "
                     f"{info['err']:.2f} px")
        lines += [f"  - {e}" for e in errs[:20]]
        print(lines[-1 - min(20, len(errs))], flush=True)
    summary = (f"RESULT: {sum(1 for x in lines if not x.startswith('  '))} clips, {tot['frames']} frames, clock "
               f"template match {tot['clock_ok']}/{tot['frames']}, {tot['q_checked']} questions re-derived from "
               f"decoded frames, max box error {tot['err']:.2f} px, {total} problems")
    (root / args.report).write_text("\n".join(lines + ["", summary]) + "\n", encoding="utf8")
    print(summary)
    sys.exit(1 if total else 0)


if __name__ == "__main__":
    main()
