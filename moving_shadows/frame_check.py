"""Frame-level check of rendered clips against the stored annotations.

    python frame_check.py --dataset-dir <dir> [--report frame_check_report.txt]

For every clip it decodes the MP4 (OpenCV) and, per frame:
  * reads the clock by template match against all possible readings (OCR over
    0.0-9.9 s) and requires the reading for that frame;
  * locates the light disc by its colour and each object by its fill colour
    (bounding box of near-colour pixels around where it should be) and
    compares the centres with what the renderer meant to draw, within TOL px;
  * reads every marker plate's centre pixels and classifies it lit / shaded;
    a disagreement with the drawn state is only tolerated on a frame where
    a shadow edge is within EDGE_PX of the marker centre (blur), and counted;
  * checks that no object touches the clock box or a marker number, and
    that numbers do not overlap each other.
Marker numbers are read by template match (digits 1-9 drawn on the bare
ground) on the first, middle and last frames and must equal scene["labels"].
Then it rebuilds a track from the DECODED light centres, object offsets and
plate shading and re-derives every answer with answers.answer(..., tr=...);
all must equal the stored answers. Clock times named in a question or given
as an answer must match the decoded clock at the matching frame, and an
answer naming a marker must match the number read under that marker.
"""
from __future__ import annotations
import argparse
import functools
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

TOL = 3.0            # px, decoded centre vs drawn centre (colour-box detector loses up to a row at patterned edges)
EDGE_PX = 2.0        # px, plate readings this close to a shadow edge may flip in the encoded video


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


# ------------------------------------------------------------------ clock
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


# ------------------------------------------------------------------ labels
@functools.lru_cache(maxsize=8)
def _bare(key):
    scene = json.loads(key)
    return np.asarray(render._static(render.static_key(dict(scene, labels=[]))), dtype=np.int16)


def read_label(frame, scene, lb):
    """Digit (1-9) best matching the decoded label box, by template match on the bare ground."""
    bare = _bare(render.static_key(scene))
    x0, y0, x1, y1 = [int(round(v)) for v in lb["box"]]
    x0, y0, x1, y1 = x0 - 3, y0 - 3, x1 + 3, y1 + 3
    crop = frame[y0:y1, x0:x1].astype(np.int16)
    best, bd = None, 1e18
    for k in range(1, 10):
        img = Image.fromarray(bare[y0:y1, x0:x1].astype(np.uint8))
        ImageDraw.Draw(img).text((lb["x"] - x0, lb["y"] - y0), str(k), fill=(255, 255, 255), font=font(lb["size"]),
                                 anchor="mm", stroke_width=3, stroke_fill=render.INK)
        d = float(np.abs(crop - np.asarray(img, dtype=np.int16)).mean())
        if d < bd:
            best, bd = str(k), d
    return best


# ------------------------------------------------------------------ objects and light
def colour_box(frame, rgb, box, thr=45):
    x0, y0, x1, y1 = [int(v) for v in box]
    x0, y0, x1, y1 = max(0, x0), max(0, y0), min(render.SIZE[0], x1), min(render.SIZE[1], y1)
    roi = frame[y0:y1, x0:x1].astype(np.int32)
    m = ((roi - np.array(rgb)) ** 2).sum(-1) < thr ** 2
    ys, xs = np.nonzero(m)
    if len(xs) < 12:
        return None
    return x0 + xs.min(), y0 + ys.min(), x0 + xs.max() + 1, y0 + ys.max() + 1


def fill_box(scene, o, pts):
    """Bounding box of the drawn fill colour, from the renderer itself on a blank canvas (the reference)."""
    img = Image.new("RGB", render.SIZE, (0, 0, 0))
    render.draw_object(img, o, pts, scene["style"]["outline"])
    xs = [p[0] for p in pts]
    ys = [p[1] for p in pts]
    return colour_box(np.asarray(img), sim.COLOURS[o["color"]],
                      (min(xs) - 8, min(ys) - 8, max(xs) + 8, max(ys) + 8))


def centre(b):
    return (b[0] + b[2]) / 2, (b[1] + b[3]) / 2


def light_centre(frame, scene, expect):
    r = scene["style"]["sun_r"]
    ex, ey = expect
    b = colour_box(frame, sim.SUNS[scene["style"]["sun"]], (ex - r - 6, ey - r - 6, ex + r + 7, ey + r + 7), 40)
    return None if b is None else centre(b)


def plate_value(frame, scene, j):
    y0, y1 = render.strip_rows(scene)
    cy = (y0 + y1) // 2
    c = scene["markers"][j]
    return frame[cy - 3:cy + 4, c:c + 1].reshape(-1, 3).astype(float).mean(0)


def boxes_overlap(a, b, pad=0):
    return not (a[2] + pad <= b[0] or b[2] + pad <= a[0] or a[3] + pad <= b[1] or b[3] + pad <= a[1])


# ------------------------------------------------------------------ per clip
def check_clip(s, root):
    scene = s["params"]
    errs = []
    info = {"frames": 0, "clock_ok": 0, "err_max": 0.0, "plates": 0, "plate_tol": 0, "labels_ok": 0,
            "labels": 0, "q_checked": 0}
    frames = decode(root / s["media_path"])
    N = sim.n_frames(scene)
    if len(frames) != N:
        return [f"decoded {len(frames)} frames, expected {N}"], info
    info["frames"] = N
    nc, nm = len(scene["casters"]), len(scene["markers"])
    lit = np.array(render.PLATE_RGB, dtype=float)
    k = scene["style"]["shade"]
    dark = lit * k + np.array([4, 6, 14]) * (1 - k)
    base_c = [centre(fill_box(scene, o, sim.outline(o, 0.0))) for o in scene["casters"]]
    lights, shifts, shaded = [], [[] for _ in range(nc)], []
    lab_boxes = [lb["box"] for lb in scene["labels"]]
    for a in range(nm):
        for b in range(a + 1, nm):
            if boxes_overlap(lab_boxes[a], lab_boxes[b], 4):
                errs.append(f"labels {a + 1} and {b + 1} overlap")
    for f, fr in enumerate(frames):
        if read_clock(fr) != sim.clock_tenths(f):
            errs.append(f"frame {f}: clock misread")
        else:
            info["clock_ok"] += 1
        st = render.state(scene, f)
        lc = light_centre(fr, scene, st["light"])
        if lc is None:
            errs.append(f"frame {f}: light not found")
            lc = st["light"]
        e = math.dist(lc, st["light"])
        info["err_max"] = max(info["err_max"], e)
        if e > TOL:
            errs.append(f"frame {f}: light at {lc}, drawn at {st['light']}")
        lights.append(lc)
        cbox = render.CLOCK_BOX
        for i, o in enumerate(scene["casters"]):
            pts = st["casters"][i]
            want = fill_box(scene, o, pts) if (o.get("vx") or o.get("vy")) else None
            want_c = centre(want) if want else base_c[i]
            xs, ys = [p[0] for p in pts], [p[1] for p in pts]
            ybot = scene["ground"] - 1 if o["kind"] == "stand" else max(ys) + 10    # nothing stands below the ground
            got = colour_box(fr, sim.COLOURS[o["color"]], (min(xs) - 10, min(ys) - 10, max(xs) + 10, ybot))
            if got is None:
                errs.append(f"frame {f}: {o['color']} object not found")
                gc = want_c
            else:
                gc = centre(got)
                if boxes_overlap(got, cbox, 4):
                    errs.append(f"frame {f}: {o['color']} touches the clock")
                if any(boxes_overlap(got, lb) for lb in lab_boxes):
                    errs.append(f"frame {f}: {o['color']} touches a marker number")
            e = math.dist(gc, want_c)
            info["err_max"] = max(info["err_max"], e)
            if e > TOL:
                errs.append(f"frame {f}: {o['color']} centre {gc}, drawn {want_c}")
            shifts[i].append((gc[0] - base_c[i][0], gc[1] - base_c[i][1]))
        on_drawn = render.shaded_columns(scene, st["shadows"])
        row = []
        for j in range(nm):
            v = plate_value(fr, scene, j)
            got = bool(np.abs(v - dark).sum() < np.abs(v - lit).sum())
            want = bool(on_drawn[scene["markers"][j]])
            info["plates"] += 1
            if got != want:
                m = sim.mx(scene, j)
                near = min((min(abs(lo - m), abs(hi - m)) for lo, hi in st["shadows"]), default=1e9)
                if near < EDGE_PX:
                    info["plate_tol"] += 1
                else:
                    errs.append(f"frame {f}: marker {j + 1} reads {'shaded' if got else 'lit'}")
            row.append(got)
        shaded.append(row)
    for f in (0, N // 2, N - 1):
        for lb in scene["labels"]:
            info["labels"] += 1
            r = read_label(frames[f], scene, lb)
            if r != lb["text"]:
                errs.append(f"frame {f}: label reads {r}, scene says {lb['text']}")
            else:
                info["labels_ok"] += 1
    tr = {"light": lights, "shift": shifts, "shaded": shaded}
    T = answers.decoded_track(scene, tr)
    for q in s["questions"]:
        info["q_checked"] += 1
        try:
            got = answers.answer(q["derivation_key"], q["params"], scene, tr=tr)
        except Exception as ex:  # noqa: BLE001
            got = f"error {ex!r}"
        if str(got) != str(q["groundtruth"]):
            errs.append(f"{q['qid']} {q['derivation_key']}: from decoded frames {got!r}, stored {q['groundtruth']!r}")
        p = q["params"]
        if "t" in p:
            f = answers.frame_of_clock(p["t"])
            if read_clock(frames[f]) != round(p["t"] * 10):
                errs.append(f"{q['qid']}: clock at frame {f} does not show {p['t']:.1f} s")
        if q["derivation_key"] in ("cover_time", "uncover_time"):
            r = T.m_runs(p["j"])[0]
            f = r[0] if q["derivation_key"] == "cover_time" else r[1]
            if f"{read_clock(frames[f]) / 10:.1f} s" != q["groundtruth"]:
                errs.append(f"{q['qid']}: decoded clock at the event frame {f} != {q['groundtruth']}")
        gt = str(q["groundtruth"])
        if gt.startswith("marker "):
            j = int(gt.split()[1]) - 1
            if read_label(frames[N // 2], scene, scene["labels"][j]) != gt.split()[1]:
                errs.append(f"{q['qid']}: number under the answer marker does not read {gt.split()[1]}")
    return errs, info


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset-dir", required=True)
    ap.add_argument("--report", default="frame_check_report.txt")
    args = ap.parse_args(argv)
    root = Path(args.dataset_dir).resolve()
    lines, total = [], 0
    tot = {k: 0 for k in ("frames", "clock_ok", "plates", "plate_tol", "labels", "labels_ok", "q_checked")}
    tot["err_max"] = 0.0
    with open(root / "annotations.jsonl", encoding="utf8") as fh:
        samples = [json.loads(line) for line in fh]
    for s in samples:
        errs, info = check_clip(s, root)
        for k in tot:
            tot[k] = max(tot[k], info[k]) if k == "err_max" else tot[k] + info[k]
        total += len(errs)
        lines.append(f"{s['sample_id']} ({s['params']['scenario']}): {len(errs)} problems, max centre error "
                     f"{info['err_max']:.2f} px, plate readings tolerated at an edge {info['plate_tol']}")
        lines += [f"  - {e}" for e in errs[:20]]
        print(lines[-1 - min(20, len(errs))], flush=True)
    summary = (f"RESULT: {sum(1 for x in lines if not x.startswith('  '))} clips, {tot['frames']} frames, "
               f"clock template match {tot['clock_ok']}/{tot['frames']}, marker numbers read "
               f"{tot['labels_ok']}/{tot['labels']}, plate readings {tot['plates']} "
               f"({tot['plate_tol']} within {EDGE_PX} px of a shadow edge tolerated), "
               f"{tot['q_checked']} answers re-derived from decoded frames, max light/object centre error "
               f"{tot['err_max']:.2f} px, {total} problems")
    (root / args.report).write_text("\n".join(lines + ["", summary]) + "\n", encoding="utf8")
    print(summary)
    sys.exit(1 if total else 0)


if __name__ == "__main__":
    main()
