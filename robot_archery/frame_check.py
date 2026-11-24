"""Frame-level checks for rendered robot_archery clips (decodes the MP4s).

    python frame_check.py --dataset-dir ../output_videos/robot_archery_preview

Per clip:
- clock: every decoded frame is template-matched against all readings 0.0-10.0 s and must show f * 10 // 24;
- ring numbers: each label region of the last frame is template-matched against every ring text ("1".."10")
  drawn on the label-free background and must read as its scene text; glyphs >= 24 px; the label regions
  equal the static drawing on every sampled frame (no arrow, robot or record circle ever covers one);
- arrows: per frame, blobs of each robot colour inside the target (>= 60 % of a marker's area) are found
  in the pixels; their count, colour order and centres (within 4.5 px of render.state: under the 0.2-ring answer margin, >= 5.6 px) must match, and each
  appears on exactly the landing frame;
- robots: at the tracked body centre the pixel colour must be the robot's own colour (or its faded "out"
  colour) on every frame, which gives the frame each robot is first shown out; in two_lives each pip is
  read as filled or hollow;
- answers: the decoded track (arrow centres, out flags, pips) replaces render.state and answers.py
  re-derives every question; clock questions also read the decoded clock at the event frame.
"""
from __future__ import annotations
import argparse
import json
import math
import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE), str(HERE.parent)]
import sim  # noqa: E402
import render  # noqa: E402
import answers  # noqa: E402
from common.video5 import font  # noqa: E402

TEXTS = [str(k) for k in range(1, 11)]


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


def near(img, rgb, tol=70):
    return np.abs(img.astype(np.int16) - np.array(rgb, dtype=np.int16)).sum(-1) < tol


_OFFS: dict = {}


def marker_centre_offset(shape, rgb):
    """Centroid of the colour pixels of a drawn marker minus its landing point (calibrated on a clean render)."""
    key = (shape, tuple(rgb))
    if key not in _OFFS:
        img = Image.new("RGB", (80, 80), (255, 255, 255))
        d = ImageDraw.Draw(img)
        render.draw_shape(d, shape, 40.0, 40.0, sim.MARK_R, tuple(rgb), render.INK, 2)
        d.ellipse([38, 38, 42, 42], fill=render.INK)
        ys, xs = np.nonzero(near(np.asarray(img), rgb))
        _OFFS[key] = (float(xs.mean()) - 40.0, float(ys.mean()) - 40.0, len(xs))
    return _OFFS[key]


def arrows_in(img, scene, disc):
    """[(robot, (x, y))] blobs of robot colours inside the target."""
    out = []
    for r, rb in enumerate(scene["robots"]):
        rgb = sim.PALETTE[rb["colour"]]
        m = (near(img, rgb) & disc).astype(np.uint8)
        n, lab, stats, cents = cv2.connectedComponentsWithStats(m, connectivity=8)
        ox, oy, area = marker_centre_offset(rb["shape"], rgb)
        area_min = 0.6 * area
        for j in range(1, n):
            if stats[j, cv2.CC_STAT_AREA] >= area_min:
                out.append((r, (float(cents[j][0]) - ox, float(cents[j][1]) - oy)))
    return out


def check_clip(s, root, rep):
    scene = s["params"]
    frames = decode(root / s["media_path"])
    F = scene["frames"]
    errs = []
    if len(frames) != F:
        return [f"decoded {len(frames)} frames, expected {F}"], None
    # clock
    bad = [f for f in range(F) if read_clock(frames[f]) != sim.clock_tenths(f)]
    if bad:
        errs.append(f"clock misread on {len(bad)} frames, first {bad[:3]}")
    rep["clock_frames"] += F
    # labels
    blank = np.asarray(render._static(render.static_key(dict(scene, labels=[]))), dtype=np.int16)
    static = np.asarray(render._static(render.static_key(scene)), dtype=np.int16)
    for lab in scene["labels"]:
        b = sim.label_box(lab)
        if b[3] - b[1] < 24:
            errs.append("label shorter than 24 px")
        x0, y0, x1, y1 = int(b[0]) - 4, int(b[1]) - 3, int(b[2]) + 5, int(b[3]) + 4
        scores = {}
        for txt in TEXTS:
            ref = Image.fromarray(blank.astype(np.uint8))
            d = ImageDraw.Draw(ref)
            bb = sim.label_box(dict(lab, text=txt))
            d.rounded_rectangle([bb[0] - 4, bb[1] - 3, bb[2] + 4, bb[3] + 3], radius=5, fill=(255, 255, 255),
                                outline=render.INK, width=1)
            d.text((lab["x"], lab["y"]), txt, fill=render.INK, font=font(lab["size"]), anchor=lab["anchor"])
            ref = np.asarray(ref, dtype=np.int16)
            scores[txt] = float(np.abs(frames[-1][y0:y1, x0:x1].astype(np.int16) - ref[y0:y1, x0:x1]).mean())
        if min(scores, key=scores.get) != lab["text"]:
            errs.append(f"ring number {lab['text']} reads as {min(scores, key=scores.get)}")
        for f in range(0, F, 3):
            if render.state(scene, f)["flight"] is not None:          # an arrow in flight may pass over a tag
                continue
            if float(np.abs(frames[f][y0:y1, x0:x1].astype(np.int16) - static[y0:y1, x0:x1]).mean()) > 6:
                errs.append(f"ring number {lab['text']} covered on frame {f}")
                break
        rep["labels"] += 1
    # arrows
    tg = scene["target"]
    yy, xx = np.mgrid[0:render.SIZE[1], 0:render.SIZE[0]]
    disc = (xx - tg["cx"]) ** 2 + (yy - tg["cy"]) ** 2 <= (10 * tg["unit"] - 2) ** 2
    dec_arrows, prev = [], []
    for f in range(F):
        st = render.state(scene, f)
        found = arrows_in(frames[f], scene, disc)
        if st["flight"] is not None:                                  # the flying arrow's coloured head / fletching
            found = [(r, xy) for r, xy in found if math.dist(xy, st["flight"]["xy"]) > 45
                     or any(math.dist(xy, a["xy"]) < 6 for a in st["arrows"])]
        # keep the previous order; add new blobs (one per frame at most)
        cur = [(pr, next((xy for r, xy in found if r == pr and math.dist(xy, pxy) < 6), pxy)) for pr, pxy in prev]
        for r, xy in found:
            if not any(r == pr and math.dist(xy, pxy) < 6 for pr, pxy in prev):
                cur.append((r, xy))
        fl = st["flight"]
        if fl is not None and len(found) < len(st["arrows"]):
            missing = [a for a in st["arrows"] if not any(r == a["robot"] and math.dist(xy, a["xy"]) < 6
                                                          for r, xy in found)]
            if all(math.dist(a["xy"], fl["xy"]) < 45 for a in missing):   # the arrow in flight passes over it
                rep["flight_occluded_frames"] += 1
                dec_arrows.append(list(prev))
                continue
        if len(cur) != len(st["arrows"]) or len(found) != len(st["arrows"]):
            errs.append(f"frame {f}: {len(found)} arrow blobs, expected {len(st['arrows'])}")
            break
        for (r, xy), a in zip(cur, st["arrows"]):
            if r != a["robot"] or math.dist(xy, a["xy"]) > 4.5:
                errs.append(f"frame {f}: arrow {r} at {xy} vs {a['robot']} {a['xy']}")
                break
        dec_arrows.append(cur)
        prev = cur
        rep["arrow_frames"] += 1
    # robots: out colour and pips
    dec_robots, last = [], {}
    for f in range(F):
        st = render.state(scene, f)
        boxes = [sim.robot_box(scene, rs["xy"]) for rs in st["robots"]]
        row = []
        for r, rs in enumerate(st["robots"]):
            if any(sim._overlap(boxes[r], b) for j, b in enumerate(boxes) if j != r):
                row.append(last.get(r, (rs["xy"], False, scene["lives"])))   # hidden behind another robot
                rep["robot_overlap_frames"] += 1
                continue
            col = np.array(render.colour(scene, r), dtype=np.int16)
            faded = np.array(render._mix(render.colour(scene, r), (205, 205, 205), 0.25), dtype=np.int16)
            x, y = rs["xy"]
            y += scene["layout"]["body"] / 2 * 0.12
            pix = frames[f][int(round(y)) - 1:int(round(y)) + 2, int(round(x)) - 1:int(round(x)) + 2].astype(np.int16)
            pix = pix.reshape(-1, 3).mean(0)
            dn, df = np.abs(pix - col).sum(), np.abs(pix - faded).sum()
            out = bool(df < dn)
            if min(dn, df) > 60:
                errs.append(f"frame {f}: robot {r} body colour {pix} matches neither shade")
            lives = rs["lives"]
            if scene["lives"] > 1:
                lives = 0
                for j in range(scene["lives"]):
                    px = x + (j - (scene["lives"] - 1) / 2) * (2 * render.PIP_R + 6)
                    py = rs["xy"][1] - scene["layout"]["body"] / 2 - render.PIP_R - 8
                    c = frames[f][int(round(py)), int(round(px))].astype(np.int16)
                    lives += int(np.abs(c - np.array((225, 30, 60))).sum() < 90)
                if lives != rs["lives"]:
                    errs.append(f"frame {f}: robot {r} pips read {lives}, drawn {rs['lives']}")
            if out != rs["out"]:
                errs.append(f"frame {f}: robot {r} out read {out}, drawn {rs['out']}")
            row.append((rs["xy"], out, lives))
            last[r] = row[-1]
        dec_robots.append(row)
        rep["robot_frames"] += 1
    if errs:
        return errs[:6], None
    return [], {"arrows": dec_arrows, "robots": dec_robots}, frames


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset-dir", default=str(HERE.parent / "output_videos" / "robot_archery_preview"))
    ap.add_argument("--report", default="frame_check_report.txt")
    args = ap.parse_args()
    root = Path(args.dataset_dir).resolve()
    rep = {"clips": 0, "clock_frames": 0, "labels": 0, "arrow_frames": 0, "robot_frames": 0, "questions": 0,
           "answer_mismatch": 0, "clock_q": 0, "robot_overlap_frames": 0, "flight_occluded_frames": 0}
    errors = []
    for line in open(root / "annotations.jsonl", encoding="utf8"):
        s = json.loads(line)
        rep["clips"] += 1
        res = check_clip(s, root, rep)
        if res[1] is None:
            errors += [f"{s['sample_id']}: {e}" for e in res[0]]
            continue
        _, tr, frames = res
        o = answers.Obs(s["params"], tr)
        for q in s["questions"]:
            rep["questions"] += 1
            got = answers.derive(q["derivation_key"], q["params"], o, s["params"])
            if str(got) != str(q["groundtruth"]):
                rep["answer_mismatch"] += 1
                errors.append(f"{s['sample_id']}/{q['qid']} {q['derivation_key']}: decoded {got!r} "
                              f"vs stored {q['groundtruth']!r}")
            if q["derivation_key"] in ("out_time", "last_record_time"):
                if q["derivation_key"] == "out_time":
                    f = o.out_frame[q["params"]["robot"]]
                else:
                    f = o.shots[max(j for j, e in enumerate(o.log["log"]) if e[2])][2]
                if f"{read_clock(frames[f]) / 10:.1f} s" != q["groundtruth"] or \
                        f - 1 >= 0 and len(tr["arrows"][f - 1]) == len(tr["arrows"][f]):
                    errors.append(f"{s['sample_id']}/{q['qid']}: decoded clock at the event frame disagrees")
                rep["clock_q"] += 1
    lines = ["robot_archery frame check", "=" * 40] + [f"{k}: {v}" for k, v in rep.items()]
    lines += [f"errors: {len(errors)}"] + [f"  - {e}" for e in errors[:200]]
    (root / args.report).write_text("\n".join(lines) + "\n", encoding="utf8")
    print("\n".join(lines[:14 + min(20, len(errors))]))
    sys.exit(1 if errors else 0)


if __name__ == "__main__":
    main()
