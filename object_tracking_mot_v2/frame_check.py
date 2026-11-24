"""Frame-level check of rendered object_tracking_mot_v2 clips against the stored annotations.

    python frame_check.py --dataset-dir <dir> [--report frame_check_report.txt]

Per clip it decodes the MP4 (OpenCV) and, per frame:
  * reads the clock: the clock box must match the rendered text for that frame
    better than any other reading 0.0-10.0 s (template match);
  * locates every object: unique objects by their colour (largest blob of
    pixels nearest that palette colour), identical objects (targets) as all
    blobs of the shared colour; an object counts as seen only if its blob has
    its full size (a screen covering part of it makes it "hidden", as in
    render.state); centres must agree with the drawn ones within TOL px and
    the seen / hidden status must agree (frames where two orbiting objects
    overlap are skipped: the top one covers the other);
  * frame 0 of the targets scenario: the rings are found around the blobs
    (ring-colour pixels on a circle of radius r + 7) and must be exactly the
    stored targets;
then it rebuilds the tracks from the DECODED centres and re-derives every
answer (all keys but shape_of / pattern_of, which are sprite looks) with
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

TOL = 3.0
JITTER = 2.0
STATIC = {"shape_of", "pattern_of"}


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


def palette(scene):
    """Object colours first, then the screens' colours as extra classes (so screen pixels never count as an object)."""
    cols = [tuple(scene["style"]["same"])] if scene["scenario"] == "targets" else \
        [sim.COLOURS[o["colour"]] for o in scene["objects"]]
    for sc in scene["screens"]:
        c = sim.SCREENS[sc["name"]]
        cols += [c, tuple(max(0, v - 25) for v in c), tuple(max(0, v - 50) for v in c)]
    return np.array(cols, dtype=np.int32)


def classify(frame, scene):
    """Per-pixel nearest palette class inside the arena (and whether it is close enough to count)."""
    a = scene["arena"]
    roi = frame[a["y0"]:a["y1"], a["x0"]:a["x1"]].astype(np.int32)
    pal = palette(scene)
    d = ((roi[:, :, None, :] - pal[None, None]) ** 2).sum(-1)
    return d.argmin(-1), d.min(-1) < 50 ** 2


def blobs(cls, scene, k_colour):
    """Connected blobs (x0, y0, x1, y1, cx, cy) of class k_colour (cls = classify(frame, scene))."""
    a = scene["arena"]
    near, ok = cls
    m = ((near == k_colour) & ok).astype(np.uint8)
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
    n, lab, stats, cent = cv2.connectedComponentsWithStats(m, connectivity=8)
    out = []
    for i in range(1, n):
        x, y, w, h, area = stats[i]
        if area < 60:
            continue
        out.append((a["x0"] + x, a["y0"] + y, a["x0"] + x + w, a["y0"] + y + h,
                    a["x0"] + x + (w - 1) / 2, a["y0"] + y + (h - 1) / 2, int(area)))
    return out


def full_area(scene, o):
    """Pixel count of the drawn sprite (its alpha), to tell a whole object from a partly covered one."""
    spr = render.sprite(render.fill_of(scene, o), o["shape"], o["pattern"], 2 * o["r"])
    return int((np.asarray(spr.getchannel("A")) > 127).sum())


def full_size(scene, o):
    """Bounding-box size of the drawn sprite (its alpha), so a partly covered object can be told apart."""
    spr = render.sprite(render.fill_of(scene, o), o["shape"], o["pattern"], 2 * o["r"])
    bb = spr.getchannel("A").getbbox()
    return bb[2] - bb[0], bb[3] - bb[1]


def screen_depth(scene, o, x, y):
    """How deep (px) the drawn shape's outline reaches into a screen (<= 0: no overlap)."""
    best = -1.0
    for sc in scene["screens"]:
        for px, py in sim.outline(o, x, y):
            best = max(best, min(px - sc["x0"], sc["x1"] - px, py - sc["y0"], sc["y1"] - py))
    return best


def offset(scene, o):
    """Centre of the drawn sprite's box minus the disc centre (non-zero for triangles)."""
    spr = render.sprite(render.fill_of(scene, o), o["shape"], o["pattern"], 2 * o["r"])
    bb = spr.getchannel("A").getbbox()
    return (bb[0] + bb[2] - 1) / 2 - (2 * o["r"] - 1) / 2, (bb[1] + bb[3] - 1) / 2 - (2 * o["r"] - 1) / 2


def overlapping(scene, S, i):
    o = scene["objects"][i]
    a = next(x for x in S["objects"] if x["id"] == i)
    for b in S["objects"]:
        if b["id"] != i and math.dist((a["x"], a["y"]), (b["x"], b["y"])) < o["r"] + scene["objects"][b["id"]]["r"] + 8:
            return True
    return False


def rings_at_start(frame, scene, pts):
    col = np.array(scene["style"]["ring"], dtype=np.int16)
    out = []
    r = scene["objects"][0]["r"] + 5                 # the 4 px ring outline is drawn inward from r + 7
    for x, y in pts:
        hits = 0
        for k in range(16):
            a = k * math.pi / 8
            px, py = int(round(x + r * math.cos(a))), int(round(y + r * math.sin(a)))
            if 0 <= px < 1280 and 0 <= py < 720 and np.abs(frame[py, px].astype(np.int16) - col).sum() < 120:
                hits += 1
        if hits >= 12:
            out.append((x, y))
    return out


def check_clip(s, root):
    scene = s["params"]
    errs, info = [], {"frames": 0, "clock_ok": 0, "q_checked": 0, "err": 0.0, "seen": 0}
    frames = decode(root / s["media_path"])
    N = render.n_frames(scene)
    if len(frames) != N:
        return [f"decoded {len(frames)} frames, expected {N}"], info
    info["frames"] = N
    decoded = []
    targets = scene["scenario"] == "targets"
    sizes = {o["id"]: full_size(scene, o) for o in scene["objects"]}
    ref_area = {}
    if not targets and scene["screens"]:
        samples = {o["id"]: [] for o in scene["objects"]}
        for f in range(0, N, 4):
            S0 = render.state(scene, f)
            cls0 = classify(frames[f], scene)
            for k, o in enumerate(scene["objects"]):
                st0 = S0["objects"][k]
                if screen_depth(scene, o, st0["x"], st0["y"]) > -3 or overlapping(scene, S0, o["id"]):
                    continue
                bl0 = blobs(cls0, scene, k)
                if bl0:
                    b0 = min(bl0, key=lambda b: math.dist(b[4:6], (st0["x"], st0["y"])))
                    samples[o["id"]].append(b0[6])
        # the object's own area in this video when nothing covers it (median of its clear frames)
        ref_area = {i: sorted(v)[len(v) // 2] for i, v in samples.items() if len(v) >= 5}
    for f, fr in enumerate(frames):
        if read_clock(fr) == sim.clock_tenths(f):
            info["clock_ok"] += 1
        else:
            errs.append(f"frame {f}: clock misread")
        S = render.state(scene, f)
        cls = classify(fr, scene)
        objs = []
        if targets:
            bl = blobs(cls, scene, 0)
            want = [(o["x"], o["y"]) for o in S["objects"]]
            if len(bl) != len(want) and not any(overlapping(scene, S, o["id"]) for o in S["objects"]):
                errs.append(f"frame {f}: {len(bl)} objects found, {len(want)} drawn")
            o0 = scene["objects"][0]
            ox, oy = offset(scene, o0)                 # all objects share one shape: blob box centre -> disc centre
            bl = [b[:4] + (b[4] - ox, b[5] - oy) + b[6:] for b in bl]
            for k, b in enumerate(bl):
                objs.append({"id": k, "x": b[4], "y": b[5], "visible": True})
                e = min(math.dist(b[4:6], w) for w in want)
                if len(bl) == len(want):                   # merged blobs (touching objects) are not measured
                    info["err"] = max(info["err"], e)
                if e > TOL and len(bl) == len(want):
                    errs.append(f"frame {f}: an object found {e:.1f} px from any drawn one")
            entry = {"objects": objs}
            if f == 0:
                entry["ring_xy"] = rings_at_start(fr, scene, [(b[4], b[5]) for b in bl])
                drawn_t = sorted((round(o["x"]), round(o["y"])) for o in S["objects"] if o["id"] in scene["targets"])
                found_t = sorted((round(x), round(y)) for x, y in entry["ring_xy"])
                if len(found_t) != len(drawn_t) or any(math.dist(a, b) > TOL + 1 for a, b in zip(found_t, drawn_t)):
                    errs.append(f"frame 0: rings found around {found_t}, drawn around {drawn_t}")
            decoded.append(entry)
            continue
        for k, o in enumerate(scene["objects"]):
            st = next(x for x in S["objects"] if x["id"] == o["id"])
            bl = blobs(cls, scene, k)
            b = min(bl, key=lambda b: math.dist(b[4:6], (st["x"], st["y"]))) if bl else None
            fw, fh = sizes[o["id"]]
            spr = render.sprite(render.fill_of(scene, o), o["shape"], o["pattern"], 2 * o["r"])
            sb = spr.getchannel("A").getbbox()
            ex0, ey0 = round(st["x"] - o["r"]), round(st["y"] - o["r"])
            want_box = (ex0 + sb[0], ey0 + sb[1], ex0 + sb[2], ey0 + sb[3])
            # fully seen: every side of the blob box within 3 px of the drawn sprite's box (a screen clips a side)
            tol = 5 if o["shape"] in ("triangle", "diamond") else 3      # sharp tips blur away in the video
            # fully seen: the blob's box matches the drawn sprite's box AND it has (almost) the sprite's area
            # (a screen corner can bite a notch out without changing the box)
            area_ok = b is not None and (o["id"] not in ref_area or b[6] >= 0.93 * ref_area[o["id"]])
            seen = b is not None and all(abs(b[k] - want_box[k]) <= tol for k in range(4)) and area_ok
            if b is not None:                          # blob box centre -> disc centre
                ox, oy = offset(scene, o)
                b = b[:4] + (b[4] - ox, b[5] - oy) + b[6:]
            if overlapping(scene, S, o["id"]):         # touching / crossing objects: colour bleed moves the box
                objs.append({"id": o["id"], "x": st["x"], "y": st["y"], "visible": st["visible"]})
                continue
            depth = screen_depth(scene, o, st["x"], st["y"])
            bx0, by0, bx1, by1 = sim.shape_box(o, st["x"], st["y"])
            corner = depth > 0 and any(bx0 <= cx <= bx1 and by0 <= cy <= by1 for sc in scene["screens"]
                                       for cx in (sc["x0"], sc["x1"]) for cy in (sc["y0"], sc["y1"]))
            if 0 < depth < 12 or corner:               # a graze, or a screen corner nibbling a corner: no call
                objs.append({"id": o["id"], "x": st["x"], "y": st["y"], "visible": st["visible"]})
                continue
            transition = any(sim.hidden(scene, o["id"], (f + d) / 24) == st["visible"] for d in (-2, -1, 1, 2)
                             if 0 <= f + d < N)
            if seen != st["visible"] and not transition:   # within 2 frames of sliding under / out: sub-pixel overlap
                errs.append(f"frame {f}: {o['colour']} {'seen' if seen else 'not seen'}, drawn "
                            f"{'visible' if st['visible'] else 'behind a screen'}")
            if seen:
                info["seen"] += 1
                e = math.dist(b[4:6], (st["x"], st["y"]))
                info["err"] = max(info["err"], e)
                if e > TOL:
                    errs.append(f"frame {f}: {o['colour']} found {e:.1f} px from where it is drawn")
                objs.append({"id": o["id"], "x": b[4], "y": b[5], "visible": True})
            else:
                objs.append({"id": o["id"], "x": st["x"], "y": st["y"], "visible": False})
        decoded.append({"objects": objs})
    for q in s["questions"]:
        key = q["derivation_key"]
        if "t" in q["params"]:
            f = answers.frame_of_clock(q["params"]["t"])
            if read_clock(frames[f]) != round(q["params"]["t"] * 10):
                errs.append(f"{q['qid']}: clock at frame {f} does not show {q['params']['t']:.1f} s")
        if key in STATIC:
            continue
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
        lines.append(f"{s['sample_id']} ({s['params']['scenario']}): {len(errs)} problems, max centre error "
                     f"{info['err']:.2f} px")
        lines += [f"  - {e}" for e in errs[:20]]
        print(lines[-1 - min(20, len(errs))], flush=True)
    summary = (f"RESULT: {sum(1 for x in lines if not x.startswith('  '))} clips, {tot['frames']} frames, clock "
               f"template match {tot['clock_ok']}/{tot['frames']}, {tot['q_checked']} questions re-derived from "
               f"decoded frames, max centre error {tot['err']:.2f} px, {total} problems")
    (root / args.report).write_text("\n".join(lines + ["", summary]) + "\n", encoding="utf8")
    print(summary)
    sys.exit(1 if total else 0)


if __name__ == "__main__":
    main()
