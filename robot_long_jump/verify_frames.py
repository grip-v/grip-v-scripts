"""Frame-level check of rendered robot_long_jump clips against the ground truth.

    python verify_frames.py --dataset-dir <dir with annotations.jsonl + videos/>

It decodes every MP4 and reads the answers back off the pixels. No answer
comes from sim.py; render.py is used only to draw reference patches (timer
and label templates) and for the label / object bounding boxes.

  * ruler labels: the five number blobs are found in the ruler band; each is
    template-matched against freshly drawn "0.0" .. "2.0" patches and must
    match its scene label, sit within 3 px of its scene position and be at
    least 24 px tall. Their centres give the pixel -> ruler mapping used below;
  * robots: each robot is tracked by its (unique) body colour in every frame.
    Its landing frame (first frame of its final resting spot), its take-off
    frame (last frame on its take-off spot) and whether it ends toppled
    (its body colour fills >= 8% of the band just above its strip, where a standing robot shows only
    its legs) all come from that track;
  * footprints: in the last frame, white blobs on each runway and dark blobs
    on each sand strip are counted and located on the ruler;
  * timer: at every take-off / landing frame the timer patch is template-
    matched against the readings 0.1 s before / at / after;
  * then the questions whose answer is a count, position, time or order
    (hops_jump, hops_foul, n_fouls, takeoff, jump_length, score, longest_hop,
    start_mark, land_time, takeoff_time, first_land) are answered from these
    readings and compared with the stored answer;
  * geometry: over every frame of the layout, no label or the timer box
    overlaps a robot, and labels don't overlap each other.
"""
from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))
import render  # noqa: E402
from common.video5 import font  # noqa: E402

TEXTS = ("0.0", "0.5", "1.0", "1.5", "2.0")


def _decode(path):
    cap = cv2.VideoCapture(str(path))
    out = []
    while True:
        ok, fr = cap.read()
        if not ok:
            break
        out.append(cv2.cvtColor(fr, cv2.COLOR_BGR2RGB))
    cap.release()
    return out


def _timer_patch(scene, tenths):
    f = next(g for g in range(0, 400) if (g * 10) // 24 == tenths)
    img = Image.new("RGB", (render.W * render.SS, render.H * render.SS), (0, 0, 0))
    render._draw_timer(ImageDraw.Draw(img), scene, f)
    img = img.resize((render.W, render.H), Image.LANCZOS)
    x0, y0, x1, y1 = scene["timer"]["box"]
    return np.asarray(img, dtype=np.float32)[y0 + 4:y1 - 4, x0 + 4:x1 - 4]


def _label_box(lab):
    l, t, r, b = font(lab["size"]).getbbox(lab["text"], anchor="mm")
    return lab["x"] + l, lab["y"] + t, lab["x"] + r, lab["y"] + b


def _label_patch(lab, text, bg):
    S = render.SS
    img = Image.new("RGB", (render.W * S, render.H * S), bg)
    ImageDraw.Draw(img).text((lab["x"] * S, lab["y"] * S), text, fill=render.INK, font=font(lab["size"] * S),
                             anchor="mm")
    img = img.resize((render.W, render.H), Image.LANCZOS)
    x0, y0, x1, y1 = (int(v) for v in _label_box(lab))
    return np.asarray(img, dtype=np.float32)[y0 - 3:y1 + 3, x0 - 6:x1 + 6]


def _read_labels(fr, scene, errs):
    """Find the ruler numbers in the frame; return {text: centre x} read from the pixels."""
    ry = scene["view"]["ruler_y"]
    band = fr[ry + 22:716].astype(np.float32)
    dark = band.mean(axis=2) < 110
    cols = np.nonzero(dark.any(axis=0))[0]
    groups, cur = [], [cols[0]]
    for c in cols[1:]:
        if c - cur[-1] > 25:
            groups.append(cur)
            cur = []
        cur.append(c)
    groups.append(cur)
    if len(groups) != len(scene["labels"]):
        errs.append(f"found {len(groups)} label blobs in the ruler band, expected {len(scene['labels'])}")
        return None
    got = {}
    bg = tuple(int(v) for v in np.median(fr[ry + 50:ry + 60, 2:40].reshape(-1, 3), axis=0))
    for g, lab in zip(groups, scene["labels"]):
        xs = (g[0] + g[-1]) / 2
        rows = np.nonzero(dark[:, g[0]:g[-1] + 1].any(axis=1))[0]
        height = rows[-1] - rows[0] + 1
        if abs(xs - lab["x"]) > 3:
            errs.append(f"label {lab['text']} centre at x={xs:.1f}, scene says {lab['x']}")
        if height < 24:
            errs.append(f"label {lab['text']} is only {height} px tall")
        x0, y0, x1, y1 = (int(v) for v in _label_box(lab))
        patch = fr[y0 - 3:y1 + 3, x0 - 6:x1 + 6].astype(np.float32)
        err = {t: float(np.abs(patch - _label_patch(lab, t, bg)).mean()) for t in TEXTS}
        best = min(err, key=err.get)
        if best != lab["text"]:
            errs.append(f"label at x={lab['x']} reads {best!r}, scene says {lab['text']!r}")
        got[best] = xs
    return got


def _track(frames, scene):
    """Per robot, per frame: (pixel count, centroid x, centroid y) of its body colour."""
    cut = 632
    tr = {r["id"]: [] for r in scene["robots"]}
    refs = {r["id"]: np.array(render.RGB[r["colour"]], np.float32) for r in scene["robots"]}
    for fr in frames:
        px = fr[:cut].astype(np.float32)
        for i, ref in refs.items():
            m = np.abs(px - ref).sum(axis=2) < 60
            ys, xs = np.nonzero(m)
            tr[i].append((len(xs), xs.mean() if len(xs) else np.nan, ys.mean() if len(ys) else np.nan))
    return tr


def _low_fill(fr, cx, size, lane_y, colour):
    """Share of the band just above the strip (rows lane_y-8 .. lane_y-2, +-0.5 size around the robot) that shows
    its body colour: ~0 for a standing robot (only its legs and background there), high for one lying on its side."""
    x0, x1 = int(cx - 0.5 * size), int(cx + 0.5 * size) + 1
    band = fr[lane_y - 8:lane_y - 1, x0:x1].astype(np.float32)
    return float((np.abs(band - np.array(render.RGB[colour], np.float32)).sum(axis=2) < 90).mean())


def check_clip(s, root):
    scene = s["params"]
    errs, n_checks = [], 0
    frames = _decode(root / s["media_path"])
    if len(frames) != scene["n_frames"]:
        return [f"{len(frames)} frames decoded, expected {scene['n_frames']}"], 1
    v = scene["view"]
    last = frames[-1]
    # ----- labels and mapping
    labs = _read_labels(last, scene, errs)
    n_checks += 5
    if not labs:
        return errs, n_checks
    lab0 = _read_labels(frames[0], scene, errs)
    if lab0 != labs:
        errs.append("ruler labels differ between the first and last frame")
    x0, x1 = labs["0.0"], labs["1.0"]
    to_t = lambda x: (x - x0) / (x1 - x0) * 10
    # ----- robot tracks
    tr = _track(frames, scene)
    seen = {}
    for r in scene["robots"]:
        i = r["id"]
        ly = v["lane_y"][r["lane"]]
        t = tr[i]
        cnt0, cx0, cy0 = t[0]
        lay0 = render.layout(scene, 0)["robots"][i]
        off = (cx0 - lay0["x"], cy0 - lay0["y"])            # colour centroid vs body centre (eye / pattern)
        if cnt0 < 150:
            errs.append(f"robot {r['colour']}: only {cnt0} body pixels in frame 0")
        # final resting spot and the frame the robot first reaches it
        fx, fy = t[-1][1], t[-1][2]
        k = len(t) - 1
        while k > 0 and abs(t[k - 1][1] - fx) <= 3 and abs(t[k - 1][2] - fy) <= 3:
            k -= 1
        # toppled: the robot's body (not just its legs) fills the band just above its strip
        toppled = _low_fill(last, fx, r["size"], ly, r["colour"]) >= 0.08      # standing: 0.00, lying: >= 0.14 on the previews
        rec = {"toppled": bool(toppled), "start": to_t(cx0 - off[0])}
        if not toppled:
            rec["land_f"] = k
            rec["land"] = to_t(fx - off[0])
            # take-off: walking back from the landing, the robot is airborne (body above its standing height)
            # until the last frame it stood on the runway -- the take-off frame
            g = k - 1
            while g > 0 and abs(t[g][2] - cy0) > 2:
                g -= 1
            rec["jump_f"] = g
            rec["takeoff_px"] = to_t(t[g][1] - off[0])
        seen[i] = rec
        n_checks += 1
    # ----- footprints in the last frame
    fl_x = render.x_of(scene, scene["foul"])
    for r in scene["robots"]:
        ly = v["lane_y"][r["lane"]]
        sh = v["strip_h"]
        mid = ly + sh // 2
        rows = last[mid - 2:mid + 3].astype(np.int32)                 # footprint rows, clear of bodies and legs
        sx = render.x_of(scene, r["start"])
        white = (rows.min(axis=2) > 215).any(axis=0)
        dark = ((rows * np.array([0.299, 0.587, 0.114])).sum(axis=2) < 70).any(axis=0)
        xs = np.arange(rows.shape[1])
        white &= (xs > render.x_of(scene, 0) - 30) & (xs < fl_x - 6) & (np.abs(xs - sx) > 20)
        dark &= (xs > fl_x + 6) & (xs < render.x_of(scene, 20) + 14)
        feet = []
        for mask in (white, dark):
            run = None
            for x in range(len(mask)):
                if mask[x] and run is None:
                    run = x
                if (not mask[x] or x == len(mask) - 1) and run is not None:
                    end = x - 1 if not mask[x] else x
                    if end - run >= 12:
                        feet.append((run + end) / 2)
                    run = None
        feet.sort()
        seen[r["id"]]["feet"] = [to_t(x) for x in feet]
        n_checks += 1
    # positions must sit on gridlines when read off the ruler
    for i, rec in seen.items():
        for kname in ("land", "takeoff_px", "start"):
            if kname in rec and abs(rec[kname] - round(rec[kname])) > 0.25:
                errs.append(f"robot {i}: {kname} reads {rec[kname]:.2f} tenths, not on a gridline")
        for x in rec["feet"]:
            if abs(x - round(x)) > 0.25:
                errs.append(f"robot {i}: footprint reads {x:.2f} tenths, not on a gridline")
    R = lambda x: int(round(x))
    # ----- timer at take-off / landing frames
    x0b, y0b, x1b, y1b = scene["timer"]["box"]
    for rec in seen.values():
        for key in ("land_f", "jump_f"):
            if key not in rec:
                continue
            f = rec[key]
            got = frames[f].astype(np.float32)[y0b + 4:y1b - 4, x0b + 4:x1b - 4]
            tt = (f * 10) // 24
            err = {d: float(np.abs(got - _timer_patch(scene, tt + d)).mean()) for d in (-1, 0, 1) if tt + d >= 0}
            if min(err, key=err.get) != 0:
                errs.append(f"timer at frame {f} doesn't read {render.timer_text(f)}")
            n_checks += 1
    # ----- answers read off the frames vs stored answers
    fmt = lambda t: f"{t // 10}.{t % 10}"
    for q in s["questions"]:
        key, p = q["derivation_key"], q["params"]
        rec = seen.get(p.get("r"))
        got = None
        if key in ("hops_jump", "hops_foul"):
            got = str(len(rec["feet"]))
        elif key == "n_fouls":
            got = str(sum(1 for x in seen.values() if x["toppled"]))
        elif key == "takeoff":
            got = fmt(R(rec["feet"][-1]))
            if R(rec["takeoff_px"]) != R(rec["feet"][-1]):
                errs.append(f"{q['qid']}: robot left from {rec['takeoff_px']:.2f}, last footprint {rec['feet'][-1]:.2f}")
        elif key == "jump_length":
            got = fmt(R(rec["land"]) - R(rec["feet"][-1]))
        elif key == "score":
            got = fmt(R(rec["land"]))
        elif key == "longest_hop":
            pts = [R(rec["start"])] + [R(x) for x in rec["feet"]]
            got = fmt(max(b - a for a, b in zip(pts, pts[1:])))
        elif key == "start_mark":
            got = fmt(R(rec["start"]))
        elif key == "land_time":
            got = render.timer_text(rec["land_f"])
        elif key == "takeoff_time":
            got = render.timer_text(rec["jump_f"])
        elif key == "first_land":
            a, b = seen[p["a"]], seen[p["b"]]
            got = f"the {scene['robots'][p['a'] if a['land_f'] < b['land_f'] else p['b']]['colour']} robot"
        if got is not None:
            n_checks += 1
            if got != q["answer_text"]:
                errs.append(f"{q['qid']} {key}: frames say {got!r}, stored {q['answer_text']!r}")
    # ----- label / timer / robot overlap over the whole clip (layout geometry)
    boxes = [_label_box(lab) for lab in scene["labels"]]
    for a in range(len(boxes)):
        for b in range(a + 1, len(boxes)):
            A, B = boxes[a], boxes[b]
            if A[0] < B[2] and B[0] < A[2] and A[1] < B[3] and B[1] < A[3]:
                errs.append("two ruler labels overlap")
    tb = scene["timer"]["box"]
    for f in range(scene["n_frames"]):
        lay = render.layout(scene, f)
        for i, st in lay["robots"].items():
            e = st["size"] * 1.25 + render.LEG
            rb = (st["x"] - e, st["y"] - e, st["x"] + e, st["y"] + e)
            for A in boxes + [tuple(tb), render.RULER_BAND]:
                if A[0] < rb[2] and rb[0] < A[2] and A[1] < rb[3] and rb[1] < A[3]:
                    errs.append(f"frame {f}: robot {i} overlaps a label / the timer / the ruler band")
    n_checks += 1
    return errs, n_checks


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset-dir", required=True)
    args = ap.parse_args()
    root = Path(args.dataset_dir)
    total, bad = 0, 0
    for line in open(root / "annotations.jsonl", encoding="utf8"):
        s = json.loads(line)
        errs, n = check_clip(s, root)
        total += n
        bad += len(errs)
        for e in errs[:6]:
            print(f"{s['sample_id']}: {e}")
        print(f"{s['sample_id']} {s['params']['scenario']}: {n} checks, {len(errs)} failures", flush=True)
    print(f"frame checks: {total}, failures: {bad}")
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
