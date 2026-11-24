"""Frame-level check of rendered clips against the scene data.

    python frame_check.py --dataset-dir <dir> [--report frame_check.txt] [--jobs 4]

For every clip it decodes the MP4 (OpenCV) and checks:
  * clock: template-matches the timer panel against all readings 0.0-9.9 s
    on every 6th frame, on every event frame and on every frame a question
    names by its clock reading;
  * no other text: the clip's label list is empty;
  * objects: on every 3rd frame and on every event frame (out of the tunnel,
    at the gate, the frame before and the frame of coming to rest, the last
    frame) every object that is fully out of the tunnel is found by its
    colour around the centre the track gives (enough pixels of its colour,
    their centroid within 3 px); wholly hidden objects show none of their
    colour at the tunnel mouth;
  * rest: on the frame an object comes to rest it sits on its slot (2.5 px),
    on the frame before it is still at least v - 2.5 px away;
  * answers read off the frames: clock answers on the event frame; objects
    resting in a bin counted as occupied slots (an object sits still on the
    slot over 3 frames) on every frame showing the clock reading; objects on
    the belt counted as colour blobs between the tunnel and the gate; the
    objects resting in each bin at the end; the order objects come to rest.
Prints a summary and exits 1 on any failure.
"""
from __future__ import annotations
import argparse
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
import questions  # noqa: E402
from common.video5 import font  # noqa: E402

COL_TOL = 60          # RGB distance for "this object's colour"


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
    crop = fr[tb[1] + 4:tb[3] - 4, tb[0] + 4:tb[2] - 4]
    tm = _timer_templates()
    return min(tm, key=lambda k: np.abs(crop - tm[k]).mean())


def colour_mask(fr, rgb, box):
    x0, y0, x1, y1 = (int(max(0, v)) for v in box)
    x1, y1 = min(x1, sim.W), min(y1, sim.H)
    crop = fr[y0:y1, x0:x1]
    m = np.abs(crop - np.array(rgb, dtype=np.int16)).sum(axis=2) < COL_TOL
    return m, x0, y0


def find(fr, rgb, cx, cy, rad):
    """(pixel count, centroid) of `rgb` within `rad` px of (cx, cy)."""
    m, x0, y0 = colour_mask(fr, rgb, (cx - rad, cy - rad, cx + rad + 1, cy + rad + 1))
    ys, xs = np.nonzero(m)
    xs = xs + x0
    ys = ys + y0
    keep = (xs - cx) ** 2 + (ys - cy) ** 2 <= rad * rad
    xs, ys = xs[keep], ys[keep]
    if len(xs) == 0:
        return 0, None
    return len(xs), (float(xs.mean()), float(ys.mean()))


_FOOT = {}


def footprint(o, outline):
    """(pixels of the object's own colour, their centroid's offset from the centre)
    in the drawn sprite at 1x: striped triangles and stars are not point-symmetric."""
    key = (o["colour"], o["shape"], o["pattern"], o["r"], outline)
    if key not in _FOOT:
        sp = render.sprite(*key).reduce(render.SS)
        a = np.asarray(sp, dtype=np.int16)
        m = (a[:, :, 3] > 200) & (np.abs(a[:, :, :3] - np.array(sim.COLOURS[o["colour"]])).sum(axis=2) < COL_TOL)
        ys, xs = np.nonzero(m)
        c = (sp.size[0] - 1) / 2
        _FOOT[key] = (len(xs), float(xs.mean()) - c, float(ys.mean()) - c)
    return _FOOT[key]


def at(fr, o, x, y, tol=2.5, outline=3):
    cnt0, dx, dy = footprint(o, outline)
    cnt, c = find(fr, sim.COLOURS[o["colour"]], x + dx, y + dy, sim.ext(o) + 2)
    ok = 0.6 * cnt0 <= cnt <= 1.4 * cnt0 and c is not None and math.hypot(c[0] - x - dx, c[1] - y - dy) <= tol
    return ok, cnt, c


def resting_on(frames, f, o_list, slot, outline):
    """Some object of the clip sits still on `slot` over frames f-1, f, f+1."""
    for o in o_list:
        ok = True
        for g in (f - 1, f, f + 1):
            if not 0 <= g < len(frames):
                continue
            good, cnt, c = at(frames[g], o, slot[0], slot[1], outline=outline)
            if not good:
                ok = False
                break
        if ok:
            return True
    return False


def belt_blobs(fr, scene):
    """Objects on the belt: colour blobs between the tunnel mouth and the gate."""
    import cv2
    x0, x1 = sorted((scene["mouth"], scene["gate"][0]))
    y = scene["belt_y"]
    h = scene["half"]
    box = (x0, y - h, x1, y + h)
    n = 0
    seen = {o["colour"] for o in scene["objects"]}
    for c in sorted(seen):
        m, bx, by = colour_mask(fr, sim.COLOURS[c], box)
        m = cv2.dilate(m.astype(np.uint8), np.ones((9, 9), np.uint8))
        k, lab, stats, _ = cv2.connectedComponentsWithStats(m)
        r = min(o["r"] for o in scene["objects"] if o["colour"] == c)
        n += sum(1 for i in range(1, k) if stats[i, cv2.CC_STAT_AREA] >= 0.4 * math.pi * r * r)
    return n


def check_clip(scene, qs, frames) -> list[str]:
    errs = []
    n = scene["n_frames"]
    if len(frames) != n:
        return [f"decoded {len(frames)} frames, expected {n}"]
    if scene["labels"]:
        errs.append("labels present")
    F = questions.Facts(scene)
    tr = render.track(scene)
    objs = scene["objects"]
    v = scene["v2"] / 2
    ol = scene["style"]["outline"]
    ev = set()
    for i, o in enumerate(objs):
        ev |= {F.e[i], F.e[i] + F.fz[i]}
        if not o["pending"]:
            ev |= {F.gate[i], F.land[i] - 1, F.land[i]}
    ev = {f for f in ev if 0 <= f < n} | {n - 1}
    named = set()
    for q in qs:
        p = q["params"]
        for kk in ("k", "k1", "k2"):
            if kk in p and isinstance(p[kk], int) and q["derivation_key"] in ("count_at", "belt_count_at",
                                                                             "landed_between"):
                lo, hi = sim.frames_showing(p[kk])
                named |= set(range(lo, hi + 1))
    # clock
    for f in sorted(set(range(0, n, 6)) | ev | named):
        got = read_timer(frames[f])
        if got != sim.tenths(f):
            errs.append(f"frame {f}: clock reads {sim.clock_text(got)}, should be {sim.clock_text(sim.tenths(f))}")
    # objects where the track puts them
    for f in sorted(set(range(0, n, 3)) | ev):
        for j, o in enumerate(objs):
            x, y = tr["pos"][f, j]
            if np.isnan(x):
                # wholly inside the tunnel: none of its colour just inside the mouth
                if F.e[j] - f <= 2 * F.fz[j] + 4:
                    mx = scene["mouth"] - scene["dir"] * (sim.ext(o) + 4)
                    cnt, _ = find(frames[f], sim.COLOURS[o["colour"]], mx, scene["belt_y"], sim.ext(o))
                    if cnt > 0.05 * footprint(o, ol)[0]:
                        errs.append(f"frame {f}: {sim.name(o)} shows through the tunnel")
                continue
            if (x - scene["mouth"]) * scene["dir"] < sim.ext(o) + 2:
                continue
            good, cnt, c = at(frames[f], o, x, y, outline=ol)
            if not good:
                errs.append(f"frame {f}: {sim.name(o)} not found at ({x:.0f}, {y:.0f}) ({cnt} px, {c})")
    # coming to rest
    for j, o in enumerate(objs):
        if o["pending"]:
            continue
        sx, sy = scene["bins"][o["bin"]]["slots"][o["slot"]]
        good, _, c = at(frames[F.land[j]], o, sx, sy, outline=ol)
        if not good:
            errs.append(f"{sim.name(o)} not on its slot at frame {F.land[j]}")
        _, dx, dy = footprint(o, ol)
        _, _, c = at(frames[F.land[j] - 1], o, sx, sy, tol=99, outline=ol)
        if c is None or math.hypot(c[0] - sx - dx, c[1] - sy - dy) < v - 2.5:
            errs.append(f"{sim.name(o)} already still on frame {F.land[j] - 1}")
    # answers read off the frames
    def occupied(b, f):
        return sum(1 for s in scene["bins"][b]["slots"] if resting_on(frames, f, objs, s, ol))
    for q in qs:
        key, p, gt = q["derivation_key"], q["params"], q["groundtruth"]
        if key in ("land_time", "gate_time"):
            j = F.nm.index(p["x"])
            fe = F.land[j] if key == "land_time" else F.gate[j]
            if sim.clock_text(read_timer(frames[fe])) != gt:
                errs.append(f"{q['qid']} {key}: clock on frame {fe} reads {sim.clock_text(read_timer(frames[fe]))}")
        if key == "count_at":
            lo, hi = sim.frames_showing(p["k"])
            for f in range(lo, hi + 1):
                if occupied(p["bin"], f) != gt:
                    errs.append(f"{q['qid']} count_at: {occupied(p['bin'], f)} resting on frame {f}, answer {gt}")
        if key == "count_end" and occupied(p["bin"], n - 2) != gt:
            errs.append(f"{q['qid']} count_end: {occupied(p['bin'], n - 2)} resting at the end, answer {gt}")
        if key == "belt_count_at":
            lo, hi = sim.frames_showing(p["k"])
            for f in range(lo, hi + 1):
                if belt_blobs(frames[f], scene) != gt:
                    errs.append(f"{q['qid']} belt_count_at: {belt_blobs(frames[f], scene)} blobs on frame {f}, "
                                f"answer {gt}")
        if key == "on_belt_end":
            j = F.nm.index(gt)
            x, y = tr["pos"][n - 1, j]
            if not at(frames[n - 1], objs[j], x, y, outline=ol)[0] or not (
                    min(scene["mouth"], scene["gate"][0]) < x < max(scene["mouth"], scene["gate"][0])):
                errs.append(f"{q['qid']} on_belt_end: {gt} not seen on the belt at the end")
        if key == "in_bin_end":
            j = F.nm.index(gt)
            s = scene["bins"][p["bin"]]["slots"][objs[j]["slot"]]
            if not at(frames[n - 1], objs[j], s[0], s[1], outline=ol)[0]:
                errs.append(f"{q['qid']} in_bin_end: {gt} not seen in {p['bin']} at the end")
        if key == "kth_land":
            # read the order of coming to rest off the frames: first frame each object sits still on its
            # slot (an object of the same colour sliding past to a farther slot does not count)
            seen = []
            for j in F.done:
                s = scene["bins"][objs[j]["bin"]]["slots"][objs[j]["slot"]]
                f0 = next((f for f in range(n - 2) if all(at(frames[g], objs[j], s[0], s[1], outline=ol)[0]
                                                          for g in (f, f + 1, f + 2))), None)
                if f0 is None:
                    errs.append(f"{q['qid']} kth_land: {F.nm[j]} never seen on its slot")
                    break
                seen.append((f0, j))
            seen.sort()
            if len(seen) == len(F.done) and F.nm[seen[p["k"] - 1][1]] != gt:
                errs.append(f"{q['qid']} kth_land: decoded order gives {F.nm[seen[p['k'] - 1][1]]}")
    return errs


def _one(arg):
    root, line = arg
    s = json.loads(line)
    frames = frames_of(Path(root) / s["media_path"])
    try:
        errs = check_clip(s["params"], s["questions"], frames)
    except Exception as e:  # report, don't crash the pool
        errs = [f"checker error {e!r}"]
    return s["sample_id"], s["params"]["scenario"], errs


def main():
    from multiprocessing import Pool
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset-dir", required=True)
    ap.add_argument("--report", default="frame_check.txt")
    ap.add_argument("--jobs", type=int, default=4)
    args = ap.parse_args()
    root = Path(args.dataset_dir)
    rows = open(root / "annotations.jsonl", encoding="utf8").read().splitlines()
    lines, total = [], 0
    with Pool(args.jobs) as pool:
        for sid, sc, errs in pool.imap(_one, [(str(root), r) for r in rows]):
            total += len(errs)
            lines += [f"{sid}: {e}" for e in errs]
            print(f"{sid} ({sc}): {len(errs)} failures", flush=True)
    lines.append(f"RESULT: {len(rows)} clips, {total} frame-check failures")
    (root / args.report).write_text("\n".join(lines) + "\n")
    print(lines[-1])
    for e in lines[:-1][:20]:
        print("  -", e)
    sys.exit(1 if total else 0)


if __name__ == "__main__":
    main()
