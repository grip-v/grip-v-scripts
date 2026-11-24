"""Frame-level check of rendered clips (decoded MP4 pixels vs. the stored answers).

    python frame_check.py --dataset-dir <dir>

For every clip it decodes the MP4 and, on every frame:
  * outlines: samples 48-64 points on each outline's drawn path (dash centres for
    dashed lines, dot centres for dotted ones) and calls a point "ink" when its
    3x3 colour is nearer the outline's colour than the clean background there.
    An outline is on screen when >= 55 % of its points are ink, off when
    <= 12 %; anything between is a failure. Its colour is the nearest palette
    colour to the median stroke-core colour (per point, the pixel of its 3x3
    neighbourhood farthest from the background);
  * rotation (every 2nd frame, outlines with a visible symmetry): the ink share
    at the drawn rotation must beat the ink share a quarter symmetry period
    either way, so the outline really sits at the angle the track says;
  * marks: at the centre and +-0.3 r sideways, the colour must be close to the
    mark's colour and nearer it than the background and the outline (present,
    >= 2 of 3 points) or not (absent);
  * builds a pixel track (visibility, colours, marks from the pixels; rotation
    and positions from the drawing state after the check above), re-derives
    every answer through answers.py on it, and re-derives it again from every
    k-th frame (k = floor(24 / min_fps)), every phase;
  * timer: template-matches the timer crop at every event frame, every
    question moment and the detected event of every timer-reading question
    (which must read the stored answer);
  * geometry: outlines >= 20 px across, lines >= 3 px, nothing under the timer
    plate, marks never touch each other or another outline.
There is no text besides the timer, so there are no labels to check.
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
import answers  # noqa: E402
import checks  # noqa: E402
import render  # noqa: E402
import sim  # noqa: E402


def decode(path: Path) -> list[np.ndarray]:
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


_TPL: dict = {}


def _timer_tpl(text: str) -> np.ndarray:
    if text not in _TPL:
        img = Image.new("RGB", (sim.W, sim.H), (0, 0, 0))
        d = ImageDraw.Draw(img)
        d.rounded_rectangle(render.TIMER_PLATE, radius=10, fill=render.TIMER_BG)
        d.text(render.TIMER_XY, text, fill=render.TIMER_INK, font=render.font(render.TIMER_SIZE))
        x0, y0, x1, y1 = render.TIMER_PLATE
        _TPL[text] = np.asarray(img, dtype=np.int16)[y0 + 4:y1 - 4, x0 + 4:x1 - 4]
    return _TPL[text]


def timer_reads(fr: np.ndarray, tn: int) -> bool:
    x0, y0, x1, y1 = render.TIMER_PLATE
    crop = fr[y0 + 4:y1 - 4, x0 + 4:x1 - 4]
    d_true = np.abs(crop - _timer_tpl(sim.fmt_reading(tn))).mean()
    d_other = min(np.abs(crop - _timer_tpl(sim.fmt_reading(t))).mean() for t in (tn - 1, tn + 1) if t >= 0)
    return d_true < 4 and d_true < 0.6 * d_other


def path_fracs(scene, k: int) -> list[float]:
    """Arc fractions of the sample points on ring k's drawn path."""
    r = scene["rings"][k]
    tot = sim.perimeter(r["shape"], r["aspect"])[2] * r["size"]
    w = r["width"]
    if r["line"] == "dashed":
        on, off = 3.2 * w + 6, 1.6 * w + 4
        n = max(1, round(tot / (on + off)))
        return [(i + 0.5 * on / (on + off)) / n for i in range(n)]
    if r["line"] == "dotted":
        n = max(3, round(tot / (2.7 * w + 4)))
        return [i / n for i in range(n)]
    return [i / 56 for i in range(56)]


def ring_points(scene, k: int, st_ring: dict, drot: float = 0.0, fracs=None) -> np.ndarray:
    r = scene["rings"][k]
    pts = []
    for u in fracs:
        ux, uy = sim.unit_point(r["shape"], r["aspect"], u)
        pts.append(sim.place(r, ux, uy, st_ring["x"], st_ring["y"], st_ring["rot"] + drot))
    return np.array(pts)


def patch(fr, pts):
    xs = np.clip(np.round(pts[:, 0]).astype(int), 1, sim.W - 2)
    ys = np.clip(np.round(pts[:, 1]).astype(int), 1, sim.H - 2)
    acc = np.zeros((len(pts), 3))
    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            acc += fr[ys + dy, xs + dx]
    return acc / 9, xs, ys


def core(fr, bg, pts):
    """Per point, the pixel of its 3x3 neighbourhood farthest from the background (the stroke's core)."""
    xs = np.clip(np.round(pts[:, 0]).astype(int), 1, sim.W - 2)
    ys = np.clip(np.round(pts[:, 1]).astype(int), 1, sim.H - 2)
    best = np.zeros((len(pts), 3))
    bd = np.full(len(pts), -1.0)
    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            c = fr[ys + dy, xs + dx]
            d = np.abs(c - bg[ys + dy, xs + dx]).sum(axis=1)
            take = d > bd
            best[take], bd[take] = c[take], d[take]
    return best


def ink(fr, bg, pts, rgb):
    c, xs, ys = patch(fr, pts)
    b = bg[ys, xs]
    dc = np.abs(c - np.array(rgb)).sum(axis=1)
    db = np.abs(c - b).sum(axis=1)
    return (dc < db) & (dc < 300), c


def band_points(scene, k: int, s: dict, st: dict):
    """Points just outside (and, for single lines, just inside) the drawn stroke of ring k,
    w/2 + 2.5 px off its path along the normal, away from its tab and from marks."""
    r = scene["rings"][k]
    n = 96
    us = [i / n for i in range(n)]
    pts = ring_points(scene, k, s, 0.0, us)
    nxt = ring_points(scene, k, s, 0.0, [u + 0.002 for u in us])
    prv = ring_points(scene, k, s, 0.0, [u - 0.002 for u in us])
    tang = nxt - prv
    nrm = np.stack([tang[:, 1], -tang[:, 0]], axis=1)
    nrm /= np.maximum(np.linalg.norm(nrm, axis=1, keepdims=True), 1e-9)
    radial = pts - np.array([s["x"], s["y"]])
    nrm *= np.sign((nrm * radial).sum(axis=1, keepdims=True) + 1e-9)        # outward
    off = r["width"] / 2 + 2.5
    # skip points near a corner (the stroke's joint bulges there)
    du = (r["width"] + 8) / (sim.perimeter(r["shape"], r["aspect"])[2] * r["size"])
    a1 = pts - ring_points(scene, k, s, 0.0, [u - du for u in us])
    a2 = ring_points(scene, k, s, 0.0, [u + du for u in us]) - pts
    cosang = (a1 * a2).sum(axis=1) / np.maximum(np.linalg.norm(a1, axis=1) * np.linalg.norm(a2, axis=1), 1e-9)
    smooth = cosang > math.cos(math.radians(12))
    sides = [1] if r["line"] == "double" else [1, -1]
    cand = np.concatenate([(pts + sd * off * nrm)[smooth] for sd in sides])
    keep = np.ones(len(cand), dtype=bool)
    avoid = [(m["x"], m["y"], m["r"] + 4) for m in st["marks"] if m["vis"]]
    if r["tab"]:
        tx, ty = render.tab_point(s)
        avoid.append((tx, ty, r["tab_r"] + 4))
    for x, y, rr in avoid:
        keep &= np.hypot(cand[:, 0] - x, cand[:, 1] - y) > rr
    return cand[keep]


def rot_check(scene, k, s, st, fr, bg, stats, errs, sid, f):
    """The drawn outline sits where the state says (position and rotation): for solid and
    double lines no ink in a band just outside / inside the stroke; for a tab, ink at the
    tab's tip and not at the point half a turn away."""
    r = scene["rings"][k]
    ok = True
    if r["tab"]:
        x, y = render.tab_point(s)
        d = math.hypot(x - s["x"], y - s["y"])
        out = r["tab_r"] - 2
        tips = []
        for a in (0.0, 180.0):
            th = math.atan2(y - s["y"], x - s["x"]) + math.radians(a)
            tips.append([s["x"] + (d + out) * math.cos(th), s["y"] + (d + out) * math.sin(th)])
        here, there = ink(fr, bg, np.array(tips), s["rgb"])[0]
        ok = bool(here) and not bool(there)
    if r["line"] in ("solid", "double"):
        band = band_points(scene, k, s, st)
        stray = ink(fr, bg, band, s["rgb"])[0].mean() if len(band) else 0.0
        ok = ok and stray <= 0.05
    elif not r["tab"]:
        return                      # dash / dot centres already pin the angle (they turn with the outline)
    stats["rot_checks"] += 1
    if not ok:
        stats["rot_off"] += 1
        errs.append(f"{sid}: frame {f} outline {k} is not where the state puts it (rotation / position)")


def ink_any(fr, bg, pts, rgb):
    """Point by point: is any pixel of its 3x3 neighbourhood nearer the colour than the background?"""
    xs = np.clip(np.round(pts[:, 0]).astype(int), 1, sim.W - 2)
    ys = np.clip(np.round(pts[:, 1]).astype(int), 1, sim.H - 2)
    hit = np.zeros(len(pts), dtype=bool)
    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            c = fr[ys + dy, xs + dx]
            dc = np.abs(c - np.array(rgb)).sum(axis=1)
            hit |= (dc < np.abs(c - bg[ys + dy, xs + dx]).sum(axis=1)) & (dc < 300)
    return hit


def pixel_states(scene, frames, stats, errs, sid):
    nf = scene["n_frames"]
    bg = np.asarray(render.background(scene), dtype=np.int16)
    pal = sim.THEMES[scene["style"]["theme"]]["colors"]
    fr_k = [path_fracs(scene, k) for k in range(len(scene["rings"]))]
    out = []
    for f in range(nf):
        st = render.state(scene, f)
        fr = frames[f]
        pst = json.loads(json.dumps(st))
        for k, s in enumerate(st["rings"]):
            pts = ring_points(scene, k, s, 0.0, fr_k[k])
            for m in st["marks"]:                     # marks sit on top of outlines: skip what they cover
                if m["vis"]:
                    far = np.hypot(pts[:, 0] - m["x"], pts[:, 1] - m["y"]) > m["r"] + 4
                    if far.sum() >= 8:
                        pts = pts[far]
            hit, cols = ink(fr, bg, pts, s["rgb"])
            share = hit.mean()
            stats["ring_frames"] += 1
            if share >= 0.55:
                vis = True
            elif share <= 0.12:
                vis = False
            else:
                errs.append(f"{sid}: frame {f} outline {k} is half-drawn ({share:.2f} ink)")
                vis = share > 0.33
            if vis != s["vis"]:
                stats["vis_off"] += 1
                errs.append(f"{sid}: frame {f} outline {k} visible={vis} in pixels, {s['vis']} in the state")
            pst["rings"][k]["vis"] = vis
            if vis:
                med = np.median(core(fr, bg, pts[hit]), axis=0)
                name = min(pal, key=lambda c: float(np.abs(med - np.array(sim.COLORS[c])).sum()))
                pst["rings"][k]["rgb"] = sim.COLORS[name]
                if sim.COLORS[name] != tuple(s["rgb"]):
                    stats["color_off"] += 1
                    errs.append(f"{sid}: frame {f} outline {k} reads {name} in pixels")
                if f % 2 == 0:
                    rot_check(scene, k, s, st, fr, bg, stats, errs, sid, f)
        for i, m in enumerate(st["marks"]):
            # centre and +-0.3 r sideways (inside every mark shape): a point is the mark's when its
            # colour is near the mark colour (< 110) and nearer it than the background and the outline
            pts = np.array([[m["x"] + dx * m["r"], m["y"]] for dx in (-0.3, 0.0, 0.3)])
            c, xs, ys = patch(fr, pts)
            ring_rgb = np.array(st["rings"][scene["marks"][i]["ring"]]["rgb"])
            dm = np.abs(c - np.array(m["rgb"])).sum(axis=1)
            own = (dm < 110) & (dm < np.abs(c - bg[ys, xs]).sum(axis=1)) & (dm < np.abs(c - ring_rgb).sum(axis=1))
            near = int(own.sum()) >= 2
            stats["mark_frames"] += 1
            if near != m["vis"]:
                stats["mark_off"] += 1
                errs.append(f"{sid}: frame {f} mark {i} present={near} in pixels")
            pst["marks"][i]["vis"] = bool(near)
        out.append(pst)
    return out


def check_clip(s: dict, root: Path, stats: dict) -> list[str]:
    scene, sid = s["params"], s["sample_id"]
    nf = scene["n_frames"]
    frames = decode(root / s["media_path"])
    if len(frames) != nf:
        return [f"{sid}: {len(frames)} frames decoded, expected {nf}"]
    if frames[0].shape[:2] != (sim.H, sim.W):
        return [f"{sid}: resolution {frames[0].shape[1]}x{frames[0].shape[0]}"]
    errs = []
    psts = pixel_states(scene, frames, stats, errs, sid)
    ptr = answers.Track(scene, range(nf), psts)
    for q in s["questions"]:
        stats["questions"] += 1
        try:
            val = answers.answer(q["derivation_key"], q["params"], scene, ptr)
        except (IndexError, ValueError, KeyError, StopIteration, TypeError) as e:
            val = f"error {e!r}"
        if str(val) != str(q["groundtruth"]):
            errs.append(f"{sid}/{q['qid']} {q['derivation_key']}: pixels give {val!r}, stored {q['groundtruth']!r}")
    bad = checks.sampling_failures(s, ptr)
    stats["sampled_q"] += len(s["questions"])
    errs += [f"pixels, {b}" for b in bad]
    # timer at every event, every named moment, and at every detected event a timer question names
    moments = {f for f, _ in sim.events(scene)}
    v = answers.View(scene, ptr)
    for q in s["questions"]:
        p = q["params"]
        for t in [p.get("a"), p.get("b"), p.get("t")]:
            if t is not None:
                f = sim.frame_of_reading(t)
                moments.update({f, f - 1})
        if q["derivation_key"] in checks.TIME_KEYS:
            key, x = q["derivation_key"], p.get("x")
            try:
                f = {"appear_time": lambda: v.appear(x), "change_time": lambda: v.changes(x)[0],
                     "mark_time": lambda: min(m["add"] for m in v.marks() if m["ring"] == x),
                     "recolor_time": lambda: v.colors(x)[1][0], "hit_time": lambda: v.hits()[0][0],
                     "hit_wall_time": lambda: next(ff for ff, w in v.hits() if w == p["wall"])}[key]()
            except (IndexError, ValueError, StopIteration, TypeError):
                errs.append(f"{sid}/{q['qid']}: the event a timer question names isn't found in the pixels")
                continue
            stats["timer_answers"] += 1
            if sim.timer_text(f) != q["answer_text"] or not timer_reads(frames[f], sim.tenths(f)):
                errs.append(f"{sid}/{q['qid']}: timer at the detected event doesn't read {q['answer_text']}")
    for f in sorted(moments):
        if 0 <= f < nf:
            stats["timer_checks"] += 1
            if not timer_reads(frames[f], sim.tenths(f)):
                errs.append(f"{sid}: timer pixels at frame {f} don't read {sim.timer_text(f)}")
    # geometry
    for k, r in enumerate(scene["rings"]):
        if 2 * sim.reach_out(r) < 20 or r["width"] < 3:
            errs.append(f"{sid}: outline {k} too small or too thin")
    for f in range(0, nf, 6):
        st = render.state(scene, f)
        x, y = st["nest"]
        if y - scene["R"] < render.TIMER_PLATE[3] + 8:
            errs.append(f"{sid}: the nest reaches the timer band at frame {f}")
        ms = [m for m in st["marks"] if m["vis"]]
        for i, a in enumerate(ms):
            for b in ms[i + 1:]:
                if math.dist((a["x"], a["y"]), (b["x"], b["y"])) < a["r"] + b["r"] + 2:
                    errs.append(f"{sid}: marks touch at frame {f}")
            for k, rs in enumerate(st["rings"]):
                if k == scene["marks"][a["id"]]["ring"] or not rs["vis"]:
                    continue
                pts = render.ring_points(rs)
                dmin = min(sim._seg_dist((p0[0] - a["x"], p0[1] - a["y"]), (p1[0] - a["x"], p1[1] - a["y"]))
                           for p0, p1 in zip(pts, pts[1:] + pts[:1]))
                if dmin < a["r"] + rs["width"] / 2 + 1:
                    errs.append(f"{sid}: a mark touches outline {k} at frame {f}")
    return errs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset-dir", required=True)
    args = ap.parse_args()
    root = Path(args.dataset_dir)
    stats = dict.fromkeys(("ring_frames", "vis_off", "color_off", "rot_checks", "rot_off", "mark_frames", "mark_off",
                           "questions", "sampled_q", "timer_checks", "timer_answers"), 0)
    errs, n = [], 0
    with open(root / "annotations.jsonl", encoding="utf8") as f:
        for line in f:
            s = json.loads(line)
            n += 1
            e = check_clip(s, root, stats)
            errs += e
            print(f"{s['sample_id']} {s['params']['scenario']}: {len(e)} problems", flush=True)
    print(f"frame check: {n} clips, {stats['ring_frames']} outline-frames read ({stats['vis_off']} visibility and "
          f"{stats['color_off']} colour disagreements), {stats['rot_checks']} rotation checks ({stats['rot_off']} off), "
          f"{stats['mark_frames']} mark-frames ({stats['mark_off']} off), {stats['questions']} questions re-derived "
          f"from pixels, {stats['sampled_q']} re-derived from every k-th frame, {stats['timer_checks']} timer "
          f"readings matched, {stats['timer_answers']} timer answers read at the detected event, {len(errs)} failures")
    for e in errs[:40]:
        print("  -", e)
    sys.exit(1 if errs else 0)


if __name__ == "__main__":
    main()
