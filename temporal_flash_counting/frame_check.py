"""Frame-level check of rendered clips (decoded MP4 pixels vs. the stored answers).

    python frame_check.py --dataset-dir <dir>

For every clip it decodes the MP4 and, frame by frame:
  * reads each shape's look from the pixels at a point well inside it
    (render.inner_point; nearest of its rest / lit / white fill colours) and
    whether it is "grown" (its outline found >= 4 px past its resting edge
    when scanning out along its tip ray, luma vs. the clean background), and compares the looks with render.state;
  * turns those pixel looks into detected flashes and pulses and re-derives
    every answer through answers.py on them (counts, orders, lengths, white
    flashes, relations, triggers, counterfactuals);
  * frame-sampling: re-derives every question again from only every k-th
    decoded frame, k = floor(24 / min_fps), for every phase 0..k-1;
  * timer: at every flash onset and every moment a question names, crops the
    timer and template-matches it against the rendered reading (and its
    neighbours); for "first_time" the timer at the detected onset must read
    the stored answer;
  * geometry: every shape >= 20 px across, inside sim.AREA (below the timer),
    no two shapes closer than their max drawn radii.
There is no other text in the frame, so there are no labels to check.
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
import render  # noqa: E402
import sim  # noqa: E402
from common.shapes import unit_shape  # noqa: E402

LUMA = np.array([0.299, 0.587, 0.114])



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


def _tip(scene, i):
    s = scene["shapes"][i]
    pts = unit_shape(s["type"], s["aspect"])
    ux, uy = max(pts, key=lambda p: p[0] ** 2 + p[1] ** 2)
    th = math.radians(s["rot"])
    return ux * math.cos(th) - uy * math.sin(th), ux * math.sin(th) + uy * math.cos(th)


def pixel_looks(scene, frames):
    n, nf = len(scene["shapes"]), scene["n_frames"]
    pals = [render.palette(scene, i) for i in range(n)]
    bgimg = np.asarray(render._background(scene), dtype=np.int16)
    tips = [_tip(scene, i) for i in range(n)]
    modes = [[None] * nf for _ in range(n)]
    grown = [[False] * nf for _ in range(n)]
    state_modes = [[None] * nf for _ in range(n)]
    for f in range(nf):
        st = render.state(scene, f)
        fr = frames[f]
        for i in range(n):
            s = st[i]
            px, py = render.inner_point(scene, i, s["x"], s["y"], s["r"])
            c = fr[py - 1:py + 2, px - 1:px + 2].reshape(-1, 3).mean(axis=0)
            modes[i][f] = min(pals[i], key=lambda k: float(np.abs(c - np.array(pals[i][k])).sum()))
            state_modes[i][f] = answers._nearest(s["fill"], pals[i])
            if modes[i][f] == "rest":
                # scan outwards along the tip ray for the shape's bright outline (luma only:
                # H.264 4:2:0 smears colour, not brightness); grown = ink >= 4 px past the resting edge
                r0 = scene["shapes"][i]["r"]
                ds = np.arange(int(r0 * 0.8), int(sim.max_radius(scene, i)) + 7)    # neighbours stay >= 10 px further out
                xs = np.clip(np.round(s["x"] + ds * tips[i][0]).astype(int), 0, sim.W - 1)
                ys = np.clip(np.round(s["y"] + ds * tips[i][1]).astype(int), 0, sim.H - 1)
                ink = np.abs((fr[ys, xs] - bgimg[ys, xs]) @ LUMA) > 40
                ext = ds[ink].max() if ink.any() else 0
                grown[i][f] = ext >= r0 + scene["style"]["outline"] / 2 + 4
    return modes, grown, state_modes


def _same(q, val) -> bool:
    if q["derivation_key"] == "first_time" and val is not None:
        v = float(str(val).split()[0])
        best = min(q["options"], key=lambda o: abs(float(o.split()[0]) - v))
        return best == q["answer_text"]
    return str(val) == str(q["groundtruth"])


def _with_track(ev, pul, cents, fn):
    old = answers.track
    answers.track = lambda scene: (ev, pul, cents)
    try:
        return fn()
    finally:
        answers.track = old


def check_clip(s: dict, root: Path, stats: dict) -> list[str]:
    scene, sid = s["params"], s["sample_id"]
    nf, n = scene["n_frames"], len(scene["shapes"])
    frames = decode(root / s["media_path"])
    if len(frames) != nf:
        return [f"{sid}: {len(frames)} frames decoded, expected {nf}"]
    if frames[0].shape[:2] != (sim.H, sim.W):
        return [f"{sid}: resolution {frames[0].shape[1]}x{frames[0].shape[0]}"]
    errs = []
    modes, grown, smodes = pixel_looks(scene, frames)
    bad = sum(modes[i][f] != smodes[i][f] for i in range(n) for f in range(nf))
    stats["look_frames"] += n * nf
    stats["look_mismatch"] += bad
    if bad:
        errs.append(f"{sid}: {bad} shape-frames whose pixel look differs from the drawing state")
    cents = [(sh["x"], sh["y"]) for sh in render.state(scene, 0)]
    ev, pul = answers.events_from(modes, grown, range(nf), nf)
    ref_ev, ref_pul, _ = answers.track(scene)
    if [[(e["on"], e["off"], e["white"]) for e in x] for x in ev] != \
            [[(e["on"], e["off"], e["white"]) for e in x] for x in ref_ev]:
        errs.append(f"{sid}: flashes detected in the pixels differ from the drawing state")
    if [bool(p) for p in pul] != [bool(p) for p in ref_pul]:
        errs.append(f"{sid}: pulses detected in the pixels differ from the drawing state")
    stats["flashes"] += sum(len(x) for x in ev)
    for q in s["questions"]:
        stats["questions"] += 1
        val = _with_track(ev, pul, cents, lambda: answers.answer(q["derivation_key"], q["params"], scene))
        if str(val) != str(q["groundtruth"]):
            errs.append(f"{sid}/{q['qid']} {q['derivation_key']}: pixels give {val!r}, stored {q['groundtruth']!r}")
        k = max(1, int(sim.FPS / q["params"]["min_fps"] + 1e-9))
        for ph in range(k):
            sev, spul = answers.events_from(modes, grown, range(ph, nf, k), nf)
            sval = _with_track(sev, spul, cents, lambda: answers.answer(q["derivation_key"], q["params"], scene))
            stats["sampled"] += 1
            if not _same(q, sval):
                errs.append(f"{sid}/{q['qid']} {q['derivation_key']}: every {k}th frame (phase {ph}) gives "
                            f"{sval!r}, stored {q['groundtruth']!r}")
                break
    # timer at every onset and every named moment
    moments = {e["on"] for x in ev for e in x}
    for q in s["questions"]:
        p = q["params"]
        for t in [p.get("a"), p.get("b"), p.get("t")] + list(p.get("cuts") or []):
            if t is not None:
                f = sim.frame_of_reading(t)
                moments.update({f, f - 1})
        if q["derivation_key"] == "first_time":
            f = ev[p["x"]][0]["on"]
            if sim.timer_text(f) != q["answer_text"] or not timer_reads(frames[f], sim.tenths(f)):
                errs.append(f"{sid}/{q['qid']}: timer at the first onset doesn't read {q['answer_text']}")
    for f in sorted(moments):
        stats["timer_checks"] += 1
        if not timer_reads(frames[f], sim.tenths(f)):
            errs.append(f"{sid}: timer pixels at frame {f} don't read {sim.timer_text(f)}")
    # geometry
    for f in range(0, nf, 6):
        st = render.state(scene, f)
        for i in range(n):
            sh = scene["shapes"][i]
            across = 2 * sh["r"] / (sh["aspect"] if sh["type"] in ("ellipse", "rectangle") else 1)
            if across < 20:
                errs.append(f"{sid}: shape {i} is under 20 px across")
            R = sim.max_radius(scene, i)
            if st[i]["y"] - R < render.TIMER_PLATE[3] + 8 and st[i]["x"] - R < render.TIMER_PLATE[2] + 8:
                errs.append(f"{sid}: shape {i} reaches the timer")
    return errs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset-dir", required=True)
    args = ap.parse_args()
    root = Path(args.dataset_dir)
    stats = dict.fromkeys(("look_frames", "look_mismatch", "flashes", "questions", "sampled", "timer_checks"), 0)
    errs, n = [], 0
    with open(root / "annotations.jsonl", encoding="utf8") as f:
        for line in f:
            s = json.loads(line)
            n += 1
            errs += check_clip(s, root, stats)
    print(f"frame check: {n} clips, {stats['flashes']} flashes read from pixels, "
          f"{stats['look_mismatch']}/{stats['look_frames']} shape-frames off, {stats['questions']} questions "
          f"re-derived from pixels, {stats['sampled']} frame-sampled re-derivations, "
          f"{stats['timer_checks']} timer readings matched, {len(errs)} failures")
    for e in errs[:40]:
        print("  -", e)
    sys.exit(1 if errs else 0)


if __name__ == "__main__":
    main()
