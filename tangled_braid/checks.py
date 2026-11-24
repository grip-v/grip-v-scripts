"""Extra checks for tangled_braid, driven from run.py:

  python run.py stats --n 5000 [--seed 0]        diversity + ladder stats over dry-run scenes
  python run.py tamper --dataset-dir DIR [--k 3] corrupt k answers, count what the validator flags
  python run.py framecheck --dataset-dir DIR     read the rendered MP4s back and compare with the scene
  python run.py framehash --seed S               sha256 of the raw rendered frames of one seed

framecheck, per clip, from the decoded frames only:
  * reads the clock (template match against every reading 0.0-9.9 s) on every
    frame where strands meet, on every frame showing a clock reading that a
    question names, and checks it equals the frame's reading;
  * reads the strand order on the scanline at the loom bar (each pixel
    classified to the nearest strand colour) in the middle of every still
    stretch; it must equal the scene's order there;
  * finds each crossing's meeting frame as the frame where the over strand's
    blob on the bar scanline is closest to the midpoint between the two
    places, and the over strand as the colour that fills a patch at the
    meeting point three frames later (the halo hides the under strand there);
  * builds a measurement (start order, crossings with frame / place / over
    strand) purely from those pixel reads and re-derives all 10 answers of
    the clip from it with answers.py: every one must equal the stored answer;
  * checks the timer box and the braid never overlap (layout) and that no
    strand colour shows inside the timer box.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import random
import statistics
import sys
from collections import Counter
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))
import answers  # noqa: E402
import questions  # noqa: E402
import render  # noqa: E402
import sim  # noqa: E402
from common import ladder5 as L5  # noqa: E402
from common.video5 import build_sample, _roundtrip, font, hex_rgb, run_validate  # noqa: E402

DATASET = "tangled_braid"


# ----------------------------------------------------------------- stats
def stats(argv):
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=5000)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args(argv)
    import warnings
    warnings.simplefilter("error")
    sigs, scen, pairs, letters = set(), Counter(), set(), Counter()
    levels, fps, keys, motion = Counter(), Counter(), Counter(), Counter()
    knobs = {k: [] for k in ("strands", "crossings", "steps", "duration_s", "gap_px", "L_px", "speed", "c_base",
                             "window_px")}
    cats = {k: Counter() for k in ("orient_grow", "bg", "pattern", "loom", "curve", "style", "thickness",
                                   "colour", "halo")}
    bad_ladder = bad_check = mism = 0
    sim.STATS.update(builds=0, tries=0)
    per_level = {lv: Counter() for lv in range(1, 6)}
    n_q = 0
    for i in range(a.n):
        s = _roundtrip(build_sample(DATASET, sim, questions, a.seed + i))
        sc = s["params"]
        sigs.add(sim.signature(sc))
        scen[sc["scenario"]] += 1
        _, ev = sim.replay(sc["start"], sc["steps"])
        knobs["strands"].append(len(sc["start"]))
        knobs["crossings"].append(len(ev))
        knobs["steps"].append(len(sc["steps"]))
        knobs["duration_s"].append(sc["duration"])
        knobs["gap_px"].append(sc["layout"]["gap"])
        knobs["L_px"].append(sc["timing"]["L"])
        knobs["speed"].append(round(sc["timing"]["speed"], 1))
        knobs["c_base"].append(sc["timing"]["c_base"])
        knobs["window_px"].append(sim.window(sc))
        lay, st = sc["layout"], sc["style"]
        cats["orient_grow"][f"{lay['orient']}{lay['grow']:+d}"] += 1
        for k in ("bg", "pattern", "loom", "curve", "halo"):
            cats[k][st[k]] += 1
        for x in sc["strands"]:
            cats["style"][x["style"]] += 1
            cats["thickness"][x["thickness"]] += 1
            cats["colour"][x["color_name"]] += 1
        if sim.check(sc):
            bad_check += 1
        if L5.ladder_errors(s["questions"], "ten"):
            bad_ladder += 1
        cnt = Counter(q["difficulty_level"] for q in s["questions"])
        levels[tuple(cnt[lv] for lv in range(1, 6))] += 1
        for lv in range(1, 6):
            per_level[lv][cnt[lv]] += 1
        for q in s["questions"]:
            n_q += 1
            pairs.add((q["question"], q["answer_text"]))
            letters[q["answer"]] += 1
            fps[q["params"]["min_fps"]] += 1
            keys[q["derivation_key"]] += 1
            motion[(q["difficulty_level"] >= 2, q["requires_motion"])] += 1
            if str(answers.answer(q["derivation_key"], q["params"], sc)) != str(q["groundtruth"]):
                mism += 1
    out = [f"scenes: {a.n}  unique signatures: {len(sigs)} ({len(sigs) / a.n:.2%})",
           f"builds {sim.STATS['builds']}, tries {sim.STATS['tries']} "
           f"(retry rate {(sim.STATS['tries'] - sim.STATS['builds']) / max(1, sim.STATS['tries']):.1%})",
           "scenario mix: " + ", ".join(f"{k} {v / a.n:.1%}" for k, v in sorted(scen.items()))]
    for k, v in knobs.items():
        c = Counter(v)
        if len(c) <= 12:
            out.append(f"  {k}: " + ", ".join(f"{x}:{c[x]}" for x in sorted(c)))
        else:
            out.append(f"  {k}: min {min(v)} median {statistics.median(v)} max {max(v)} ({len(c)} distinct)")
    for k, c in cats.items():
        out.append(f"  {k}: " + ", ".join(f"{x}:{c[x]}" for x in sorted(c, key=str)))
    out.append(f"questions: {n_q}  distinct (question, answer) pairs: {len(pairs)}")
    out.append("answer letters: " + ", ".join(f"{k} {v / n_q:.2%}" for k, v in sorted(letters.items())))
    out.append("level counts (top 6): " + ", ".join(f"{k}:{v}" for k, v in levels.most_common(6)))
    for lv in range(1, 6):
        out.append(f"  L{lv} per clip: " + ", ".join(f"{k}:{v}" for k, v in sorted(per_level[lv].items())))
    out.append("min_fps: " + ", ".join(f"{k}:{v}" for k, v in sorted(fps.items())))
    for t in (1, 2, 4):
        out.append(f"  answerable at <= {t} fps: {sum(v for k, v in fps.items() if k <= t) / n_q:.1%}")
    l25 = motion[(True, True)] + motion[(True, False)]
    out.append(f"L2-L5 requiring motion: {motion[(True, True)] / max(1, l25):.1%}")
    out.append(f"derivation keys used: {len(keys)}: " + ", ".join(f"{k}:{v}" for k, v in sorted(keys.items())))
    out.append(f"check failures {bad_check}, ladder errors {bad_ladder}, answer mismatches {mism}")
    print("\n".join(out))
    return bad_check + bad_ladder + mism + (len(sigs) != a.n)


# ----------------------------------------------------------------- tamper
def tamper(argv):
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset-dir", required=True)
    ap.add_argument("--k", type=int, default=3)
    a = ap.parse_args(argv)
    root = Path(a.dataset_dir)
    rows = [json.loads(x) for x in (root / "annotations.jsonl").read_text().splitlines()]
    rng = random.Random(7)
    hit = rng.sample(range(len(rows)), a.k)
    for r in hit:
        q = rows[r]["questions"][rng.randrange(10)]
        wrong = next(o for o in q["options"] if o != q["answer_text"])
        q["groundtruth"], q["answer_text"], q["answer"] = wrong, wrong, "ABCD"[q["options"].index(wrong)]
    (root / "annotations_tampered.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    try:
        run_validate(DATASET, HERE, sim, questions, answers, render,
                     ["--dataset-dir", str(root), "--ann", "annotations_tampered.jsonl",
                      "--report", "tamper_report.txt", "--skip-video"])
    except SystemExit:
        pass
    rep = (root / "tamper_report.txt").read_text()
    flagged = {ln.split(" (")[0].strip(" -") for ln in rep.splitlines() if "independent answer" in ln}
    print(f"tampered {a.k} answers, validator flagged {len(flagged)}: {sorted(flagged)}")
    return 0 if len(flagged) == a.k else 1


# ----------------------------------------------------------------- frame reading
def read_frames(path) -> list:
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


def _timer_templates(scene):
    key = (tuple(scene["timer"]["box"]), scene["timer"]["size"])
    if key not in _TIMER:
        tb = scene["timer"]["box"]
        out = {}
        for k in range(100):
            img = Image.new("RGB", (sim.W, sim.H), (0, 0, 0))
            d = ImageDraw.Draw(img)
            d.rounded_rectangle(tb, radius=10, fill=render.TIMER_BG, outline=render.TIMER_EDGE, width=2)
            d.text(((tb[0] + tb[2]) / 2, (tb[1] + tb[3]) / 2), sim.clock_text(k), fill=render.TIMER_FG,
                   font=font(scene["timer"]["size"]), anchor="mm")
            out[k] = np.asarray(img, dtype=np.int16)[tb[1] + 4:tb[3] - 4, tb[0] + 4:tb[2] - 4]
        _TIMER[key] = out
    return _TIMER[key]


def read_timer(scene, fr) -> int:
    tb = scene["timer"]["box"]
    crop = fr[tb[1] + 4:tb[3] - 4, tb[0] + 4:tb[2] - 4]
    tm = _timer_templates(scene)
    return min(tm, key=lambda k: np.abs(crop - tm[k]).mean())


def _palette(scene, bands_are_strands=False):
    """(colours, labels): the strand colours, plus the colours that belong to
    no strand (background, ink, bar, white, the dark bands of banded strands)."""
    cols, labs = [], []
    for s in scene["strands"]:
        cols.append(hex_rgb(s["color"]))
        labs.append(s["name"])
    bg = hex_rgb(scene["style"]["bg_hex"])
    for v in [bg, render._mix(bg, (0, 0, 0), 0.08), (255, 255, 255), render.INK,
              hex_rgb(sim.LOOMS[scene["style"]["loom"]])] + \
            ([] if bands_are_strands else
             [render.dark(hex_rgb(s["color"])) for s in scene["strands"] if s["style"] == "banded"]):
        cols.append(v)
        labs.append(None)
    if bands_are_strands:              # inside a halo only the over strand is drawn: its bands count
        for s in scene["strands"]:
            if s["style"] == "banded":
                cols.append(render.dark(hex_rgb(s["color"])))
                labs.append(s["name"])
    return np.array(cols, dtype=np.int32), labs


def classify(px, pal, tol=45) -> list:
    cols, labs = pal
    d = ((px[:, None, :].astype(np.int32) - cols[None, :, :]) ** 2).sum(axis=2)
    j = d.argmin(axis=1)
    return [labs[k] if d[i, k] < tol ** 2 else None for i, k in enumerate(j)]


def scanline(scene, fr, pal, offsets=(1,)):
    """Blobs of strand colour on the line(s) just inside the loom bar:
    list of (name, centre u). With several lines each pixel column takes the
    label most lines agree on (dots and beads don't split a strand). Runs
    under 3 px (anti-aliased edges) are noise."""
    lay = scene["layout"]
    cols = sim.columns(scene)
    lo, hi = int(cols[0] - lay["gap"] / 2), int(cols[-1] + lay["gap"] / 2)
    labs = []
    for off in offsets:
        a = int(round(lay["loom"] - lay["grow"] * off))
        px = fr[a, lo:hi] if lay["orient"] == "v" else fr[lo:hi, a]
        labs.append(classify(px, pal))
    if len(labs) == 1:
        lab = labs[0]
    else:
        lab = []
        for col in zip(*labs):
            c = Counter(x for x in col if x is not None).most_common(1)
            lab.append(c[0][0] if c and c[0][1] >= 2 else None)
    runs = []                                  # [name, first, last]
    for i, n in enumerate(lab):
        if runs and runs[-1][0] == n and runs[-1][2] == i - 1:
            runs[-1][2] = i
        else:
            runs.append([n, i, i])
    runs = [r for r in runs if r[0] is not None and r[2] - r[1] >= 2]
    blobs = []
    for n, f, l in runs:
        if blobs and blobs[-1][0] == n and f - blobs[-1][2] <= 0.6 * lay["gap"]:
            blobs[-1][2] = l
        else:
            blobs.append([n, f, l])
    return [(n, lo + (f + l) / 2) for n, f, l in blobs]


def patch_owner(scene, fr, pal, x, y, r):
    x0, x1, y0, y1 = int(x - r), int(x + r) + 1, int(y - r), int(y + r) + 1
    px = fr[max(0, y0):y1, max(0, x0):x1].reshape(-1, 3)
    c = Counter(n for n in classify(px, pal) if n is not None).most_common(1)
    return c[0][0] if c and c[0][1] >= 5 else None


def pixel_measure(scene, frames, errs) -> dict:
    """answers.measure()-style dict built from decoded frames only."""
    pal = _palette(scene)
    pal_band = _palette(scene, bands_are_strands=True)
    spans = sim.quiet_spans(scene)
    cols = sim.columns(scene)
    lay = scene["layout"]
    read = []
    for lo, hi in spans:
        bl = scanline(scene, frames[(lo + hi) // 2], pal, offsets=tuple(range(1, 32, 3)))
        read.append([n for n, _ in bl])
    exp_orders, exp_ev = sim.replay(scene["start"], scene["steps"])
    for j, (o, e) in enumerate(zip(read, exp_orders)):
        if o != e:
            errs.append(f"still stretch {j}: order read {[x.split()[0] for x in o]} != scene "
                        f"{[x.split()[0] for x in e]}")
    events = []
    G = render._geom(render._key(scene))
    for k, st in enumerate(scene["steps"]):
        before, after = read[k], read[k + 1]
        if len(before) != len(after) or len(before) != len(scene["start"]):
            continue
        places = [p for p in range(len(before) - 1) if before[p] == after[p + 1] and before[p + 1] == after[p]
                  and before[p] != after[p]]
        for p in places:
            lo_n, hi_n = before[p], before[p + 1]
            mid = (cols[p] + cols[p + 1]) / 2
            # over strand: colour at the meeting point 3 frames after the nominal meet
            f = st["m"] + 3
            s_mid = st["S"] + scene["timing"]["L"] / 2
            along = lay["loom"] - lay["grow"] * (sim.head_s(scene, f) - s_mid)
            x, y = (mid, along) if lay["orient"] == "v" else (along, mid)
            r = 3
            over = patch_owner(scene, frames[f], pal_band, x, y, r)
            # meeting frame: where a straight line fitted to the over strand's blob
            # centre on the bar scanline (frames near the meeting) reaches the midpoint
            pts = []
            for g in range(st["a"], st["a"] + st["c"] + 1):
                us = [u for n, u in scanline(scene, frames[g], pal_band) if n == over]
                if us and abs(us[0] - mid) <= 0.35 * lay["gap"]:
                    pts.append((g, us[0]))
            bf = None
            if len(pts) >= 2:
                gs, us = np.array(pts, dtype=float).T
                slope, icpt = np.polyfit(gs, us, 1)
                keep = np.abs(us - (slope * gs + icpt)) <= 4      # drop misreads (bands, beads) and refit
                if keep.sum() >= 2 and not keep.all():
                    slope, icpt = np.polyfit(gs[keep], us[keep], 1)
                if abs(slope) > 1e-6:
                    bf = int(round((mid - icpt) / slope))
            if bf != st["m"]:
                errs.append(f"step {k}: meeting frame read {bf} != {st['m']}")
            want = next((e["over"] for e in exp_ev if e["step"] == k and e["i"] == p), None)
            if over != want:
                errs.append(f"step {k}: over strand read {over} != scene {want}")
            events.append({"f": bf if bf is not None else st["m"], "place": p, "low": lo_n, "high": hi_n,
                           "over": over, "under": hi_n if over == lo_n else lo_n})
        if sorted(places) != sorted(mv["i"] for mv in st["moves"]):
            errs.append(f"step {k}: swapped places read {places} != {[mv['i'] for mv in st['moves']]}")
    order = list(read[0])
    orders_at = []
    by_f = {}
    for e in events:
        by_f.setdefault(e["f"], []).append(e)
    for f in range(len(frames)):
        for e in by_f.get(f, []):
            if e["place"] + 1 >= len(order):
                continue
            order[e["place"]], order[e["place"] + 1] = order[e["place"] + 1], order[e["place"]]
        orders_at.append(list(order))
    return {"start": read[0], "events": events, "orders_at": orders_at,
            "color": {s["name"]: s["color_name"] for s in scene["strands"]}, "orient": lay["orient"],
            "nf": len(frames)}


def check_clip(scene, qs, frames) -> list:
    errs = []
    if len(frames) != scene["n_frames"]:
        return [f"decoded {len(frames)} frames, expected {scene['n_frames']}"]
    # clock at every meeting frame and every moment a question names
    want = {st["m"] for st in scene["steps"]}
    for q in qs:
        for k in ("t", "t0", "t1"):
            if k in q["params"]:
                lo, hi = sim.frames_showing(q["params"][k])
                want |= set(range(lo, hi + 1))
    for f in sorted(want):
        if read_timer(scene, frames[f]) != sim.timer_tenths(f):
            errs.append(f"frame {f}: clock misread")
    M = pixel_measure(scene, frames, errs)
    for q in qs:
        try:
            got = answers.answer(q["derivation_key"], q["params"], scene, M=M)
        except Exception as e:  # noqa: BLE001
            got = f"error {e!r}"
        if str(got) != str(q["groundtruth"]):
            errs.append(f"{q['qid']} ({q['derivation_key']}): answer from pixels {got!r} != {q['groundtruth']!r}")
        if q["derivation_key"] == "cross_time":
            e = next((e for e in M["events"] if {e["low"], e["high"]} == {q["params"]["a"], q["params"]["b"]}), None)
            if e is None or sim.clock_text(read_timer(scene, frames[e["f"]])) != q["groundtruth"]:
                errs.append(f"{q['qid']}: clock read at the meeting frame != {q['groundtruth']}")
    # nothing of a strand's colour inside the timer box; layout rules
    tb = scene["timer"]["box"]
    pal = _palette(scene)
    for f in (0, scene["n_frames"] // 2, scene["n_frames"] - 1):
        ring = np.concatenate([frames[f][tb[3] + 2:tb[3] + 14, tb[0]:tb[2] + 14].reshape(-1, 3),
                               frames[f][tb[1]:tb[3] + 14, tb[2] + 2:tb[2] + 14].reshape(-1, 3)])
        if any(n is not None for n in classify(ring, pal)):
            errs.append(f"frame {f}: strand colour near the timer")
    errs += sim.layout_errors(scene)
    return errs


def framecheck(argv):
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset-dir", required=True)
    ap.add_argument("--report", default="frame_check.txt")
    a = ap.parse_args(argv)
    root = Path(a.dataset_dir)
    lines, total, n, nq = [], 0, 0, 0
    for line in open(root / "annotations.jsonl", encoding="utf8"):
        s = json.loads(line)
        n += 1
        nq += len(s["questions"])
        errs = check_clip(s["params"], s["questions"], read_frames(root / s["media_path"]))
        total += len(errs)
        lines += [f"{s['sample_id']}: {e}" for e in errs]
        print(f"{s['sample_id']}: {len(errs)} failures", flush=True)
    lines.append(f"RESULT: {n} clips, {nq} answers re-derived from pixels, {total} frame-check failures")
    (root / a.report).write_text("\n".join(lines) + "\n")
    print(lines[-1])
    for e in lines[:-1][:25]:
        print("  -", e)
    return total


def framehash(argv):
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, required=True)
    a = ap.parse_args(argv)
    sc = _roundtrip(build_sample(DATASET, sim, questions, a.seed))["params"]
    h = hashlib.sha256()
    for f in range(sc["n_frames"]):
        h.update(np.asarray(render.frame_at(sc, f), dtype=np.uint8).tobytes())
    print(h.hexdigest())
    return 0


COMMANDS = {"stats": stats, "tamper": tamper, "framecheck": framecheck, "framehash": framehash}
