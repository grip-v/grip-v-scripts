"""Extra checks for blinking_networks, driven from run.py:

  python run.py stats --n 5000 [--seed 0]        diversity + ladder stats over dry-run scenes
  python run.py tamper --dataset-dir DIR [--k 3] corrupt k answers, count what the validator flags
  python run.py framecheck --dataset-dir DIR     read the rendered MP4s back and compare with the scene
  python run.py framehash --seed S               sha256 of the raw rendered frames of one seed
"""
from __future__ import annotations
import argparse
import hashlib
import json
import math
import random
import statistics
import sys
from collections import Counter
from pathlib import Path

import numpy as np

import answers
import questions
import render
import sim

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from common import ladder5 as L5  # noqa: E402
from common.video5 import FPS, build_sample, _roundtrip  # noqa: E402

DATASET = "blinking_networks"


# ----------------------------------------------------------------- stats
def stats(argv):
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=5000)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args(argv)
    sigs, scen, pairs, letters = set(), Counter(), set(), Counter()
    levels, fps, tries, keys = Counter(), [], [], Counter()
    knobs = {k: [] for k in ("nodes", "links", "R", "speed", "duration", "link_w")}
    cats = {k: Counter() for k in ("theme", "style", "layout", "shapes", "patterns", "rail")}
    ladder_bad = check_bad = mism = 0
    for i in range(a.n):
        s = _roundtrip(build_sample(DATASET, sim, questions, a.seed + i))
        sc = s["params"]
        sigs.add(hashlib.sha256(sim.signature(sc).encode()).hexdigest())
        scen[sc["scenario"]] += 1
        tries.append(sc["tries"])
        knobs["nodes"].append(len(sc["nodes"]))
        knobs["links"].append(len(sc["edges"]))
        knobs["R"].append(sc["R"])
        knobs["speed"].append(sc["speed"])
        knobs["duration"].append(sc["end_t"])
        knobs["link_w"].append(sc["theme"]["link_w"])
        cats["theme"][sc["theme"]["name"]] += 1
        cats["style"][sc["theme"]["style"]] += 1
        cats["layout"][sc["layout"]] += 1
        cats["rail"][sc["theme"]["rail"]] += 1
        cats["shapes"][len({n["shape"] for n in sc["nodes"]})] += 1
        cats["patterns"][",".join(sorted({n["pattern"] for n in sc["nodes"]}))] += 1
        ladder_bad += bool(L5.ladder_errors(s["questions"], "ten"))
        check_bad += bool(sim.check(sc))
        lc = Counter(q["difficulty_level"] for q in s["questions"])
        levels[tuple(lc[k] for k in range(1, 6))] += 1
        for q in s["questions"]:
            pairs.add((q["question"], q["answer_text"]))
            letters[q["answer"]] += 1
            fps.append(q["params"]["min_fps"])
            keys[(q["difficulty_level"], q["derivation_key"])] += 1
            if str(answers.answer(q["derivation_key"], q["params"], sc)) != str(q["groundtruth"]):
                mism += 1
    nq = len(fps)
    print(f"scenes: {a.n}  unique signatures: {len(sigs)} ({100 * len(sigs) / a.n:.2f}%)")
    print("scenario mix:", dict(sorted(scen.items())))
    for k, v in knobs.items():
        print(f"  {k:9s} min {min(v)}  median {statistics.median(v)}  max {max(v)}  distinct {len(set(v))}")
    for k, v in cats.items():
        print(f"  {k}: {dict(sorted(v.items(), key=lambda kv: str(kv[0])))}")
    print(f"questions: {nq}  distinct (question, answer) pairs: {len(pairs)}")
    print("answer letters:", {k: f"{100 * v / nq:.1f}%" for k, v in sorted(letters.items())})
    print("level counts (L1..L5):", {k: v for k, v in levels.most_common()})
    print(f"min_fps: <=1: {100 * sum(f <= 1 for f in fps) / nq:.1f}%  <=2: {100 * sum(f <= 2 for f in fps) / nq:.1f}%"
          f"  <=4: {100 * sum(f <= 4 for f in fps) / nq:.1f}%  max {max(fps)}")
    print(f"retries: mean tries/scene {statistics.mean(tries):.2f}, max {max(tries)}, "
          f"rejected attempts {sum(tries) - a.n} ({100 * (sum(tries) - a.n) / sum(tries):.1f}% of attempts)")
    print("templates:", {f"L{k[0]}:{k[1]}": v for k, v in sorted(keys.items())})
    print(f"ladder errors: {ladder_bad}  check failures: {check_bad}  answer mismatches: {mism}")


# ---------------------------------------------------------------- tamper
def tamper(argv):
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset-dir", required=True)
    ap.add_argument("--k", type=int, default=3)
    ap.add_argument("--seed", type=int, default=7)
    a = ap.parse_args(argv)
    root = Path(a.dataset_dir)
    rows = [json.loads(x) for x in open(root / "annotations.jsonl", encoding="utf8")]
    rng = random.Random(a.seed)
    targets = rng.sample([(i, j) for i, r in enumerate(rows) for j in range(len(r["questions"]))], a.k)
    for i, j in targets:
        q = rows[i]["questions"][j]
        wrong = next(L for L in "ABCD"[:len(q["options"])] if L != q["answer"])
        q["answer"] = wrong
        q["answer_text"] = q["options"]["ABCD".index(wrong)]
        q["groundtruth"] = q["answer_text"]
    flagged = 0
    for r in rows:
        for q in r["questions"]:
            if str(answers.answer(q["derivation_key"], q["params"], r["params"])) != str(q["groundtruth"]):
                flagged += 1
    print(f"tamper: corrupted {a.k} answers -> {flagged} flagged by the independent re-derivation")
    tf = root / "annotations_tampered.jsonl"
    tf.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf8")
    return flagged


# ------------------------------------------------------------ frame check
def _read_frames(path):
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


def _ring_frac(img, x, y, r, col, skip, lo=0.0, hi=360.0):
    hit = tot = 0
    for k in range(120):
        ang = 3 * k
        if not lo <= ang < hi:
            continue
        if any(min(abs(ang - s), 360 - abs(ang - s)) < 16 for s in skip):
            continue
        px = int(round(x + r * math.cos(math.radians(ang))))
        py = int(round(y + r * math.sin(math.radians(ang))))
        if 0 <= px < img.shape[1] and 0 <= py < img.shape[0]:
            tot += 1
            hit += np.abs(img[py, px] - col).sum() < 120
    return hit / max(1, tot)


def framecheck_sample(s, video_path) -> tuple[list[str], dict]:
    sc = s["params"]
    frames = _read_frames(video_path)
    errs, n = [], Counter()
    nfr = len(frames)
    halo = np.array(sc["theme"]["halo"])
    R = sc["R"]
    pos = {d["id"]: (d["x"], d["y"]) for d in sc["nodes"]}
    skip = {k: [] for k in pos}
    for e in sc["edges"]:
        for u, v in ((e["a"], e["b"]), (e["b"], e["a"])):
            skip[u].append(math.degrees(math.atan2(pos[v][1] - pos[u][1], pos[v][0] - pos[u][0])) % 360)
    rr = R + (render.HALO_IN + render.HALO_OUT) / 2

    def state(img, k):
        full = _ring_frac(img, *pos[k], rr, halo, skip[k])
        top = _ring_frac(img, *pos[k], rr, halo, skip[k], 190, 350)
        bot = _ring_frac(img, *pos[k], rr, halo, skip[k], 10, 170)
        if full >= 0.8:
            return "lit"
        if top >= 0.8 and bot <= 0.2:
            return "half"
        if full <= 0.2:
            return "dark"
        return f"unclear({full:.2f},{top:.2f},{bot:.2f})"
    # node identity: the colour fills a good part of the shape
    mid = frames[nfr // 2]
    for d in sc["nodes"]:
        m = np.array(render.shape_mask(d["shape"], R * 0.85, 4 * R)) > 0
        x0, y0 = int(round(d["x"] - 2 * R)), int(round(d["y"] - 2 * R))
        patch = mid[y0:y0 + 4 * R, x0:x0 + 4 * R][m]
        rgb = np.array(d["rgb"])
        near = lambda c: np.abs(patch - c).sum(axis=1) < 90  # noqa: E731
        frac = near(rgb).mean()                      # the plain colour
        drawn = (near(rgb) | near(rgb // 2) | near(halo)).mean()   # colour, its pattern shade, double outline
        n["node_colour"] += 1
        if frac < 0.12 or drawn < 0.6:
            errs.append(f"{d['id']} colour covers only {frac:.2f} of its shape ({drawn:.2f} incl. pattern)")
    # light-up events: dark 2 frames before, lit 2 frames after; timer reads the frame time
    for k, tf in sc["fire"].items():
        kf = math.ceil(tf * FPS - 1e-9)
        for kk, want in ((kf - 2, ("dark", "half")), (kf + 2, ("lit",))):
            st = state(frames[kk], k)
            n["event_frames"] += 1
            if st not in want:
                errs.append(f"{k} at frame {kk} (t={kk / FPS:.2f}, fires {tf:.2f}) looks {st}, expected {want}")
            ref = render.frame(sc, kk / FPS)
            x0, y0, x1, y1 = sim.TIMER_BOX
            got = frames[kk][y0:y1, x0:x1]
            exp = np.asarray(ref, dtype=np.int16)[y0:y1, x0:x1]
            other = np.asarray(render.frame(sc, kk / FPS + 0.3), dtype=np.int16)[y0:y1, x0:x1]
            mae, mae_o = np.abs(got - exp).mean(), np.abs(got - other).mean()
            n["timer_reads"] += 1
            if not mae < 4 or not mae < mae_o:
                errs.append(f"timer at frame {kk} doesn't read {kk / FPS:.1f} s (mae {mae:.1f} vs {mae_o:.1f})")
    for k, th in sc["half"].items():
        kf = math.ceil(th * FPS - 1e-9)
        st = state(frames[kf + 2], k)
        n["event_frames"] += 1
        if st != "half":
            errs.append(f"{k} should show a half halo at frame {kf + 2}, looks {st}")
    # final frame: count lit nodes (blob count of full halos)
    last = frames[-1]
    lit = sum(state(last, k) == "lit" for k in pos)
    n["final_counts"] += 1
    if lit != len(sc["fire"]):
        errs.append(f"final frame shows {lit} lit nodes, scene says {len(sc['fire'])}")
    # pulses: a pulse-coloured dot at the predicted spot halfway along
    pcol = np.array(sc["theme"]["pulse"])
    for p in sc["pulses"]:
        tm = (p["t0"] + p["t1"]) / 2
        kk = int(round(tm * FPS))
        f = (kk / FPS - p["t0"]) / (p["t1"] - p["t0"])
        a, b = pos[p["from"]], pos[p["to"]]
        x, y = a[0] + (b[0] - a[0]) * f, a[1] + (b[1] - a[1]) * f
        if not 0 < f < 1 or kk >= nfr:
            continue
        patch = frames[kk][int(y) - 3:int(y) + 4, int(x) - 3:int(x) + 4].reshape(-1, 3)
        n["pulse_positions"] += 1
        if (np.abs(patch - pcol).sum(axis=1) < 90).mean() < 0.6:
            errs.append(f"no pulse dot at ({x:.0f},{y:.0f}) frame {kk} for {p['from']}->{p['to']}")
    if abs(nfr / FPS - sc["end_t"]) > 1.5 / FPS:
        errs.append(f"video has {nfr} frames for {sc['end_t']} s")
    return errs, n


def framecheck(argv):
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset-dir", required=True)
    a = ap.parse_args(argv)
    root = Path(a.dataset_dir)
    tot, bad = Counter(), []
    for line in open(root / "annotations.jsonl", encoding="utf8"):
        s = json.loads(line)
        errs, n = framecheck_sample(s, root / s["media_path"])
        tot.update(n)
        bad += [f"{s['sample_id']}: {e}" for e in errs]
    print("frame check:", dict(tot), f"errors: {len(bad)}")
    for e in bad[:30]:
        print("  -", e)
    return len(bad)


def framehash(argv):
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, required=True)
    a = ap.parse_args(argv)
    s = _roundtrip(build_sample(DATASET, sim, questions, a.seed))
    h = hashlib.sha256()
    n = int(round(render.duration(s["params"]) * FPS))
    for i in range(n):
        h.update(np.asarray(render.frame(s["params"], i / FPS).convert("RGB"), dtype=np.uint8).tobytes())
    print(h.hexdigest())


COMMANDS = {"stats": stats, "tamper": tamper, "framecheck": framecheck, "framehash": framehash}
