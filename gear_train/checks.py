"""Extra checks for gear_train, driven from run.py:

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

DATASET = "gear_train"


# ----------------------------------------------------------------- stats
def stats(argv):
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=5000)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args(argv)
    sigs, scen, pairs, letters = set(), Counter(), set(), Counter()
    levels, fps, tries, keys, motion = Counter(), [], [], Counter(), Counter()
    knobs = {k: [] for k in ("gears", "train", "module_px", "teeth", "w_fast", "duration", "fill")}
    cats = {k: Counter() for k in ("theme", "bg_style", "topology", "teeth_style", "patterns", "loose",
                                   "reversals", "speed_factor")}
    ladder_bad = check_bad = mism = 0
    for i in range(a.n):
        s = _roundtrip(build_sample(DATASET, sim, questions, a.seed + i))
        sc = s["params"]
        sigs.add(hashlib.sha256(sim.signature(sc).encode()).hexdigest())
        scen[sc["scenario"]] += 1
        tries.append(sc["tries"])
        tr = [g for g in sc["gears"] if not g["loose"]]
        knobs["gears"].append(len(sc["gears"]))
        knobs["train"].append(len(tr))
        knobs["module_px"].append(round(sc["m"], 1))
        knobs["teeth"].extend(g["z"] for g in sc["gears"])
        knobs["w_fast"].append(round(sc["K"] / min(g["z"] for g in tr), 2))
        knobs["duration"].append(sc["duration"])
        knobs["fill"].append(sc["fill"])
        cats["theme"][sc["theme"]["name"]] += 1
        cats["bg_style"][sc["theme"]["style"]] += 1
        cats["topology"][sc["topology"]] += 1
        cats["teeth_style"][sc["teeth_style"]] += 1
        cats["patterns"].update(g["pattern"] for g in sc["gears"])
        cats["loose"][any(g["loose"] for g in sc["gears"])] += 1
        if sc["scenario"] == "reversal":
            cats["reversals"][len(sc["program"]["reversals"])] += 1
        if sc["scenario"] == "speed_change":
            cats["speed_factor"][round(sc["program"]["speed"]["factor"], 2)] += 1
        ladder_bad += bool(L5.ladder_errors(s["questions"], "ten"))
        check_bad += bool(sim.check(sc))
        lc = Counter(q["difficulty_level"] for q in s["questions"])
        levels[tuple(lc[k] for k in range(1, 6))] += 1
        for q in s["questions"]:
            pairs.add((q["question"], q["answer_text"]))
            letters[q["answer"]] += 1
            fps.append(q["params"]["min_fps"])
            keys[(q["difficulty_level"], q["derivation_key"])] += 1
            if q["difficulty_level"] >= 2:
                motion[q["requires_motion"]] += 1
            if str(answers.answer(q["derivation_key"], q["params"], sc)) != str(q["groundtruth"]):
                mism += 1
    nq = len(fps)
    print(f"scenes: {a.n}  unique signatures: {len(sigs)} ({100 * len(sigs) / a.n:.2f}%)")
    print("scenario mix:", dict(sorted(scen.items())))
    for k, v in knobs.items():
        print(f"  {k:10s} min {min(v)}  median {statistics.median(v)}  max {max(v)}  distinct {len(set(v))}")
    for k, v in cats.items():
        print(f"  {k}: {dict(sorted(v.items(), key=lambda kv: str(kv[0])))}")
    print(f"questions: {nq}  distinct (question, answer) pairs: {len(pairs)}")
    print("answer letters:", {k: f"{100 * v / nq:.1f}%" for k, v in sorted(letters.items())})
    print("level counts (L1..L5):", {k: v for k, v in levels.most_common()})
    print(f"L2-L5 requires_motion: {100 * motion[True] / max(1, sum(motion.values())):.1f}%")
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
        wrong = next(x for x in "ABCD"[:len(q["options"])] if x != q["answer"])
        q["answer"] = wrong
        q["answer_text"] = q["options"]["ABCD".index(wrong)]
        q["groundtruth"] = q["answer_text"]
    flagged = 0
    for r in rows:
        for q in r["questions"]:
            if str(answers.answer(q["derivation_key"], q["params"], r["params"])) != str(q["groundtruth"]):
                flagged += 1
    print(f"tamper: corrupted {a.k} answers -> {flagged} flagged by the independent re-derivation")
    (root / "annotations_tampered.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf8")
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


def _ring(img, cx, cy, r, n=240):
    a = np.arange(n) * 2 * math.pi / n
    xs = np.clip(np.floor(cx + r * np.cos(a)).astype(int), 0, img.shape[1] - 1)
    ys = np.clip(np.floor(cy - r * np.sin(a)).astype(int), 0, img.shape[0] - 1)
    return a, img[ys, xs]


def marker_angle(img, scene, g, t):
    """Screen angle (CCW +) of gear g's white marker, read off the frame; None if not found."""
    cx, cy = sim.centre(scene, g["id"], t)
    rf = sim.root_r(scene, g)
    a, px = _ring(img, cx, cy, render.MARK_R * rf)
    white = np.abs(px - 255).sum(axis=1) < 110
    if white.sum() < 2:
        return None
    return math.atan2(np.sin(a[white]).mean(), np.cos(a[white]).mean())


def _adiff(x, y):
    return (x - y + math.pi) % (2 * math.pi) - math.pi


def _track(frames, scene, g, f0, f1, step=6):
    """Unwrapped marker angle read off frames f0, f0+step, ..., f1."""
    out, prev = [], None
    for f in list(range(f0, f1, step)) + [f1]:
        a = marker_angle(frames[f], scene, g, f / FPS)
        if a is None:
            return None
        if prev is None:
            out.append(a)
        else:
            out.append(out[-1] + _adiff(a, prev))
        prev = a
    return out


def _count_runs(img, scene, g):
    cx, cy = sim.centre(scene, g["id"], 0.0)
    _, px = _ring(img, cx, cy, sim.radius(scene, g), 1440)
    own = np.abs(px - np.array(g["rgb"])).sum(axis=1) < 90
    # runs of the gear's own colour around the pitch circle; slivers (< 1/4 tooth width, from the module, not the tooth count) are anti-aliasing or
    # a similar-coloured neighbour's tooth tip, not teeth
    start = int(np.argmin(own))
    own = np.roll(own, -start)
    runs, cur = [], 0
    for v in own:
        if v:
            cur += 1
        elif cur:
            runs.append(cur)
            cur = 0
    if cur:
        runs.append(cur)
    step = 2 * math.pi * sim.radius(scene, g) / len(own)            # px per sample
    return int(sum(r * step >= math.pi * scene["m"] / 8 for r in runs))


def _timer_ok(frames, scene, f):
    x0, y0, x1, y1 = sim.TIMER_BOX
    got = frames[f][y0:y1, x0:x1]
    exp = np.asarray(render.frame(scene, f / FPS), dtype=np.int16)[y0:y1, x0:x1]
    other = np.asarray(render.frame(scene, f / FPS + 0.3), dtype=np.int16)[y0:y1, x0:x1]
    mae, mae_o = np.abs(got - exp).mean(), np.abs(got - other).mean()
    return mae < 4 and mae < mae_o, mae


def framecheck_sample(s, video_path) -> tuple[list, Counter]:
    sc = s["params"]
    frames = _read_frames(video_path)
    errs, n = [], Counter()
    nfr = len(frames)
    gm = sim.gmap(sc)
    last = nfr - 1
    if abs(nfr - int(round(sc["duration"] * FPS))) > 1:
        errs.append(f"video has {nfr} frames for {sc['duration']} s")
    # 1. every marker sits where the scene says, sampled at 4 fps
    for f in range(0, nfr, 6):
        for g in sc["gears"]:
            a = marker_angle(frames[f], sc, g, f / FPS)
            n["marker_reads"] += 1
            exp = sim.theta(sc, g["id"], f / FPS)
            if a is None or abs(_adiff(a, exp)) > math.radians(12):
                errs.append(f"{g['id']} marker at frame {f}: read {a}, expected {exp % (2 * math.pi):.2f}")
    # 2. colours and teeth read off the first frame
    for g in sc["gears"]:
        cx, cy = g["x"], g["y"]
        _, px = _ring(frames[0], cx, cy, 0.9 * sim.root_r(sc, g))
        frac = (np.abs(px - np.array(g["rgb"])).sum(axis=1) < 110).mean()
        n["gear_colour"] += 1
        if frac < 0.6:
            errs.append(f"{g['id']} body colour covers only {frac:.2f} of its ring")
    tracked_end = {}
    for q in s["questions"]:
        key, p = q["derivation_key"], q["params"]
        gt = q["groundtruth"]
        if key == "teeth":
            got = _count_runs(frames[0], sc, gm[p["g"]])
            n["teeth_counts"] += 1
            if got != gt:
                errs.append(f"{q['qid']} teeth: counted {got} on the frame, answer {gt}")
        elif key in ("largest", "smallest"):
            area = {}
            for g in sc["gears"]:
                x0, y0 = int(g["x"] - sim.tip_r(sc, g)), int(g["y"] - sim.tip_r(sc, g))
                x1, y1 = int(g["x"] + sim.tip_r(sc, g)) + 1, int(g["y"] + sim.tip_r(sc, g)) + 1
                patch = frames[0][max(0, y0):y1, max(0, x0):x1]
                area[g["colour"]] = int((np.abs(patch - np.array(g["rgb"])).sum(axis=2) < 110).sum())
            pick = (max if key == "largest" else min)(area, key=area.get)
            n["size_reads"] += 1
            if pick != gt:
                errs.append(f"{q['qid']} {key}: pixel areas pick {pick}, answer {gt}")
        elif key == "turns":
            tr = _track(frames, sc, gm[p["g"]], int(round(p["t0"] * FPS)), min(last, int(round(p["t1"] * FPS))))
            n["turn_counts"] += 1
            got = None if tr is None else int(math.floor(abs(tr[-1] - tr[0]) / (2 * math.pi)))
            if got != gt:
                errs.append(f"{q['qid']} turns: frames give {got}, answer {gt}")
        elif key in ("start_dirs", "end_dirs"):
            w0, w1 = sc["log"]["windows"]["start" if key == "start_dirs" else "end"]
            f0, f1 = int(math.ceil(w0 * FPS)), min(last, int(math.floor(w1 * FPS)))
            ds = []
            for g in (p["a"], p["b"]):
                tr = _track(frames, sc, gm[g], f0, f1)
                ds.append(0 if tr is None else (1 if tr[-1] > tr[0] else -1))
            got = questions.combo_text(questions.Ctx(sc), p["a"], p["b"], ds[0], ds[1])
            n["direction_reads"] += 1
            if got != gt:
                errs.append(f"{q['qid']} {key}: frames give {got!r}, answer {gt!r}")
        elif key in ("stop_duration", "still_count", "keeps_turning", "stopped_followers") and sc["log"]["still"]:
            s0, s1 = sc["log"]["still"]
            f0, f1 = int(math.ceil((s0 + 0.1) * FPS)), int(math.floor((s1 - 0.1) * FPS))
            for g in sc["gears"]:
                tr = _track(frames, sc, g, f0, f1)
                moved = tr is not None and abs(tr[-1] - tr[0]) > math.radians(4)
                should = g["id"] not in sc["log"]["stopped"] and not g["loose"]
                n["still_reads"] += 1
                if moved != should:
                    errs.append(f"{q['qid']} {g['id']} {'moves' if moved else 'still'} in the stop window")
        elif key == "event_time":
            got = _event_time_from_frames(frames, sc, p)
            n["event_times"] += 1
            if got != gt:
                errs.append(f"{q['qid']} event_time: frames give {got}, answer {gt}")
        elif key == "loose_which":
            lg = gm[sim.loose_id(sc)]
            tr = _track(frames, sc, lg, 0, last)
            n["loose_reads"] += 1
            if tr is None or max(tr) - min(tr) > math.radians(4):
                errs.append(f"{q['qid']} the loose gear seems to move")
    del tracked_end
    # 3. events leave visible, persistent results
    prog = sc["program"]
    if prog.get("disengage"):
        d = prog["disengage"]
        g, par = gm[d["gear"]], gm[d["parent"]]
        for tm in (d["still0"], (d["still0"] + d["still1"]) / 2, d["still1"]):
            f = int(round(tm * FPS))
            cg = sim.centre(sc, g["id"], f / FPS)
            L = math.dist(cg, (par["x"], par["y"]))
            ux, uy = (cg[0] - par["x"]) / L, (cg[1] - par["y"]) / L
            rr = sim.tip_r(sc, par) + (L - sim.tip_r(sc, par) - sim.tip_r(sc, g)) / 2
            x, y = par["x"] + ux * rr, par["y"] + uy * rr
            pix = frames[f][int(y) - 2:int(y) + 3, int(x) - 2:int(x) + 3].reshape(-1, 3)
            near = lambda c: (np.abs(pix - np.array(c)).sum(axis=1) < 110).mean()  # noqa: E731
            n["gap_reads"] += 1
            if near(g["rgb"]) > 0.2 or near(par["rgb"]) > 0.2:
                errs.append(f"no visible gap between {g['id']} and {par['id']} at frame {f}")
            ok, mae = _timer_ok(frames, sc, f)
            n["timer_reads"] += 1
            if not ok:
                errs.append(f"timer at frame {f} doesn't read {f / FPS:.1f} s (mae {mae:.1f})")
    if prog.get("brake"):
        b = prog["brake"]
        g = gm[b["gear"]]
        ux, uy = math.cos(b["angle"]), -math.sin(b["angle"])
        rr = sim.tip_r(sc, g) + max(22.0, 2.0 * sc["m"]) / 2
        x, y = g["x"] + ux * rr, g["y"] + uy * rr
        for tm, want in ((0.0, False), ((b["still0"] + b["release"]) / 2, True), (sc["duration"] - 0.1, False)):
            f = min(last, int(round(tm * FPS)))
            pix = frames[f][int(y) - 2:int(y) + 3, int(x) - 2:int(x) + 3].reshape(-1, 3)
            on = (np.abs(pix - np.array(sim.BRAKE_RGB)).sum(axis=1) < 70).mean() > 0.6
            n["brake_reads"] += 1
            if on != want:
                errs.append(f"brake block {'present' if on else 'missing'} at frame {f}")
    for t in sc["log"]["event_t"] + ([sc["log"]["still"][0], sc["log"]["still"][1]] if sc["log"]["still"] else []):
        for f in (int(math.floor(t * FPS)), int(math.ceil(t * FPS))):
            ok, mae = _timer_ok(frames, sc, min(last, f))
            n["timer_reads"] += 1
            if not ok:
                errs.append(f"timer at frame {f} doesn't read {f / FPS:.1f} s (mae {mae:.1f})")
    for q in s["questions"]:
        if q["derivation_key"] == "still_count":
            f = int(q["params"]["T"]) * FPS
            ok, mae = _timer_ok(frames, sc, f)
            n["timer_reads"] += 1
            if not ok:
                errs.append(f"timer at frame {f} doesn't read {q['params']['T']}.0 s")
            cnt = 0
            for g in sc["gears"]:
                tr = _track(frames, sc, g, f - 3, f + 3, step=3)
                cnt += tr is not None and abs(tr[-1] - tr[0]) < math.radians(0.5)
            n["still_counts"] += 1
            if cnt != q["groundtruth"]:
                errs.append(f"{q['qid']} still_count at {q['params']['T']} s: frames give {cnt}, answer {q['groundtruth']}")
    return errs, n


def _event_time_from_frames(frames, sc, p):
    """Read the event moment off the decoded frames (driver marker / brake block / gear position)."""
    last = len(frames) - 1
    prog = sc["program"]
    drv = sim.gmap(sc)[sc["driver"]]
    if sc["scenario"] == "reversal":
        tr = _track(frames, sc, drv, 0, last, step=2)
        if tr is None:
            return None
        ext = [i for i in range(1, len(tr) - 1) if (tr[i] - tr[i - 1]) * (tr[i + 1] - tr[i]) < 0
               or (tr[i] == tr[i - 1])]
        flips = []
        for i in ext:
            if not flips or i - flips[-1] > 12:
                flips.append(i)
        return round(flips[p["i"]] * 2 / FPS) if p["i"] < len(flips) else None
    if sc["scenario"] == "speed_change":
        tr = _track(frames, sc, drv, 0, last, step=2)
        if tr is None:
            return None
        v = np.abs(np.diff(tr))
        a, b = v[:12].mean(), v[-12:].mean()
        mid = (a + b) / 2
        i = int(np.argmax(v > mid) if b > a else np.argmax(v < mid))
        return round((i + 0.5) * 2 / FPS)
    if sc["scenario"] == "brake":
        b = prog["brake"]
        g = sim.gmap(sc)[b["gear"]]
        ux, uy = math.cos(b["angle"]), -math.sin(b["angle"])
        rr = sim.tip_r(sc, g) + 4
        x, y = g["x"] + ux * rr, g["y"] + uy * rr
        for f in range(last + 1):
            pix = frames[f][int(y) - 1:int(y) + 2, int(x) - 1:int(x) + 2].reshape(-1, 3)
            if (np.abs(pix - np.array(sim.BRAKE_RGB)).sum(axis=1) < 70).mean() > 0.6:
                return round(f / FPS)
        return None
    if sc["scenario"] == "disengage":
        d = prog["disengage"]
        g = sim.gmap(sc)[d["gear"]]
        bg = np.asarray(render._static(sc)[0], dtype=np.int16)
        cx, cy = g["x"], g["y"]
        R = int(sim.root_r(sc, g) * 0.9)
        yy, xx = np.mgrid[-R:R, -R:R]
        disk = xx ** 2 + yy ** 2 < R * R
        y0, x0 = int(cy) - R, int(cx) - R
        bgp = bg[y0:y0 + 2 * R, x0:x0 + 2 * R]
        base = None
        for f in range(last + 1):
            patch = frames[f][y0:y0 + 2 * R, x0:x0 + 2 * R]
            mask = (np.abs(patch - bgp).sum(axis=2) > 45) & disk
            ys, xs = np.nonzero(mask)
            c = (xs.mean(), ys.mean())
            if base is None:
                base = c
            elif math.dist(c, base) > 0.6:
                return round(f / FPS)
        return None
    return None


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
    for e in bad[:40]:
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
