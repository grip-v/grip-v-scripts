"""Extra checks for overlapping_shapes, driven from run.py:

  python run.py stats --n 5000 [--seed 0] [--jobs 8]   diversity + ladder stats over dry-run scenes
  python run.py tamper --dataset-dir DIR [--k 3]        corrupt k answers, count what gets flagged
  python run.py framecheck --dataset-dir DIR            read the rendered MP4s back, compare with the scene
  python run.py framehash --seed S                      sha256 of the raw rendered frames of one seed
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
from multiprocessing import Pool
from pathlib import Path

import numpy as np
from PIL import Image
from shapely.geometry import Polygon

import answers
import questions
import render
import sim

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from common import ladder5 as L5  # noqa: E402
from common.video5 import FPS, build_sample, _roundtrip  # noqa: E402

DATASET = "overlapping_shapes"


# ----------------------------------------------------------------- stats
def _one(seed):
    import warnings
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        s = _roundtrip(build_sample(DATASET, sim, questions, seed))
        sc = s["params"]
        mism = sum(str(answers.answer(q["derivation_key"], q["params"], sc)) != str(q["groundtruth"])
                   for q in s["questions"])
        out = {
            "sig": sim.signature(sc), "scenario": sc["scenario"], "tries": sc["tries"],
            "n": len(sc["shapes"]), "types": len({x["type"] for x in sc["shapes"]}),
            "theme": sc["theme"]["name"], "layout": sc["pattern"], "anim": sc["anim"],
            "sizes": [x["size"] for x in sc["shapes"]], "colors": [x["color"] for x in sc["shapes"]],
            "styles": [x["style"] for x in sc["shapes"]], "fills": [x["fill"] for x in sc["shapes"]],
            "kinds": [x["type"] for x in sc["shapes"]], "duration": sc["duration"], "pairs": len(sim.pairs(sc)),
            "board_w": sc["board"][2] - sc["board"][0], "alpha": sc["alpha"], "line_w": sc["line_w"],
            "speeds": [round(math.hypot(x["path"]["x1"] - x["x"], x["path"]["y1"] - x["y"]) /
                             ((x["path"]["f1"] - x["path"]["f0"]) / FPS)) for x in sc["shapes"] if "path" in x],
            "ladder": L5.ladder_errors(s["questions"], "ten"), "check": sim.check(sc), "mism": mism,
            "qs": [(q["difficulty_level"], q["derivation_key"], q["question"], q["answer_text"], q["answer"],
                    q["params"]["min_fps"], q["requires_motion"]) for q in s["questions"]],
        }
    out["warnings"] = [str(x.message) for x in w]
    return out


def _spread(v):
    return f"min {min(v)}  median {statistics.median(v)}  max {max(v)}  distinct {len(set(v))}"


def stats(argv):
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=5000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--jobs", type=int, default=8)
    a = ap.parse_args(argv)
    with Pool(a.jobs) as pool:
        rows = pool.map(_one, range(a.seed, a.seed + a.n), chunksize=8)
    sigs = {r["sig"] for r in rows}
    print(f"scenes: {a.n}  unique signatures: {len(sigs)} ({100 * len(sigs) / a.n:.2f}%)")
    print("scenario mix:", dict(sorted(Counter(r["scenario"] for r in rows).items())))
    for k in ("n", "types", "duration", "pairs", "board_w", "alpha", "line_w"):
        print(f"  {k:9s}", _spread([r[k] for r in rows]))
    print(f"  {'size px':9s}", _spread([x for r in rows for x in r["sizes"]]))
    print(f"  {'speed':9s}", _spread([x for r in rows for x in r["speeds"]]))
    for k in ("theme", "layout", "anim"):
        print(f"  {k}: {dict(sorted(Counter(r[k] for r in rows).items()))}")
    for k in ("colors", "styles", "fills", "kinds"):
        print(f"  {k}: {dict(sorted(Counter(x for r in rows for x in r[k]).items()))}")
    qs = [q for r in rows for q in r["qs"]]
    nq = len(qs)
    print(f"questions: {nq}  distinct (question, answer) pairs: {len({(q[2], q[3]) for q in qs})}  "
          f"distinct question texts: {len({q[2] for q in qs})}")
    letters = Counter(q[4] for q in qs)
    print("answer letters:", {k: f"{100 * v / nq:.1f}%" for k, v in sorted(letters.items())})
    lv = Counter(tuple(Counter(q[0] for q in r["qs"])[k] for k in range(1, 6)) for r in rows)
    print("level counts (L1..L5):", dict(lv.most_common()))
    per_level = {k: Counter(r_q for r in rows for r_q in [sum(q[0] == k for q in r["qs"])]) for k in range(1, 6)}
    print("questions per level:", {f"L{k}": dict(sorted(v.items())) for k, v in per_level.items()})
    mot = [q[6] for q in qs if q[0] >= 2]
    print(f"L2-L5 requires_motion: {100 * sum(mot) / len(mot):.1f}%")
    fps = [q[5] for q in qs]
    print(f"min_fps: answerable at 1 fps {100 * sum(f <= 1 for f in fps) / nq:.1f}%  at 2 fps "
          f"{100 * sum(f <= 2 for f in fps) / nq:.1f}%  at 4 fps {100 * sum(f <= 4 for f in fps) / nq:.1f}%  "
          f"median {statistics.median(fps)}  max {max(fps)}")
    tries = [r["tries"] for r in rows]
    print(f"retries: seeds needing a retry {sum(t > 1 for t in tries)} ({100 * sum(t > 1 for t in tries) / a.n:.1f}%)"
          f", mean attempts/scene {statistics.mean(tries):.2f}, max {max(tries)}, rejected attempts "
          f"{sum(tries) - a.n} ({100 * (sum(tries) - a.n) / sum(tries):.1f}% of attempts)")
    print("templates:", dict(sorted(Counter(f"L{q[0]}:{q[1]}" for q in qs).items())))
    print(f"ladder errors: {sum(bool(r['ladder']) for r in rows)}  check failures: {sum(bool(r['check']) for r in rows)}"
          f"  answer mismatches: {sum(r['mism'] for r in rows)}  warnings: {sum(len(r['warnings']) for r in rows)}")


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
    flagged = sum(str(answers.answer(q["derivation_key"], q["params"], r["params"])) != str(q["groundtruth"])
                  for r in rows for q in r["questions"])
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


def _timer_ref(f):
    img = Image.new("RGB", (sim.W, sim.H), (0, 0, 0))
    render.draw_timer(img, f)
    x0, y0, x1, y1 = sim.TIMER_BOX
    return np.asarray(img, dtype=np.int16)[y0:y1, x0:x1]


def timer_ok(frame, f):
    """The clock region shows this frame's reading, not the readings 0.1 s either side."""
    x0, y0, x1, y1 = sim.TIMER_BOX
    crop = frame[y0:y1, x0:x1]
    d = np.abs(crop - _timer_ref(f)).mean()
    others = [np.abs(crop - _timer_ref(g)).mean() for g in (f - 3, f + 3) if g >= 0 and
              sim.timer_text(g) != sim.timer_text(f)]
    return d < 6 and all(d < o - 0.5 for o in others), d


def _board_img(sc):
    return np.asarray(render._board(json.dumps({"theme": sc["theme"], "board": sc["board"]}, sort_keys=True))
                      .convert("RGB"), dtype=np.int16)


def _probe(sc, i, others):
    """Pixels (ys, xs) well inside shape i and clear of every other shape: where only i can show."""
    sh = sc["shapes"][i]
    p = sim.polygon(sh).buffer(-6)
    for j in others:
        if j != i:
            p = p.difference(sim.polygon(sc["shapes"][j]).buffer(8))
    if p.is_empty or p.area < 40:
        return None
    m = answers._mask(list(p.exterior.coords) if p.geom_type == "Polygon" else
                      list(max(p.geoms, key=lambda g: g.area).exterior.coords))
    ys, xs = np.nonzero(m)
    return (ys, xs) if len(ys) >= 30 else None


def _present(frame, bg, probe):
    ys, xs = probe
    return float(np.abs(frame[ys, xs] - bg[ys, xs]).sum(axis=1).mean()) > 30


def framecheck_sample(s, video_path):
    sc = s["params"]
    frames = _read_frames(video_path)
    errs, n = [], Counter()
    N = len(frames)
    if N != sc["n_frames"]:
        errs.append(f"video has {N} frames, scene has {sc['n_frames']}")
        return errs, n
    bg = _board_img(sc)
    kind = sc["scenario"]
    st = sim.statics(sc)
    # 1. nothing but the clock outside the board: the frame margin is plain background
    x0, y0, x1, y1 = sc["board"]
    for f in (0, N // 2, N - 1):
        strip = frames[f][y0 + 20:y1 - 20, max(0, x0 - 12):max(1, x0 - 4)]
        n["margin_reads"] += 1
        if strip.size and np.abs(strip - np.array(sc["theme"]["bg"])).max() > 24:
            errs.append(f"unexpected marks outside the board at frame {f}")
    # 2. presence of every shape read off probe pixels, frame by frame
    times = {}
    if kind != "glide":
        for i in range(len(sc["shapes"])):
            pr = _probe(sc, i, range(len(sc["shapes"])))
            if pr is None:
                n["no_probe"] += 1
                continue
            # skip frames in which another shape is mid-animation (a dropping / spinning shape may sweep
            # briefly over this one's probe pixels); every other frame must show a single clean change
            busy = {f for e in sc["events"] if e["shape"] != i
                    for f in range(e["f"] - 1, e["f"] + sc["anim_f"] + 1)}
            fs = [f for f in range(N) if f not in busy]
            seq = {f: _present(frames[f], bg, pr) for f in fs}
            n["presence_tracks"] += 1
            changes = [b for a, b in zip(fs, fs[1:]) if seq[a] != seq[b]]
            seq = [seq.get(f, False) for f in range(N)]
            if len(changes) > 1:
                errs.append(f"{sim.label(sc['shapes'][i])} flickers (changes at {changes[:4]})")
                continue
            ev = next((e["f"] for e in sc["events"] if e["shape"] == i), None)
            if ev is None:
                if changes or not seq[fs[0]]:
                    errs.append(f"{sim.label(sc['shapes'][i])} should stay on the board")
                continue
            if not changes or abs(changes[0] - ev) > sc["anim_f"]:
                errs.append(f"{sim.label(sc['shapes'][i])}: frames show a change at {changes}, event at {ev}")
                continue
            times[i] = changes[0]
        # order and clock-time answers recomputed from the frames
        for q in s["questions"]:
            key, p = q["derivation_key"], q["params"]
            got = None
            if key in ("first_shape", "first_gone", "last_shape", "earlier4", "earlier2", "gone_after"):
                pool = p.get("cands") or p.get("pair") or [e["shape"] for e in sc["events"]]
                if any(i not in times for i in pool):
                    continue
                if key == "last_shape":
                    got = sim.label(sc["shapes"][max(pool, key=times.get)])
                elif key == "gone_after":
                    later = [i for i in pool if times[i] > times[p["x"]]]
                    got = sim.label(sc["shapes"][min(later, key=times.get)]) if later else None
                else:
                    got = sim.label(sc["shapes"][min(pool, key=times.get)])
            elif key == "time_of" and p["x"] in times:
                f = times[p["x"]]
                ok, d = timer_ok(frames[f], f)
                n["timer_reads"] += 1
                if not ok:
                    errs.append(f"{q['qid']} clock at frame {f} doesn't read {sim.timer_text(f)}")
                got = f"{int(round(f / FPS)):g} s"
            elif key == "count_at_T":
                f = p["f"]
                ok, d = timer_ok(frames[f], f)
                n["timer_reads"] += 1
                if not ok:
                    errs.append(f"{q['qid']} clock at frame {f} doesn't read {sim.timer_text(f)}")
                if len(times) + n["no_probe"] < len(sc["shapes"]):
                    continue
                if kind == "depart":
                    got = sum(1 for i in range(len(sc["shapes"])) if i not in times or times[i] > f)
                else:
                    got = sum(1 for i in times if times[i] <= f)
                if n["no_probe"]:
                    got = None
            if got is not None:
                n["frame_answers"] += 1
                if str(got) != str(q["groundtruth"]):
                    errs.append(f"{q['qid']} {key}: frames give {got!r}, answer {q['groundtruth']!r}")
        for e in sc["events"]:
            for f in (e["f"], e["f"] + sc["anim_f"] - 1):
                ok, d = timer_ok(frames[f], f)
                n["timer_reads"] += 1
                if not ok:
                    errs.append(f"clock at event frame {f} doesn't read {sim.timer_text(f)} (mae {d:.1f})")
    # 3. overlaps read off the full board: shared regions show a blend (or the top sheet), gaps show board
    f_full = 0 if kind == "depart" else N - 1
    if kind == "glide":
        f_full = 0
    img = frames[f_full]
    polys = {i: sim.polygon(sc["shapes"][i]) for i in st}
    seen_pairs = 0
    for a in st:
        for b in st:
            if b <= a:
                continue
            inter = polys[a].intersection(polys[b]).buffer(-5)
            if (a, b) in set(sim.pairs(sc)):
                rest = inter
                for c in st:
                    if c not in (a, b):
                        rest = rest.difference(polys[c].buffer(6))
                if rest.is_empty or rest.area < 30:
                    n["pair_unreadable"] += 1
                    seen_pairs += 1
                    continue
                g = rest if rest.geom_type == "Polygon" else max(rest.geoms, key=lambda x: x.area)
                ys, xs = np.nonzero(answers._mask(list(g.exterior.coords)))
                if len(ys) == 0:
                    n["pair_unreadable"] += 1
                    seen_pairs += 1
                    continue
                blend = img[ys, xs].mean(axis=0)
                pa, pb = _probe(sc, a, st), _probe(sc, b, st)
                n["pair_reads"] += 1
                same_look = sc["shapes"][a]["color"] == sc["shapes"][b]["color"]
                if pa is not None and pb is not None and not (kind == "paper" and same_look):
                    ca, cb = img[pa].mean(axis=0), img[pb].mean(axis=0)
                    if kind == "paper":
                        later = max((a, b), key=lambda i: [e["shape"] for e in sc["events"]].index(i))
                        top = a if np.abs(blend - ca).sum() < np.abs(blend - cb).sum() else b
                        if top != later:
                            errs.append(f"paper overlap of {sim.label(sc['shapes'][a])} / "
                                        f"{sim.label(sc['shapes'][b])}: frames show the wrong sheet on top")
                    elif min(np.abs(blend - ca).sum(), np.abs(blend - cb).sum()) < 12:
                        errs.append(f"overlap of {sim.label(sc['shapes'][a])} / {sim.label(sc['shapes'][b])} "
                                    f"doesn't look blended")
                seen_pairs += 1
            else:
                d = polys[a].distance(polys[b])
                if d < 60:
                    from shapely.ops import nearest_points
                    pa_, pb_ = nearest_points(polys[a], polys[b])
                    mx, my = int(round((pa_.x + pb_.x) / 2)), int(round((pa_.y + pb_.y) / 2))
                    covered = any(polys[c].buffer(3).contains(Polygon([(mx, my), (mx + 1, my), (mx, my + 1)]))
                                  for c in st)
                    if not covered:
                        n["gap_reads"] += 1
                        if np.abs(img[my, mx] - bg[my, mx]).sum() > 40:
                            errs.append(f"no visible gap between {sim.label(sc['shapes'][a])} and "
                                        f"{sim.label(sc['shapes'][b])}")
    for q in s["questions"]:
        if q["derivation_key"] == "n_pairs":
            n["frame_answers"] += 1
            if seen_pairs != q["groundtruth"]:
                errs.append(f"{q['qid']} n_pairs: {seen_pairs} overlaps checked on the frame, answer {q['groundtruth']}")
    # 4. glide: track each mover off the frames, then recompute its overlaps from the measured track
    if kind == "glide":
        mv = sim.movers(sc)
        measured = {m: {} for m in mv}
        base = np.asarray(_still_render(sc), dtype=np.int16)
        for f in range(0, N):
            ch = np.abs(frames[f] - base).sum(axis=2) > 60
            x0, y0, x1, y1 = sim.TIMER_BOX
            ch[y0 - 4:y1 + 4, x0 - 4:x1 + 4] = False
            for m in mv:
                x, y, r = sim.pose(sc["shapes"][m], f / FPS)
                exp = Polygon(sim.outline_at(sc["shapes"][m], x, y, r)).buffer(0)
                bx0, by0, bx1, by1 = [int(v) for v in exp.buffer(20).bounds]
                sub = ch[max(0, by0):by1, max(0, bx0):bx1]
                ys, xs = np.nonzero(sub)
                if len(ys) < 50:
                    errs.append(f"moving {sim.label(sc['shapes'][m])} not found at frame {f}")
                    break
                cx, cy = xs.mean() + max(0, bx0), ys.mean() + max(0, by0)
                ex, ey = exp.centroid.x, exp.centroid.y
                measured[m][f] = (x + cx - ex, y + cy - ey, r)
                if f % 6 == 0:
                    n["track_reads"] += 1
                    if math.hypot(cx - ex, cy - ey) > 9:
                        errs.append(f"moving {sim.label(sc['shapes'][m])} at frame {f} is "
                                    f"{math.hypot(cx - ex, cy - ey):.1f} px off its path")
        for m in mv:
            if len(measured[m]) != N:
                continue
            first = {}
            for f in range(N):
                x, y, r = measured[m][f]
                pm = Polygon(sim.outline_at(sc["shapes"][m], x, y, r)).buffer(0)
                for s_ in st:
                    if s_ not in first and pm.intersection(polys[s_]).area >= sim.MIN_OVERLAP:
                        first[s_] = f
            seq = [sim.label(sc["shapes"][i]) for i in sorted(first, key=first.get)]
            for q in s["questions"]:
                key, p = q["derivation_key"], q["params"]
                if p.get("m") != m:
                    continue
                got = None
                if key == "first_touch":
                    got = seq[0] if seq else None
                elif key == "last_touch":
                    got = seq[-1] if seq else None
                elif key == "n_touched":
                    got = len(seq)
                elif key == "touch_order":
                    got = ", then ".join(f"the {x}" for x in seq if
                                         x in {sim.label(sc['shapes'][i]) for i in p["ids"]})
                if got is not None:
                    n["frame_answers"] += 1
                    if str(got) != str(q["groundtruth"]):
                        errs.append(f"{q['qid']} {key}: frames give {got!r}, answer {q['groundtruth']!r}")
                if "f" in p:
                    ok, d = timer_ok(frames[p["f"]], p["f"])
                    n["timer_reads"] += 1
                    if not ok:
                        errs.append(f"{q['qid']} clock at frame {p['f']} doesn't read {sim.timer_text(p['f'])}")
        for t in sim.glide_events(sc):
            f = int(round(t * FPS))
            ok, d = timer_ok(frames[f], f)
            n["timer_reads"] += 1
            if not ok:
                errs.append(f"clock at glide event frame {f} doesn't read {sim.timer_text(f)}")
    return errs, n


def _still_render(sc):
    """Board + still shapes (no movers, no clock), as the renderer draws them."""
    s2 = json.loads(json.dumps(sc))
    s2["shapes"] = [x for x in s2["shapes"] if "path" not in x]
    img = render._board(json.dumps({"theme": sc["theme"], "board": sc["board"]}, sort_keys=True)).copy()
    key = json.dumps(s2, sort_keys=True)
    for i in range(len(s2["shapes"])):
        tile, off = render._tile(key, i)
        img.alpha_composite(tile, dest=off)
    return img.convert("RGB")


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
    print("frame check:", dict(sorted(tot.items())), f"errors: {len(bad)}")
    for e in bad[:40]:
        print("  -", e)
    return len(bad)


def framehash(argv):
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, required=True)
    a = ap.parse_args(argv)
    s = _roundtrip(build_sample(DATASET, sim, questions, a.seed))
    h = hashlib.sha256()
    for f in range(s["params"]["n_frames"]):
        h.update(np.asarray(render.frame_at(s["params"], f), dtype=np.uint8).tobytes())
    print(h.hexdigest())


COMMANDS = {"stats": stats, "tamper": tamper, "framecheck": framecheck, "framehash": framehash}
