"""Extra checks for nested_shapes, driven from run.py:

  python run.py stats --n 5000 [--seed 0] [--jobs 8] [--no-sampling]
        diversity + ladder + min_fps stats over dry-run scenes, and the frame-sampling
        check: every question re-derived from every k-th frame (k = floor(24 / min_fps)),
        every phase, through answers.py
  python run.py tamper --dataset-dir DIR [--k 3]   corrupt k answers, count what gets flagged
  python run.py framehash --seed S                 sha256 of the raw rendered frames of one seed
"""
from __future__ import annotations
import argparse
import hashlib
import json
import random
import statistics
import sys
import warnings
from collections import Counter
from multiprocessing import Pool
from pathlib import Path

import answers
import questions
import render
import sim

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from common import ladder5 as L5  # noqa: E402
from common.video5 import build_sample, _roundtrip  # noqa: E402

DATASET = "nested_shapes"
TIME_KEYS = ("appear_time", "change_time", "mark_time", "recolor_time", "hit_time", "hit_wall_time")


def same(q, val) -> bool:
    """A sampled answer counts as right if it equals the stored one (or, for a
    timer reading, if it is still closest to the stored option)."""
    if q["derivation_key"] in TIME_KEYS and val is not None:
        x = float(str(val).split()[0])
        return min(q["options"], key=lambda o: abs(float(o.split()[0]) - x)) == q["answer_text"]
    return str(val) == str(q["groundtruth"])


def sampling_failures(s: dict, tr=None) -> list[str]:
    """Re-derive every question from every k-th frame, every phase."""
    sc = s["params"]
    tr = tr or answers.track(sc)
    cache, bad = {}, []
    for q in s["questions"]:
        k = max(1, int(sim.FPS / q["params"]["min_fps"] + 1e-9))
        for ph in range(k):
            if (k, ph) not in cache:
                cache[(k, ph)] = answers.subsample(tr, k, ph)
            try:
                val = answers.answer(q["derivation_key"], q["params"], sc, cache[(k, ph)])
            except (IndexError, ValueError, KeyError, StopIteration, TypeError) as e:
                val = f"error {e!r}"
            if not same(q, val):
                bad.append(f"{s['sample_id']}/{q['qid']} {q['derivation_key']} every {k}th frame phase {ph}: "
                           f"{val!r} vs {q['groundtruth']!r}")
                break
    return bad


def _one(args):
    seed, sampling = args
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        s = _roundtrip(build_sample(DATASET, sim, questions, seed))
        sc = s["params"]
        mism = sum(str(answers.answer(q["derivation_key"], q["params"], sc)) != str(q["groundtruth"])
                   for q in s["questions"])
        rings = sc["rings"]
        out = {
            "sig": sim.signature(sc), "scenario": sc["scenario"], "tries": sc["tries"], "n": len(rings),
            "dur": round(sc["n_frames"] / sim.FPS, 2), "theme": sc["style"]["theme"], "bg": sc["style"]["bg_name"],
            "texture": sc["style"]["texture"], "shapes": [r["shape"] for r in rings],
            "colors": [r["color"] for r in rings], "lines": [r["line"] for r in rings],
            "widths": [r["width"] for r in rings], "tabs": sum(r["tab"] for r in rings),
            "outer": round(sim.reach_out(rings[0])), "inner": round(sim.reach_out(rings[-1])),
            "offset": rings[-1]["dx"] != 0 or rings[-1]["dy"] != 0,
            "cx": round(sc["nest"]["x"]), "speeds": [abs(w) for r in rings for _, w in r["spin"] if w],
            "events": len(sim.events(sc)), "marks": len(sc["marks"]), "hits": len(sim.hits(sc)),
            "order": sc["order"],
            "ladder": L5.ladder_errors(s["questions"], "ten"), "check": sim.check(sc), "mism": mism,
            "qs": [(q["difficulty_level"], q["derivation_key"], q["question"], q["answer_text"], q["answer"],
                    q["params"]["min_fps"], q["requires_motion"]) for q in s["questions"]],
            "sampling": sampling_failures(s) if sampling else [],
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
    ap.add_argument("--no-sampling", action="store_true")
    a = ap.parse_args(argv)
    with Pool(a.jobs) as pool:
        rows = pool.map(_one, [(s, not a.no_sampling) for s in range(a.seed, a.seed + a.n)], chunksize=8)
    sigs = {r["sig"] for r in rows}
    print(f"scenes: {a.n}  unique signatures: {len(sigs)} ({100 * len(sigs) / a.n:.2f}%)")
    print("scenario mix:", {k: f"{v} ({100 * v / a.n:.1f}%)" for k, v in sorted(Counter(r["scenario"] for r in rows).items())})
    for k in ("n", "dur", "outer", "inner", "cx", "tabs", "events", "marks", "hits"):
        print(f"  {k:8s}", _spread([r[k] for r in rows]))
    print(f"  {'turn deg/s':8s}", _spread([round(x) for r in rows for x in r["speeds"]]))
    print(f"  rings per scene by scenario:", {sc: dict(sorted(Counter(r["n"] for r in rows if r["scenario"] == sc).items()))
                                             for sc in sim.SCENARIOS})
    print(f"  offset nests: {100 * sum(r['offset'] for r in rows) / a.n:.1f}%")
    for k in ("theme", "bg", "texture", "order"):
        print(f"  {k}: {dict(sorted(Counter(str(r[k]) for r in rows).items()))}")
    for k in ("shapes", "colors", "lines", "widths"):
        print(f"  {k}: {dict(sorted(Counter(x for r in rows for x in r[k]).items()))}")
    qs = [q for r in rows for q in r["qs"]]
    nq = len(qs)
    print(f"questions: {nq}  distinct (question, answer) pairs: {len({(q[2], q[3]) for q in qs})}  "
          f"distinct question texts: {len({q[2] for q in qs})}")
    letters = Counter(q[4] for q in qs)
    print("answer letters:", {k: f"{100 * v / nq:.1f}%" for k, v in sorted(letters.items())})
    lv = Counter(tuple(Counter(q[0] for q in r["qs"])[k] for k in range(1, 6)) for r in rows)
    print("level-count combos (L1..L5):", dict(lv.most_common()))
    per_level = {k: Counter(sum(q[0] == k for q in r["qs"]) for r in rows) for k in range(1, 6)}
    print("questions per level:", {f"L{k}": dict(sorted(v.items())) for k, v in per_level.items()})
    mot = [q[6] for q in qs if q[0] >= 2]
    print(f"L2-L5 requires_motion: {100 * sum(mot) / len(mot):.1f}%")
    fps = [q[5] for q in qs]
    print(f"min_fps: answerable at 1 fps {100 * sum(f <= 1 for f in fps) / nq:.1f}%  at 2 fps "
          f"{100 * sum(f <= 2 for f in fps) / nq:.1f}%  at 4 fps {100 * sum(f <= 4 for f in fps) / nq:.1f}%  "
          f"at 8 fps {100 * sum(f <= 8 for f in fps) / nq:.1f}%  median {statistics.median(fps)}  max {max(fps)}")
    tries = [r["tries"] for r in rows]
    print(f"retries: seeds needing a retry {sum(t > 1 for t in tries)} ({100 * sum(t > 1 for t in tries) / a.n:.1f}%)"
          f", mean attempts/scene {statistics.mean(tries):.2f}, max {max(tries)}, rejected attempts "
          f"{sum(tries) - a.n} ({100 * (sum(tries) - a.n) / sum(tries):.1f}% of attempts)")
    print("templates:", dict(sorted(Counter(f"L{q[0]}:{q[1]}" for q in qs).items())))
    samp = [x for r in rows for x in r["sampling"]]
    print(f"ladder errors: {sum(bool(r['ladder']) for r in rows)}  check failures: {sum(bool(r['check']) for r in rows)}"
          f"  answer mismatches: {sum(r['mism'] for r in rows)}  warnings: {sum(len(r['warnings']) for r in rows)}"
          f"  frame-sampling failures: {len(samp)}")
    for x in samp[:20]:
        print("   ", x)


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
    return flagged


def framehash(argv):
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, required=True)
    a = ap.parse_args(argv)
    sc = _roundtrip(build_sample(DATASET, sim, questions, a.seed))["params"]
    h = hashlib.sha256()
    for f in range(sc["n_frames"]):
        h.update(render.frame_at(sc, f).tobytes())
    print(h.hexdigest())
    return h.hexdigest()


COMMANDS = {"stats": stats, "tamper": tamper, "framehash": framehash}
