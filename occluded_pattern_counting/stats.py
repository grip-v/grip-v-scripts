"""Diversity / balance report over a dataset dir or annotations.jsonl files (dry runs are fine).

    python stats.py out_a [out_b/annotations.jsonl ...]

Reports (occluded_pattern_counting): unique scene signatures (hash of the scene rounded to 3 decimals, without
build_attempts), scenario and family mix, parameter spreads (objects, spacing, radius, stops, occluders, clip
length, hidden fraction per hold), build_attempts, per-clip level splits, requires_motion share for L2-L5,
distinct (question, answer) pairs, answer-letter balance and min_fps shares.
"""
from __future__ import annotations
import hashlib
import json
import statistics
import sys
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE), str(HERE.parent)]
import sim  # noqa: E402


def _round(x):
    if isinstance(x, float):
        return round(x, 3)
    if isinstance(x, dict):
        return {k: _round(v) for k, v in x.items() if k != "build_attempts"}
    if isinstance(x, list):
        return [_round(v) for v in x]
    return x


def signature(scene):
    return hashlib.sha256(json.dumps(_round(scene), sort_keys=True).encode()).hexdigest()


def spread(name, vals):
    vals = sorted(vals) or [0]
    q = lambda p: vals[min(len(vals) - 1, int(p * len(vals)))]  # noqa: E731
    return (f"  {name:<24} min {vals[0]:.3g}  p10 {q(.1):.3g}  median {statistics.median(vals):.3g}  "
            f"p90 {q(.9):.3g}  max {vals[-1]:.3g}")


def counts(name, vals):
    return f"  {name}: " + ", ".join(f"{k} {v}" for k, v in sorted(Counter(vals).items()))


def hidden_fracs(scene):
    """Fraction of objects completely hidden at the middle of each hold."""
    tr, n = sim.track(scene), len(scene["objects"])
    out = []
    for a, b in scene["holds"]:
        rects = tr[(a + b) // 2]
        out.append(sum(max([sim.cover(rc, o["x"], o["y"], o["r"]) for rc in rects] or [0]) == 2
                       for o in scene["objects"]) / n)
    return out


def main(paths):
    files = []
    for p in paths:
        p = Path(p)
        files.append(p / "annotations.jsonl" if p.is_dir() else p)
    samples = [json.loads(line) for p in files for line in open(p, encoding="utf8")]
    n = len(samples)
    P = [s["params"] for s in samples]
    sigs = Counter(signature(p) for p in P)
    scen = Counter(p["scenario"] for p in P)
    fam = Counter(p["family"] for p in P)
    letters, levels, fps, pairs = Counter(), Counter(), Counter(), set()
    motion = {lv: [0, 0] for lv in range(1, 6)}
    splits, keys = Counter(), Counter()
    for s in samples:
        per = Counter(q["difficulty_level"] for q in s["questions"])
        splits["/".join(str(per[lv]) for lv in range(1, 6))] += 1
        for q in s["questions"]:
            letters[q["answer"]] += 1
            levels[q["difficulty_level"]] += 1
            keys[q["derivation_key"]] += 1
            pairs.add((q["question"], q["answer_text"]))
            motion[q["difficulty_level"]][0] += bool(q["requires_motion"])
            motion[q["difficulty_level"]][1] += 1
            f = q["params"]["min_fps"]
            fps["<=1" if f <= 1 else "<=2" if f <= 2 else "<=4" if f <= 4 else ">4"] += 1
    nq = sum(letters.values())
    att = [p.get("build_attempts", 1) for p in P]
    out = [f"samples: {n}", f"unique scene signatures: {len(sigs)} ({100 * len(sigs) / n:.2f} %)",
           "scenario mix: " + ", ".join(f"{k} {v} ({100 * v / n:.1f} %)" for k, v in sorted(scen.items())),
           "family mix: " + ", ".join(f"{k} {v} ({100 * v / n:.1f} %)" for k, v in sorted(fam.items())),
           "parameter spread:",
           spread("objects per clip", [len(p["objects"]) for p in P]),
           spread("spacing (px)", [p["spacing"] for p in P]),
           spread("object radius", [p["objects"][0]["r"] for p in P]),
           spread("stops per clip", [len(p["holds"]) for p in P]),
           spread("occluders per clip", [len(p["occluders"]) for p in P]),
           spread("clip length (s)", [p["frames"] / sim.FPS for p in P]),
           spread("hidden fraction/hold", [v for p in P for v in hidden_fracs(p)]),
           counts("shape", [p["shape"] for p in P]),
           counts("background", [p["background"] for p in P]),
           counts("occluder texture", [o["texture"] for p in P for o in p["occluders"]]),
           counts("occluder colour", [o["color"] for p in P for o in p["occluders"]]),
           counts("object colour", [o["color"] for p in P for o in p["objects"]]),
           "build_attempts:",
           spread("attempts", att),
           f"  redrawn at least once: {sum(a > 1 for a in att)} ({100 * sum(a > 1 for a in att) / n:.1f} %), "
           f"mean {statistics.mean(att):.1f}",
           "per-clip level split (L1/L2/L3/L4/L5): " + ", ".join(f"{k} {v}" for k, v in sorted(splits.items())),
           "questions per level: " + ", ".join(f"L{k} {v}" for k, v in sorted(levels.items())),
           "requires_motion share: " + ", ".join(f"L{lv} {100 * m[0] / max(1, m[1]):.1f} %"
                                                  for lv, m in sorted(motion.items()) if lv >= 2),
           "derivation keys: " + ", ".join(f"{k} {v}" for k, v in sorted(keys.items())),
           f"distinct (question, answer) pairs: {len(pairs)} of {nq}",
           "answer letters: " + ", ".join(f"{k} {v} ({100 * v / nq:.2f} %)" for k, v in sorted(letters.items())),
           "min_fps: " + ", ".join(f"{k} {v} ({100 * v / nq:.1f} %)" for k, v in sorted(fps.items()))]
    print("\n".join(out))


if __name__ == "__main__":
    main(sys.argv[1:])
