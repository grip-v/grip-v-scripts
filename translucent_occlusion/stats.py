"""Diversity / balance report over one or more annotations.jsonl files (dry runs are fine).

    python stats.py out_a/annotations.jsonl [out_b/annotations.jsonl ...]

Reports (translucent_occlusion): unique scene signatures (hash of the scene rounded to 3 decimals, without
build_attempts), scenario mix, parameter spreads (actors, panels, radius, alpha, duration, speeds, events),
per-clip level-split distribution, requires_motion share for L2-L5, build_attempts distribution, distinct
(question, answer) pairs, answer-letter balance and min_fps shares.
"""
from __future__ import annotations
import hashlib
import json
import math
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


def speeds(scene):
    """px/s of every keyframed segment with movement."""
    out = []
    for a in scene["actors"]:
        for (fa, xa, ya), (fb, xb, yb) in zip(a["kf"], a["kf"][1:]):
            d = math.hypot(xb - xa, yb - ya)
            if fb > fa and d > 0:
                out.append(d / (fb - fa) * sim.FPS)
    return out


def main(paths):
    samples = [json.loads(line) for p in paths for line in open(p, encoding="utf8")]
    n = len(samples)
    P = [s["params"] for s in samples]
    sigs = Counter(signature(p) for p in P)
    scen = Counter(p["scenario"] for p in P)
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
    A = [a for p in P for a in p["actors"]]
    ev = [sim.events(p) for p in P]
    att = [p.get("build_attempts", 1) for p in P]
    out = [f"samples: {n}", f"unique scene signatures: {len(sigs)} ({100 * len(sigs) / n:.2f} %)",
           "scenario mix: " + ", ".join(f"{k} {v} ({100 * v / n:.1f} %)" for k, v in sorted(scen.items())),
           "parameter spread:",
           spread("actors per clip", [len(p["actors"]) for p in P]),
           spread("panels per clip", [len(p["panels"]) for p in P]),
           spread("actor radius", [a["r"] for a in A]),
           spread("panel alpha", [p["alpha"] for p in P]),
           spread("clip length (s)", [p["frames"] / sim.FPS for p in P]),
           spread("segment speed (px/s)", [v for p in P for v in speeds(p)]),
           spread("events per clip", [len(e) for e in ev]),
           spread("first event (s)", [e[0][0] / sim.FPS for e in ev if e]),
           spread("changes per clip", [sum(bool(a["changes"]) for a in p["actors"]) for p in P]),
           counts("actor kind", [a["kind"] for a in A]),
           counts("colours", [a["color"] for a in A]),
           counts("shapes", [a["shape"] for a in A]),
           counts("panel kind", [pp["kind"] for p in P for pp in p["panels"]]),
           counts("background", [p["background"] for p in P]),
           counts("floor", [p["floor"] for p in P]),
           counts("tint", [p["tint"] for p in P]),
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
