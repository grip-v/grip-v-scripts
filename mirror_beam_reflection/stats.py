"""Diversity / balance report over one or more annotations.jsonl files (dry runs are fine).

    python stats.py out_a/annotations.jsonl [out_b/annotations.jsonl ...]

Reports: unique scene signatures (hash of the scene rounded to 3 decimals),
scenario mix, parameter spreads, distinct (question, answer) pairs,
answer-letter balance, per-level question counts, min_fps shares,
requires_motion share on L2-L5 and the derivation_key mix.
"""
from __future__ import annotations
import hashlib
import json
import statistics
import sys
from collections import Counter


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
    vals = sorted(vals)
    q = lambda p: vals[min(len(vals) - 1, int(p * len(vals)))]
    return (f"  {name:<22} min {vals[0]:.3g}  p10 {q(.1):.3g}  median {statistics.median(vals):.3g}  "
            f"p90 {q(.9):.3g}  max {vals[-1]:.3g}")


def main(paths):
    samples = [json.loads(line) for p in paths for line in open(p, encoding="utf8")]
    n = len(samples)
    sigs = Counter(signature(s["params"]) for s in samples)
    scen = Counter(s["params"]["scenario"] for s in samples)
    letters, levels, fps, keys, pairs = Counter(), Counter(), Counter(), Counter(), set()
    motion = [0, 0]
    splits = Counter()
    for s in samples:
        splits[tuple(sum(q["difficulty_level"] == lv for q in s["questions"]) for lv in range(1, 6))] += 1
        for q in s["questions"]:
            letters[q["answer"]] += 1
            levels[q["difficulty_level"]] += 1
            keys[q["derivation_key"]] += 1
            pairs.add((q["question"], q["answer_text"]))
            f = q["params"]["min_fps"]
            fps["<=1" if f <= 1 else "<=2" if f <= 2 else "<=4" if f <= 4 else ">4"] += 1
            if q["difficulty_level"] >= 2:
                motion[0] += q["requires_motion"]
                motion[1] += 1
    nq = sum(letters.values())
    P = [s["params"] for s in samples]
    out = [f"samples: {n}", f"unique scene signatures: {len(sigs)} ({100 * len(sigs) / n:.2f} %)",
           "scenario mix: " + ", ".join(f"{k} {v} ({100 * v / n:.1f} %)" for k, v in sorted(scen.items())),
           "parameter spread:",
           spread("board columns", [p["cols"] for p in P]), spread("board rows", [p["rows"] for p in P]),
           spread("mirrors", [len(p["mirrors"]) for p in P]), spread("absorbers", [len(p["absorbers"]) for p in P]),
           spread("cell size (px)", [p["style"]["cell"] for p in P]),
           spread("speed (cells/s)", [1 / p["step"] for p in P]),
           spread("clip length (s)", [p["frames"] / 24 for p in P]),
           spread("beams", [len(p["emitters"]) for p in P]),
           "  entry edges: " + ", ".join(f"{k} {v}" for k, v in sorted(Counter(e["edge"] for p in P for e in p["emitters"]).items())),
           "  beam colours: " + ", ".join(f"{k} {v}" for k, v in sorted(Counter(e["colour"] for p in P for e in p["emitters"]).items())),
           "  themes / board patterns / grid styles: " + ", ".join(f"{k} {v}" for k, v in sorted(Counter(
               (p["style"]["theme"], p["style"]["board_pat"], p["style"]["grid_style"]) for p in P).items())),
           f"distinct (question, answer) pairs: {len(pairs)} of {nq}",
           "answer letters: " + ", ".join(f"{k} {v} ({100 * v / nq:.2f} %)" for k, v in sorted(letters.items())),
           "questions per level: " + ", ".join(f"L{k} {v}" for k, v in sorted(levels.items())),
           "per-clip level splits (L1..L5): " + ", ".join(f"{k} {v}" for k, v in splits.most_common()),
           "min_fps: " + ", ".join(f"{k} {v} ({100 * v / nq:.1f} %)" for k, v in sorted(fps.items())),
           f"requires_motion on L2-L5: {motion[0]} / {motion[1]} ({100 * motion[0] / max(1, motion[1]):.1f} %)",
           f"derivation keys ({len(keys)}): " + ", ".join(f"{k} {v}" for k, v in sorted(keys.items()))]
    print("\n".join(out))


if __name__ == "__main__":
    main(sys.argv[1:])
