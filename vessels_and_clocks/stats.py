"""Diversity / balance report over one or more annotations.jsonl files (dry runs are fine).

    python stats.py out_a/annotations.jsonl [out_b/annotations.jsonl ...]

Reports: unique scene signatures (hash of the scene rounded to 3 decimals),
scenario mix, parameter spreads, distinct (question, answer) pairs,
answer-letter balance, per-level question counts, min_fps shares,
requires_motion share on L2-L5, the derivation_key mix and build retries.
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
    q = lambda p: vals[min(len(vals) - 1, int(p * len(vals)))]  # noqa: E731
    return (f"  {name:<24} min {vals[0]:.3g}  p10 {q(.1):.3g}  median {statistics.median(vals):.3g}  "
            f"p90 {q(.9):.3g}  max {vals[-1]:.3g}")


def counts(name, vals):
    return f"  {name}: " + ", ".join(f"{k} {v}" for k, v in sorted(Counter(vals).items()))


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
    V = [v for p in P for v in p["vessels"]]
    att = Counter(p.get("build_attempts", 1) for p in P)
    retried = sum(v for k, v in att.items() if k > 1)
    out = [f"samples: {n}", f"unique scene signatures: {len(sigs)} ({100 * len(sigs) / n:.2f} %)",
           "scenario mix: " + ", ".join(f"{k} {v} ({100 * v / n:.1f} %)" for k, v in sorted(scen.items())),
           "parameter spread:",
           spread("vessels per clip", [len(p["vessels"]) for p in P]),
           spread("clip length (s)", [p["frames"] / 24 for p in P]),
           spread("vessel width (px)", [v["w"] * 100 for v in V]),
           spread("start level (marks)", [v["h0"] for v in V]),
           spread("tap rise (marks/s)", [v["q"] / v["w"] for v in V if v["role"] == "tap"]),
           spread("steady fall (marks/s)", [v["q"] / v["w"] for v in V if v["role"] == "steady"]),
           spread("leaky first fall (mk/s)", [v["k"] * v["h0"] ** 0.5 / v["w"] for v in V if v["role"] == "leaky"]),
           spread("valve opens (s)", [v["valve"] for v in V if v["role"] == "pair"]),
           spread("in-clip events", [sum(1 for v in p["vessels"] if v["role"] != "pair" and
                                         _law(v) <= (p["frames"] - 1) / 24) + len(p["pairs"]) for p in P]),
           spread("px per mark", [p["layout"]["M"] for p in P]),
           spread("bench y (px)", [p["layout"]["bottom"] for p in P]),
           spread("row left x (px)", [p["layout"]["x"][0][0] for p in P]),
           counts("roles", [v["role"] for v in V]),
           counts("colours", [v["colour"] for v in V]),
           counts("themes", [p["style"]["theme"] for p in P]),
           counts("backgrounds", [p["style"]["bg_pat"] for p in P]),
           counts("grid styles", [p["style"]["grid_style"] for p in P]),
           counts("bench patterns", [p["style"]["bench_pat"] for p in P]),
           counts("plinth patterns", [p["style"]["plinth_pat"] for p in P]),
           counts("rims", [p["style"]["rim"] for p in P]),
           counts("ruler side", [p["layout"]["ruler_side"] for p in P]),
           f"distinct (question, answer) pairs: {len(pairs)} of {nq}",
           "answer letters: " + ", ".join(f"{k} {v} ({100 * v / nq:.2f} %)" for k, v in sorted(letters.items())),
           "questions per level: " + ", ".join(f"L{k} {v}" for k, v in sorted(levels.items())),
           "per-clip level splits (L1..L5): " + ", ".join(f"{k} {v}" for k, v in splits.most_common()),
           "min_fps: " + ", ".join(f"{k} {v} ({100 * v / nq:.1f} %)" for k, v in sorted(fps.items())),
           f"requires_motion on L2-L5: {motion[0]} / {motion[1]} ({100 * motion[0] / max(1, motion[1]):.1f} %)",
           f"derivation keys ({len(keys)}): " + ", ".join(f"{k} {v}" for k, v in sorted(keys.items())),
           f"build retries: {retried} of {n} seeds redrew ({100 * retried / n:.1f} %), "
           f"mean attempts {sum(k * v for k, v in att.items()) / n:.2f}, max {max(att)}"]
    print("\n".join(out))


def _law(v):
    if v["role"] == "tap":
        return (10 - v["h0"]) * v["w"] / v["q"]
    if v["role"] == "steady":
        return v["h0"] * v["w"] / v["q"]
    return 2 * v["w"] * v["h0"] ** 0.5 / v["k"]


if __name__ == "__main__":
    main(sys.argv[1:])
