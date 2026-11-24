"""Diversity / balance report over one or more annotations.jsonl files (dry runs are fine).

    python stats.py out_a/annotations.jsonl [out_b/annotations.jsonl ...]

Reports (multi_agent_chase): unique scene signatures (hash of the scene rounded to 3 decimals),
scenario mix, parameter spreads, distinct (question, answer) pairs,
answer-letter balance, per-level question counts and min_fps shares.
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
    letters, levels, fps, pairs = Counter(), Counter(), Counter(), set()
    for s in samples:
        for q in s["questions"]:
            letters[q["answer"]] += 1
            levels[q["difficulty_level"]] += 1
            pairs.add((q["question"], q["answer_text"]))
            f = q["params"]["min_fps"]
            fps["<=1" if f <= 1 else "<=2" if f <= 2 else "<=4" if f <= 4 else ">4"] += 1
    nq = sum(letters.values())
    P = [s["params"] for s in samples]
    B = [b for p in P for b in p["balls"]]
    out = [f"samples: {n}", f"unique scene signatures: {len(sigs)} ({100 * len(sigs) / n:.2f} %)",
           "scenario mix: " + ", ".join(f"{k} {v} ({100 * v / n:.1f} %)" for k, v in sorted(scen.items())),
           "parameter spread:",
           spread("balls per clip", [len(p["balls"]) for p in P]),
           spread("hunters per clip", [sum(b["role"] == "hunter" for b in p["balls"]) for p in P]),
           spread("catches per clip", [len(sim.catches(p)) for p in P]),
           spread("clip length (s)", [p["frames"] / 24 for p in P]),
           spread("arena width", [p["arena"][2] - p["arena"][0] for p in P]),
           spread("arena height", [p["arena"][3] - p["arena"][1] for p in P]),
           spread("arena x0", [p["arena"][0] for p in P]),
           spread("arena y0", [p["arena"][1] for p in P]),
           spread("ball radius", [b["r"] for b in B]),
           spread("ball speed", [b["v"] for b in B]),
           spread("pillars (pillars clips)", [len(p["pillars"]) for p in P if "pillars" in p] or [0]),
           spread("zone radius (safe_zone)", [p["zone"]["r"] for p in P if "zone" in p] or [0]),
           counts("colours", [b["color"] for b in B]),
           counts("patterns", [b["pattern"] for b in B]),
           counts("scenario", [p["scenario"] for p in P]),
           counts("style palette", [p["style"]["palette"] for p in P]),
           counts("style floor", [p["style"]["floor"] for p in P]),
           counts("style wall", [p["style"]["wall"] for p in P]),
           counts("layout", [p["layout"] for p in P]),
           counts("pillars count", [len(p.get("pillars", [])) for p in P]),
           f"distinct (question, answer) pairs: {len(pairs)} of {nq}",
           "answer letters: " + ", ".join(f"{k} {v} ({100 * v / nq:.2f} %)" for k, v in sorted(letters.items())),
           "questions per level: " + ", ".join(f"L{k} {v}" for k, v in sorted(levels.items())),
           "min_fps: " + ", ".join(f"{k} {v} ({100 * v / nq:.1f} %)" for k, v in sorted(fps.items()))]
    print("\n".join(out))



if __name__ == "__main__":
    main(sys.argv[1:])
