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
from pathlib import Path

sys.path[:0] = [str(Path(__file__).resolve().parent)]
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
    q = lambda p: vals[min(len(vals) - 1, int(p * len(vals)))]
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
    runs = [[sim.ball_run(p, b) for b in p["balls"]] for p in P]
    att = [p["build_attempts"] for p in P]
    cum = lambda k: sum(v for kk, v in fps.items() if kk in {"<=1": ["<=1"], "<=2": ["<=1", "<=2"],
                                                                "<=4": ["<=1", "<=2", "<=4"]}[k])
    out = [f"samples: {n}", f"unique scene signatures: {len(sigs)} ({100 * len(sigs) / n:.2f} %)",
           "scenario mix: " + ", ".join(f"{k} {v} ({100 * v / n:.1f} %)" for k, v in sorted(scen.items())),
           f"races with no gap: {sum(1 for p in P if not any(b.get('gaps') for b in p['balls']))} "
           f"({100 * sum(1 for p in P if not any(b.get('gaps') for b in p['balls'])) / n:.1f} %)",
           "parameter spread:",
           spread("balls", [len(p["balls"]) for p in P]),
           spread("gapped tracks", [sum(1 for b in p["balls"] if b.get("gaps")) for p in P]),
           spread("balls that fall", [sum(r["fell"] is not None for r in rr) for rr in runs]),
           spread("arena width W (m)", [p["W"] for p in P]), spread("arena drop D (m)", [p["D"] for p in P]),
           spread("push v0 (m/s)", [p["v0"] for p in P]), spread("slow-motion factor", [p["slow"] for p in P]),
           spread("release t0 (s)", [p["t0"] for p in P]), spread("clip length (s)", [p["frames"] / 24 for p in P]),
           spread("scale (px/m)", [p["style"]["S"] for p in P]), spread("ball radius (px)", [p["style"]["R"] for p in P]),
           spread("track width (px)", [p["style"]["track_w"] for p in P]),
           spread("origin x (px)", [p["style"]["ox"] for p in P]), spread("origin y (px)", [p["style"]["oy"] for p in P]),
           spread("gap length (m)", [g["x1"] - g["x0"] for p in P for b in p["balls"] for g in b.get("gaps", [])]),
           spread("lip angle (deg)", [g["lip"] for p in P for b in p["balls"] for g in b.get("gaps", [])]),
           counts("track families", [b["family"] for p in P for b in p["balls"]]),
           counts("ball colours", [b["colour"] for p in P for b in p["balls"]]),
           counts("ball patterns", [b["pattern"] for p in P for b in p["balls"]]),
           counts("themes", [p["style"]["theme"] for p in P]),
           counts("backgrounds", [p["style"]["bg_pat"] for p in P]),
           counts("track styles", [p["style"]["track_style"] for p in P]),
           counts("mirrored (start on the right)", [p["style"]["mirror"] for p in P]),
           f"build attempts: mean {statistics.mean(att):.2f}, max {max(att)}, "
           f"seeds that redrew at least once: {sum(a > 1 for a in att)} ({100 * sum(a > 1 for a in att) / n:.1f} %)",
           f"distinct (question, answer) pairs: {len(pairs)} of {nq}",
           "answer letters: " + ", ".join(f"{k} {v} ({100 * v / nq:.2f} %)" for k, v in sorted(letters.items())),
           "questions per level: " + ", ".join(f"L{k} {v}" for k, v in sorted(levels.items())),
           f"per-clip level splits (L1..L5), {len(splits)} used: " + ", ".join(f"{k} {v}" for k, v in splits.most_common()),
           "min_fps: " + ", ".join(f"{k} {v} ({100 * v / nq:.1f} %)" for k, v in sorted(fps.items())),
           "answerable at: " + ", ".join(f"{k[2:]} fps {100 * cum(k) / nq:.1f} %" for k in ("<=1", "<=2", "<=4")),
           f"requires_motion on L2-L5: {motion[0]} / {motion[1]} ({100 * motion[0] / max(1, motion[1]):.1f} %)",
           f"derivation keys ({len(keys)}): " + ", ".join(f"{k} {v}" for k, v in sorted(keys.items()))]
    print("\n".join(out))


if __name__ == "__main__":
    main(sys.argv[1:])
