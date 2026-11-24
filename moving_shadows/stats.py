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
import math
import sys
from collections import Counter
from pathlib import Path

sys.path[:0] = [str(Path(__file__).resolve().parent), str(Path(__file__).resolve().parent.parent)]
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
    return (f"  {name:<26} min {vals[0]:.3g}  p10 {q(.1):.3g}  median {statistics.median(vals):.3g}  "
            f"p90 {q(.9):.3g}  max {vals[-1]:.3g}")


def counts(name, c):
    return f"  {name}: " + ", ".join(f"{k} {v}" for k, v in sorted(c.items(), key=lambda kv: str(kv[0])))


def main(paths):
    samples = [json.loads(line) for p in paths for line in open(p, encoding="utf8")]
    n = len(samples)
    sc = [s["params"] for s in samples]
    sigs = Counter(signature(x) for x in sc)
    scen = Counter(x["scenario"] for x in sc)
    letters, levels, fps, keys, pairs = Counter(), Counter(), Counter(), Counter(), set()
    motion, splits = [0, 0], Counter()
    for s in samples:
        splits[tuple(sum(q["difficulty_level"] == lv for q in s["questions"]) for lv in range(1, 6))] += 1
        for q in s["questions"]:
            letters[q["answer"]] += 1
            levels[q["difficulty_level"]] += 1
            keys[q["derivation_key"]] += 1
            pairs.add((q["question"], q["answer_text"]))
            f = q["params"]["min_fps"]
            fps["1" if f <= 1 else "2" if f <= 2 else "3-4" if f <= 4 else ">4"] += 1
            if q["difficulty_level"] >= 2:
                motion[0] += q["requires_motion"]
                motion[1] += 1
    nq = sum(letters.values())
    objs = [o for x in sc for o in x["casters"]]
    fl = [o for o in objs if o["kind"] == "float"]
    stn = [o for o in objs if o["kind"] == "stand"]
    arcs = [x["light"] for x in sc if x["light"]["kind"] == "arc"]
    att = Counter(x["build_attempts"] for x in sc)
    retried = sum(v for k, v in att.items() if k > 1)
    out = [f"samples: {n}", f"unique scene signatures: {len(sigs)} ({100 * len(sigs) / n:.2f} %)",
           "scenario mix: " + ", ".join(f"{k} {v} ({100 * v / n:.1f} %)" for k, v in sorted(scen.items())),
           "parameter spread:",
           spread("objects per clip", [len(x["casters"]) for x in sc]),
           spread("markers per clip", [len(x["markers"]) for x in sc]),
           spread("clip length (s)", [x["frames"] / 24 for x in sc]),
           spread("ground line y (px)", [x["ground"] for x in sc]),
           spread("floating size (radius px)", [o["s"] for o in fl]),
           spread("floating height (px)", [o["ground"] - o["y"] for o in fl]),
           spread("standing height (px)", [o["h"] + o.get("roof", 0) for o in stn]),
           spread("standing width (px)", [o["w"] for o in stn]),
           spread("sun arc width (px)", [a["Rx"] for a in arcs]),
           spread("sun sweep (deg)", [abs(a["phi1"] - a["phi0"]) for a in arcs]),
           spread("light speed (px/s)", [abs(_lx(x, 1) - _lx(x, 0)) / (x["frames"] - 1) * 24 for x in sc]),
           spread("drift speed (px/s)", [abs(o["vx"]) for o in objs if o.get("vx")]),
           spread("rise/sink speed (px/s)", [abs(o["vy"]) for o in objs if o.get("vy")]),
           spread("shaded markers per clip", [_shaded(s) for s in samples]),
           counts("light direction", Counter(_dir(x) for x in sc)),
           counts("floating shapes", Counter(o["shape"] for o in fl)),
           counts("standing kinds", Counter(o["shape"] for o in stn)),
           counts("colours", Counter(o["color"] for o in objs)),
           counts("patterns", Counter(o["pattern"] for o in objs)),
           counts("skies", Counter(x["style"]["sky"] for x in sc)),
           counts("sky patterns", Counter(x["style"]["sky_pattern"] for x in sc)),
           counts("grounds", Counter(x["style"]["ground"] for x in sc)),
           counts("ground textures", Counter(x["style"]["ground_tex"] for x in sc)),
           counts("plates", Counter(x["style"]["plate"] for x in sc)),
           counts("sun colour / style", Counter((x["style"]["sun"], x["style"]["sun_style"]) for x in sc)),
           f"build retries: {retried} of {n} scenes ({100 * retried / n:.1f} %) needed more than one attempt; "
           f"mean attempts {sum(k * v for k, v in att.items()) / n:.2f}, max {max(att)}",
           f"distinct (question, answer) pairs: {len(pairs)} of {nq}",
           "answer letters: " + ", ".join(f"{k} {v} ({100 * v / nq:.2f} %)" for k, v in sorted(letters.items())),
           "questions per level: " + ", ".join(f"L{k} {v}" for k, v in sorted(levels.items())),
           "per-clip level splits (L1..L5): " + ", ".join(f"{k} {v}" for k, v in splits.most_common()),
           "min_fps: " + ", ".join(f"{k} {v} ({100 * v / nq:.1f} %)" for k, v in
                                   sorted(fps.items(), key=lambda kv: ["1", "2", "3-4", ">4"].index(kv[0]))),
           f"requires_motion on L2-L5: {motion[0]} / {motion[1]} ({100 * motion[0] / max(1, motion[1]):.1f} %)",
           f"derivation keys ({len(keys)}): " + ", ".join(f"{k} {v}" for k, v in sorted(keys.items()))]
    print("\n".join(out))


def _lx(x, end):
    lt = x["light"]
    if lt["kind"] == "lamp":
        return lt["x1"] if end else lt["x0"]
    return lt["cx"] + lt["Rx"] * math.cos(math.radians(lt["phi1"] if end else lt["phi0"]))


def _dir(x):
    return "left to right" if _lx(x, 1) > _lx(x, 0) else "right to left"


def _shaded(s):
    un = sim.union(s["params"])
    return sum(1 for j in un if un[j])


if __name__ == "__main__":
    main(sys.argv[1:])
