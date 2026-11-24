"""Independent check of the 5-question ladder (see CHANGES.md).

Checks, for every sample:
  - exactly 5 questions q01..q05, one per ladder level, with the ladder's names
    (Direct Observation, Relational Reasoning, Measurement, Multi-step
    Integration, Counterfactual Reasoning) and only that level's variants
  - every answer re-derived here from the saved path / orientations with its
    own die-rolling table (not questions.py or die.py), and exactly one option
    is correct: the stated answer

usage:
    python verify_ladder.py                    # 3000 seeds, built in memory
    python verify_ladder.py --dataset <dir>    # a generated annotations.jsonl
"""
import argparse
import json
import random
import re
import sys
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

LADDER = {
    1: ("Direct Observation", {"start_top", "final_top"}),
    2: ("Relational Reasoning", {"first_direction", "revisit", "opposite_start"}),
    3: ("Measurement", {"total_rolls", "unique_cells", "value_count"}),
    4: ("Multi-step Integration", {"broken_or_confirm", "value_at_n", "bottom_now"}),
    5: ("Counterfactual Reasoning", {"alternate_start", "alternate_direction", "counterfactual_correction"}),
}
LABEL = {"north": "up", "south": "down", "east": "right", "west": "left"}
FROM_LABEL = {v: k for k, v in LABEL.items()}

# rolling written out from the definition (which slot each slot's value moves to)
MOVES = {
    "north": {"top": "north", "north": "bottom", "bottom": "south", "south": "top"},
    "south": {"top": "south", "south": "bottom", "bottom": "north", "north": "top"},
    "east": {"top": "east", "east": "bottom", "bottom": "west", "west": "top"},
    "west": {"top": "west", "west": "bottom", "bottom": "east", "east": "top"},
}


def roll(o, d):
    n = dict(o)
    for src, dst in MOVES[d].items():
        n[dst] = o[src]
    return n


def truth(p, q):
    """The correct answer TEXT for question q, worked out from params p."""
    cells = [tuple(c) for c in p["path_cells"]]
    seq = p["displayed_roll_sequence"]
    shown = [p["initial_orientation"]] + [s["resulting_orientation"] for s in seq]
    dirs = [s["direction"] for s in seq]
    v, text = q["variant"], q["prompt"]
    if v == "start_top":
        return str(shown[0]["top"])
    if v == "final_top":
        return str(shown[-1]["top"])
    if v == "first_direction":
        dr, dc = cells[1][0] - cells[0][0], cells[1][1] - cells[0][1]
        return {(-1, 0): "up", (1, 0): "down", (0, 1): "right", (0, -1): "left"}[(dr, dc)]
    if v == "revisit":
        return "yes" if len(set(cells)) < len(cells) else "no"
    if v == "opposite_start":
        return str(shown[0]["bottom"])
    if v == "total_rolls":
        return str(len(cells) - 1)
    if v == "unique_cells":
        return str(len(set(cells)))
    if v == "value_count":
        target = int(re.search(r"the number (\d+) on its top face", text)[1])
        return str(sum(1 for o in shown[1:] if o["top"] == target))
    if v == "value_at_n":
        n = int(re.search(r"After exactly (\d+) rolls", text)[1])
        return str(shown[n]["top"])
    if v == "bottom_now":
        return f"yes, it is {shown[-1]['bottom']}"
    if v == "broken_or_confirm":
        o = dict(p["initial_orientation"])
        for i, s in enumerate(seq):
            if roll(o, s["direction"]) != s["resulting_orientation"]:
                return f"roll #{i + 1}, should show {roll(o, s['direction'])['top']} on top"
            o = s["resulting_orientation"]
        return "true"
    if v == "counterfactual_correction":
        o = dict(p["initial_orientation"])
        for d in dirs:
            o = roll(o, d)
        return str(o["top"])
    if v == "alternate_start":
        pre = FROM_LABEL[re.search(r"once more, toward the (\w+),", text)[1]]
        o = roll(dict(p["initial_orientation"]), pre)
        if f"bringing the number {o['top']} to the top" not in text:
            return None
        for d in dirs:
            o = roll(o, d)
        return str(o["top"])
    if v == "alternate_direction":
        m = re.search(r"on roll #(\d+) the die had rolled (\w+) instead of (\w+),", text)
        k, alt, was = int(m[1]) - 1, FROM_LABEL[m[2]], FROM_LABEL[m[3]]
        if dirs[k] != was or alt == was:
            return None
        o = dict(p["initial_orientation"])
        for i, d in enumerate(dirs):
            o = roll(o, alt if i == k else d)
        return str(o["top"])
    return None


ap = argparse.ArgumentParser()
ap.add_argument("--seeds", type=int, default=3000)
ap.add_argument("--dataset")
args = ap.parse_args()

if args.dataset:
    samples = [json.loads(l) for l in open(Path(args.dataset) / "annotations.jsonl", encoding="utf8")]
else:
    import generate as G
    samples = [G.build_one(seed) for seed in range(args.seeds)]

fails, variants, letters = [], Counter(), Counter()


def fail(m):
    fails.append(m)
    if len(fails) <= 15:
        print("  FAIL", m)


for s in samples:
    qs, p, sid = s["questions"], s["params"], s["sample_id"]
    if [q["qid"] for q in qs] != ["q01", "q02", "q03", "q04", "q05"]:
        fail(f"{sid}: qids {[q['qid'] for q in qs]}")
        continue
    for lvl, q in enumerate(qs, start=1):
        name, allowed = LADDER[lvl]
        if q["difficulty_level"] != lvl or q["task"] != name or q.get("variant") not in allowed:
            fail(f"{sid} {q['qid']}: level {q['difficulty_level']} / {q['task']} / {q.get('variant')} breaks the ladder")
            continue
        want = truth(p, q)
        if want is None:
            fail(f"{sid} {q['qid']}: prompt does not match the data")
            continue
        right = [k for k, txt in q["options"].items() if str(txt) == want]
        if right != [q["answer"]] or len(set(q["options"].values())) != len(q["options"]):
            fail(f"{sid} {q['qid']} {q['variant']}: correct option(s) {right}, stated {q['answer']}, want {want!r}")
        variants[(lvl, q["variant"])] += 1
        letters[q["answer"]] += 1

print(f"samples: {len(samples)}; questions checked: {sum(variants.values())}")
for lvl in range(1, 6):
    print(f"  L{lvl} {LADDER[lvl][0]}: " + ", ".join(f"{v} {n}" for (l, v), n in sorted(variants.items()) if l == lvl))
print("  answer letters:", dict(sorted(letters.items())))
print(f"RESULT: {'PASS' if not fails else f'{len(fails)} FAILURES'}")
sys.exit(1 if fails else 0)
