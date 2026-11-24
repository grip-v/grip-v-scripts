"""Independent check: every question can be answered from what the video shows.

The video shows only the die's TOP number (right-hand digit) and its moves on
the board; the other faces are never seen. For each sample this lists every
die (6 distinct numbers 1-9 on its faces) that explains the shown top numbers:
with no broken roll for normal videos, or with exactly one broken roll (any
roll, any wrong pose of the same die) for broken-roll videos. Every question
whose answer depends on hidden faces must give the SAME answer under all of
them, and that answer must be the stated one.

Own die-rolling table and own search: nothing from questions.py or die.py.

usage:
    python verify_answerable.py --dataset <dir>
    python verify_answerable.py --seeds 300
"""
import argparse
import itertools
import json
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

MOVES = {
    "north": {"top": "north", "north": "bottom", "bottom": "south", "south": "top"},
    "south": {"top": "south", "south": "bottom", "bottom": "north", "north": "top"},
    "east": {"top": "east", "east": "bottom", "bottom": "west", "west": "top"},
    "west": {"top": "west", "west": "bottom", "bottom": "east", "east": "top"},
}
FROM_LABEL = {"up": "north", "down": "south", "right": "east", "left": "west"}


def roll(o, d):
    n = dict(o)
    for a, b in MOVES[d].items():
        n[b] = o[a]
    return n


def poses(o):
    """All 24 poses of a die, by breadth-first rolling."""
    seen, todo, out = set(), [o], []
    while todo:
        x = todo.pop()
        k = tuple(sorted(x.items()))
        if k in seen:
            continue
        seen.add(k)
        out.append(x)
        todo += [roll(x, d) for d in MOVES]
    return out


def explanations(p):
    seq = p["displayed_roll_sequence"]
    dirs = [s["direction"] for s in seq]
    shown = [p["initial_orientation"]["top"]] + [s["resulting_orientation"]["top"] for s in seq]
    broken = p["broken_step_index"] is not None
    out = []
    for combo in itertools.permutations([v for v in range(1, 10) if v != shown[0]], 5):
        h0 = dict(zip(("north", "south", "east", "west", "bottom"), combo), top=shown[0])
        states, o = [h0], h0
        for i, d in enumerate(dirs):
            o = roll(o, d)
            if o["top"] != shown[i + 1]:
                break
            states.append(o)
        if not broken:
            if len(states) == len(dirs) + 1:
                out.append((h0, None, None))
            continue
        all_poses = None
        for j in range(min(len(states), len(dirs))):
            correct = roll(states[j], dirs[j])
            all_poses = all_poses or poses(h0)
            for ow in all_poses:
                if ow["top"] != shown[j + 1] or ow == correct:
                    continue
                o, good = ow, True
                for i in range(j + 1, len(dirs)):
                    o = roll(o, dirs[i])
                    if o["top"] != shown[i + 1]:
                        good = False
                        break
                if good:
                    out.append((h0, j, ow))
    return out, dirs


def physical(o, dirs):
    for d in dirs:
        o = roll(o, d)
    return o


def answer_under(q, h0, j, ow, dirs):
    """The answer text this one explanation gives, or 'n/a' if the question
    does not depend on hidden faces."""
    v, text = q["variant"], q["prompt"]
    if v == "opposite_start":
        return str(h0["bottom"])
    if v == "bottom_now":
        o = h0
        for i, d in enumerate(dirs):
            o = ow if i == j else roll(o, d)
        return f"yes, it is {o['bottom']}"
    if v == "broken_or_confirm":
        if j is None:
            return "true"
        return f"roll #{j + 1}, should show {roll(physical(h0, dirs[:j]), dirs[j])['top']} on top"
    if v == "counterfactual_correction":
        return str(physical(h0, dirs)["top"])
    if v == "alternate_start":
        pre = FROM_LABEL[re.search(r"once more, toward the (\w+),", text)[1]]
        return str(physical(roll(h0, pre), dirs)["top"])
    if v == "alternate_direction":
        m = re.search(r"on roll #(\d+) the die had rolled (\w+) instead", text)
        k, alt = int(m[1]) - 1, FROM_LABEL[m[2]]
        return str(physical(h0, dirs[:k] + [alt] + dirs[k + 1:])["top"])
    return "n/a"


ap = argparse.ArgumentParser()
ap.add_argument("--seeds", type=int, default=300)
ap.add_argument("--dataset")
args = ap.parse_args()
if args.dataset:
    samples = [json.loads(l) for l in open(Path(args.dataset) / "annotations.jsonl", encoding="utf8")]
else:
    import generate as G
    samples = [G.build_one(seed) for seed in range(args.seeds)]

fails = checked = 0
for s in samples:
    expl, dirs = explanations(s["params"])
    if not expl:
        fails += 1
        print("  FAIL", s["sample_id"], "no die explains what is shown")
        continue
    for q in s["questions"]:
        answers = {answer_under(q, *e, dirs) for e in expl}
        if answers == {"n/a"}:
            continue
        checked += 1
        stated = q["options"][q["answer"]]
        if len(answers) != 1 or answers.pop() != stated:
            fails += 1
            if fails <= 10:
                print("  FAIL", s["sample_id"], q["qid"], q["variant"], "not uniquely answerable from the video")
print(f"samples: {len(samples)}; hidden-face questions checked: {checked}")
print(f"RESULT: {'PASS' if not fails else f'{fails} FAILURES'}")
sys.exit(1 if fails else 0)
