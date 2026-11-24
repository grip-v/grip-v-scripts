"""Independent check of the q04/q05 multiple-choice fix (see CHANGES.md 1b).

For every seed, builds the scene and questions and, for the two fixed
question variants (q04 broken roll, q05 bottom value), checks:
  - 4 options, all different
  - exactly one option is correct, and it is the stated answer
The correct answer is worked out here without using questions.py: q04 by
re-running the rolls with die.roll(), q05 by reading the bottom face of the
final orientation directly.

Optional --original DIR: also builds the questions with an original
(pre-change) questions.py and checks that every other question is identical
and that no groundtruth or prompt changed.

usage:
    python verify_questions.py                       # 5000 seeds
    python verify_questions.py --seeds 500
    python verify_questions.py --original <dir with the old questions.py>
    python verify_questions.py --original <dir> --check-original   # must FAIL
"""
import argparse
import importlib.util
import random
import re
import sys
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import die as D           # noqa: E402
import puzzle as P        # noqa: E402
import questions as Q     # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--seeds", type=int, default=5000)
ap.add_argument("--original", help="folder containing the original questions.py")
ap.add_argument("--check-original", action="store_true",
                help="negative control: run the checks on the ORIGINAL questions.py (must fail)")
args = ap.parse_args()

Qorig = None
if args.original:
    spec = importlib.util.spec_from_file_location("questions_orig", Path(args.original) / "questions.py")
    Qorig = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(Qorig)
if args.check_original:
    if Qorig is None:
        sys.exit("--check-original needs --original")
    Q = Qorig

fails, letters, seen = [], Counter(), Counter()


def fail(msg):
    fails.append(msg)
    if len(fails) <= 15:
        print("  FAIL", msg)


def is_fixed(q, scene):
    # found by variant, not qid: the ladder can place a variant at any level
    return (q.get("variant") == "broken_or_confirm" and scene.content_mode == "broken_rolls") or \
           q.get("variant") == "bottom_now"


for seed in range(args.seeds):
    rng = random.Random(seed)
    sc = P.build_scene(rng)
    new = Q.build_questions(sc, rng)

    if Qorig is not None:
        rng2 = random.Random(seed)
        old = Qorig.build_questions(P.build_scene(rng2), rng2)
        for qn, qo in zip(new, old):
            for k in ("qid", "variant", "prompt", "groundtruth"):
                if qn.get(k) != qo.get(k):
                    fail(f"seed {seed} {qn['qid']}: {k} changed")
            if not is_fixed(qn, sc) and qn != qo:
                fail(f"seed {seed} {qn['qid']}: question that should be untouched changed")

    for q in new:
        if not is_fixed(q, sc):
            continue
        seen[q["variant"]] += 1
        opts = q["options"]
        if len(opts) != 4 or len(set(opts.values())) != 4:
            fail(f"seed {seed} {q['qid']}: options {opts}")

        if q["variant"] == "broken_or_confirm":
            o, truth = dict(sc.initial_orientation), None
            for i, st in enumerate(sc.displayed_roll_sequence):
                correct_here = D.roll(o, st["direction"])
                if truth is None and correct_here != st["resulting_orientation"]:
                    truth = (i + 1, correct_here["top"])
                o = st["resulting_orientation"]

            def is_right(txt):
                m = re.fullmatch(r"roll #(\d+), should show (\d+) on top", txt)
                return m is not None and (int(m[1]), int(m[2])) == truth
        else:
            bottom = sc.displayed_roll_sequence[-1]["resulting_orientation"]["bottom"]

            def is_right(txt):
                return txt == f"yes, it is {bottom}"

        right = [k for k, v in opts.items() if is_right(v)]
        if right != [q["answer"]]:
            fail(f"seed {seed} {q['qid']}: correct options {right}, answer {q['answer']}, {opts}")
        letters[(q["variant"], q["answer"])] += 1

print(f"seeds: {args.seeds}; fixed questions checked: {dict(seen)}"
      + ("" if Qorig is None else "; compared with original questions.py"))
print("answer letters:", {q: {l: letters[(q, l)] for l in "ABCD"} for q in sorted({k[0] for k in letters})})
print(f"RESULT: {'PASS' if not fails else f'{len(fails)} FAILURES'}")
sys.exit(1 if fails else 0)
