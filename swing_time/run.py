"""python run.py generate --n 20 --seed 0   |   python run.py validate --dataset-dir <dir>

`validate` runs the shared checks (common/video5.py) and then checks every
question's `option_errors` (the mistake behind each wrong option) against
labels re-derived from the rendered frames (answers.option_errors)."""
import argparse
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))
from common.video5 import dataset_cli  # noqa: E402
from common import ladder5 as L  # noqa: E402
import sim, questions, answers, render  # noqa: E402,F401


def label_errors(q, scene):
    errs = []
    got = q.get("option_errors")
    if not isinstance(got, list) or len(got) != len(q["options"]):
        return ["option_errors missing or not aligned with the options"]
    k = L.LETTERS.index(q["answer"])
    if got[k] is not None or any(t is None for i, t in enumerate(got) if i != k):
        errs.append("option_errors must be empty exactly at the correct option")
    if any(t is not None and t not in answers.MISTAKES for t in got):
        errs.append(f"unknown mistake tag in {got}")
    want = answers.option_errors(q, scene)
    if want != got:
        errs.append(f"mistake tags {got} != re-derived from the frames {want}")
    return errs


def check_labels(argv):
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset-dir", default=str(HERE.parent / "output_videos" / "swing_time"))
    ap.add_argument("--ann", default="annotations.jsonl")
    ap.add_argument("--report", default="validation_report.txt")
    args, _ = ap.parse_known_args(argv)
    root = Path(args.dataset_dir).resolve()
    errs, n = [], 0
    with open(root / args.ann, encoding="utf8") as fh:
        lines_in = fh.readlines()
    for line in lines_in:
        s = json.loads(line)
        for q in s["questions"]:
            n += 1
            errs += [f"{s['sample_id']}/{q['qid']} ({q['derivation_key']}): {e}" for e in label_errors(q, s["params"])]
    lines = ["", f"wrong-option labels: {n} questions checked, {len(errs)} mismatches"] + [f"  - {e}" for e in errs[:300]]
    with open(root / args.report, "a", encoding="utf8") as fh:
        fh.write("\n".join(lines) + "\n")
    print(f"label check done: {n} questions, {len(errs)} mismatches")
    for e in errs[:15]:
        print("  -", e)
    if errs:
        sys.exit(1)


if __name__ == "__main__":
    dataset_cli("swing_time", HERE, sim, questions, answers, render)    # exits 1 on any shared-check failure
    if len(sys.argv) > 1 and sys.argv[1] == "validate":
        check_labels(sys.argv[2:])
