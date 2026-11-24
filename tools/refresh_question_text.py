"""Refresh question text in an existing annotations.jsonl without re-rendering.

    python tools/refresh_question_text.py <dataset> <dataset_dir>

Rebuilds every sample from its seed with the current questions.py and copies
over only the "question" field. Everything else (params, answers, options,
render_meta, media paths) is kept as stored, and the script refuses to write
if any sample differs in anything but question text, since that would mean
the videos are stale too. The old file is kept as annotations.jsonl.bak.
"""
import json
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def main():
    dataset, ddir = sys.argv[1], Path(sys.argv[2])
    sys.path.insert(0, str(ROOT / dataset))
    sys.path.insert(0, str(ROOT))
    from common.video5 import build_sample, _roundtrip
    import sim, questions  # noqa: E401

    ann = ddir / "annotations.jsonl"
    rows = [json.loads(line) for line in ann.read_text(encoding="utf8").splitlines() if line.strip()]
    changed = 0
    for s in rows:
        new = _roundtrip(build_sample(dataset, sim, questions, s["seed"]))
        if new["params"] != s["params"] or len(new["questions"]) != len(s["questions"]):
            sys.exit(f"{s['sample_id']}: scene changed - re-render this dataset instead")
        for old_q, new_q in zip(s["questions"], new["questions"]):
            if {k: v for k, v in old_q.items() if k != "question"} != \
               {k: v for k, v in new_q.items() if k != "question"}:
                sys.exit(f"{s['sample_id']} {old_q['qid']}: more than the text changed - re-render instead")
            if old_q["question"] != new_q["question"]:
                old_q["question"] = new_q["question"]
                changed += 1
    shutil.copy(ann, ann.with_suffix(".jsonl.bak"))
    with open(ann, "w", encoding="utf8") as f:
        for s in rows:
            f.write(json.dumps(s) + "\n")
    print(f"{dataset}: {len(rows)} samples, {changed} question texts updated -> {ann}")


if __name__ == "__main__":
    main()
