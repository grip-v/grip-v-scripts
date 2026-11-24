"""annotations.jsonl -> dataset_final.csv / dataset_final.jsonl (one row per question).

    python flatten.py --dataset-dir <dir>
"""
from __future__ import annotations
import argparse
import csv
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset-dir", default=str(HERE.parent / "output_videos" / "nested_shapes"))
    ap.add_argument("--ann", default="annotations.jsonl")
    args = ap.parse_args()
    root = Path(args.dataset_dir).resolve()
    rows = []
    for line in open(root / args.ann, encoding="utf8"):
        s = json.loads(line)
        for q in s["questions"]:
            rows.append({
                "task": q["task"],
                "video": s["media_path"],
                "prompt": q["question"] + " Options: " + ", ".join(f"{l}) {o}" for l, o in zip("ABCD", q["options"])),
                "groundtruth": q["answer"],
                "metadata": json.dumps({"sample_id": s["sample_id"], "qid": q["qid"], "rung": q["rung"],
                                        "derivation_key": q["derivation_key"], "groundtruth_value": q["groundtruth"],
                                        "difficulty_level": q["difficulty_level"], "scenario": s["params"]["scenario"],
                                        "n_rings": len(s["params"]["rings"]), "min_fps": q["params"].get("min_fps"),
                                        "requires_motion": q["requires_motion"]}),
            })
    with open(root / "dataset_final.csv", "w", newline="", encoding="utf8") as f:
        w = csv.DictWriter(f, fieldnames=["task", "video", "prompt", "groundtruth", "metadata"])
        w.writeheader()
        w.writerows(rows)
    with open(root / "dataset_final.jsonl", "w", encoding="utf8") as f:
        f.writelines(json.dumps(r) + "\n" for r in rows)
    print(f"wrote {len(rows)} rows -> {root / 'dataset_final.csv'}")


if __name__ == "__main__":
    main()
