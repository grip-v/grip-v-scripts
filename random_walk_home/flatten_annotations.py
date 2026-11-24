"""annotations.jsonl -> dataset_final.csv / dataset_final.jsonl for the
random_walk_home."""
from __future__ import annotations
import argparse
import csv
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent


def flatten(dataset_dir: Path):
    ann_path = dataset_dir / "annotations.jsonl"
    rows = []
    with open(ann_path, encoding="utf8") as f:
        for line in f:
            sample = json.loads(line)
            p = sample["params"]
            for q in sample["questions"]:
                metadata = {
                    "difficulty_score": q["difficulty_level"], "n_ants": p["n_ants"],
                    "n_ants_home": p["n_ants_home"], "n_ants_dead": p["n_ants_dead"],
                    "n_predators": p["n_predators"], "n_holes": p["n_holes"],
                    "home_moved": p["home_moved"], "seed": sample["seed"],
                }
                rows.append({
                    "task": q["task"], "video": sample.get("video_path", ""),
                    "prompt": q["prompt"], "groundtruth": json.dumps(q["groundtruth"]),
                    "answer": q["answer"], "options": json.dumps(q["options"]),
                    "sample_id": sample["sample_id"], "qid": q["qid"],
                    "metadata": json.dumps(metadata),
                })

    csv_path = dataset_dir / "dataset_final.csv"
    jsonl_path = dataset_dir / "dataset_final.jsonl"
    fieldnames = ["sample_id", "qid", "task", "video", "prompt", "groundtruth", "answer", "options", "metadata"]
    with open(csv_path, "w", newline="", encoding="utf8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in rows:
            writer.writerow({k: r[k] for k in fieldnames})
    with open(jsonl_path, "w", encoding="utf8") as f:
        for r in rows:
            f.write(json.dumps({k: r[k] for k in fieldnames}) + "\n")

    print(f"wrote {len(rows)} rows -> {csv_path}, {jsonl_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset-dir", default=str(HERE.parent / "output_videos" / "random_walk_home"))
    args = ap.parse_args()
    flatten(Path(args.dataset_dir))


if __name__ == "__main__":
    main()
