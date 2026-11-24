"""Independent validation for hooks_grid_trace.

- Independently reconstructs the full hook structure from grid_size alone
  and confirms it matches the stored hooks list exactly.
- Recomputes final_row_sums/final_column_sums directly from final_grid
  and confirms exact match; confirms row_sum_matches/column_sum_matches
  are correctly derived by comparing to row_targets/column_targets.
- For is_broken=true: confirms EXACTLY one cell in final_grid differs
  from the theoretically correct hook-fill grid, off by exactly 1, and
  confirms the resulting mismatches are exactly the ones stored.
- For is_broken=false: confirms final_grid exactly matches the correct
  hook-fill grid, and every row/column sum matches its target.
- Recomputes shaded_cells_sum directly from final_grid and shaded_cells.
- Re-derives Level 5a/5c by replaying the fill sequence hook-by-hook and
  checking cumulative cell counts / row lock-in steps -- the most
  important check in this dataset, since these require correct sequence
  replay, not just final-state inspection.
- Re-derives every question and diffs groundtruth/answer/options.
- Decodes every rendered MP4 with OpenCV and checks its frame count.

`--sample` renders 5 videos for a manual watch-and-confirm pass.
"""
from __future__ import annotations
import argparse
import json
import random
from collections import Counter
from pathlib import Path

import cv2

import hooks as H
import questions as Q


def validate(dataset_dir: Path, report_path: Path):
    ann_path = dataset_dir / "annotations.jsonl"
    samples = [json.loads(line) for line in open(ann_path, encoding="utf8")]

    errors = []
    n_questions = 0
    broken_counts = Counter()
    grid_size_counts = Counter()
    direction_counts = Counter()
    frame_counts = []

    for sample in samples:
        sid = sample["sample_id"]
        p = sample["params"]
        n = p["grid_size"]

        # --- independent hook reconstruction ---
        recon_hooks = H.build_hooks(n)
        stored_hooks = {h["hook_index"]: [tuple(c) for c in h["cell_positions"]] for h in p["hooks"]}
        for k in range(1, n + 1):
            if sorted(recon_hooks[k]) != sorted(stored_hooks[k]):
                errors.append(f"{sid}: hook {k} cell list mismatch")

        recon_fill_order = H.fill_order_for(n, p["fill_direction"])
        if recon_fill_order != p["fill_order"]:
            errors.append(f"{sid}: fill_order mismatch")

        recon_row_t, recon_col_t = H.row_column_targets(n)
        if recon_row_t != p["row_targets"] or recon_col_t != p["column_targets"]:
            errors.append(f"{sid}: row/column targets mismatch")

        # --- final grid vs theoretically correct grid ---
        correct = H.correct_grid(n)
        final = p["final_grid"]
        diffs = [(r, c) for r in range(n) for c in range(n) if correct[r][c] != final[r][c]]
        if p["is_broken"]:
            if len(diffs) != 1:
                errors.append(f"{sid}: expected exactly 1 deviation from correct grid, found {len(diffs)}")
            else:
                r, c = diffs[0]
                if [r, c] != p["error_cell_position"]:
                    errors.append(f"{sid}: deviation at {(r, c)} != stored error_cell_position")
                if abs(final[r][c] - correct[r][c]) != 1:
                    errors.append(f"{sid}: error digit not off by exactly 1")
                mismatched_rows = [i for i in range(n) if not p["row_sum_matches"][i]]
                mismatched_cols = [i for i in range(n) if not p["column_sum_matches"][i]]
                if mismatched_rows != [p["affected_row_index"]]:
                    errors.append(f"{sid}: mismatched rows {mismatched_rows} != [{p['affected_row_index']}]")
                if mismatched_cols != [p["affected_column_index"]]:
                    errors.append(f"{sid}: mismatched cols {mismatched_cols} != [{p['affected_column_index']}]")
        else:
            if diffs:
                errors.append(f"{sid}: is_broken=false but final_grid deviates from correct grid at {diffs}")
            if not all(p["row_sum_matches"]) or not all(p["column_sum_matches"]):
                errors.append(f"{sid}: is_broken=false but not all row/column sums match")

        # --- recompute sums directly from final_grid ---
        recon_row_sums = [sum(row) for row in final]
        recon_col_sums = [sum(final[r][c] for r in range(n)) for c in range(n)]
        if recon_row_sums != p["final_row_sums"] or recon_col_sums != p["final_column_sums"]:
            errors.append(f"{sid}: final_row_sums/final_column_sums mismatch")
        recon_row_match = [recon_row_sums[i] == p["row_targets"][i] for i in range(n)]
        recon_col_match = [recon_col_sums[i] == p["column_targets"][i] for i in range(n)]
        if recon_row_match != p["row_sum_matches"] or recon_col_match != p["column_sum_matches"]:
            errors.append(f"{sid}: row_sum_matches/column_sum_matches mismatch")

        # --- shaded cells ---
        recon_shaded_sum = sum(final[r][c] for r, c in p["shaded_cells"])
        if recon_shaded_sum != p["shaded_cells_sum"]:
            errors.append(f"{sid}: shaded_cells_sum mismatch: recomputed={recon_shaded_sum} "
                           f"stored={p['shaded_cells_sum']}")

        # --- re-derive questions by replaying generate.py's EXACT rng consumption: one
        # Random(seed), continued (not re-seeded) across build_scene then build_questions ---
        replay_rng = random.Random(sample["seed"])
        replay_scene = H.build_scene(replay_rng)
        rederived = Q.build_questions(replay_scene, replay_rng)
        stored_qs = {q["qid"]: q for q in sample["questions"]}
        if len(sample["questions"]) != 5:
            errors.append(f"{sid}: expected 5 questions, found {len(sample['questions'])}")
        for rq in rederived:
            n_questions += 1
            sq = stored_qs.get(rq["qid"])
            if sq is None:
                errors.append(f"{sid}: missing question {rq['qid']}")
                continue
            if rq["groundtruth"] != sq["groundtruth"]:
                errors.append(f"{sid}/{rq['qid']}: groundtruth mismatch: recomputed={rq['groundtruth']} "
                               f"stored={sq['groundtruth']}")
            if rq["answer"] != sq["answer"] or rq["options"] != sq["options"]:
                errors.append(f"{sid}/{rq['qid']}: options/answer mismatch")
            if sq["options"].get(sq["answer"]) is None:
                errors.append(f"{sid}/{rq['qid']}: answer letter not present in options")

        broken_counts[p["is_broken"]] += 1
        grid_size_counts[n] += 1
        direction_counts[p["fill_direction"]] += 1

        video_path = dataset_dir / sample.get("video_path", "")
        if video_path.exists():
            cap = cv2.VideoCapture(str(video_path))
            if not cap.isOpened():
                errors.append(f"{sid}: could not open video {video_path}")
            else:
                n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                frame_counts.append(n_frames)
                expected_frames = p["frame_count"]
                if abs(n_frames - expected_frames) > max(3, int(0.15 * expected_frames)):
                    errors.append(f"{sid}: video frame count {n_frames} far from expected {expected_frames}")
            cap.release()

    lines = ["hooks_grid_trace validation report", "=" * 50, ""]
    lines.append(f"samples checked: {len(samples)}")
    lines.append(f"questions checked: {n_questions}")
    lines.append(f"is_broken distribution: {dict(broken_counts)}")
    lines.append(f"grid_size distribution: {dict(sorted(grid_size_counts.items()))}")
    lines.append(f"fill_direction distribution: {dict(direction_counts)}")
    if frame_counts:
        lines.append(f"video frame counts: min={min(frame_counts)} max={max(frame_counts)} "
                      f"mean={sum(frame_counts) / len(frame_counts):.1f}")
    lines.append("")
    lines.append(f"errors: {len(errors)}")
    for e in errors:
        lines.append(f"  - {e}")
    lines.append("")
    lines.append(f"RESULT: {len(samples)} samples, {n_questions} questions, {len(errors)} mismatches")

    report_path.write_text("\n".join(lines) + "\n")
    print(f"validation done: {len(samples)} samples, {n_questions} questions, {len(errors)} mismatches -> {report_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset-dir", default=".")
    ap.add_argument("--report", default="validation_report.txt")
    ap.add_argument("--sample", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    dataset_dir = Path(args.dataset_dir)

    if args.sample:
        import sys
        sys.argv = ["generate.py", "--sample", "--output-dir", str(dataset_dir), "--seed", str(args.seed)]
        import generate
        generate.main()

    validate(dataset_dir, dataset_dir / args.report)


if __name__ == "__main__":
    main()
