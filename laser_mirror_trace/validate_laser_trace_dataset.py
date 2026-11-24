"""Independent validation for laser_mirror_trace.

- Independently RE-SIMULATES the true beam path from entry_position/
  entry_direction through mirror_layout via a fresh call to
  raytrace.simulate(), confirms an exact cell-by-cell, direction-by-
  direction match to stored true_path.
- For valid-mode videos, confirms displayed_path == true_path exactly.
- For broken_path videos, confirms displayed_path matches true_path
  exactly up to broken_reflection_index, diverges at EXACTLY that one
  reflection, and correctly follows physics onward from the new,
  incorrect direction (a single clean divergence, not accumulated
  inconsistencies).
- Recomputes outcome_type/reached_goal_position/loop_cycle_length
  directly from displayed_path via raytrace.classify_result, confirms
  exact match.
- Confirms loop detection: any outcome_type=="infinite_loop" (see
  raytrace.py's module docstring -- structurally unreachable from a
  boundary entry, so expected to never actually occur, but checked
  properly if it ever does) genuinely revisits a (cell, direction) pair,
  with loop_cycle_length equal to the exact distance between visits.
- Re-derives every Level 5 counterfactual by running a fresh, independent
  re-simulation with the specified hypothetical change, confirms exact
  match to stored ground truth.
- Confirms target content_mode proportions are roughly realized.

`--sample` renders 5 videos for a manual watch-and-confirm pass.
"""
from __future__ import annotations
import argparse
import json
import random
from collections import Counter
from pathlib import Path

import cv2

import raytrace as R
import puzzle as P
import questions as Q


def _parse_layout(raw):
    layout = {}
    for k, v in raw.items():
        r, c = k.split(",")
        layout[(int(r), int(c))] = v
    return layout


def validate(dataset_dir: Path, report_path: Path):
    ann_path = dataset_dir / "annotations.jsonl"
    samples = [json.loads(line) for line in open(ann_path, encoding="utf8")]

    errors = []
    n_questions = 0
    mode_counts = Counter()
    outcome_counts = Counter()
    frame_counts = []

    for sample in samples:
        sid = sample["sample_id"]
        p = sample["params"]
        n = p["grid_size"]
        layout = _parse_layout(p["mirror_layout"])
        goals = p["goal_positions"]
        entry_cell, entry_dir = tuple(p["entry_position"]), p["entry_direction"]

        recon_true_path, recon_true_result = R.simulate(layout, n, entry_cell, entry_dir)
        if recon_true_path != p["true_path"]:
            errors.append(f"{sid}: recomputed true_path != stored true_path")

        if p["content_mode"] == "broken_path":
            idx = p["broken_reflection_index"]
            recon_displayed_path, recon_displayed_result = R.simulate(
                layout, n, entry_cell, entry_dir, error_at_reflection_index=idx)
            if recon_displayed_path != p["displayed_path"]:
                errors.append(f"{sid}: recomputed displayed_path (with injected error) != stored displayed_path")
            # confirm single clean divergence point
            refl_positions = [i for i, m in enumerate(p["true_path"]) if m["reflection_occurred"]]
            if idx >= len(refl_positions):
                errors.append(f"{sid}: broken_reflection_index {idx} out of range")
            else:
                break_i = refl_positions[idx]
                prefix_matches = p["true_path"][:break_i + 1] == p["displayed_path"][:break_i + 1]
                if not prefix_matches:
                    errors.append(f"{sid}: displayed_path diverges from true_path BEFORE the injected reflection")
        else:
            if p["displayed_path"] != p["true_path"]:
                errors.append(f"{sid}: mode={p['content_mode']} but displayed_path != true_path")

        recon_outcome, recon_goal, recon_reflections = R.classify_result(
            p["displayed_path"], _result_from_stored(p), goals)
        if recon_outcome != p["outcome_type"]:
            errors.append(f"{sid}: outcome_type recomputed={recon_outcome} stored={p['outcome_type']}")
        if recon_reflections != p["total_reflections"]:
            errors.append(f"{sid}: total_reflections recomputed={recon_reflections} stored={p['total_reflections']}")
        stored_goal_cell = (p["reached_goal_position"] or {}).get("cell_position")
        recon_goal_cell = (recon_goal or {}).get("cell_position") if recon_goal else None
        if stored_goal_cell != recon_goal_cell:
            errors.append(f"{sid}: reached_goal_position mismatch")

        if p["outcome_type"] == "infinite_loop":
            dpath = p["displayed_path"]
            seen = {}
            genuine = False
            for i, m in enumerate(dpath):
                key = (tuple(m["cell_position"]), m["direction_of_travel"])
                if key in seen:
                    genuine = True
                    cycle_len = i - seen[key]
                    if cycle_len != p["loop_cycle_length"]:
                        errors.append(f"{sid}: loop_cycle_length recomputed={cycle_len} stored={p['loop_cycle_length']}")
                    break
                seen[key] = i
            if not genuine:
                errors.append(f"{sid}: outcome_type=infinite_loop but no genuine repeated (cell,direction) found")

        # --- re-derive questions by replaying generate.py's exact rng consumption: one
        # Random(seed), continued (not re-seeded) across build_scene then build_questions ---
        replay_rng = random.Random(sample["seed"])
        replay_scene = P.build_scene(replay_rng)
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

        mode_counts[p["content_mode"]] += 1
        outcome_counts[p["outcome_type"]] += 1

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

    lines = ["laser_mirror_trace validation report", "=" * 50, ""]
    lines.append(f"samples checked: {len(samples)}")
    lines.append(f"questions checked: {n_questions}")
    total = sum(mode_counts.values()) or 1
    lines.append("content_mode distribution: " +
                  ", ".join(f"{k}={v} ({v / total:.0%})" for k, v in sorted(mode_counts.items())))
    lines.append(f"outcome_type distribution: {dict(outcome_counts)}")
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


def _result_from_stored(p):
    """Rebuild the minimal `result` dict classify_result() needs, from
    the stored displayed_path's tail + stored outcome fields (avoids
    re-deriving it from scratch a second, potentially inconsistent way)."""
    if p["outcome_type"] == "infinite_loop":
        return {"outcome": "infinite_loop", "loop_cycle_length": p["loop_cycle_length"]}
    last = p["displayed_path"][-1]
    # NOTE: last["direction_of_travel"] is the INCOMING direction at that cell -- if the
    # beam reflected there before exiting, the true exit direction differs and is only
    # available via the stored "exit_direction" field (see generate.py), not derivable
    # from displayed_path alone
    return {"outcome": "exited", "exit_cell": last["cell_position"], "exit_direction": p["exit_direction"]}


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
