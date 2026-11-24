"""Independent validation for random_walk_home.

- Re-simulates every scenario from its stored seed and confirms every
  ant's outcome/steps_taken/reset_count and every scenario-level counter
  (home_moved, tile_swap_count, hole_reset_count, death_count,
  total_events, total_ticks) matches exactly.
- Confirms n_ants_home + n_ants_dead == n_ants and that every ant's
  outcome is exactly "home" or "dead" (never left unresolved -- a
  scenario hitting MAX_TICKS unresolved is rejected at generation time,
  never stored).
- Re-derives all 5 questions per video and diffs `groundtruth`/`answer`/
  `options` against storage.
- Decodes every rendered MP4 with OpenCV and checks its frame count is
  close to the value computed from the stored timing schedule.

`--sample` renders 5 videos for a manual watch-and-confirm pass.
"""
from __future__ import annotations
import argparse
import json
import random
from collections import Counter
from pathlib import Path

import cv2

import questions as Q
import simulate as S

HERE = Path(__file__).resolve().parent


def validate(dataset_dir: Path, report_path: Path):
    ann_path = dataset_dir / "annotations.jsonl"
    samples = [json.loads(line) for line in open(ann_path, encoding="utf8")]

    errors = []
    n_questions = 0
    n_ants_dist = Counter()
    outcome_counts = Counter()
    layer_counts = Counter()
    ticks_list = []
    frame_counts = []

    for sample in samples:
        sid = sample["sample_id"]
        p = sample["params"]
        seed = sample["seed"]

        scenario = S.simulate_scenario(seed)

        if scenario.n_ants != p["n_ants"]:
            errors.append(f"{sid}: n_ants mismatch: recomputed={scenario.n_ants} stored={p['n_ants']}")
        if scenario.total_ticks != p["total_ticks"]:
            errors.append(f"{sid}: total_ticks mismatch: recomputed={scenario.total_ticks} stored={p['total_ticks']}")
        if scenario.home_moved != p["home_moved"]:
            errors.append(f"{sid}: home_moved mismatch")
        if scenario.tile_swap_count != p["tile_swap_count"]:
            errors.append(f"{sid}: tile_swap_count mismatch: recomputed={scenario.tile_swap_count} "
                           f"stored={p['tile_swap_count']}")
        if scenario.hole_reset_count != p["hole_reset_count"]:
            errors.append(f"{sid}: hole_reset_count mismatch")
        if scenario.death_count != p["death_count"]:
            errors.append(f"{sid}: death_count mismatch")
        if (scenario.wind_strength != p.get("wind_strength")
                or scenario.wind_shift_tick != p.get("wind_shift_tick")):   # CHANGE: wind
            errors.append(f"{sid}: wind mismatch")
        if [k for k, g in enumerate(scenario.gust_history) if g] != p.get("gust_ticks"):   # CHANGE: gusts
            errors.append(f"{sid}: gust mismatch")
        if scenario.total_events != p["total_events"]:
            errors.append(f"{sid}: total_events mismatch: recomputed={scenario.total_events} "
                           f"stored={p['total_events']}")

        stored_ants = {a["symbol"]: a for a in p["ants"]}
        for a in scenario.ants:
            sa = stored_ants.get(a.symbol)
            if sa is None:
                errors.append(f"{sid}: ant {a.symbol} missing from stored params")
                continue
            if a.outcome != sa["outcome"] or a.steps_taken != sa["steps_taken"] or a.reset_count != sa["reset_count"]:
                errors.append(f"{sid}/{a.symbol}: ant outcome/steps_taken/reset_count mismatch")
            if a.outcome not in ("home", "dead"):
                errors.append(f"{sid}/{a.symbol}: ant left unresolved (outcome={a.outcome})")
            outcome_counts[a.outcome] += 1

        n_home = sum(1 for a in scenario.ants if a.outcome == "home")
        n_dead = scenario.n_ants - n_home
        if n_home != p["n_ants_home"] or n_dead != p["n_ants_dead"]:
            errors.append(f"{sid}: n_ants_home/n_ants_dead mismatch")
        if n_home + n_dead != scenario.n_ants:
            errors.append(f"{sid}: n_ants_home + n_ants_dead != n_ants")

        # --- re-derive questions (no preceding aux_rng draw before build_questions -- see generate.py) ---
        aux_rng = random.Random(seed)
        rederived = Q.build_questions(scenario, aux_rng)
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

        n_ants_dist[p["n_ants"]] += 1
        layer_counts["home_moved"] += int(p["home_moved"])
        layer_counts["has_holes"] += int(p["n_holes"] > 0)
        layer_counts["has_predators"] += int(p["n_predators"] > 0)
        layer_counts["has_tile_swap"] += int(p["tile_swap_count"] > 0)
        layer_counts["has_wind"] += int(p["wind_strength"] is not None)   # CHANGE: wind
        layer_counts["wind_shift"] += int(p["wind_shift_tick"] is not None)
        ticks_list.append(p["total_ticks"])

        video_path = dataset_dir / sample.get("video_path", "")
        if video_path.exists():
            cap = cv2.VideoCapture(str(video_path))
            if not cap.isOpened():
                errors.append(f"{sid}: could not open video {video_path}")
            else:
                n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                frame_counts.append(n_frames)
                expected_frames = p["frame_count"]
                if abs(n_frames - expected_frames) > max(3, int(0.1 * expected_frames)):
                    errors.append(f"{sid}: video frame count {n_frames} far from expected {expected_frames}")
            cap.release()

    lines = ["random_walk_home validation report", "=" * 50, ""]
    lines.append(f"samples checked: {len(samples)}")
    lines.append(f"questions checked: {n_questions}")
    lines.append(f"n_ants distribution: {dict(sorted(n_ants_dist.items()))}")
    lines.append(f"ant outcome distribution: {dict(outcome_counts)}")
    lines.append(f"layer usage: {dict(layer_counts)}")
    if ticks_list:
        lines.append(f"total_ticks: min={min(ticks_list)} max={max(ticks_list)} "
                      f"mean={sum(ticks_list) / len(ticks_list):.1f}")
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
    ap.add_argument("--dataset-dir", default=str(HERE.parent / "output_videos" / "random_walk_home"))
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
