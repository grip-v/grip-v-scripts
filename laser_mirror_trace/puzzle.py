"""Two-pass puzzle assembly for laser_mirror_trace -- picks content_mode,
retries random layouts until one naturally satisfies that mode's
requirement, then (for broken_path) injects the single reflection error.
Zero rendering imports.
"""
from __future__ import annotations
import random
from dataclasses import dataclass

import config as C
import raytrace as R


@dataclass
class SceneData:
    grid_size: int
    mirror_layout: dict            # {(r,c): "/" | "\\"}
    goal_positions: list
    entry_position: list
    entry_direction: str
    entry_edge: str
    true_path: list
    true_result: dict
    true_outcome_type: str
    true_reached_goal: dict | None
    displayed_path: list
    displayed_result: dict
    content_mode: str
    outcome_type: str
    reached_goal_position: dict | None
    total_cells_traversed: int
    total_reflections: int
    broken_reflection_index: int | None
    broken_correct_direction: str | None
    broken_displayed_direction: str | None
    loop_cycle_length: int | None


def _attempt(rng):
    n = rng.choice(range(C.GRID_SIZE_RANGE[0], C.GRID_SIZE_RANGE[1] + 1))
    density = rng.uniform(*C.MIRROR_DENSITY_RANGE)
    mirror_layout = R.random_mirror_layout(rng, n, density)
    entry_cell, entry_dir, entry_edge = R.random_entry(rng, n)
    num_goals = rng.randint(*C.NUM_GOALS_RANGE)
    goals = R.random_goals(rng, n, num_goals)
    true_path, true_result = R.simulate(mirror_layout, n, entry_cell, entry_dir)
    return n, mirror_layout, entry_cell, entry_dir, entry_edge, goals, true_path, true_result


def build_scene(rng: random.Random) -> SceneData:
    modes, weights = zip(*C.MODE_WEIGHTS.items())
    content_mode = rng.choices(modes, weights=weights, k=1)[0]

    chosen = None
    for _ in range(C.MAX_GENERATION_ATTEMPTS):
        n, mirror_layout, entry_cell, entry_dir, entry_edge, goals, true_path, true_result = _attempt(rng)
        true_outcome, true_goal, total_reflections = R.classify_result(true_path, true_result, goals)

        if content_mode == "valid_path_reaches_goal" and true_outcome == "reached_goal":
            chosen = (n, mirror_layout, entry_cell, entry_dir, entry_edge, goals, true_path, true_result,
                      true_outcome, true_goal)
            break
        # true_outcome will in practice only ever be "exited_non_goal" here -- "infinite_loop"
        # is structurally unreachable from a boundary entry (see raytrace.py's module
        # docstring); the check stays general rather than hardcoding that in, since it costs
        # nothing and keeps this branch honest about what it's actually testing for
        if content_mode == "valid_path_no_goal_or_loop" and true_outcome in ("exited_non_goal", "infinite_loop"):
            chosen = (n, mirror_layout, entry_cell, entry_dir, entry_edge, goals, true_path, true_result,
                      true_outcome, true_goal)
            break
        if content_mode == "broken_path" and total_reflections >= 1:
            chosen = (n, mirror_layout, entry_cell, entry_dir, entry_edge, goals, true_path, true_result,
                      true_outcome, true_goal)
            break
    if chosen is None:
        # fall back to whatever the last attempt naturally produced, reclassified honestly --
        # exceedingly rare at these density/size ranges, but never silently force an
        # impossible mode
        n, mirror_layout, entry_cell, entry_dir, entry_edge, goals, true_path, true_result = _attempt(rng)
        true_outcome, true_goal, total_reflections = R.classify_result(true_path, true_result, goals)
        content_mode = {"reached_goal": "valid_path_reaches_goal"}.get(
            true_outcome, "valid_path_no_goal_or_loop")
        chosen = (n, mirror_layout, entry_cell, entry_dir, entry_edge, goals, true_path, true_result,
                  true_outcome, true_goal)

    (n, mirror_layout, entry_cell, entry_dir, entry_edge, goals, true_path, true_result,
     true_outcome, true_goal) = chosen

    broken_reflection_index = broken_correct_direction = broken_displayed_direction = None
    if content_mode == "broken_path":
        n_reflections = sum(1 for p in true_path if p["reflection_occurred"])
        broken_reflection_index = rng.randrange(n_reflections)
        displayed_path, displayed_result = R.simulate(mirror_layout, n, entry_cell, entry_dir,
                                                        error_at_reflection_index=broken_reflection_index)
        # recover the true vs. displayed direction at the break point for ground truth
        true_refl_positions = [i for i, p in enumerate(true_path) if p["reflection_occurred"]]
        break_path_index = true_refl_positions[broken_reflection_index]
        broken_correct_direction = true_path[break_path_index + 1]["direction_of_travel"] \
            if break_path_index + 1 < len(true_path) else true_result.get("exit_direction")
        broken_displayed_direction = displayed_path[break_path_index + 1]["direction_of_travel"] \
            if break_path_index + 1 < len(displayed_path) else displayed_result.get("exit_direction")
    else:
        displayed_path, displayed_result = true_path, true_result

    outcome_type, reached_goal, total_reflections = R.classify_result(displayed_path, displayed_result, goals)
    loop_cycle_length = displayed_result.get("loop_cycle_length") if outcome_type == "infinite_loop" else None

    return SceneData(
        grid_size=n, mirror_layout=mirror_layout, goal_positions=goals,
        entry_position=list(entry_cell), entry_direction=entry_dir, entry_edge=entry_edge,
        true_path=true_path, true_result=true_result, true_outcome_type=true_outcome,
        true_reached_goal=true_goal,
        displayed_path=displayed_path, displayed_result=displayed_result,
        content_mode=content_mode, outcome_type=outcome_type, reached_goal_position=reached_goal,
        total_cells_traversed=len(displayed_path), total_reflections=total_reflections,
        broken_reflection_index=broken_reflection_index, broken_correct_direction=broken_correct_direction,
        broken_displayed_direction=broken_displayed_direction, loop_cycle_length=loop_cycle_length,
    )
