"""Builds the fixed 5-question ladder for one laser_mirror_trace video."""
from __future__ import annotations
import random

import config as C
import raytrace as R
import puzzle as P


def _mc(rng, correct, distractors, formatter=str):
    pool = list(dict.fromkeys([correct] + list(distractors)))[:4]
    rng.shuffle(pool)
    letters = "ABCD"[:len(pool)]
    options = {letters[i]: formatter(pool[i]) for i in range(len(pool))}
    return options, letters[pool.index(correct)]


def _numeric_distractors(rng, correct, lo=0, spreads=(1, 2, 3, 4)):
    out = set()
    for s in spreads:
        for v in (correct - s, correct + s):
            if v != correct and v >= lo:
                out.add(v)
        if len(out) >= 6:
            break
    out = list(out)
    rng.shuffle(out)
    return out[:3]


def _edge_of(cell, n):
    r, c = cell
    if r == 0:
        return "top"
    if r == n - 1:
        return "bottom"
    if c == 0:
        return "left"
    return "right"


def build_questions(scene: P.SceneData, rng: random.Random) -> list:
    questions = []
    n = scene.grid_size
    dpath = scene.displayed_path

    # --- Level 1: always ---
    correct = scene.total_reflections
    options, ans = _mc(rng, correct, _numeric_distractors(rng, correct, lo=0, spreads=(1, 2)))
    questions.append({"qid": "q01", "difficulty_level": 1, "task": C.TASK_NAME[1],
                       "prompt": "How many mirror cells did the laser pass through before the video ends?",
                       "options": options, "answer": ans, "groundtruth": correct})

    # --- Level 2: one of two ---
    l2_variant = rng.choice(["reached_goal", "stuck_in_loop"])
    if l2_variant == "reached_goal":
        correct = "yes" if scene.outcome_type == "reached_goal" else "no"
        prompt = "Did the laser reach one of the marked goal arrows? Answer yes or no."
    else:
        correct = "yes" if scene.outcome_type == "infinite_loop" else "no"
        prompt = "Did the laser get stuck in a repeating loop instead of ever exiting the grid? Answer yes or no."
    options, ans = _mc(rng, correct, ["no" if correct == "yes" else "yes"])
    questions.append({"qid": "q02", "difficulty_level": 2, "task": C.TASK_NAME[2], "variant": l2_variant,
                       "prompt": prompt, "options": options, "answer": ans, "groundtruth": correct})

    # --- Level 3: pick one of three ---
    l3_variant = rng.choice(["direction_after_last_reflection", "direction_change_count", "entry_edge"])
    if l3_variant == "direction_after_last_reflection":
        reflections = [p for p in dpath if p["reflection_occurred"]]
        if reflections:
            last_idx = dpath.index(reflections[-1])
            correct = dpath[last_idx + 1]["direction_of_travel"] if last_idx + 1 < len(dpath) \
                else scene.displayed_result.get("exit_direction", reflections[-1]["direction_of_travel"])
        else:
            correct = scene.entry_direction
        options, ans = _mc(rng, correct, [d for d in ("up", "down", "left", "right") if d != correct])
        prompt = "What direction was the laser traveling immediately AFTER its very LAST reflection shown in the video?"
        gt = correct
    elif l3_variant == "direction_change_count":
        correct = scene.total_reflections
        options, ans = _mc(rng, correct, _numeric_distractors(rng, correct, lo=0, spreads=(1, 2)))
        prompt = "How many total times did the beam change direction during this video?"
        gt = correct
    else:
        correct = scene.entry_edge
        options, ans = _mc(rng, correct, [e for e in ("top", "bottom", "left", "right") if e != correct])
        prompt = "Which edge of the grid (top, bottom, left, or right) did the laser originally enter from?"
        gt = correct
    questions.append({"qid": "q03", "difficulty_level": 3, "task": C.TASK_NAME[3], "variant": l3_variant,
                       "prompt": prompt, "options": options, "answer": ans, "groundtruth": gt})

    # --- Level 4: pick one of three ---
    l4_variant = rng.choice(["broken_or_confirm", "outcome_description", "loop_length_or_na"])
    if l4_variant == "broken_or_confirm":
        if scene.content_mode == "broken_path":
            correct = {"reflection_number": scene.broken_reflection_index + 1,
                       "correct_direction": scene.broken_correct_direction}
            options, ans = {"A": f"reflection #{correct['reflection_number']}, should go "
                                  f"{correct['correct_direction']}"}, "A"
            prompt = ("One reflection in this video did NOT match the actual orientation of the mirror it "
                      "bounced off -- which reflection number was it (counting from the first), and what "
                      "direction should the beam have correctly gone instead?")
            gt = correct
        else:
            correct = "true"
            options, ans = _mc(rng, correct, ["false"])
            prompt = "Confirm every reflection in this video correctly matches its mirror's orientation -- true or false."
            gt = correct
    elif l4_variant == "outcome_description":
        if scene.outcome_type == "reached_goal":
            correct = f"goal at {tuple(scene.reached_goal_position['cell_position'])}"
            gt = scene.reached_goal_position
        elif scene.outcome_type == "infinite_loop":
            correct = "stuck in a loop"
            gt = "infinite_loop"
        else:
            correct = "exited the grid without hitting a goal"
            gt = "exited_non_goal"
        options, ans = {"A": correct}, "A"
        prompt = ("If the laser reached a goal, which one (state its edge position)? If it did not, say "
                  "whether it exited the grid without hitting a goal, or became stuck in a loop.")
    else:
        if scene.outcome_type == "infinite_loop":
            correct = scene.loop_cycle_length
            options, ans = _mc(rng, correct, _numeric_distractors(rng, correct, lo=1, spreads=(1, 2)))
            gt = correct
        else:
            correct = "not applicable"
            options, ans = {"A": correct}, "A"
            gt = correct
        prompt = ("If the video shows the laser stuck in a loop, how many cells long is the repeating cycle? "
                  "If it is not stuck in a loop, answer 'not applicable'.")
    questions.append({"qid": "q04", "difficulty_level": 4, "task": C.TASK_NAME[4], "variant": l4_variant,
                       "prompt": prompt, "options": options, "answer": ans, "groundtruth": gt})

    # --- Level 5: pick from applicable variants ---
    l5_pool = ["flip_mirror_on_path"]
    path_cells = {tuple(p["cell_position"]) for p in scene.true_path}
    if any(cell not in path_cells for cell in scene.mirror_layout):
        l5_pool.append("remove_unrelated_mirror")
    if scene.content_mode == "broken_path":
        l5_pool.append("corrected_reflection_outcome")
    l5_variant = rng.choice(l5_pool)

    if l5_variant == "flip_mirror_on_path":
        mirror_cells_on_path = [tuple(p["cell_position"]) for p in dpath if p["is_mirror_cell"]]
        target = mirror_cells_on_path[rng.randrange(len(mirror_cells_on_path))]
        flipped_layout = dict(scene.mirror_layout)
        flipped_layout[target] = "\\" if flipped_layout[target] == "/" else "/"
        new_path, new_result = R.simulate(flipped_layout, n, tuple(scene.entry_position), scene.entry_direction)
        new_outcome, new_goal, _ = R.classify_result(new_path, new_result, scene.goal_positions)
        if new_outcome == "reached_goal":
            same_goal = scene.outcome_type == "reached_goal" and \
                tuple(new_goal["cell_position"]) == tuple((scene.reached_goal_position or {}).get("cell_position", ()))
            correct = "reaches the SAME goal" if same_goal else f"reaches a DIFFERENT goal at {tuple(new_goal['cell_position'])}"
        elif new_outcome == "infinite_loop":
            correct = "gets stuck in a loop"
        else:
            correct = "exits with no goal reached"
        options, ans = {"A": correct}, "A"
        prompt = (f"If the mirror at cell {target} (a mirror the beam actually passed through) had been the "
                  "OTHER orientation instead, would the laser have ended up reaching a different goal, no "
                  "goal at all, or gotten stuck in a loop? Trace the resulting path to find out.")
        gt = {"outcome": new_outcome, "goal": new_goal}
    elif l5_variant == "remove_unrelated_mirror":
        path_cells = {tuple(p["cell_position"]) for p in scene.true_path}
        candidates = [cell for cell in scene.mirror_layout if cell not in path_cells]
        if candidates:
            target = candidates[rng.randrange(len(candidates))]
            reduced_layout = dict(scene.mirror_layout)
            del reduced_layout[target]
            new_path, new_result = R.simulate(reduced_layout, n, tuple(scene.entry_position), scene.entry_direction)
            unaffected = new_path == scene.true_path
            correct = "no" if unaffected else "yes"
        else:
            target, correct = None, "no"
        options, ans = _mc(rng, correct, ["no" if correct == "yes" else "yes"])
        prompt = (f"If the mirror at cell {target} (a mirror the beam did NOT actually pass through in this "
                  "video) were instead REMOVED entirely, would that change this laser's path at all? Answer "
                  "yes or no.")
        gt = correct
    else:
        true_final_outcome = scene.true_outcome_type
        displayed_outcome = scene.outcome_type
        if true_final_outcome == "reached_goal" and displayed_outcome == "reached_goal" and \
                tuple((scene.true_reached_goal or {}).get("cell_position", ())) == \
                tuple((scene.reached_goal_position or {}).get("cell_position", ())):
            correct = "the SAME goal"
        elif true_final_outcome == "reached_goal":
            correct = "a DIFFERENT goal"
        else:
            correct = "no goal"
        options, ans = {"A": correct}, "A"
        prompt = ("For the video's actual broken reflection, if you continued tracing the beam using the "
                  "CORRECT reflection at that point instead of the displayed (wrong) one, would it have "
                  "reached the SAME goal shown in this video, a DIFFERENT goal, or no goal? Compute the "
                  "fully-corrected path to answer.")
        gt = {"true_outcome": true_final_outcome, "true_goal": scene.true_reached_goal, "answer": correct}
    questions.append({"qid": "q05", "difficulty_level": 5, "task": C.TASK_NAME[5], "variant": l5_variant,
                       "prompt": prompt, "options": options, "answer": ans, "groundtruth": gt})

    return questions
