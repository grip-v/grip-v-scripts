"""Builds the fixed 5-question ladder for one hooks_grid_trace video."""
from __future__ import annotations
import random

import config as C
import hooks as H


def _mc(rng, correct, distractors, formatter=str):
    pool = list(dict.fromkeys([correct] + list(distractors)))[:4]
    rng.shuffle(pool)
    letters = "ABCD"[:len(pool)]
    options = {letters[i]: formatter(pool[i]) for i in range(len(pool))}
    return options, letters[pool.index(correct)]


def _numeric_distractors(rng, correct, lo=0, spreads=(1, 2, 3, 4, 5)):
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


def build_questions(scene: H.SceneData, rng: random.Random) -> list:
    questions = []
    n = scene.grid_size

    # --- Level 1: always ---
    options, ans = _mc(rng, n, _numeric_distractors(rng, n, lo=2, spreads=(1, 2)))
    questions.append({"qid": "q01", "difficulty_level": 1, "task": C.TASK_NAME[1],
                       "prompt": "How many hooks make up this grid?",
                       "options": options, "answer": ans, "groundtruth": n})

    # --- Level 2: one of two ---
    l2_variant = rng.choice(["direction", "all_matched"])
    if l2_variant == "direction":
        correct = scene.fill_direction
        options, ans = _mc(rng, correct, ["outward" if correct == "inward" else "inward"])
        prompt = ("Was the grid filled from the outermost hook inward, or from the innermost hook "
                  "outward? Answer 'inward' or 'outward'.")
        gt = correct
    else:
        correct = "no" if scene.is_broken else "yes"
        options, ans = _mc(rng, correct, ["no" if correct == "yes" else "yes"])
        prompt = ("By the end of the video, did every row and column sum match its target value "
                  "shown on the border? Answer yes or no.")
        gt = correct
    questions.append({"qid": "q02", "difficulty_level": 2, "task": C.TASK_NAME[2], "variant": l2_variant,
                       "prompt": prompt, "options": options, "answer": ans, "groundtruth": gt})

    # --- Level 3: pick from applicable variants ---
    l3_pool = ["outermost_digit", "mismatched_row"]
    if scene.fill_direction == "outward":
        l3_pool.append("hook_after_one")
    l3_variant = rng.choice(l3_pool)
    if l3_variant == "outermost_digit":
        options, ans = _mc(rng, n, _numeric_distractors(rng, n, lo=1, spreads=(1, 2)))
        prompt = "What digit was used to fill the outermost hook?"
        gt = n
    elif l3_variant == "hook_after_one":
        correct = 2
        options, ans = _mc(rng, correct, _numeric_distractors(rng, correct, lo=2, spreads=(1, 2, 3)))
        prompt = "Which hook was filled immediately after the hook containing the single '1' cell?"
        gt = correct
    else:
        correct = (scene.affected_row_index + 1) if scene.is_broken else "none"
        if scene.is_broken:
            distractors = _numeric_distractors(rng, correct, lo=1, spreads=(1, 2, 3)) + ["none"]
        else:
            distractors = [str(i + 1) for i in range(min(3, n))]
        options, ans = _mc(rng, correct, distractors)
        prompt = ("If a row's final sum did NOT match its target, which row was it? Answer with the "
                  "row number, or 'none' if all rows matched.")
        gt = correct
    questions.append({"qid": "q03", "difficulty_level": 3, "task": C.TASK_NAME[3], "variant": l3_variant,
                       "prompt": prompt, "options": options, "answer": ans, "groundtruth": gt})

    # --- Level 4: pick from applicable variants ---
    l4_variant = rng.choice(["error_detail", "shaded_sum", "mismatch_count"])
    if l4_variant == "error_detail":
        if scene.is_broken:
            direction_word = "more" if scene.placed_digit > scene.correct_digit else "less"
            correct = f"hook {scene.error_hook_index}, one {direction_word} than correct"
            options, ans = {"A": correct}, "A"
            prompt = ("One cell was filled with an incorrect digit during this video. Which hook was "
                      "it in, and was the placed digit one MORE or one LESS than the correct value "
                      "for that hook?")
            gt = {"error_hook_index": scene.error_hook_index, "direction": direction_word}
        else:
            options, ans = _mc(rng, "true", ["false"])
            prompt = "Confirm no incorrect digit was ever placed in this video -- true or false."
            gt = "true"
    elif l4_variant == "shaded_sum":
        correct = scene.shaded_cells_sum
        options, ans = _mc(rng, correct, _numeric_distractors(rng, correct, lo=0, spreads=(1, 2, 3, 5)))
        prompt = "What is the sum of the digit values in all the shaded cells shown in this grid? Answer with a number."
        gt = correct
    else:
        correct = 2 if scene.is_broken else 0
        options, ans = _mc(rng, correct, [0, 1, 3, 4])
        prompt = ("How many total row-and-column sum mismatches resulted from the single error in "
                  "this video? Answer with a number (0 if no error occurred).")
        gt = correct
    questions.append({"qid": "q04", "difficulty_level": 4, "task": C.TASK_NAME[4], "variant": l4_variant,
                       "prompt": prompt, "options": options, "answer": ans, "groundtruth": gt})

    # --- Level 5: pick from applicable variants ---
    l5_pool = ["midpoint_fraction", "row_lock_in"]
    if scene.is_broken:
        l5_pool.append("counterfactual_fix")
    l5_variant = rng.choice(l5_pool)
    if l5_variant == "midpoint_fraction":
        middle_hook = n // 2
        cumulative = H.cumulative_cells_through(n, scene.hooks, scene.fill_order, middle_hook)
        total_cells = n * n
        correct = "more" if cumulative > total_cells / 2 else "fewer"
        options, ans = _mc(rng, correct, ["fewer" if correct == "more" else "more"])
        prompt = (f"At the exact moment the middle hook (hook number {middle_hook}) finished being "
                  "filled, had MORE than half of the grid's total cells already been filled, or "
                  "fewer? Answer 'more' or 'fewer'.")
        gt = correct
    elif l5_variant == "counterfactual_fix":
        correct = "yes"
        options, ans = _mc(rng, "yes", ["no"])
        prompt = ("If the incorrect digit had instead been placed correctly, would ALL row and "
                  "column sums have matched their targets by the end? Answer yes or no.")
        gt = correct
    else:
        row_x = rng.randrange(n)
        lock_step = H.row_lock_in_step(n, scene.fill_order, row_x)
        locked_early = lock_step < n - 1
        correct = "locked in early (before the last hook)" if locked_early else "only finalized on the very last hook"
        options, ans = _mc(rng, correct, ["only finalized on the very last hook" if locked_early
                                           else "locked in early (before the last hook)"])
        prompt = (f"Was the row-sum target for row {row_x + 1} already fully satisfied at some point "
                  "BEFORE the final hook was filled, or did it only become correct once the very "
                  "last hook was added?")
        gt = correct
    questions.append({"qid": "q05", "difficulty_level": 5, "task": C.TASK_NAME[5], "variant": l5_variant,
                       "prompt": prompt, "options": options, "answer": ans, "groundtruth": gt})

    return questions
