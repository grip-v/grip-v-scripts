"""Builds the fixed 5-question ladder for one die_roll_trace video."""
from __future__ import annotations
import itertools
import random

import config as C
import die as D
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


def _die_values(scene: P.SceneData) -> list:
    return sorted(scene.die_face_adjacency.keys())


# CHANGE: new helper. The q04 "broken roll" and q05 "bottom value" questions
# used to offer a single option ("A"), so the answer could be picked without
# watching. They now get wrong options too. Those are chosen and shuffled with
# this separate RNG, seeded from the scene, so the main `rng` stream is not
# touched and every other question/answer in the dataset stays the same.
def _side_rng(scene: P.SceneData, qid: str) -> random.Random:
    return random.Random(f"{qid}-{scene.die_net_layout}-{scene.path_cells}")


# CHANGE: new helper. Wrong options for the broken-roll question. Only one roll
# is broken, so any other roll number is wrong; at the broken roll, any value
# other than the correct one is wrong. The value actually SHOWN after the broken
# roll is offered first, since it is the natural mistake.
def _broken_roll_distractors(scene: P.SceneData, idx: int, correct_top) -> list:
    roll_no = idx + 1
    shown_top = scene.displayed_roll_sequence[idx]["resulting_orientation"]["top"]
    near = [r for r in (roll_no - 1, roll_no + 1, roll_no - 2, roll_no + 2) if 1 <= r <= scene.total_rolls]
    others = [v for v in _die_values(scene) if v not in (correct_top, shown_top)]
    cands = [(roll_no, shown_top)]
    cands += [(r, correct_top) for r in near]
    cands += [(r, shown_top) for r in near]
    cands += [(roll_no, v) for v in others]
    out = []
    for c in cands:
        if c != (roll_no, correct_top) and c not in out:
            out.append(c)
    return out[:3]


# CHANGE: new. The video shows only the TOP number (plus the board moves), so
# the side and bottom faces are never seen directly. These helpers list every
# die that explains what is shown, so a question about hidden faces is only
# asked when all of them give the same answer (it can be worked out from the
# video). Normal videos: dice that explain every top with no broken roll.
# Broken-roll videos: a die plus one broken roll (any roll, any wrong pose).
_SIDES = ("north", "south", "east", "west", "bottom")


def _yaw(o):
    """Turn the die a quarter turn about the vertical axis (top stays on top)."""
    return {"top": o["top"], "bottom": o["bottom"],
            "north": o["west"], "east": o["north"], "south": o["east"], "west": o["south"]}


def _with_top(o, v):
    """The 4 poses of this die that have value v on top."""
    slot = next(k for k, x in o.items() if x == v)
    if slot == "bottom":
        o = D.roll(D.roll(o, "north"), "north")
    elif slot != "top":
        o = D.roll(o, {"south": "north", "north": "south", "west": "east", "east": "west"}[slot])
    out = [o]
    for _ in range(3):
        out.append(_yaw(out[-1]))
    return out


def _explanations(scene):
    """[(start pose, broken roll index or None, pose shown after it or None)]"""
    seq = scene.displayed_roll_sequence
    dirs = [x["direction"] for x in seq]
    shown = [scene.initial_orientation["top"]] + [x["resulting_orientation"]["top"] for x in seq]
    broken = scene.content_mode == "broken_rolls"
    out = []
    for combo in itertools.permutations([v for v in range(1, 10) if v != shown[0]], 5):
        h0 = dict(zip(_SIDES, combo), top=shown[0])
        states, o = [h0], h0
        for i, d in enumerate(dirs):
            o = D.roll(o, d)
            if o["top"] != shown[i + 1]:
                break
            states.append(o)
        if not broken:
            if len(states) == len(dirs) + 1:
                out.append((h0, None, None))
            continue
        for j in range(min(len(states), len(dirs))):          # broken roll at roll j
            correct = D.roll(states[j], dirs[j])
            if shown[j + 1] not in h0.values():
                continue
            for ow in _with_top(h0, shown[j + 1]):
                if ow == correct:
                    continue
                o, good = ow, True
                for i in range(j + 1, len(dirs)):
                    o = D.roll(o, dirs[i])
                    if o["top"] != shown[i + 1]:
                        good = False
                        break
                if good:
                    out.append((h0, j, ow))
    return out


def _unique(expl, f):
    """f's answer if every explanation agrees on it, else None."""
    answers = {f(*e) for e in expl}
    return answers.pop() if len(answers) == 1 else None


def _final_physical(o, dirs):
    for d in dirs:
        o = D.roll(o, d)
    return o


def _final_shown(h0, j, ow, dirs):
    o = h0
    for i, d in enumerate(dirs):
        o = ow if i == j else D.roll(o, d)
    return o


def build_questions(scene: P.SceneData, rng: random.Random) -> list:
    # CHANGE: the 5 questions now form the lead's ladder, one level each:
    #   q01 L1 direct observation      -- read one thing straight off the video
    #   q02 L2 relational reasoning    -- a relation between two things shown
    #   q03 L3 measurement             -- a count/quantity over the whole video
    #   q04 L4 multi-step integration  -- chain several rolls / facts together
    #   q05 L5 counterfactual          -- "what if X had been different"
    # Every answer is closed-form: re-derived from the scene by the validator.
    questions = []
    hist = scene.top_face_value_history
    total_rolls = scene.total_rolls
    seq = scene.displayed_roll_sequence
    values = _die_values(scene)
    dir_label = {"north": "up", "south": "down", "east": "right", "west": "left"}
    dirs = [x["direction"] for x in seq]
    expl = _explanations(scene)                          # CHANGE: dice that fit what is shown

    def add(qid, level, variant, prompt, options, ans, gt):
        questions.append({"qid": qid, "difficulty_level": level, "task": C.TASK_NAME[level], "variant": variant,
                          "prompt": prompt, "options": options, "answer": ans, "groundtruth": gt})

    # --- L1 direct observation ---
    v = rng.choice(["start_top", "final_top"])
    if v == "start_top":
        correct = scene.initial_orientation["top"]
        prompt = "What number is on the top face of the die at the very START of this video, before it first rolls?"
    else:
        correct = hist[-1]
        prompt = "What number is on the top face of the die at the very END of this video?"
    options, ans = _mc(rng, correct, [x for x in values if x != correct][:3])
    add("q01", 1, v, prompt, options, ans, correct)

    # --- L2 relational reasoning ---
    l2 = ["first_direction", "revisit"]
    if _unique(expl, lambda h0, j, ow: h0["bottom"]) is not None:
        l2.append("opposite_start")                      # CHANGE: only if it can be worked out
    v = rng.choice(l2)
    if v == "first_direction":
        correct = seq[0]["direction"]
        options, ans = _mc(rng, correct, [d for d in D.DIRECTIONS if d != correct], formatter=lambda d: dir_label[d])
        prompt = "Relative to its starting cell, in which direction did the die roll on its very FIRST move -- up, down, left, or right?"
    elif v == "revisit":
        correct = "yes" if scene.unique_cells_visited < total_rolls + 1 else "no"
        options, ans = _mc(rng, correct, ["no" if correct == "yes" else "yes"])
        prompt = "Did the die ever return to a cell it had already been on earlier in the video? Answer yes or no."
    else:
        start_top = scene.initial_orientation["top"]
        correct = scene.die_face_adjacency[start_top]["opposite"]
        options, ans = _mc(rng, correct, [x for x in values if x != correct][:3])
        prompt = ("Which number is on the face directly OPPOSITE the face that was on top at the start "
                  "(i.e., which number was on the bottom at the very start)?")
    add("q02", 2, v, prompt, options, ans, correct)

    # --- L3 measurement ---
    v = rng.choice(["total_rolls", "unique_cells", "value_count"])
    if v == "total_rolls":
        correct = total_rolls
        options, ans = _mc(rng, correct, _numeric_distractors(rng, correct, lo=1, spreads=(1, 2, 3)))
        prompt = "How many times did the die roll in this video?"
    elif v == "unique_cells":
        correct = scene.unique_cells_visited
        options, ans = _mc(rng, correct, _numeric_distractors(rng, correct, lo=1, spreads=(1, 2, 3)))
        prompt = ("How many different grid cells did the die stand on in total, including its starting cell "
                  "(counting each cell only once, even if revisited)?")
    else:
        target = rng.choice(values)
        correct = sum(1 for x in hist if x == target)
        options, ans = _mc(rng, correct, _numeric_distractors(rng, correct, lo=0, spreads=(1, 2, 3)))
        prompt = (f"How many times, across the whole video, did the die land with the number {target} on its "
                  "top face (count landings after rolls only, not the starting position)?")
    add("q03", 3, v, prompt, options, ans, correct)

    # --- L4 multi-step integration ---
    l4 = ["value_at_n"]
    if scene.content_mode != "broken_rolls" or _unique(
            expl, lambda h0, j, ow: (j, D.roll(_final_physical(h0, dirs[:j]), dirs[j])["top"])) is not None:
        l4.insert(0, "broken_or_confirm")                # CHANGE: only if the break can be pinned down
    if _unique(expl, lambda h0, j, ow: _final_shown(h0, j, ow, dirs)["bottom"]) is not None:
        l4.append("bottom_now")
    v = rng.choice(l4)
    if v == "broken_or_confirm":
        if scene.content_mode == "broken_rolls":
            idx = scene.broken_step_index
            correct_top = scene.true_roll_sequence[idx]["resulting_orientation"]["top"]
            correct = {"roll_number": idx + 1, "correct_top_value": correct_top}
            options, ans = _mc(_side_rng(scene, "q04"), (idx + 1, correct_top),
                               _broken_roll_distractors(scene, idx, correct_top),
                               formatter=lambda rv: f"roll #{rv[0]}, should show {rv[1]} on top")
            prompt = ("One roll in this video resulted in a physically IMPOSSIBLE orientation given how a real "
                      "die rolls -- which roll number was it, and what value SHOULD have correctly been on top "
                      "after that roll instead?")
        else:
            correct = "true"
            options, ans = _mc(rng, correct, ["false"])
            prompt = "Confirm every roll in this video shows a physically correct resulting orientation -- true or false."
    elif v == "value_at_n":
        n = rng.randrange(1, total_rolls) if total_rolls > 1 else total_rolls
        correct = hist[n - 1]
        options, ans = _mc(rng, correct, [x for x in values if x != correct][:3])
        prompt = f"After exactly {n} rolls, what number was on the top face of the die?"
    else:
        final_top = hist[-1]
        bottom = scene.die_face_adjacency[final_top]["opposite"]
        side = [x for x in values if x not in (bottom, final_top)]
        options, ans = _mc(_side_rng(scene, "q05"), f"yes, it is {bottom}",
                           ["no, it cannot be determined", f"yes, it is {final_top}",
                            f"yes, it is {_side_rng(scene, 'q05-side').choice(side)}"])
        prompt = ("Given everything shown in this video, is it possible to determine what number is on the "
                  "BOTTOM face of the die at the end of the video, even though the bottom face is never "
                  "directly visible? If so, what is it?")
        correct = {"determinable": True, "bottom_value": bottom}
    add("q04", 4, v, prompt, options, ans, correct)

    # --- L5 counterfactual reasoning ---
    # CHANGE: only counterfactuals whose answer can be worked out from the video
    pres = [d for d in D.DIRECTIONS
            if _unique(expl, lambda h0, j, ow, d=d: _final_physical(D.roll(h0, d), dirs)["top"]) is not None]
    swaps = [(k, a) for k in range(total_rolls) for a in D.DIRECTIONS if a != dirs[k]
             and _unique(expl, lambda h0, j, ow, k=k, a=a:
                         _final_physical(h0, dirs[:k] + [a] + dirs[k + 1:])["top"]) is not None]
    pool = (["alternate_start"] if pres else []) + (["alternate_direction"] if swaps else [])
    if scene.content_mode == "broken_rolls" and _unique(
            expl, lambda h0, j, ow: _final_physical(h0, dirs)["top"]) is not None:
        pool.append("counterfactual_correction")
    if not pool:
        raise RuntimeError("no counterfactual question can be worked out from this video")
    v = rng.choice(pool)
    directions_used = [s_["direction"] for s_ in seq]
    if v == "counterfactual_correction":
        # the always-physically-correct sequence already is "the break had been correct"
        correct = scene.true_roll_sequence[-1]["resulting_orientation"]["top"]
        prompt = ("If the die's broken roll had instead shown the physically correct orientation, what would "
                  "the die's FINAL top-face number have been at the end of the video (assuming all later "
                  "rolls went in the same directions as shown)?")
    elif v == "alternate_start":
        # CHANGE: the different start is now fully stated (one extra tip in a
        # named direction). Before, the question only said "number Y on top",
        # which leaves 4 possible turns of the die; in 68% of cases those gave
        # different answers, so the question had no single correct answer.
        pre = rng.choice(pres)
        o = D.roll(dict(scene.initial_orientation), pre)
        y = o["top"]
        for d in directions_used:
            o = D.roll(o, d)
        correct = o["top"]
        prompt = (f"Suppose that before its first roll, the die had been tipped over once more, toward the "
                  f"{dir_label[pre]}, while staying on its starting cell (bringing the number {y} to the top), "
                  "and then made exactly the same sequence of rolls as shown, with real die physics. What would "
                  "its final top-face number have been?")
    else:
        k, alt = rng.choice(swaps)
        o = dict(scene.initial_orientation)
        for i, d in enumerate(directions_used):
            o = D.roll(o, alt if i == k else d)
        correct = o["top"]
        prompt = (f"Suppose that on roll #{k + 1} the die had rolled {dir_label[alt]} instead of "
                  f"{dir_label[directions_used[k]]}, and every other roll went in the same direction as shown, "
                  "with every roll following real die physics. What would the die's final top-face number be?")
    options, ans = _mc(rng, correct, [x for x in values if x != correct][:3])
    add("q05", 5, v, prompt, options, ans, correct)

    return questions
