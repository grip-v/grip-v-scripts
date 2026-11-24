"""The shared 5-rung, closed-loop question ladder.

Two layouts share these helpers: the original 5-question ladder below, and a
10-question ladder (`ten_counts`, `number`, `ladder_errors(qs, mode="ten")`)
with 1-3 / 2-3 / 2-3 / 1-3 / 1-3 questions on levels 1-5. A dataset opts in
by setting `LADDER = "ten"` in its questions.py.

Every clip gets exactly five multiple-choice questions, one per rung:

    q01  L1  observation     what you can count or see directly
    q02  L2  relational      how two things relate
    q03  L3  measurement     a quantity read off the video
    q04  L4  integration     combining several events across the whole clip
    q05  L5  counterfactual  what would happen if one thing were different

"Closed loop": every question stores a `derivation_key` and the `params`
needed to answer it, and each dataset ships an independent `answers.py`
that re-derives the answer from the stored scene alone. The validator
(common/video5.py) fails any clip where that re-derivation disagrees.
"""
from __future__ import annotations
import json
import random

RUNGS = ("observation", "relational", "measurement", "integration", "counterfactual")
TASK_NAMES = ("Direct Observation", "Relational Reasoning", "Measurement",
              "Multi-step Integration", "Counterfactual Reasoning")
LETTERS = "ABCD"


def mc(rng: random.Random, correct, distractors, fmt=str) -> tuple[list[str], str]:
    """Up to 4 options: the correct one plus the first distinct distractors,
    shuffled. Numeric answers are padded with nearby integers if needed."""
    seen = {fmt(correct)}
    opts = [correct]
    for d in distractors:
        if len(opts) == 4:
            break
        if fmt(d) not in seen:
            opts.append(d)
            seen.add(fmt(d))
    bump = 1
    while len(opts) < 4 and isinstance(correct, int) and not isinstance(correct, bool) and bump < 50:
        for cand in (correct + bump, correct - bump):
            if len(opts) < 4 and cand >= 0 and fmt(cand) not in seen:
                opts.append(cand)
                seen.add(fmt(cand))
        bump += 1
    rng.shuffle(opts)
    texts = [fmt(o) for o in opts]
    return texts, LETTERS[texts.index(fmt(correct))]


def near_ints(correct: int, lo: int = 0, hi: int = 10**6) -> list[int]:
    out, step = [], 1
    while len(out) < 3 and step < 60:
        for v in (correct + step, correct - step):
            if lo <= v <= hi and len(out) < 3:
                out.append(v)
        step += 1
    return out


def yes_no(rng: random.Random, gt: str) -> tuple[list[str], str]:
    return mc(rng, gt, ["no" if gt == "yes" else "yes"])


def question(level: int, key: str, text: str, gt, opts: list[str], letter: str,
             requires_motion: bool, params: dict | None = None, idx: int | None = None) -> dict:
    """One question record. `idx` (1-based) sets the qid for 10-question
    ladders; without it the qid is q0<level> as in the 5-question ladder."""
    return {
        "qid": f"q{idx:02d}" if idx else f"q0{level}", "difficulty_level": level, "rung": RUNGS[level - 1],
        "task": TASK_NAMES[level - 1], "question_type": "multiple_choice",
        "derivation_key": key, "question": text, "options": opts, "answer": letter,
        "answer_text": opts[LETTERS.index(letter)], "groundtruth": gt,
        "params": params or {}, "requires_motion": requires_motion,
    }


# 10-question ladder: per-level question counts (min, max); total is exactly TEN_TOTAL.
TEN_RANGES = ((1, 3), (2, 3), (2, 3), (1, 3), (1, 3))
TEN_TOTAL = 10


def ten_counts(rng: random.Random) -> list[int]:
    """Random per-level counts for a 10-question ladder, each inside TEN_RANGES."""
    options = [c for c in _count_combos() if sum(c) == TEN_TOTAL]
    return list(rng.choice(options))


def _count_combos():
    combos = [[]]
    for lo, hi in TEN_RANGES:
        combos = [c + [k] for c in combos for k in range(lo, hi + 1)]
    return [tuple(c) for c in combos]


def number(questions: list[dict]) -> list[dict]:
    """Sort by level (stable) and set qids q01..qNN in that order."""
    out = sorted(questions, key=lambda q: q["difficulty_level"])
    for i, q in enumerate(out, 1):
        q["qid"] = f"q{i:02d}"
    return out


def ladder_errors(questions: list[dict], mode: str = "five") -> list[str]:
    if mode == "ten":
        return _ten_errors(questions)
    errs = []
    if [q.get("qid") for q in questions] != ["q01", "q02", "q03", "q04", "q05"]:
        errs.append("expected qids q01..q05")
    if [q.get("difficulty_level") for q in questions] != [1, 2, 3, 4, 5]:
        errs.append("expected difficulty levels 1..5")
    if [q.get("rung") for q in questions] != list(RUNGS):
        errs.append(f"expected rungs {RUNGS}")
    for q in questions:
        opts = q.get("options", [])
        if len(opts) < 2 or len(set(opts)) != len(opts) or any(o in ("", None) for o in opts):
            errs.append(f"{q.get('qid')}: bad options {opts}")
        elif q["answer"] not in LETTERS[:len(opts)] or opts[LETTERS.index(q["answer"])] != q["answer_text"]:
            errs.append(f"{q.get('qid')}: answer letter doesn't point at answer_text")
    return errs


def _option_errors(q: dict) -> list[str]:
    opts = q.get("options", [])
    if len(opts) < 2 or len(set(opts)) != len(opts) or any(o in ("", None) for o in opts):
        return [f"{q.get('qid')}: bad options {opts}"]
    if q["answer"] not in LETTERS[:len(opts)] or opts[LETTERS.index(q["answer"])] != q["answer_text"]:
        return [f"{q.get('qid')}: answer letter doesn't point at answer_text"]
    return []


def _ten_errors(questions: list[dict]) -> list[str]:
    errs = []
    if len(questions) != TEN_TOTAL:
        errs.append(f"expected {TEN_TOTAL} questions, got {len(questions)}")
    if [q.get("qid") for q in questions] != [f"q{i:02d}" for i in range(1, len(questions) + 1)]:
        errs.append("expected qids q01..qNN in order")
    levels = [q.get("difficulty_level") for q in questions]
    if levels != sorted(levels):
        errs.append("questions are not ordered by level")
    for lv, (lo, hi) in enumerate(TEN_RANGES, 1):
        k = levels.count(lv)
        if not lo <= k <= hi:
            errs.append(f"level {lv}: {k} questions, expected {lo}-{hi}")
    seen = set()
    for q in questions:
        lv = q.get("difficulty_level")
        if lv not in (1, 2, 3, 4, 5) or q.get("rung") != RUNGS[lv - 1]:
            errs.append(f"{q.get('qid')}: level/rung mismatch")
        sig = (q.get("derivation_key"), json.dumps(q.get("params"), sort_keys=True))
        if sig in seen:
            errs.append(f"{q.get('qid')}: repeats derivation_key + params of an earlier question")
        seen.add(sig)
        mf = (q.get("params") or {}).get("min_fps")
        if not isinstance(mf, (int, float)) or isinstance(mf, bool) or mf <= 0:
            errs.append(f"{q.get('qid')}: params.min_fps missing or not a positive number")
        errs.extend(_option_errors(q))
    return errs
