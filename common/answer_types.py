"""SHARED_CONTRACT.md §5: answer-type helpers and the per-sample type-mix
assertion.

These are pure helpers for a dataset's `questions.py` to build §2-shaped
question dicts with correctly-populated answer fields, plus the one
generic, dataset-independent check §5 requires: the type-mix quota over a
single sample's 10 questions.
"""
from __future__ import annotations
import random
from collections import Counter


class AnswerTypeError(ValueError):
    pass


MC_LETTERS = ("A", "B", "C", "D")

# §5 per-sample quotas.
TYPE_MIX_MINIMUMS = {
    "multiple_choice": 3,
    "numeric": 2,
    "count": 1,        # "≥1 count"
}
# "≥1 ordering or boolean" -- an either/or minimum, not a per-type one.
ORDERING_OR_BOOLEAN_MINIMUM = 1
TYPE_MIX_MAXIMUM_ANY = 4


def is_close(a: float, b: float, tolerance: float, relative: bool = False) -> bool:
    if relative:
        return abs(a - b) <= tolerance * max(abs(b), 1e-12)
    return abs(a - b) <= tolerance


def distinct_from(candidate, correct, tolerance: float = 0.0) -> bool:
    """True if `candidate` is NOT within `tolerance` of `correct` -- the
    guard MC distractor generation must apply against the true answer
    (numeric: absolute distance; anything else: plain inequality)."""
    if isinstance(correct, (int, float)) and isinstance(candidate, (int, float)):
        return not is_close(float(candidate), float(correct), tolerance)
    return candidate != correct


def numeric_answer(value: float, tolerance: float, unit: str | None = None,
                    text: str | None = None) -> dict:
    return dict(
        question_type="numeric",
        answer=value,
        answer_text=text if text is not None else (f"{value} {unit}" if unit else str(value)),
        answer_tolerance=tolerance,
        unit=unit,
    )


def count_answer(value: int, text: str | None = None) -> dict:
    return dict(
        question_type="count",
        answer=value,
        answer_text=text if text is not None else str(value),
    )


def boolean_answer(value: bool, text: str | None = None) -> dict:
    return dict(
        question_type="boolean",
        answer=bool(value),
        answer_text=text if text is not None else ("yes" if value else "no"),
    )


def ordering_answer(order: list[str], text: str | None = None) -> dict:
    return dict(
        question_type="ordering",
        answer=list(order),
        answer_text=text if text is not None else " < ".join(str(x) for x in order),
    )


def short_text_answer(value: str) -> dict:
    return dict(question_type="short_text", answer=value, answer_text=value)


def make_mc_options(correct_text: str, distractors: list[str], rng: random.Random,
                     letters: tuple[str, ...] = MC_LETTERS) -> tuple[list[str], str]:
    """Shuffle `correct_text` in among `distractors` (must supply exactly
    len(letters)-1 of them, already vetted with `distinct_from`) and return
    (options_in_letter_order, correct_letter). Uses the caller's own `rng`
    so option placement is reproducible under the dataset's own --seed."""
    if len(distractors) != len(letters) - 1:
        raise AnswerTypeError(f"need exactly {len(letters) - 1} distractors, got {len(distractors)}")
    pool = list(distractors) + [correct_text]
    rng.shuffle(pool)
    correct_letter = letters[pool.index(correct_text)]
    return pool, correct_letter


def assert_mc_option_sanity(question: dict) -> list[str]:
    """§7 assertion 4: MC distractors distinct, non-empty, none numerically
    within `answer_tolerance` of the correct answer."""
    errors = []
    if question.get("question_type") != "multiple_choice":
        return errors
    options = question.get("options") or []
    if len(options) != 4:
        errors.append(f"{question.get('qid')}: expected 4 MC options, got {len(options)}")
        return errors
    if any(not o for o in options):
        errors.append(f"{question.get('qid')}: an MC option is empty")
    if len(set(options)) != len(options):
        errors.append(f"{question.get('qid')}: MC options are not all distinct: {options}")
    tol = question.get("answer_tolerance")
    answer_letter = question.get("answer")
    if tol is not None and answer_letter in MC_LETTERS:
        try:
            correct_val = float(options[MC_LETTERS.index(answer_letter)])
        except (ValueError, TypeError):
            correct_val = None
        if correct_val is not None:
            for letter, opt in zip(MC_LETTERS, options):
                if letter == answer_letter:
                    continue
                try:
                    val = float(opt)
                except (ValueError, TypeError):
                    continue
                if is_close(val, correct_val, tol):
                    errors.append(
                        f"{question.get('qid')}: distractor {letter}={opt!r} is within "
                        f"answer_tolerance={tol} of the correct answer {correct_val}"
                    )
    return errors


def assert_type_mix(questions: list[dict]) -> dict:
    """§5's per-sample quota check. Returns the type counts on success;
    raises AnswerTypeError (naming exactly which quota failed) otherwise."""
    counts = Counter(q["question_type"] for q in questions)
    errors = []
    for qtype, minimum in TYPE_MIX_MINIMUMS.items():
        if counts.get(qtype, 0) < minimum:
            errors.append(f"need >= {minimum} {qtype!r} questions, got {counts.get(qtype, 0)}")
    if counts.get("ordering", 0) + counts.get("boolean", 0) < ORDERING_OR_BOOLEAN_MINIMUM:
        errors.append(
            f"need >= {ORDERING_OR_BOOLEAN_MINIMUM} 'ordering' or 'boolean' question(s), "
            f"got ordering={counts.get('ordering', 0)}, boolean={counts.get('boolean', 0)}"
        )
    for qtype, n in counts.items():
        if n > TYPE_MIX_MAXIMUM_ANY:
            errors.append(f"no more than {TYPE_MIX_MAXIMUM_ANY} questions of one type allowed, "
                           f"got {n} of {qtype!r}")
    if errors:
        raise AnswerTypeError("type-mix quota violated: " + "; ".join(errors))
    return dict(counts)
