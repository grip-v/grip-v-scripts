"""SHARED_CONTRACT.md §7 assertion 5 ("answer distribution"), as a
post-generation REBALANCING pass rather than just a check: if a dataset's
MC correct-option letter distribution is skewed, permute affected
questions' option ORDER (never which content is correct, never the answer's
actual meaning) so the letter carrying the correct answer is redistributed
towards uniform, then the caller rewrites annotations.jsonl with the
mutated records.

(EXPANSION_PROMPTS.md's Prompt 0 refers to this as "§7.5" -- SHARED_CONTRACT.md
itself numbers this as item 5 under §7 ("Answer distribution"), there is no
literal "§7.5" heading. Treated as the same thing; noted here rather than
left as a silent mismatch.)
"""
from __future__ import annotations
from collections import Counter

from .validate_common import assert_answer_distribution

MC_LETTERS = ("A", "B", "C", "D")


def check_distribution(records: list[dict]) -> dict:
    """Returns {"issues": [...], "mc_letter_counts": Counter, ...} -- the
    same checks as validate_common.assert_answer_distribution, but returning
    the raw counts too so a caller can decide HOW to rebalance, not just
    THAT it's skewed."""
    mc_answers = [q["answer"] for r in records for q in r.get("questions", [])
                  if q.get("question_type") == "multiple_choice"]
    bool_answers = [bool(q["answer"]) for r in records for q in r.get("questions", [])
                    if q.get("question_type") == "boolean"]
    return dict(
        issues=assert_answer_distribution(records),
        mc_letter_counts=Counter(mc_answers),
        boolean_true_count=sum(bool_answers),
        boolean_total=len(bool_answers),
        mc_total=len(mc_answers),
    )


def _reorder_mc_to_letter(question: dict, target_letter: str) -> None:
    """Swaps the correct option into `target_letter`'s slot (swapping with
    whichever option currently sits there) -- every option's CONTENT is
    preserved, repositioned only, and `answer` is updated to the new letter.
    A no-op if the question isn't MC, already sits at the target letter, or
    its stored answer isn't a plain letter (shouldn't happen per §2, but
    left untouched rather than guessed at if it somehow isn't)."""
    if question.get("question_type") != "multiple_choice":
        return
    current_letter = question.get("answer")
    if current_letter not in MC_LETTERS or target_letter not in MC_LETTERS:
        return
    if current_letter == target_letter:
        return
    options = list(question.get("options") or [])
    if len(options) != 4:
        return
    idx_map = {L: i for i, L in enumerate(MC_LETTERS)}
    ci, ti = idx_map[current_letter], idx_map[target_letter]
    options[ci], options[ti] = options[ti], options[ci]
    question["options"] = options
    question["answer"] = target_letter


def rebalance_mc_option_order(records: list[dict]) -> list[dict]:
    """Deterministically round-robins every MC question's correct letter
    across A/B/C/D in the order the questions are encountered (record order,
    then qid order) -- after this, the letter distribution is as close to
    uniform as integer counts allow, by construction (not "hopefully" --
    directly checkable). Mutates and returns `records`."""
    mc_questions = [q for r in records for q in r.get("questions", [])
                    if q.get("question_type") == "multiple_choice"]
    for i, q in enumerate(mc_questions):
        _reorder_mc_to_letter(q, MC_LETTERS[i % len(MC_LETTERS)])
    return records


def rebalance_if_skewed(records: list[dict]) -> tuple[list[dict], bool]:
    """Checks the MC-letter skew and only rewrites option order if it's
    actually outside SHARED_CONTRACT.md's band -- so a dataset whose
    distribution already happens to be fine is left untouched (less
    unnecessary churn on annotations.jsonl). Returns (records, did_rebalance)."""
    report = check_distribution(records)
    mc_skewed = any("MC answer letter" in issue for issue in report["issues"])
    if not mc_skewed:
        return records, False
    return rebalance_mc_option_order(records), True
