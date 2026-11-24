"""SHARED_CONTRACT.md §3: the 10-question ladder builder.

`LADDER_TABLE` is the ONE place the qid -> (difficulty_level, tier,
task_family) table lives; every dataset's `questions.py` builds its 10
questions by constructing a `QuestionLadder` from 10 callables (one per
slot) and calling `.build()` -- it never hardcodes the table itself, so the
table can't drift between datasets or between a dataset and its validator.
"""
from __future__ import annotations
from typing import Callable

from . import schema
from .answer_types import assert_type_mix

# (qid, difficulty_level, tier, task_family) -- SHARED_CONTRACT.md §3, verbatim.
LADDER_TABLE: tuple[tuple[str, int, str, str], ...] = (
    ("q01", 1, "easy", "description"),
    ("q02", 1, "easy", "description"),
    ("q03", 2, "easy", "relational"),
    ("q04", 2, "medium", "relational"),
    ("q05", 2, "medium", "relational"),
    ("q06", 3, "medium", "comparative"),
    ("q07", 3, "medium", "structural"),
    ("q08", 3, "hard", "counterfactual"),
    ("q09", 4, "hard", "compound"),
    ("q10", 4, "hard", "compound_extrapolate"),
)
LADDER_QIDS: tuple[str, ...] = tuple(row[0] for row in LADDER_TABLE)
LADDER_BY_QID: dict[str, tuple[str, int, str, str]] = {row[0]: row for row in LADDER_TABLE}

MIN_NON_SINGLE_FRAME = 7  # §6: >= 7 of 10 questions must have single_frame_answerable == False


class LadderError(ValueError):
    pass


class QuestionLadder:
    """Takes 10 callables (`builders`, keyed by qid), each returning a
    `schema.Question` or an equivalent dict, and enforces §3 exactly when
    `.build()` is called -- raising on the first slot whose returned
    difficulty_level/tier/task_family doesn't match `LADDER_TABLE`, so a
    dataset can never silently drift from the shared table."""

    def __init__(self, builders: dict[str, Callable[[], "schema.Question | dict"]]):
        missing = [qid for qid in LADDER_QIDS if qid not in builders]
        extra = sorted(set(builders) - set(LADDER_QIDS))
        if missing:
            raise LadderError(f"QuestionLadder missing builder(s) for slot(s): {missing}")
        if extra:
            raise LadderError(f"QuestionLadder given builder(s) for unknown slot(s): {extra} "
                               f"(valid slots are {LADDER_QIDS})")
        self.builders = builders

    def build(self, video: bool = False) -> list[dict]:
        questions: list[schema.Question] = []
        seen_derivation_pairs: set[tuple[str, str]] = set()

        for qid, expected_level, expected_tier, expected_family in LADDER_TABLE:
            raw = self.builders[qid]()
            q = raw if isinstance(raw, schema.Question) else schema.Question.from_dict(raw)

            if q.qid != qid:
                raise LadderError(f"slot {qid}: builder returned qid={q.qid!r}, expected {qid!r}")
            if q.difficulty_level != expected_level:
                raise LadderError(
                    f"slot {qid}: difficulty_level={q.difficulty_level!r}, expected {expected_level!r}")
            if q.tier != expected_tier:
                raise LadderError(f"slot {qid}: tier={q.tier!r}, expected {expected_tier!r}")
            if q.task_family != expected_family:
                raise LadderError(
                    f"slot {qid}: task_family={q.task_family!r}, expected {expected_family!r}")

            errors = q.validate(video=video)
            if errors:
                raise LadderError(f"slot {qid} failed schema validation: {'; '.join(errors)}")

            if q.derivation_key:
                pair = (q.derivation_key, q.question_type)
                if pair in seen_derivation_pairs:
                    raise LadderError(
                        f"slot {qid}: (derivation_key={q.derivation_key!r}, "
                        f"question_type={q.question_type!r}) was already used by an earlier slot "
                        f"-- §3 forbids two questions sharing both"
                    )
                seen_derivation_pairs.add(pair)

            questions.append(q)

        dicts = [q.to_dict() for q in questions]
        assert_type_mix(dicts)  # raises AnswerTypeError naming the failed quota

        if video:
            n_non_single_frame = sum(1 for q in questions if q.single_frame_answerable is False)
            if n_non_single_frame < MIN_NON_SINGLE_FRAME:
                raise LadderError(
                    f"§6 motion requirement: only {n_non_single_frame}/10 questions marked "
                    f"single_frame_answerable=False, need >= {MIN_NON_SINGLE_FRAME}"
                )

        return dicts


def check_record_matches_ladder(questions: list[dict]) -> list[str]:
    """Independent (validator-side) re-check that a STORED list of 10
    question dicts matches LADDER_TABLE exactly -- used by
    validate_common.assert_schema so the validator does not simply trust
    that the generator ran its data through QuestionLadder."""
    errors = []
    if len(questions) != len(LADDER_TABLE):
        return [f"expected {len(LADDER_TABLE)} questions, got {len(questions)}"]
    for q, (qid, level, tier, family) in zip(questions, LADDER_TABLE):
        if q.get("qid") != qid:
            errors.append(f"position {qid}: found qid={q.get('qid')!r}")
            continue
        if q.get("difficulty_level") != level:
            errors.append(f"{qid}: difficulty_level={q.get('difficulty_level')!r}, expected {level!r}")
        if q.get("tier") != tier:
            errors.append(f"{qid}: tier={q.get('tier')!r}, expected {tier!r}")
        if q.get("task_family") != family:
            errors.append(f"{qid}: task_family={q.get('task_family')!r}, expected {family!r}")
    return errors
