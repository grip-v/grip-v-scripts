"""SHARED_CONTRACT.md §2: the record and question object schema.

Uses plain dataclasses rather than pydantic (SHARED_CONTRACT.md §0.1 allows
"pydantic (or dataclass + jsonschema)") -- pydantic isn't in requirements.txt
and this suite's existing code (bouncing_ball_restitution/config.py) already
uses dataclasses throughout, so this keeps the dependency footprint at zero
and the style consistent with the rest of the repo.

Every dataclass here has a `.validate() -> list[str]` method that returns a
list of human-readable error strings (empty = valid) rather than raising, so
callers (validate_common.py, ladder.py) can aggregate multiple problems into
one report instead of stopping at the first. `.assert_valid()` is a thin
wrapper that raises `SchemaError` if `.validate()` is non-empty.
"""
from __future__ import annotations
import json
from dataclasses import dataclass, field, asdict
from typing import Any


class SchemaError(ValueError):
    pass


# ---------------------------------------------------------------------------
# §5 answer-type / §4 task_family / §10.2 unit vocabularies
# ---------------------------------------------------------------------------

QUESTION_TYPES = ("multiple_choice", "numeric", "boolean", "count", "ordering", "short_text")

# §4: base vocabulary (every modality) + video-only additions.
TASK_FAMILIES_BASE = (
    "description", "relational", "comparative", "structural",
    "counterfactual", "compound", "compound_extrapolate",
)
TASK_FAMILIES_VIDEO_ONLY = (
    "event_counting", "temporal_order", "duration", "rate", "prediction", "persistence",
)
TASK_FAMILIES = TASK_FAMILIES_BASE + TASK_FAMILIES_VIDEO_ONLY

TIERS = ("easy", "medium", "hard")
DIFFICULTY_LEVELS = (1, 2, 3, 4)

# §10.2: fixed unit vocabulary for numeric physics answers. None is allowed
# (non-numeric questions, or a numeric answer that is genuinely unitless).
UNITS = (
    "m", "m/s", "m/s^2", "s", "kg", "N", "J", "W", "deg", "rad", "rad/s", "Hz", "Pa", "N*m",
)

MODALITIES = ("image", "video")

# §10.5: labelled physics-distractor error types (physics sub-benchmarks only).
DISTRACTOR_TYPES = (
    "dropped_factor_two", "sin_for_cos", "ignored_component",
    "wrong_power_of_r", "unit_confusion", "sign_error",
)


@dataclass
class Question:
    """§2 question object."""
    qid: str
    difficulty_level: int
    tier: str
    task_family: str
    question_type: str
    question: str
    answer: Any
    answer_text: str
    options: list[str] | None = None
    answer_tolerance: float | None = None
    unit: str | None = None
    rationale: str = ""
    derivation_key: str = ""
    requires_motion: bool | None = None          # video only
    single_frame_answerable: bool | None = None  # video only
    legacy_qid: str | None = None                # §8 migration
    distractor_types: list[str] | None = None    # §10.5 physics

    def validate(self, video: bool = False) -> list[str]:
        errors = []
        if not self.qid or not self.qid.startswith("q"):
            errors.append(f"qid {self.qid!r} does not look like 'q01'..'q10'")
        if self.difficulty_level not in DIFFICULTY_LEVELS:
            errors.append(f"difficulty_level {self.difficulty_level!r} not in {DIFFICULTY_LEVELS}")
        if self.tier not in TIERS:
            errors.append(f"tier {self.tier!r} not in {TIERS}")
        allowed_families = TASK_FAMILIES if video else TASK_FAMILIES_BASE
        if self.task_family not in allowed_families:
            errors.append(f"task_family {self.task_family!r} not in {allowed_families}"
                           + ("" if video else " (video-only families need modality='video')"))
        if self.question_type not in QUESTION_TYPES:
            errors.append(f"question_type {self.question_type!r} not in {QUESTION_TYPES}")
        if not self.question:
            errors.append("question text is empty")
        if self.answer is None:
            errors.append("answer is None -- every question must have a canonical answer")
        if not self.answer_text:
            errors.append("answer_text is empty")
        if self.question_type == "multiple_choice":
            if not self.options or len(self.options) != 4:
                errors.append(f"multiple_choice requires exactly 4 options, got {self.options!r}")
            elif self.answer not in ("A", "B", "C", "D"):
                errors.append(f"multiple_choice answer must be an option letter A-D, got {self.answer!r}")
        if self.question_type == "numeric":
            if self.answer_tolerance is None:
                errors.append("numeric question missing answer_tolerance")
            if self.unit is not None and self.unit not in UNITS:
                errors.append(f"unit {self.unit!r} not in the §10.2 vocabulary {UNITS}")
        if self.question_type == "ordering" and not isinstance(self.answer, list):
            errors.append(f"ordering answer must be a list of ids, got {type(self.answer).__name__}")
        if self.question_type == "boolean" and not isinstance(self.answer, bool):
            errors.append(f"boolean answer must be a bool, got {type(self.answer).__name__}")
        if self.distractor_types is not None:
            bad = [d for d in self.distractor_types if d not in DISTRACTOR_TYPES]
            if bad:
                errors.append(f"distractor_types {bad} not in {DISTRACTOR_TYPES}")
        if video and self.single_frame_answerable is None:
            errors.append("video modality requires single_frame_answerable to be set (True or False)")
        return errors

    def assert_valid(self, video: bool = False) -> None:
        errors = self.validate(video=video)
        if errors:
            raise SchemaError(f"question {self.qid}: " + "; ".join(errors))

    def to_dict(self) -> dict:
        d = asdict(self)
        return {k: v for k, v in d.items() if v is not None}

    @classmethod
    def from_dict(cls, d: dict) -> "Question":
        known = {f for f in cls.__dataclass_fields__}
        return cls(**{k: v for k, v in d.items() if k in known})


@dataclass
class Record:
    """§2 top-level record."""
    sample_id: str
    dataset: str
    preset: str
    modality: str
    media_path: str
    seed: int
    params: dict
    derived: dict
    render_meta: dict
    questions: list[Question] = field(default_factory=list)
    media_meta: dict | None = None
    schema_version: int = 2

    def validate(self) -> list[str]:
        errors = []
        if self.schema_version != 2:
            errors.append(f"schema_version {self.schema_version!r} != 2")
        if self.modality not in MODALITIES:
            errors.append(f"modality {self.modality!r} not in {MODALITIES}")
        video = self.modality == "video"
        if video and not self.media_meta:
            errors.append("video modality requires media_meta (fps, duration_s, frame_count, resolution)")
        elif video:
            for key in ("fps", "duration_s", "frame_count", "resolution"):
                if key not in self.media_meta:
                    errors.append(f"media_meta missing required video key {key!r}")
        if len(self.questions) != 10:
            errors.append(f"expected exactly 10 questions, got {len(self.questions)}")
        expected_qids = tuple(f"q{i:02d}" for i in range(1, 11))
        got_qids = tuple(q.qid for q in self.questions)
        if got_qids != expected_qids:
            errors.append(f"question qids out of order or missing: got {got_qids}, expected {expected_qids}")
        for q in self.questions:
            errors.extend(f"{q.qid}: {e}" for e in q.validate(video=video))
        return errors

    def assert_valid(self) -> None:
        errors = self.validate()
        if errors:
            raise SchemaError(f"record {self.sample_id}: " + "; ".join(errors))

    def to_dict(self) -> dict:
        d = dict(
            sample_id=self.sample_id, dataset=self.dataset, preset=self.preset,
            modality=self.modality, media_path=self.media_path, seed=self.seed,
            params=self.params, derived=self.derived, render_meta=self.render_meta,
            questions=[q.to_dict() for q in self.questions],
            schema_version=self.schema_version,
        )
        if self.media_meta is not None:
            d["media_meta"] = self.media_meta
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "Record":
        questions = [Question.from_dict(q) for q in d.get("questions", [])]
        known = {f for f in cls.__dataclass_fields__} - {"questions"}
        kwargs = {k: v for k, v in d.items() if k in known}
        return cls(questions=questions, **kwargs)


def is_v2(raw: dict) -> bool:
    return raw.get("schema_version") == 2


def read_record(raw: dict) -> dict:
    """§8's v1->v2 reader shim.

    What this function can do GENERICALLY: detect whether a raw annotation
    dict already conforms to v2 (schema_version == 2) and, if so, return it
    unchanged. What it deliberately does NOT do: guess a v1->v2 field mapping
    for an arbitrary dataset's old schema -- every pre-existing dataset in
    this suite (today: only `bouncing_ball_restitution`, which predates this
    contract entirely and has its own, unrelated old record shape) has a
    DIFFERENT legacy shape, and §8 itself says the real mapping ("map legacy
    questions onto ladder slots... call the dataset's questions.py to fill
    the remaining slots") is dataset-specific knowledge. That real migration
    logic lives in `tools/migrate_v1_to_v2.py` plus each dataset's own
    `ladder_migration.py` plugin (see that tool's module docstring) -- this
    function's job is just the version check + a safe, uniform wrapper a
    migration tool can pattern-match on, not the mapping itself.
    """
    if is_v2(raw):
        return raw
    return {"schema_version": 1, "legacy": raw}


def load_jsonl(path) -> list[dict]:
    with open(path, encoding="utf8") as f:
        return [json.loads(line) for line in f if line.strip()]
