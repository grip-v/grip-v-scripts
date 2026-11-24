"""SHARED_CONTRACT.md §7: the eight validator assertions, as reusable
functions, plus the `RESULT:` line writer.

Assertions 2 (re-derivation) and 3 (pixel check) are inherently
dataset-specific (recomputing a question's answer, or locating an object in
a frame, needs that dataset's own `geometry.py`/`physics.py` and detection
code) -- this module provides the GENERIC part of each (comparison logic,
error aggregation, the shared numeric-tolerance rule) and takes the
dataset-specific part as a callback or as already-computed values, the same
division of labour SHARED_CONTRACT.md itself describes ("dataset validators
import these and add only their own pixel-detection logic").

Each `assert_*` function returns a list of human-readable issue strings
(empty = passed) rather than raising, so a dataset's `validate_<name>.py` can
run all eight over every sample and report everything wrong at once --
exactly like every existing `validate_*.py` in this repo already does.
"""
from __future__ import annotations
import math
from pathlib import Path
from typing import Callable

from . import schema
from .ladder import check_record_matches_ladder
from .answer_types import assert_mc_option_sanity, assert_type_mix, is_close, AnswerTypeError

# §7.5 ("answer distribution") thresholds. "Within ±20% / ±15% of uniform" is
# read here as an ADDITIVE percentage-point band around the uniform/50-50
# baseline (e.g. a 4-option MC's uniform share is 25%, so the allowed band is
# 5%-45%) -- a disclosed judgment call, per this suite's convention of
# stating interpretation choices rather than leaving them implicit, since
# SHARED_CONTRACT.md doesn't itself say additive vs. relative.
MC_LETTER_BAND_PP = 0.20
BOOLEAN_BAND_PP = 0.15
NUMERIC_MAX_SHARE = 0.10


# ---------------------------------------------------------------------------
# 1. Schema
# ---------------------------------------------------------------------------

def assert_schema(record: dict) -> list[str]:
    errors = []
    try:
        rec = schema.Record.from_dict(record)
    except TypeError as e:
        return [f"record does not match the §2 schema shape: {e}"]
    errors.extend(rec.validate())
    errors.extend(f"ladder mismatch: {e}" for e in check_record_matches_ladder(record.get("questions", [])))
    return errors


# ---------------------------------------------------------------------------
# 2. Re-derivation
# ---------------------------------------------------------------------------

def assert_rederivation(record: dict,
                         recompute_fn: Callable[[dict, dict], object]) -> list[str]:
    """`recompute_fn(record, question) -> recomputed_answer`, using ONLY
    `record["params"]` (never `record["derived"]` or the stored answer
    itself) -- the independence discipline every validator in this suite
    already follows. Numeric answers are compared within `answer_tolerance`;
    everything else by exact equality."""
    errors = []
    for q in record.get("questions", []):
        try:
            expected = recompute_fn(record, q)
        except Exception as e:  # noqa: BLE001 -- report as a mismatch, not a crash
            errors.append(f"{q.get('qid')}: recompute_fn raised {type(e).__name__}: {e}")
            continue
        stored = q.get("answer")
        if q.get("question_type") == "numeric":
            tol = q.get("answer_tolerance", 0.0) or 0.0
            if not is_close(float(stored), float(expected), tol):
                errors.append(f"{q.get('qid')}: numeric mismatch (stored {stored}, recomputed {expected}, tol {tol})")
        elif expected != stored:
            errors.append(f"{q.get('qid')}: answer mismatch (stored {stored!r}, recomputed {expected!r})")
    return errors


# ---------------------------------------------------------------------------
# 3. Pixel check
# ---------------------------------------------------------------------------

def pixel_distance(detected: tuple[float, float], expected: tuple[float, float]) -> float:
    return math.hypot(detected[0] - expected[0], detected[1] - expected[1])


def assert_pixel_positions(detections: list[tuple[float, float] | None],
                            expecteds: list[tuple[float, float]],
                            pixel_tol: float, labels: list[str] | None = None) -> dict:
    """Generic centroid-distance check: the dataset supplies its own
    color-blob/template detections and closed-form expected positions (in
    the SAME pixel coordinate system); this just does the distance math and
    aggregation. A `None` detection (object not found at all) is reported as
    its own issue, not silently skipped."""
    labels = labels or [str(i) for i in range(len(expecteds))]
    issues = []
    errors = []
    for label, det, exp in zip(labels, detections, expecteds):
        if det is None:
            issues.append(f"{label}: not detected")
            continue
        d = pixel_distance(det, exp)
        errors.append(d)
        if d > pixel_tol:
            issues.append(f"{label}: position mismatch (expected {exp}, detected {det}, dist={d:.2f}px > tol={pixel_tol})")
    return dict(
        issues=issues,
        max_error=max(errors) if errors else None,
        mean_error=(sum(errors) / len(errors)) if errors else None,
    )


# ---------------------------------------------------------------------------
# 4. Option sanity
# ---------------------------------------------------------------------------

def assert_option_sanity(record: dict) -> list[str]:
    errors = []
    for q in record.get("questions", []):
        errors.extend(assert_mc_option_sanity(q))
    return errors


# ---------------------------------------------------------------------------
# 5. Answer distribution (run over a whole batch, not one record)
# ---------------------------------------------------------------------------

def assert_answer_distribution(records: list[dict]) -> list[str]:
    from collections import Counter
    errors = []

    mc_answers = [q["answer"] for r in records for q in r.get("questions", [])
                  if q.get("question_type") == "multiple_choice"]
    if mc_answers:
        n = len(mc_answers)
        uniform = 1.0 / 4  # §5: 4 options
        counts = Counter(mc_answers)
        for letter in ("A", "B", "C", "D"):
            share = counts.get(letter, 0) / n
            if abs(share - uniform) > MC_LETTER_BAND_PP:
                errors.append(f"MC answer letter {letter!r} share={share:.3f}, "
                               f"outside uniform({uniform:.2f}) +/- {MC_LETTER_BAND_PP}")

    bool_answers = [bool(q["answer"]) for r in records for q in r.get("questions", [])
                    if q.get("question_type") == "boolean"]
    if bool_answers:
        n = len(bool_answers)
        share_true = sum(bool_answers) / n
        if abs(share_true - 0.5) > BOOLEAN_BAND_PP:
            errors.append(f"boolean answers: True share={share_true:.3f}, "
                           f"outside 0.50 +/- {BOOLEAN_BAND_PP}")

    numeric_answers = [q["answer"] for r in records for q in r.get("questions", [])
                        if q.get("question_type") == "numeric"]
    if numeric_answers:
        n = len(numeric_answers)
        counts = Counter(round(float(v), 6) for v in numeric_answers)
        for value, count in counts.items():
            share = count / n
            if share > NUMERIC_MAX_SHARE:
                errors.append(f"numeric answer value {value} covers {share:.1%} of samples "
                               f"(> {NUMERIC_MAX_SHARE:.0%} max)")
    return errors


# ---------------------------------------------------------------------------
# 6. Type mix
# ---------------------------------------------------------------------------

def assert_type_mix_record(record: dict) -> list[str]:
    try:
        assert_type_mix(record.get("questions", []))
    except AnswerTypeError as e:
        return [str(e)]
    return []


# ---------------------------------------------------------------------------
# 7. Motion (video only, §6)
# ---------------------------------------------------------------------------

def assert_motion_requirements(record: dict) -> list[str]:
    """Structural half of §6: >=7/10 questions flagged single_frame_
    answerable=False, and (when any are) a non-empty
    `derived.ablation_counterexample` is present to back that claim. The
    SEMANTIC half -- does the stored counterexample actually produce a
    different answer at an identical sampled frame -- is inherently
    dataset-specific (needs that dataset's own scene generator) and belongs
    in that dataset's own `tests/test_single_frame_ablation.py`, per
    SHARED_CONTRACT.md §6's own wording ("assert that... the generator CAN
    construct a second scene...")."""
    if record.get("modality") != "video":
        return []
    errors = []
    questions = record.get("questions", [])
    non_single_frame = [q for q in questions if q.get("single_frame_answerable") is False]
    from .ladder import MIN_NON_SINGLE_FRAME
    if len(non_single_frame) < MIN_NON_SINGLE_FRAME:
        errors.append(f"only {len(non_single_frame)}/10 questions marked single_frame_answerable=False, "
                       f"need >= {MIN_NON_SINGLE_FRAME}")
    if non_single_frame and not record.get("derived", {}).get("ablation_counterexample"):
        errors.append("questions are marked single_frame_answerable=False but "
                       "derived.ablation_counterexample is missing/empty")
    return errors


# ---------------------------------------------------------------------------
# 8. Media integrity
# ---------------------------------------------------------------------------

def assert_media_integrity(media_path: Path, media_meta: dict | None, modality: str,
                            frame_reader: Callable[[Path], list] | None = None) -> list[str]:
    """`frame_reader(path) -> list of frames (as arrays/arrays-like)` is
    supplied by the dataset (it knows whether to use cv2/PIL/manim's own
    reader); this function just does the file-existence and
    all-identical/all-black variance checks generically."""
    errors = []
    media_path = Path(media_path)
    if not media_path.exists():
        return [f"media file missing: {media_path}"]
    if frame_reader is None:
        return errors  # dataset didn't opt into the frame-decode part
    try:
        frames = frame_reader(media_path)
    except Exception as e:  # noqa: BLE001
        return [f"media file failed to decode: {type(e).__name__}: {e}"]
    if not frames:
        return [f"media file decoded to zero frames: {media_path}"]
    if modality == "video" and media_meta:
        expected_count = media_meta.get("frame_count")
        if expected_count is not None and abs(len(frames) - expected_count) > 1:
            errors.append(f"frame count mismatch: decoded {len(frames)}, media_meta says {expected_count}")
    try:
        import numpy as np
        first = np.asarray(frames[0])
        if first.max() == first.min():
            errors.append("first frame is a single flat color (all-black or all-identical)")
        if len(frames) > 1 and all(np.array_equal(np.asarray(f), first) for f in frames[1:]):
            errors.append("every sampled frame is pixel-identical")
    except ImportError:
        pass  # numpy not available -- skip the variance check, file-level checks above still ran
    return errors


# ---------------------------------------------------------------------------
# RESULT line
# ---------------------------------------------------------------------------

def format_result_line(n_samples: int, n_questions: int, n_mismatches: int) -> str:
    return f"RESULT: {n_samples} samples, {n_questions} questions, {n_mismatches} mismatches"


def write_validation_report(path: Path, report_lines: list[str], n_samples: int,
                             n_questions: int, n_mismatches: int) -> bool:
    """Writes `report_lines` followed by the machine-greppable RESULT: line
    to `path`. Returns True iff n_mismatches == 0 (the pass/fail a dataset's
    validate script should exit on)."""
    text = "\n".join(report_lines) + ("\n" if report_lines else "") + format_result_line(
        n_samples, n_questions, n_mismatches) + "\n"
    Path(path).write_text(text, encoding="utf8")
    return n_mismatches == 0
