import json
from pathlib import Path

import numpy as np
import pytest

from common import validate_common as VC
from common.tests.test_ladder import _good_builders
from common.ladder import QuestionLadder


def _valid_record(video=False):
    questions = QuestionLadder(_good_builders(video=video)).build(video=video)
    rec = dict(
        sample_id="ds_classic_0001", dataset="ds", preset="classic",
        modality="video" if video else "image",
        media_path="videos/0001.mp4" if video else "images/0001.png",
        seed=1, params={"h0": 2.0}, derived={}, render_meta={}, questions=questions,
        schema_version=2,
    )
    if video:
        rec["media_meta"] = dict(fps=15, duration_s=4.0, frame_count=60, resolution=[854, 480])
    return rec


def test_assert_schema_passes_for_valid_record():
    assert VC.assert_schema(_valid_record()) == []


def test_assert_schema_flags_wrong_order():
    rec = _valid_record()
    rec["questions"][0], rec["questions"][1] = rec["questions"][1], rec["questions"][0]
    errors = VC.assert_schema(rec)
    assert any("ladder mismatch" in e for e in errors)


def test_assert_rederivation_catches_numeric_and_exact_mismatch():
    rec = _valid_record()

    def bad_recompute(record, q):
        if q["question_type"] == "numeric":
            return q["answer"] + 100  # force a mismatch outside tolerance
        return q["answer"]

    errors = VC.assert_rederivation(rec, bad_recompute)
    assert any("numeric mismatch" in e for e in errors)


def test_assert_rederivation_passes_when_recompute_matches():
    rec = _valid_record()
    errors = VC.assert_rederivation(rec, lambda record, q: q["answer"])
    assert errors == []


def test_assert_rederivation_reports_exceptions_as_mismatches():
    rec = _valid_record()

    def raising_recompute(record, q):
        raise ValueError("boom")

    errors = VC.assert_rederivation(rec, raising_recompute)
    assert any("boom" in e for e in errors)


def test_assert_pixel_positions_reports_out_of_tolerance_and_missing():
    result = VC.assert_pixel_positions(
        detections=[(10.0, 10.0), None, (5.0, 5.0)],
        expecteds=[(10.0, 10.0), (0.0, 0.0), (5.0, 20.0)],
        pixel_tol=2.0,
        labels=["a", "b", "c"],
    )
    assert any(i.startswith("b:") and "not detected" in i for i in result["issues"])
    assert any(i.startswith("c:") and "mismatch" in i for i in result["issues"])
    assert not any(i.startswith("a:") for i in result["issues"])  # a was within tolerance
    assert result["max_error"] is not None


def test_assert_option_sanity_delegates_to_answer_types():
    rec = _valid_record()
    mc = next(q for q in rec["questions"] if q["question_type"] == "multiple_choice")
    mc["options"] = [mc["options"][0], mc["options"][0], mc["options"][2], mc["options"][3]]
    errors = VC.assert_option_sanity(rec)
    assert any("distinct" in e for e in errors)


def test_assert_answer_distribution_flags_skewed_mc():
    records = []
    for i in range(20):
        rec = _valid_record()
        for q in rec["questions"]:
            if q["question_type"] == "multiple_choice":
                q["answer"] = "A"  # every MC question always answers 'A'
        records.append(rec)
    errors = VC.assert_answer_distribution(records)
    assert any("letter 'A'" in e for e in errors)


def test_assert_answer_distribution_passes_balanced_mc():
    records = []
    letters = ["A", "B", "C", "D"]
    for i in range(40):
        rec = _valid_record()
        mcs = [q for q in rec["questions"] if q["question_type"] == "multiple_choice"]
        for j, q in enumerate(mcs):
            q["answer"] = letters[(i + j) % 4]
        records.append(rec)
    errors = VC.assert_answer_distribution(records)
    assert not any("letter" in e for e in errors)


def test_assert_motion_requirements_image_is_noop():
    assert VC.assert_motion_requirements(_valid_record(video=False)) == []


def test_assert_motion_requirements_video_passes_when_flagged_and_counterexample_present():
    rec = _valid_record(video=True)
    rec["derived"]["ablation_counterexample"] = {"params": {"x": 2}}
    assert VC.assert_motion_requirements(rec) == []


def test_assert_motion_requirements_video_flags_missing_counterexample():
    rec = _valid_record(video=True)
    rec["derived"] = {}
    errors = VC.assert_motion_requirements(rec)
    assert any("ablation_counterexample" in e for e in errors)


def test_assert_media_integrity_missing_file():
    errors = VC.assert_media_integrity(Path("/nonexistent/file.mp4"), None, "video")
    assert any("missing" in e for e in errors)


def test_assert_media_integrity_flat_frame_detected(tmp_path):
    media_path = tmp_path / "fake.bin"
    media_path.write_bytes(b"x")
    flat_frame = np.zeros((4, 4, 3), dtype=np.uint8)
    errors = VC.assert_media_integrity(media_path, None, "image", frame_reader=lambda p: [flat_frame])
    assert any("flat color" in e for e in errors)


def test_assert_media_integrity_ok_frame(tmp_path):
    media_path = tmp_path / "fake.bin"
    media_path.write_bytes(b"x")
    frame = np.random.randint(0, 255, (4, 4, 3), dtype=np.uint8)
    errors = VC.assert_media_integrity(media_path, None, "image", frame_reader=lambda p: [frame])
    assert errors == []


def test_format_result_line():
    assert VC.format_result_line(10, 100, 0) == "RESULT: 10 samples, 100 questions, 0 mismatches"


def test_write_validation_report(tmp_path):
    path = tmp_path / "validation_report.txt"
    ok = VC.write_validation_report(path, ["some issue"], n_samples=5, n_questions=50, n_mismatches=1)
    assert ok is False
    text = path.read_text()
    assert "some issue" in text
    assert text.strip().endswith("RESULT: 5 samples, 50 questions, 1 mismatches")

    ok2 = VC.write_validation_report(path, [], n_samples=5, n_questions=50, n_mismatches=0)
    assert ok2 is True
