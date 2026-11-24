import pytest

from common import schema
from common.ladder import QuestionLadder, LadderError, LADDER_TABLE, check_record_matches_ladder


def _q(qid, level, tier, family, qtype="short_text", **kw):
    base = dict(qid=qid, difficulty_level=level, tier=tier, task_family=family,
                question_type=qtype, question=f"{qid}?", answer="x", answer_text="x")
    base.update(kw)
    return schema.Question(**base)


def _good_builders(video: bool = False):
    """A full valid set of 10 builders satisfying both LADDER_TABLE and the
    §5 type-mix quota (>=3 MC, >=2 numeric, >=1 count, >=1 ordering/boolean)."""
    plan = {
        "q01": ("count", dict(answer=3, answer_text="3")),
        "q02": ("multiple_choice", dict(options=["a", "b", "c", "d"], answer="A", answer_text="a")),
        "q03": ("multiple_choice", dict(options=["a", "b", "c", "d"], answer="B", answer_text="b")),
        "q04": ("numeric", dict(answer=1.0, answer_text="1.0", answer_tolerance=0.1, unit="m")),
        "q05": ("multiple_choice", dict(options=["a", "b", "c", "d"], answer="C", answer_text="c")),
        "q06": ("multiple_choice", dict(options=["a", "b", "c", "d"], answer="D", answer_text="d")),
        "q07": ("boolean", dict(answer=True, answer_text="yes")),
        "q08": ("numeric", dict(answer=2.0, answer_text="2.0", answer_tolerance=0.1, unit="s")),
        "q09": ("ordering", dict(answer=["a", "b"], answer_text="a < b")),
        "q10": ("short_text", dict(answer="foo", answer_text="foo")),
    }
    builders = {}
    for qid, level, tier, family in LADDER_TABLE:
        qtype, extra = plan[qid]
        kwargs = dict(qid=qid, difficulty_level=level, tier=tier, task_family=family,
                      question_type=qtype, question=f"{qid}?", derivation_key=f"key_{qid}")
        kwargs.update(extra)
        if video:
            # q01/q02 stay single-frame-answerable=True (calibration anchors,
            # per §6); the other 8 slots are False, satisfying the >=7 minimum.
            kwargs["single_frame_answerable"] = qid in ("q01", "q02")
        builders[qid] = (lambda q=schema.Question(**kwargs): q)
    return builders


def test_ladder_builds_successfully_for_image():
    ladder = QuestionLadder(_good_builders())
    questions = ladder.build(video=False)
    assert [q["qid"] for q in questions] == [row[0] for row in LADDER_TABLE]


def test_ladder_builds_successfully_for_video():
    ladder = QuestionLadder(_good_builders(video=True))
    questions = ladder.build(video=True)
    n_non_single_frame = sum(1 for q in questions if q.get("single_frame_answerable") is False)
    assert n_non_single_frame >= 7


def test_ladder_missing_builder_raises_at_construction():
    builders = _good_builders()
    del builders["q05"]
    with pytest.raises(LadderError, match="missing builder"):
        QuestionLadder(builders)


def test_ladder_unknown_slot_raises_at_construction():
    builders = _good_builders()
    builders["q11"] = builders["q10"]
    with pytest.raises(LadderError, match="unknown slot"):
        QuestionLadder(builders)


def test_ladder_wrong_difficulty_level_caught():
    builders = _good_builders()
    bad = _q("q01", 2, "easy", "description", "count", answer=1, answer_text="1")  # should be level 1
    builders["q01"] = lambda: bad
    with pytest.raises(LadderError, match="difficulty_level"):
        QuestionLadder(builders).build()


def test_ladder_wrong_task_family_caught():
    builders = _good_builders()
    bad = _q("q07", 3, "medium", "comparative")  # q07 must be 'structural'
    builders["q07"] = lambda: bad
    with pytest.raises(LadderError, match="task_family"):
        QuestionLadder(builders).build()


def test_ladder_duplicate_derivation_key_and_type_rejected():
    builders = _good_builders()
    # force q02 and q03 to share both derivation_key and question_type
    q02 = builders["q02"]()
    q03_dict = q02.to_dict()
    q03_dict["qid"] = "q03"
    q03_dict["difficulty_level"] = 2
    q03_dict["tier"] = "easy"
    q03_dict["task_family"] = "relational"
    builders["q03"] = lambda d=q03_dict: schema.Question.from_dict(d)
    with pytest.raises(LadderError, match="already used"):
        QuestionLadder(builders).build()


def test_ladder_video_motion_requirement_enforced():
    builders = _good_builders(video=True)
    # flip q01..q07 all to single_frame_answerable=True, leaving only 3 False
    for qid in ("q03", "q04", "q05", "q06", "q07"):
        q = builders[qid]()
        d = q.to_dict()
        d["single_frame_answerable"] = True
        builders[qid] = (lambda d=d: schema.Question.from_dict(d))
    with pytest.raises(LadderError, match="motion requirement"):
        QuestionLadder(builders).build(video=True)


def test_ladder_type_mix_violation_propagates():
    builders = _good_builders()
    # break the type mix: turn q04's numeric into another short_text (drops numeric count to 1)
    q = builders["q04"]()
    d = q.to_dict()
    d["question_type"] = "short_text"
    d["answer"] = "x"
    d.pop("answer_tolerance", None)
    d.pop("unit", None)
    builders["q04"] = (lambda d=d: schema.Question.from_dict(d))
    from common.answer_types import AnswerTypeError
    with pytest.raises(AnswerTypeError, match="numeric"):
        QuestionLadder(builders).build()


def test_check_record_matches_ladder_detects_swapped_slots():
    good = QuestionLadder(_good_builders()).build()
    swapped = list(good)
    swapped[0], swapped[1] = swapped[1], swapped[0]
    errors = check_record_matches_ladder(swapped)
    assert errors  # position q01 now holds q02's content -> mismatch


def test_check_record_matches_ladder_clean():
    good = QuestionLadder(_good_builders()).build()
    assert check_record_matches_ladder(good) == []
