from common import schema


def make_question(**overrides):
    base = dict(
        qid="q01", difficulty_level=1, tier="easy", task_family="description",
        question_type="count", question="How many?", answer=3, answer_text="3",
    )
    base.update(overrides)
    return schema.Question(**base)


def test_valid_question_has_no_errors():
    assert make_question().validate() == []


def test_bad_tier_is_caught():
    errors = make_question(tier="bogus").validate()
    assert any("tier" in e for e in errors)


def test_video_only_task_family_rejected_for_image():
    errors = make_question(task_family="prediction").validate(video=False)
    assert any("task_family" in e for e in errors)


def test_video_only_task_family_ok_for_video():
    q = make_question(task_family="prediction", single_frame_answerable=False)
    assert q.validate(video=True) == []


def test_video_requires_single_frame_answerable_flag():
    errors = make_question(single_frame_answerable=None).validate(video=True)
    assert any("single_frame_answerable" in e for e in errors)


def test_multiple_choice_requires_four_distinct_options_and_letter_answer():
    q = make_question(question_type="multiple_choice", options=["a", "b", "c"], answer="A")
    errors = q.validate()
    assert any("4 options" in e for e in errors)

    q2 = make_question(question_type="multiple_choice", options=["a", "b", "c", "d"], answer="Z")
    errors2 = q2.validate()
    assert any("option letter" in e for e in errors2)


def test_numeric_requires_tolerance_and_known_unit():
    q = make_question(question_type="numeric", answer=1.5, unit="bogus_unit")
    errors = q.validate()
    assert any("answer_tolerance" in e for e in errors)
    assert any("unit" in e for e in errors)


def test_question_round_trips_through_dict():
    q = make_question()
    d = q.to_dict()
    q2 = schema.Question.from_dict(d)
    assert q2.qid == q.qid
    assert q2.answer == q.answer


def make_record(questions):
    return schema.Record(
        sample_id="ds_classic_0001", dataset="ds", preset="classic", modality="image",
        media_path="images/0001.png", seed=1, params={"a": 1}, derived={}, render_meta={},
        questions=questions,
    )


def _ten_valid_image_questions():
    from common.ladder import LADDER_TABLE
    qs = []
    for qid, level, tier, family in LADDER_TABLE:
        qtype = "numeric" if qid == "q04" else ("count" if qid == "q01" else
                 ("multiple_choice" if qid in ("q02", "q03", "q05", "q06") else
                  ("boolean" if qid == "q07" else
                   ("ordering" if qid == "q08" else "multiple_choice"))))
        kwargs = dict(qid=qid, difficulty_level=level, tier=tier, task_family=family,
                      question_type=qtype, question=f"question {qid}?")
        if qtype == "multiple_choice":
            kwargs.update(options=["w", "x", "y", "z"], answer="A", answer_text="w")
        elif qtype == "numeric":
            kwargs.update(answer=1.0, answer_text="1.0", answer_tolerance=0.1, unit="m")
        elif qtype == "boolean":
            kwargs.update(answer=True, answer_text="yes")
        elif qtype == "ordering":
            kwargs.update(answer=["a", "b"], answer_text="a < b")
        else:
            kwargs.update(answer=2, answer_text="2")
        qs.append(schema.Question(**kwargs))
    return qs


def test_record_validates_with_ten_correct_questions():
    rec = make_record(_ten_valid_image_questions())
    assert rec.validate() == []


def test_record_rejects_wrong_question_count():
    rec = make_record(_ten_valid_image_questions()[:5])
    errors = rec.validate()
    assert any("exactly 10 questions" in e for e in errors)


def test_video_record_requires_media_meta():
    rec = make_record(_ten_valid_image_questions())
    rec.modality = "video"
    errors = rec.validate()
    assert any("media_meta" in e for e in errors)


def test_read_record_v2_passthrough_and_v1_wrap():
    v2 = {"schema_version": 2, "x": 1}
    assert schema.read_record(v2) is v2

    v1 = {"id": "foo", "questions": []}
    wrapped = schema.read_record(v1)
    assert wrapped["schema_version"] == 1
    assert wrapped["legacy"] == v1


def test_record_round_trip():
    rec = make_record(_ten_valid_image_questions())
    d = rec.to_dict()
    rec2 = schema.Record.from_dict(d)
    assert rec2.sample_id == rec.sample_id
    assert len(rec2.questions) == 10
