import random

import pytest

from common import answer_types as AT


def test_distinct_from_numeric():
    assert AT.distinct_from(10.0, 5.0, tolerance=1.0) is True
    assert AT.distinct_from(5.4, 5.0, tolerance=1.0) is False


def test_distinct_from_non_numeric():
    assert AT.distinct_from("blue", "red") is True
    assert AT.distinct_from("blue", "blue") is False


def test_make_mc_options_places_correct_answer_and_is_reproducible():
    rng1 = random.Random(42)
    options1, letter1 = AT.make_mc_options("correct", ["d1", "d2", "d3"], rng1)
    rng2 = random.Random(42)
    options2, letter2 = AT.make_mc_options("correct", ["d1", "d2", "d3"], rng2)
    assert options1 == options2 and letter1 == letter2
    assert set(options1) == {"correct", "d1", "d2", "d3"}
    assert options1[AT.MC_LETTERS.index(letter1)] == "correct"


def test_make_mc_options_wrong_distractor_count_raises():
    with pytest.raises(AT.AnswerTypeError):
        AT.make_mc_options("correct", ["d1", "d2"], random.Random(1))


def test_numeric_answer_shape():
    d = AT.numeric_answer(3.5, tolerance=0.2, unit="m")
    assert d["question_type"] == "numeric"
    assert d["answer"] == 3.5
    assert d["answer_tolerance"] == 0.2
    assert d["unit"] == "m"


def test_assert_mc_option_sanity_catches_duplicate_and_close_distractor():
    q_dup = dict(qid="q02", question_type="multiple_choice", options=["a", "a", "b", "c"], answer="A")
    errors = AT.assert_mc_option_sanity(q_dup)
    assert any("distinct" in e for e in errors)

    q_close = dict(qid="q04", question_type="multiple_choice",
                    options=["10.0", "10.05", "20.0", "30.0"], answer="A", answer_tolerance=0.5)
    errors2 = AT.assert_mc_option_sanity(q_close)
    assert any("within answer_tolerance" in e for e in errors2)


def test_assert_mc_option_sanity_passes_clean_options():
    q = dict(qid="q02", question_type="multiple_choice", options=["a", "b", "c", "d"], answer="A")
    assert AT.assert_mc_option_sanity(q) == []


def test_assert_mc_option_sanity_ignores_non_mc():
    q = dict(qid="q01", question_type="count", answer=3)
    assert AT.assert_mc_option_sanity(q) == []


def _valid_type_mix():
    return [
        dict(question_type="multiple_choice"), dict(question_type="multiple_choice"),
        dict(question_type="multiple_choice"),
        dict(question_type="numeric"), dict(question_type="numeric"),
        dict(question_type="count"),
        dict(question_type="boolean"),
        dict(question_type="ordering"),
        dict(question_type="short_text"), dict(question_type="short_text"),
    ]


def test_assert_type_mix_passes_valid_mix():
    counts = AT.assert_type_mix(_valid_type_mix())
    assert counts["multiple_choice"] == 3


def test_assert_type_mix_fails_missing_numeric():
    questions = _valid_type_mix()
    questions[3] = dict(question_type="short_text")  # remove one of the 2 numerics
    with pytest.raises(AT.AnswerTypeError, match="numeric"):
        AT.assert_type_mix(questions)


def test_assert_type_mix_fails_too_many_of_one_type():
    questions = [dict(question_type="multiple_choice")] * 5 + [
        dict(question_type="numeric"), dict(question_type="numeric"),
        dict(question_type="count"), dict(question_type="boolean"),
        dict(question_type="short_text"),
    ]
    with pytest.raises(AT.AnswerTypeError, match="no more than 4"):
        AT.assert_type_mix(questions)


def test_assert_type_mix_fails_missing_ordering_or_boolean():
    questions = [
        dict(question_type="multiple_choice"), dict(question_type="multiple_choice"),
        dict(question_type="multiple_choice"),
        dict(question_type="numeric"), dict(question_type="numeric"),
        dict(question_type="count"),
        dict(question_type="short_text"), dict(question_type="short_text"),
        dict(question_type="short_text"), dict(question_type="short_text"),
    ]
    with pytest.raises(AT.AnswerTypeError, match="ordering.*or.*boolean"):
        AT.assert_type_mix(questions)
