import copy

from common import distribution as D
from common.tests.test_ladder import _good_builders
from common.ladder import QuestionLadder


def _skewed_records(n=20):
    records = []
    for i in range(n):
        questions = copy.deepcopy(QuestionLadder(_good_builders()).build())
        for q in questions:
            if q["question_type"] == "multiple_choice":
                q["answer"] = "A"  # every MC question answers 'A' -> maximally skewed
        records.append(dict(questions=questions))
    return records


def test_check_distribution_reports_skew_and_counts():
    records = _skewed_records()
    report = D.check_distribution(records)
    assert report["mc_letter_counts"]["A"] == report["mc_total"]
    assert any("letter 'A'" in i for i in report["issues"])


def test_rebalance_mc_option_order_preserves_option_content():
    records = _skewed_records()
    before_contents = [set(q["options"]) for r in records for q in r["questions"]
                        if q["question_type"] == "multiple_choice"]
    D.rebalance_mc_option_order(records)
    after_contents = [set(q["options"]) for r in records for q in r["questions"]
                       if q["question_type"] == "multiple_choice"]
    assert before_contents == after_contents  # same 4 option strings per question, just reordered


def test_rebalance_mc_option_order_actually_balances():
    records = _skewed_records()
    D.rebalance_mc_option_order(records)
    report = D.check_distribution(records)
    # The fixture's boolean/numeric answers are deliberately fixed across
    # every record (irrelevant to this test, which only exercises MC
    # rebalancing) -- so only the MC-letter issue is expected to clear.
    assert not any("letter" in i for i in report["issues"])
    counts = report["mc_letter_counts"]
    total = report["mc_total"]
    for letter in ("A", "B", "C", "D"):
        assert abs(counts.get(letter, 0) / total - 0.25) <= 0.20


def test_rebalance_answer_still_points_at_correct_content():
    records = _skewed_records()
    # before rebalancing, every MC question's correct content sits at index 0 (letter A)
    originally_correct_content = [q["options"][0] for r in records for q in r["questions"]
                                    if q["question_type"] == "multiple_choice"]
    D.rebalance_mc_option_order(records)
    mcs = [q for r in records for q in r["questions"] if q["question_type"] == "multiple_choice"]
    for q, correct_content in zip(mcs, originally_correct_content):
        letter_idx = D.MC_LETTERS.index(q["answer"])
        assert q["options"][letter_idx] == correct_content


def test_rebalance_if_skewed_noop_when_already_balanced():
    records = _skewed_records()
    D.rebalance_mc_option_order(records)  # now balanced
    before = copy.deepcopy(records)
    _, did_rebalance = D.rebalance_if_skewed(records)
    assert did_rebalance is False
    assert records == before


def test_rebalance_if_skewed_rebalances_when_needed():
    records = _skewed_records()
    _, did_rebalance = D.rebalance_if_skewed(records)
    assert did_rebalance is True
    assert not any("letter" in i for i in D.check_distribution(records)["issues"])
