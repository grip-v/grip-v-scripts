import json
import textwrap

import pytest

from tools.migrate_v1_to_v2 import migrate_dataset, migrate_record, MigrationError, hash_params
from common.ladder import LADDER_TABLE


PLUGIN_SOURCE = textwrap.dedent("""
    from common.answer_types import numeric_answer, count_answer, boolean_answer, ordering_answer

    LEGACY_QID_TO_SLOT = {"old_q1": "q01", "old_q2": "q04"}

    def legacy_to_question(record, old_question, new_qid):
        if new_qid == "q01":
            base = dict(qid="q01", difficulty_level=1, tier="easy", task_family="description",
                        question=old_question["question_text"], legacy_qid=old_question["qid"],
                        derivation_key="legacy_count")
            base.update(count_answer(old_question["ground_truth"]))
            return base
        if new_qid == "q04":
            base = dict(qid="q04", difficulty_level=2, tier="medium", task_family="relational",
                        question=old_question["question_text"], legacy_qid=old_question["qid"],
                        derivation_key="legacy_numeric")
            base.update(numeric_answer(float(old_question["ground_truth"]), tolerance=0.1, unit="m"))
            return base
        raise AssertionError(new_qid)

    def fill_remaining(record, mapped):
        from common.ladder import LADDER_TABLE
        out = {}
        plan = {
            "q02": ("multiple_choice", dict(options=["a", "b", "c", "d"], answer="A", answer_text="a")),
            "q03": ("multiple_choice", dict(options=["a", "b", "c", "d"], answer="B", answer_text="b")),
            "q05": ("multiple_choice", dict(options=["a", "b", "c", "d"], answer="C", answer_text="c")),
            "q06": ("multiple_choice", dict(options=["a", "b", "c", "d"], answer="D", answer_text="d")),
            "q07": ("boolean", dict(answer=True, answer_text="yes")),
            "q08": ("numeric", dict(answer=2.0, answer_text="2.0", answer_tolerance=0.1, unit="s")),
            "q09": ("ordering", dict(answer=["a", "b"], answer_text="a < b")),
            "q10": ("short_text", dict(answer="foo", answer_text="foo")),
        }
        for qid, level, tier, family in LADDER_TABLE:
            if qid in mapped:
                continue
            qtype, extra = plan[qid]
            q = dict(qid=qid, difficulty_level=level, tier=tier, task_family=family,
                      question_type=qtype, question=f"{qid}?", derivation_key=f"filled_{qid}")
            q.update(extra)
            out[qid] = q
        return out
""")


def _v1_record(sample_id="ds_classic_0001"):
    return dict(
        id=sample_id, dataset="ds", preset="classic", modality="image",
        media_path="images/0001.png", seed=1,
        params={"h0": 2.0, "e": 0.7},
        questions=[
            dict(qid="old_q1", question_text="how many?", ground_truth=3),
            dict(qid="old_q2", question_text="how far?", ground_truth="1.5"),
        ],
    )


@pytest.fixture
def dataset_dir(tmp_path):
    d = tmp_path / "ds"
    d.mkdir()
    (d / "ladder_migration.py").write_text(PLUGIN_SOURCE, encoding="utf8")
    records = [_v1_record(f"ds_classic_{i:04d}") for i in range(3)]
    with (d / "annotations.jsonl").open("w", encoding="utf8") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")
    return d


def test_migrate_dataset_produces_valid_v2_records(dataset_dir):
    stats = migrate_dataset(dataset_dir)
    assert stats == dict(n_records=3, n_migrated=3, n_already_v2=0, dry_run=False)

    migrated = [json.loads(l) for l in (dataset_dir / "annotations.jsonl").read_text().splitlines()]
    assert len(migrated) == 3
    for r in migrated:
        assert r["schema_version"] == 2
        assert len(r["questions"]) == 10
        assert [q["qid"] for q in r["questions"]] == [row[0] for row in LADDER_TABLE]

    q01 = migrated[0]["questions"][0]
    assert q01["legacy_qid"] == "old_q1"
    q04 = migrated[0]["questions"][3]
    assert q04["legacy_qid"] == "old_q2"


def test_migrate_dataset_writes_v1_backup(dataset_dir):
    original_text = (dataset_dir / "annotations.jsonl").read_text()
    migrate_dataset(dataset_dir)
    backup = dataset_dir / "annotations.jsonl.v1.bak"
    assert backup.exists()
    assert backup.read_text() == original_text


def test_migrate_dataset_dry_run_does_not_write(dataset_dir):
    original_text = (dataset_dir / "annotations.jsonl").read_text()
    stats = migrate_dataset(dataset_dir, dry_run=True)
    assert stats["dry_run"] is True
    assert (dataset_dir / "annotations.jsonl").read_text() == original_text
    assert not (dataset_dir / "annotations.jsonl.v1.bak").exists()


def test_migrate_dataset_missing_plugin_raises(tmp_path):
    d = tmp_path / "no_plugin"
    d.mkdir()
    (d / "annotations.jsonl").write_text(json.dumps(_v1_record()) + "\n", encoding="utf8")
    with pytest.raises(MigrationError, match="ladder_migration.py"):
        migrate_dataset(d)


def test_migrate_record_preserves_params_hash(dataset_dir):
    import importlib.util
    spec = importlib.util.spec_from_file_location("plugin", dataset_dir / "ladder_migration.py")
    plugin = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(plugin)

    record = _v1_record()
    before = hash_params(record["params"])
    migrated = migrate_record(record, plugin)
    after = hash_params(migrated["params"])
    assert before == after


def test_migrate_dataset_is_idempotent_on_already_v2_records(dataset_dir):
    migrate_dataset(dataset_dir)
    stats = migrate_dataset(dataset_dir)  # second run: everything already v2
    assert stats["n_migrated"] == 0
    assert stats["n_already_v2"] == 3


CONTENT_DEPENDENT_PLUGIN_SOURCE = textwrap.dedent("""
    from common.answer_types import count_answer, boolean_answer

    def extract_legacy_qid(old_question):
        return old_question["suffix"]  # this legacy schema has no 'qid'/'question_id' at all

    def _route_q2(old_question):
        return "q07" if old_question["kind"] == "structural_flavor" else "q08"

    LEGACY_QID_TO_SLOT = {"q1": "q01", "q2": _route_q2}

    def legacy_to_question(record, old_question, new_qid):
        if new_qid == "q01":
            base = dict(qid="q01", difficulty_level=1, tier="easy", task_family="description",
                        question="how many?", legacy_qid="q1", derivation_key="legacy_count")
            base.update(count_answer(3))
            return base
        family = "structural" if new_qid == "q07" else "counterfactual"
        base = dict(qid=new_qid, difficulty_level=3, tier="medium" if new_qid == "q07" else "hard",
                    task_family=family, question=old_question["question_text"],
                    legacy_qid="q2", derivation_key=f"legacy_{new_qid}")
        base.update(boolean_answer(old_question["ground_truth"]))
        return base

    def fill_remaining(record, mapped):
        from common.ladder import LADDER_TABLE
        from common.answer_types import numeric_answer
        plan = {
            "q02": ("multiple_choice", dict(options=["a", "b", "c", "d"], answer="A", answer_text="a")),
            "q03": ("numeric", dict(answer=0.5, answer_text="0.5", answer_tolerance=0.1, unit="s")),
            "q04": ("numeric", dict(answer=1.0, answer_text="1.0", answer_tolerance=0.1, unit="m")),
            "q05": ("multiple_choice", dict(options=["a", "b", "c", "d"], answer="C", answer_text="c")),
            "q06": ("multiple_choice", dict(options=["a", "b", "c", "d"], answer="D", answer_text="d")),
            "q07": ("boolean", dict(answer=True, answer_text="yes")),
            "q08": ("boolean", dict(answer=True, answer_text="yes")),
            "q09": ("ordering", dict(answer=["a", "b"], answer_text="a < b")),
            "q10": ("short_text", dict(answer="foo", answer_text="foo")),
        }
        out = {}
        for qid, level, tier, fam in LADDER_TABLE:
            if qid in mapped:
                continue
            qtype, extra = plan[qid]
            q = dict(qid=qid, difficulty_level=level, tier=tier, task_family=fam,
                      question_type=qtype, question=f"{qid}?", derivation_key=f"filled_{qid}")
            q.update(extra)
            out[qid] = q
        return out
""")


def test_content_dependent_slot_routing_and_custom_qid_extraction(tmp_path):
    d = tmp_path / "ds2"
    d.mkdir()
    (d / "ladder_migration.py").write_text(CONTENT_DEPENDENT_PLUGIN_SOURCE, encoding="utf8")

    structural_record = dict(
        id="r1", dataset="ds2", preset="classic", modality="image", media_path="images/1.png",
        seed=1, params={"h0": 1.0},
        questions=[
            dict(suffix="q1", question_text="count?", ground_truth=3),
            dict(suffix="q2", kind="structural_flavor", question_text="whole config true?", ground_truth=True),
        ],
    )
    counterfactual_record = dict(
        id="r2", dataset="ds2", preset="classic", modality="image", media_path="images/2.png",
        seed=2, params={"h0": 2.0},
        questions=[
            dict(suffix="q1", question_text="count?", ground_truth=5),
            dict(suffix="q2", kind="counterfactual_flavor", question_text="if X changed?", ground_truth=False),
        ],
    )
    with (d / "annotations.jsonl").open("w", encoding="utf8") as f:
        f.write(json.dumps(structural_record) + "\n")
        f.write(json.dumps(counterfactual_record) + "\n")

    migrate_dataset(d)
    migrated = [json.loads(l) for l in (d / "annotations.jsonl").read_text().splitlines()]

    q_by_id_1 = {q["qid"]: q for q in migrated[0]["questions"]}
    q_by_id_2 = {q["qid"]: q for q in migrated[1]["questions"]}
    # first record's legacy q2 was "structural_flavor" -> routed to q07
    assert q_by_id_1["q07"]["legacy_qid"] == "q2"
    assert q_by_id_1["q07"]["answer"] is True
    # second record's legacy q2 was "counterfactual_flavor" -> routed to q08 instead
    assert q_by_id_2["q08"]["legacy_qid"] == "q2"
    assert q_by_id_2["q08"]["answer"] is False


FLAT_SCHEMA_PLUGIN_SOURCE = textwrap.dedent("""
    from common.answer_types import count_answer

    LEGACY_QID_TO_SLOT = {"q1": "q01"}

    def params_for_hash(record):
        # a flat legacy schema has no "params" key -- explicitly name the
        # physics-defining fields, excluding "questions"/"derived"/
        # "schema_version" (what migration legitimately adds/changes).
        return {"h0": record["h0"], "e": record["e"]}

    def legacy_to_question(record, old_question, new_qid):
        base = dict(qid="q01", difficulty_level=1, tier="easy", task_family="description",
                    question="how many?", legacy_qid="q1", derivation_key="legacy_count")
        base.update(count_answer(old_question["ground_truth"]))
        return base

    def fill_remaining(record, mapped):
        from common.ladder import LADDER_TABLE
        plan = {
            "q02": ("multiple_choice", dict(options=["a", "b", "c", "d"], answer="A", answer_text="a")),
            "q03": ("numeric", dict(answer=0.5, answer_text="0.5", answer_tolerance=0.1, unit="s")),
            "q04": ("numeric", dict(answer=1.0, answer_text="1.0", answer_tolerance=0.1, unit="m")),
            "q05": ("multiple_choice", dict(options=["a", "b", "c", "d"], answer="B", answer_text="b")),
            "q06": ("multiple_choice", dict(options=["a", "b", "c", "d"], answer="C", answer_text="c")),
            "q07": ("boolean", dict(answer=True, answer_text="yes")),
            "q08": ("boolean", dict(answer=True, answer_text="yes")),
            "q09": ("ordering", dict(answer=["a", "b"], answer_text="a < b")),
            "q10": ("short_text", dict(answer="foo", answer_text="foo")),
        }
        out = {}
        for qid, level, tier, fam in LADDER_TABLE:
            if qid in mapped:
                continue
            qtype, extra = plan[qid]
            q = dict(qid=qid, difficulty_level=level, tier=tier, task_family=fam,
                      question_type=qtype, question=f"{qid}?", derivation_key=f"filled_{qid}")
            q.update(extra)
            out[qid] = q
        return out

    def build_derived(record, questions):
        return {"note": "populated by migration -- must NOT affect the params_for_hash guard"}

    def mutate_h0_by_mistake_variant(record):
        # simulates a bug: some migration code path accidentally alters a
        # physics-defining field instead of only restructuring questions.
        bad = dict(record)
        bad["h0"] = record["h0"] + 1.0
        return bad
""")


def test_flat_schema_dataset_needs_params_for_hash_override(tmp_path):
    d = tmp_path / "ds3"
    d.mkdir()
    (d / "ladder_migration.py").write_text(FLAT_SCHEMA_PLUGIN_SOURCE, encoding="utf8")
    record = dict(id="r1", modality="image", h0=2.0, e=0.7,
                  questions=[dict(qid="q1", question_text="count?", ground_truth=3)])
    (d / "annotations.jsonl").write_text(json.dumps(record) + "\n", encoding="utf8")

    migrate_dataset(d)
    migrated = json.loads((d / "annotations.jsonl").read_text())
    assert migrated["schema_version"] == 2
    assert migrated["derived"]["note"].startswith("populated by migration")
    assert migrated["h0"] == 2.0  # untouched


def test_params_for_hash_guard_actually_catches_a_mutated_physics_field(tmp_path):
    import importlib.util
    d = tmp_path / "ds3b"
    d.mkdir()
    (d / "ladder_migration.py").write_text(FLAT_SCHEMA_PLUGIN_SOURCE, encoding="utf8")
    spec = importlib.util.spec_from_file_location("plugin3b", d / "ladder_migration.py")
    plugin = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(plugin)

    record = dict(id="r1", modality="image", h0=2.0, e=0.7,
                  questions=[dict(qid="q1", question_text="count?", ground_truth=3)])
    # simulate the bug directly at the hash-comparison level the real
    # orchestrator uses, proving the guard's math actually distinguishes
    # "physics changed" from "only questions/derived changed".
    before = hash_params(plugin.params_for_hash(record))
    after = hash_params(plugin.params_for_hash(plugin.mutate_h0_by_mistake_variant(record)))
    assert before != after
