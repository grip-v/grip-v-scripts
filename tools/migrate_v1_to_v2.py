"""EXPANSION_PROMPTS.md Prompt 0, item 2 / SHARED_CONTRACT.md §8: migrate an
existing dataset's `annotations.jsonl` from its old (v1) shape onto the v2
schema_version=2 / 10-question ladder, WITHOUT re-rendering any media.

"Map legacy questions onto ladder slots" and "fill the remaining slots
[via] the dataset's questions.py" are inherently dataset-specific -- every
pre-existing dataset's old question set has different qids/content and a
different `questions.py` API. This script is a generic ORCHESTRATOR that
delegates those two steps to a small plugin every migrated dataset provides
at `<dataset_dir>/ladder_migration.py`, exposing:

    LEGACY_QID_TO_SLOT: dict[str, str | Callable[[dict], str]]
        old question qid (as it appears in a v1 record's "questions" list,
        or as computed by the plugin's optional `extract_legacy_qid`) -> new
        ladder slot qid ("q01".."q10") it becomes. A value may be a plain
        string, or a callable(old_question) -> qid for a legacy qid whose
        target slot depends on the question's own content (e.g. one old qid
        that was actually built from two different templates depending on a
        generation-time coin flip -- a static string can't route those
        differently, but the callable sees the full old question).

    extract_legacy_qid(old_question: dict) -> str   [optional]
        how to read the "old qid" off one legacy question dict. Defaults to
        `old_question.get("qid") or old_question.get("question_id")` --
        override this when a dataset's legacy schema encodes the qid
        differently (e.g. only as a suffix of a longer id, or implicitly via
        `difficulty_level`).

    legacy_to_question(record: dict, old_question: dict, new_qid: str) -> dict
        converts one old question dict into a full §2 question dict for
        `new_qid`, setting `legacy_qid` to the OLD qid (§8: "note the old
        qid in question.legacy_qid. Do not renumber silently"). Takes the
        whole `record` too (not just the question) because building a
        proper fixed-4-option multiple_choice question often needs other
        record-level data (e.g. every ball's color) that isn't on the old
        question dict itself.

    fill_remaining(record: dict, mapped: dict[str, dict]) -> dict[str, dict]
        given the v1 record and the ladder slots already filled from legacy
        questions, returns question dicts for whichever of the 10 slots are
        still unfilled (typically by calling the dataset's own
        `questions.py`).

No such plugin exists for any dataset yet -- `bouncing_ball_restitution`'s
own migration (EXPANSION_PROMPTS.md Prompt 23) is what will add
`bouncing_ball_restitution/ladder_migration.py`. `common/tests/` proves this
orchestrator is correct against a synthetic in-memory v1 fixture ahead of any
real plugin existing, per Prompt 0's own "do not modify any existing dataset
in this step".

Usage:
    python tools/migrate_v1_to_v2.py <dataset_dir> [--dry-run]
"""
from __future__ import annotations
import argparse
import hashlib
import importlib.util
import json
import shutil
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))

from common import schema  # noqa: E402
from common.ladder import QuestionLadder, LADDER_QIDS  # noqa: E402


class MigrationError(RuntimeError):
    pass


def hash_params(params: dict) -> str:
    return hashlib.sha256(json.dumps(params, sort_keys=True).encode("utf8")).hexdigest()


def load_plugin(dataset_dir: Path):
    plugin_path = dataset_dir / "ladder_migration.py"
    if not plugin_path.exists():
        raise MigrationError(
            f"{dataset_dir} has no ladder_migration.py plugin -- see this module's docstring "
            f"for the required LEGACY_QID_TO_SLOT / legacy_to_question / fill_remaining API"
        )
    spec = importlib.util.spec_from_file_location(f"{dataset_dir.name}_ladder_migration", plugin_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    for name in ("LEGACY_QID_TO_SLOT", "legacy_to_question", "fill_remaining"):
        if not hasattr(module, name):
            raise MigrationError(f"{plugin_path} is missing required attribute {name!r}")
    return module


def migrate_record(record: dict, plugin) -> dict:
    if schema.is_v2(record):
        raise MigrationError(f"record {record.get('sample_id') or record.get('id')} is already v2")

    new_record = dict(record)
    # `normalize_record(record) -> dict`  [optional plugin hook]
    #   §2 requires specific top-level key NAMES (sample_id, preset, modality,
    #   media_path, render_meta, ...) that a legacy dataset may spell
    #   differently (bouncing_ball_restitution uses "id"/"config_preset"/
    #   "media_type"/"video_path" and has no "render_meta" at all). Applied
    #   FIRST -- before video-detection, hashing, or anything else below --
    #   so every subsequent step sees the normalized shape too, and returns
    #   ADDITIONAL/renamed top-level keys to merge in, never removing the
    #   originals (so nothing that already reads the old flat fields breaks).
    if hasattr(plugin, "normalize_record"):
        new_record.update(plugin.normalize_record(record))

    # `params_for_hash(record) -> dict`  [optional plugin hook]
    #   what to hash for the "migration never touches physics params" guard.
    #   Defaults to `record.get("params", record)` -- fine for a dataset that
    #   already has a "params" key, but a WRONG default for one that doesn't
    #   (like bouncing_ball_restitution's flat legacy schema): hashing the
    #   whole record against itself before/after would trivially always
    #   match (comparing the same untouched dict to itself), silently
    #   defeating the guard. A dataset without a "params" key MUST override
    #   this to name exactly its physics-defining fields (e.g. balls'
    #   h0/restitution/mass/radius/x/vx, duration, fps, gravity, primitives)
    #   and exclude the fields migration legitimately adds/changes
    #   ("questions", "derived", "schema_version").
    params_for_hash = getattr(plugin, "params_for_hash", lambda r: r.get("params", r))
    before_hash = hash_params(params_for_hash(new_record))

    extract_legacy_qid = getattr(plugin, "extract_legacy_qid", lambda q: q.get("qid") or q.get("question_id"))

    mapped: dict[str, dict] = {}
    for old_q in new_record.get("questions", []):
        old_qid = extract_legacy_qid(old_q)
        slot = plugin.LEGACY_QID_TO_SLOT.get(old_qid)
        if slot is None:
            continue  # this legacy question isn't mapped onto any ladder slot -- dropped, not silently kept
        # A slot mapping may be a fixed qid string, OR a callable(old_question) -> qid,
        # for datasets (like bouncing_ball_restitution) where the SAME legacy qid
        # holds structurally different content per record (e.g. its old "q4" is
        # either a comparison or an extrapolation question, chosen at generation
        # time) -- a static string can't route those to different ladder slots,
        # but the plugin has the full old_q content to decide with.
        new_qid = slot(old_q) if callable(slot) else slot
        mapped[new_qid] = plugin.legacy_to_question(new_record, old_q, new_qid)

    remaining = plugin.fill_remaining(new_record, dict(mapped))
    mapped.update(remaining)

    missing_slots = [qid for qid in LADDER_QIDS if qid not in mapped]
    if missing_slots:
        raise MigrationError(f"record {record.get('sample_id') or record.get('id')}: "
                               f"ladder slots {missing_slots} unfilled after legacy mapping + fill_remaining")

    builders = {qid: (lambda q=mapped[qid]: q) for qid in LADDER_QIDS}
    video = new_record.get("modality") == "video"
    questions = QuestionLadder(builders).build(video=video)  # raises LadderError on any table mismatch

    new_record["questions"] = questions
    new_record["schema_version"] = 2
    # `build_derived(record, questions) -> dict`  [optional plugin hook]
    #   §2's "derived" field: exact quantities the questions consume. Called
    #   AFTER the questions are built (so it can see the final derivation_keys)
    #   but its output is deliberately excluded from the hash guard below by
    #   whatever `params_for_hash` the plugin defines -- adding new derived/
    #   support data is legitimate migration output, not a physics change.
    if hasattr(plugin, "build_derived"):
        new_record["derived"] = plugin.build_derived(new_record, questions)
    # `build_params(record) -> dict`  [optional plugin hook]
    #   §2 requires a top-level "params" dict; a dataset whose legacy schema
    #   keeps its physics-defining fields flat (no "params" key at all, like
    #   bouncing_ball_restitution) needs an explicit hook to populate it --
    #   defaulting to whatever's already there (or {}) leaves a v1-flat
    #   dataset's v2 records without a real params object, which is schema-
    #   valid but not schema-FAITHFUL. Existing flat fields are left in place
    #   either way (nothing reads "params" instead of them yet).
    if hasattr(plugin, "build_params"):
        new_record["params"] = plugin.build_params(new_record)
    new_record.setdefault("params", {})
    new_record.setdefault("derived", {})

    after_hash = hash_params(params_for_hash(new_record))
    if before_hash != after_hash:
        raise MigrationError(
            f"record {record.get('sample_id') or record.get('id')}: params hash changed during migration "
            f"({before_hash} -> {after_hash}) -- migration must never touch params (would require re-rendering)"
        )

    return new_record


def migrate_dataset(dataset_dir: Path, dry_run: bool = False) -> dict:
    dataset_dir = Path(dataset_dir)
    annotations_path = dataset_dir / "annotations.jsonl"
    if not annotations_path.exists():
        raise MigrationError(f"{annotations_path} does not exist")

    plugin = load_plugin(dataset_dir)
    lines = annotations_path.read_text(encoding="utf8").splitlines()
    records = [json.loads(l) for l in lines if l.strip()]

    migrated = []
    already_v2 = 0
    for record in records:
        if schema.is_v2(record):
            already_v2 += 1
            migrated.append(record)
            continue
        migrated.append(migrate_record(record, plugin))

    if not dry_run:
        backup_path = annotations_path.with_suffix(annotations_path.suffix + ".v1.bak")
        if not backup_path.exists():
            shutil.copy2(annotations_path, backup_path)
        with annotations_path.open("w", encoding="utf8", newline="\n") as f:
            for r in migrated:
                f.write(json.dumps(r, sort_keys=True, separators=(",", ":")) + "\n")

    return dict(n_records=len(records), n_migrated=len(records) - already_v2, n_already_v2=already_v2,
                dry_run=dry_run)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dataset_dir", type=Path)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    stats = migrate_dataset(args.dataset_dir, dry_run=args.dry_run)
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
