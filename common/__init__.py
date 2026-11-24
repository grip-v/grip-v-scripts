"""Prompt 0 groundwork: shared schema/ladder/answer/validate/distribution
helpers for every sub-benchmark in the suite (SHARED_CONTRACT.md). Every
dataset's own `questions.py`/`validate_<name>.py` imports from here instead
of re-implementing the §2 schema, §3 ladder table, §5 answer-type rules, or
§7 validator assertions locally.

No dataset's own code is modified by this package's existence -- see
EXPANSION_PROMPTS.md Prompt 0's own "do not modify any existing dataset in
this step" instruction. `bouncing_ball_restitution` in particular predates
this contract entirely and is migrated separately (Prompt 23).
"""
