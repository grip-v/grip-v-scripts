# SHARED_CONTRACT.md — the contract every sub-benchmark must satisfy

The rules every generator in this repository follows: annotation schema,
question ladder, and validation. Read it before adding a new sub-benchmark.

---

## 0. Non-negotiables (inherited from the parent suite)

1. **Closed-form ground truth.** Every sample's geometry/physics is generated
   from exact algebra in a `geometry.py` (static) or `physics.py` (video)
   module. The renderer reads positions/angles/areas directly off those
   formulas. Ground truth is **never** measured from the rendered output.
2. **Independent validation.** Every sub-benchmark ships `validate_<name>.py`
   that (a) recomputes the scene from stored parameters only, (b) decodes the
   rendered PNG/video frames and locates objects by color-blob contour or
   template match, (c) asserts detected pixel positions match the closed-form
   prediction within a stated tolerance, and (d) **re-derives every question's
   answer from scratch** and diffs it against `annotations.jsonl`. Output goes
   to `validation_report.txt`. A validator may never be loosened to make a new
   feature pass — fix the contract, not the threshold.
3. **Determinism.** `--seed` fully determines the sample set. Re-running with
   the same seed reproduces byte-identical parameters (renders may differ in
   encoder noise only).
4. **Scale is a flag, not a redesign.** `--n 3000` must work without code
   changes.

---

## 1. Directory layout for a new sub-benchmark

```
<dataset_name>/
  README.md                 # what it tests, why the questions need this modality,
                            # §5 cost table: render time + disk per sample
  config.py                 # declarative sampling config (ranges, presets)
  geometry.py | physics.py  # closed-form scene math, zero rendering imports
  generate.py               # params -> annotations.jsonl (+ render invocation)
  render.py | scene.py      # matplotlib/PIL (static) or Manim Scene (video)
  questions.py              # the 10-question ladder builder (see §3)
  validate_<name>.py        # independent re-derivation + pixel check
  flatten.py                # -> flat parquet/jsonl for training/eval harnesses
  annotations.jsonl
  images/ | videos/
  validation_report.txt
  presets/
    <preset_name>/config.yaml
  tests/                    # ad-hoc configs proving primitives not yet in a preset
```

---

## 2. `annotations.jsonl` record schema

One JSON object per line, per sample:

```jsonc
{
  "sample_id": "angle_estimation_classic_000137",
  "dataset": "angle_estimation",
  "preset": "classic",
  "modality": "image",                  // "image" | "video"
  "media_path": "images/000137.png",    // or "videos/000137.mp4"
  "media_meta": {                       // video only
    "fps": 30, "duration_s": 6.0, "frame_count": 180,
    "resolution": [854, 480]
  },
  "seed": 137,
  "params": { /* every value needed to recompute the scene, closed-form */ },
  "derived": { /* exact quantities the questions consume, e.g. angles_deg */ },
  "render_meta": { "renderer": "manim|matplotlib", "version": "...",
                   "render_time_s": 12.4, "bytes": 481233 },
  "questions": [ /* exactly 10, see §3 */ ]
}
```

Question object:

```jsonc
{
  "qid": "q07",
  "difficulty_level": 3,          // 1..4, parent-suite contract
  "tier": "medium",               // "easy" | "medium" | "hard", see §3
  "task_family": "comparative",   // see §4
  "question_type": "multiple_choice", // see §5
  "question": "…",
  "options": ["A…", "B…", "C…", "D…"],  // null for non-MC
  "answer": "B",                  // canonical answer (option letter / number / bool / list)
  "answer_text": "37°",           // human-readable form of the answer
  "answer_tolerance": 2.0,        // numeric only; absolute unless *_rel
  "unit": "deg",                  // numeric only; null otherwise
  "rationale": "…",               // one-line derivation, machine-checkable
  "derivation_key": "theta_ab",   // key in `derived` this answer comes from
  "requires_motion": true,        // video only — see §6
  "single_frame_answerable": false
}
```

---

## 3. The 10-question ladder (replaces the old 4–5 question set)

Every sample carries **exactly 10 questions**, `q01`..`q10`, in this fixed
order. The old `difficulty_level: 1..4` contract is preserved; `tier` is the
coarse easy/medium/hard grouping.

| qid | difficulty_level | tier | Family | What it asks |
|---|---:|---|---|---|
| q01 | 1 | easy | description | Count or name the primary elements present |
| q02 | 1 | easy | description | Read one salient attribute directly (colour, label, largest item) |
| q03 | 2 | easy | relational | One pairwise relation between two named elements |
| q04 | 2 | medium | relational | A single-step measurement / classification with tolerance |
| q05 | 2 | medium | relational | Localisation: *where* / *which one* satisfies a stated property |
| q06 | 3 | medium | comparative | Rank or compare ≥3 elements on one dimension |
| q07 | 3 | medium | structural | A property of the whole configuration, not any one part |
| q08 | 3 | hard | counterfactual | "If X were changed to X′, what becomes true?" |
| q09 | 4 | hard | compound | Two-step chain: derive an intermediate, then use it |
| q10 | 4 | hard | compound_extrapolate | Extrapolate beyond what is shown (next state, hidden part, N-th step) |

Rules:
- Each qid must be answerable **only** from the sample + its own question text.
- q08–q10 must not be answerable by guessing the modal option; check the
  answer-distribution assertion in §7.
- No two questions in a sample may share a `derivation_key` **and** a
  `question_type` (prevents near-duplicate phrasings).

---

## 4. `task_family` vocabulary

`description`, `relational`, `comparative`, `structural`, `counterfactual`,
`compound`, `compound_extrapolate`, plus video-only:
`event_counting`, `temporal_order`, `duration`, `rate`, `prediction`,
`persistence` (object permanence / occlusion).

---

## 5. `question_type` vocabulary and answer-type mix

`multiple_choice` (4 options), `numeric` (with `answer_tolerance` + `unit`),
`boolean`, `count`, `ordering` (list of ids), `short_text` (closed vocabulary).

Per sample, across the 10 questions, enforce:
- ≥3 `multiple_choice`, ≥2 `numeric`, ≥1 `count`, ≥1 `ordering` or `boolean`.
- No more than 4 of any single type.

---

## 6. Video-only: the motion requirement

A question belongs in the video suite only if a single frame cannot answer it.
Enforce this **mechanically**, not by intuition:

- `single_frame_answerable` must be `false` for ≥7 of the 10 questions.
- Ship `tests/test_single_frame_ablation.py`: for each question marked
  `single_frame_answerable: false`, assert that the quantity in
  `derivation_key` is **not** determined by any one frame's state — i.e. the
  generator can construct a second scene with different parameters that is
  frame-identical at the sampled frame but yields a different answer. Store
  that counterexample's params in `derived.ablation_counterexample`.
- Questions that *are* single-frame answerable (q01, q02 typically) stay, as
  calibration anchors, but must be flagged truthfully.

---

## 7. Validator assertions (all sub-benchmarks)

`validate_<name>.py` must assert, and log counts for, all of:

1. **Schema**: every record validates against the §2 schema; exactly 10
   questions; qids `q01..q10` in order; ladder table in §3 matched exactly.
2. **Re-derivation**: recompute every answer from `params` alone; 0 mismatches
   (numeric within `answer_tolerance`).
3. **Pixel check**: sample K frames (video) or the single image; detect each
   tracked object by colour-blob contour; assert detected centroid within
   `PIXEL_TOL` of the closed-form prediction. Report max/mean error.
4. **Option sanity**: MC distractors are distinct, non-empty, and none is
   numerically within `answer_tolerance` of the correct answer.
5. **Answer distribution**: per dataset, correct-option letter is within
   ±20% of uniform; boolean answers within ±15% of 50/50; no numeric answer
   value covers >10% of samples.
6. **Type mix**: §5 quotas hold for every sample.
7. **Motion** (video): §6 ablation test passes for all flagged questions.
8. **Media integrity**: file exists, decodes, frame count matches
   `media_meta`, no all-black or all-identical frames.

`validation_report.txt` must end with a machine-greppable line:
`RESULT: <n_samples> samples, <n_questions> questions, <n_mismatches> mismatches`

---

## 8. Migration rule for existing sub-benchmarks

When extending an existing dataset from 4–5 to 10 questions:
- **Keep the original questions** and their original `difficulty_level`; map
  them onto the nearest ladder slot and note the old qid in
  `question.legacy_qid`. Do not renumber silently.
- Regenerate `annotations.jsonl` in place from the **same seeds** so media
  files need not be re-rendered. Assert re-render-free migration by hashing
  `params` before and after.
- Add `schema_version: 2` at the record root; keep a reader shim for v1.

---

## 9. Definition of done for a new sub-benchmark

- [ ] `geometry.py`/`physics.py` has no rendering imports and is unit-tested
      against hand-computed values in `tests/`.
- [ ] 150-sample prototype generated, rendered, validated: `0 mismatches`.
- [ ] `validation_report.txt` committed.
- [ ] README written, including §5 cost table and, for video, an explicit
      paragraph on *why each question needs motion*.
- [ ] `flatten.py` emits one row per question with the media path.
- [ ] `--n 3000` dry-run (`--no-render`) completes without error.

---

## 10. Physics addendum (applies to every physics sub-benchmark)

Physics sub-benchmarks inherit everything above, plus:

**10.1 Closed form is the default; integration is an exception that must be
declared.** Prefer systems with exact solutions (projectile without drag,
small-angle pendulum, SHM, elastic collisions, Kepler orbits, Snell's law).
Where no closed form exists (large-angle pendulum, three-body, drag), the
sub-benchmark must:
- name the reference solver and tolerance in `params.solver`
  (e.g. `{"method": "rk4", "dt": 1e-4, "abs_tol": 1e-6}`),
- ship a convergence test in `tests/` showing the answer is stable to within
  half the question's `answer_tolerance` when `dt` is halved,
- and where a closed form exists for a limiting case (small-angle, zero-drag),
  assert the solver reproduces it. Ground truth still comes from the solver,
  never from the rendered pixels.

**10.2 Units are mandatory and explicit.** Every numeric answer carries `unit`
from a fixed vocabulary (`m`, `m/s`, `m/s^2`, `s`, `kg`, `N`, `J`, `W`, `deg`,
`rad`, `rad/s`, `Hz`, `Pa`, `N*m`). `params` records `g`, and the
pixels-per-metre scale in `render_meta.scale_px_per_m`. At least one question
per sample must require the metric scale (a visible scale reference in-frame),
so the dataset tests metric grounding, not just pixel ratios.

**10.3 Conserved-quantity ledger.** `derived` must include a per-frame ledger
of the system's conserved quantities (`energy_j`, `momentum_kg_m_s`,
`angular_momentum`), computed from the closed form. The validator asserts
these are conserved to within `CONSERVATION_TOL` for any clip not flagged
`violation_type`. This is what makes `physics_violation_spotting` possible
and catches renderer/physics drift for free.

**10.4 Prediction mode is required, not optional.** Every video physics
sub-benchmark ships at least one named truncation preset (Phase-5 style), with
q10 scored against the full simulation rather than the rendered clip.

**10.5 Distractor generation for numeric physics answers.** MC distractors
must be generated from *named physical errors*, each labelled in
`question.distractor_types`: `dropped_factor_two`, `sin_for_cos`,
`ignored_component`, `wrong_power_of_r`, `unit_confusion`, `sign_error`.
Random numeric distractors are not acceptable — labelled ones make error
analysis fall out of the annotations.

**10.6 Regime annotation.** Each sample records its regime in
`params.regime` (e.g. `small_angle` vs `large_angle`, `elastic` vs
`inelastic`, `underdamped`/`critical`/`overdamped`, `subsonic`). Presets
sweep the regime so per-regime accuracy is measurable.
