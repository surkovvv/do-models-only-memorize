# H1 experiment: current state and next steps

Last verified: 2026-08-12.

This note is the durable project memory for decisions that are easy to lose
between experiment sessions.

## Verified current state

- Branch `30.07.2026`, commit `f773a9c`, contains the work completed before
  2026-08-09: reproducible full-dataset SFT evaluation, bf16/CUDA support,
  metrics and environment artifacts, model revision pinning, and MLC/ClearML
  integration.
- Branch `09.08.2026`, commit `10b81ff`, is based on that commit and contains
  the H1 template-transfer dataset work completed on 2026-08-09.
- The registry has 32 templates: exactly two templates for every combination
  of four operations and four template families.
- The current manifest assigns `_001` globally to SFT and `_002` globally to
  `eval_seen_family_new_template`. `nominal_attribute` is the globally held-out
  family.
- Seed `30072026` produces 6,000 SFT examples and a paired evaluation sample of
  250 people / 1,000 facts. Each fact appears once in each of three slices:
  `eval_exact_recall`, `eval_seen_family_new_template`, and
  `eval_heldout_family`.
- Two comparable family-fold datasets additionally hold out `direct_question`
  and `profile_field`. Strict per-fold validation and cross-fold fact pairing
  validation pass; all three folds share the same world and 1,000 eval facts.
- An exploratory five-epoch extension was submitted for the nominal-attribute,
  direct-question, and profile-field folds. Relative to the three-epoch runs,
  only `training.epochs` changes. Treat this extension as exploratory if the
  decision to run longer used held-out-family results; do not present it as an
  untouched confirmatory estimate.
- `test.jsonl` contains all three slices. The full-run loader partitions it by
  `split`; every slice contains 250 examples per operation.
- Strict template, split, and dataset validators pass. The full suite passed
  with 43 tests and 4 subtests on the published tree.
- `microworld-lab` is a separate local nested repository. It must not be added
  to this repository or pushed with the experiment branches.

## Completed in the current branch

### Auditable evaluation artifacts

- Every full-evaluation prediction is saved with source provenance, raw and
  normalized output, normalized target, exact match, epoch, step, and split.
- Exact match is aggregated by operation, template family, template ID,
  entity, attribute value, and world.
- Same-fact paired rows and gaps are saved for exact versus seen-template,
  seen-template versus held-out-family, and exact versus held-out-family.
- The resolved config, environment, checkpoint history, final summary, and
  reloadable final model are retained by the full-run configuration.

### Pilot protocol

- `configs/h1_full.yaml` names the 0.6B full-fine-tuning pilot explicitly and
  evaluates the untouched base model at `epoch=0`.
- The concrete optimizer, effective batch, warmup, cosine schedule, sequence
  length, precision, gradient clipping, and model revision are recorded in the
  resolved config and logs.
- Exact recall must be high before a held-out-family gap is interpreted as
  surface-form dependence. Hyperparameters and checkpoints must not be selected
  by looking at `eval_heldout_family`.

## Next work, in priority order

### P2: remove template-direction and family-specific confounds

1. Add the counterbalanced manifest with `_002` in SFT and `_001` in
   seen-family/new-template evaluation.
2. Rotate the globally held-out family across all four families. The nominal,
   direct-question, and profile-field folds now exist; the imperative fold is
   still missing.
3. Keep all three evaluation slices paired on the same sampled facts in every
   manifest.

### P3: replication and inference

1. Use multiple world seeds and multiple training seeds. A reasonable first
   robust matrix is 4 held-out-family folds x 3 world seeds x 3 training seeds,
   after the pilot has frozen the protocol.
2. Treat people, not repeated template renderings, as the resampling cluster.
   Report person-cluster bootstrap confidence intervals for paired gaps.
3. Report per-operation results alongside the aggregate; 250 examples per
   operation are intended to detect a fairly large effect, not a tiny gap.

## Run hygiene and traps

- Regenerate the MLC preset from `09.08.2026`; do not reuse the old local
  `.private` full-run preset because it predates the new dataset contract.
- Record Git commit, resolved configuration, model revision, dataset metadata
  and hashes, world seed, training seed, and template-assignment manifest for
  every run.
- Do not let a seen-family evaluation template ID appear in SFT for other
  facts. The holdout is global by template ID.
- Unused possible renderings do not need to be moved into evaluation. They may
  remain ungenerated; statistical information comes primarily from distinct
  facts and people, not repeated phrasings of the same fact.
- Projection, composition, and H2 diversity experiments come after the H1
  paraphrase-invariance protocol is complete and auditable.
