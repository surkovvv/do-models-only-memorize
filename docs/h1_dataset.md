# H1 SFT/evaluation dataset

This dataset tests whether retrieval of already learned facts transfers first
to a new wording inside a seen template family and then to a globally held-out
template family.

## Controlled comparison

SFT and evaluation use facts from the same generated world. The family split
is:

- SFT families: `direct_question`, `imperative`, `profile_field`;
- held-out family: `nominal_attribute`.

Within every SFT family and operation, the split manifest assigns `_001` to
SFT and reserves `_002` globally for `eval_seen_family_new_template`. The
suffixes are an explicit assignment for this run, not a permanent direction;
later runs can swap them for counterbalancing.

Fact overlap is intentional for H1. A test fact absent from SFT would measure
generalization to a new fact or entity instead of retrieval through a new
surface form.

## Paired evaluation sample

The manifest deterministically selects 250 complete people, or 1,000 atomic
facts across the four operations. Each selected fact appears once in each
evaluation slice:

- `eval_exact_recall`: the same template ID used for that fact during SFT;
- `eval_seen_family_new_template`: the reserved sibling template from the same
  family;
- `eval_heldout_family`: one of the two globally held-out
  `nominal_attribute` templates.

Seen SFT families and held-out template variants are balanced across facts.
Both nominal templates are balanced within each operation. The remaining
possible renderings are intentionally unused; generated rows do not need to be
exhaustively partitioned between SFT and evaluation.

## Fact and example identity

An atomic fact is identified within a world by:

```text
person_id + relation_id
```

Its complete definition is `person_id + relation_id + fact_value`.
`fact_id` remains the same across renderings. `example_id` contains `world_id`,
`split`, `fact_id`, and `template_id`, so the intentional exact-recall copy is
still a distinct evaluation record.

## Record schema

Every JSONL record contains:

```text
example_id
world_id
world_seed
person_id
person_name
fact_id
relation_id
fact_value
operation_id
template_family_id
template_id
rendered_question
canonical_answer
answer_format_id
split
```

`canonical_answer` is equal to `fact_value`. Dates use ISO `YYYY-MM-DD`.
Synonym expansion is not used.

## Current generated dataset

Seed `30072026` with 500 people produces:

| Split | Birth city | Birth date | Residence | Occupation | Total |
| --- | ---: | ---: | ---: | ---: | ---: |
| train | 1,500 | 1,500 | 1,500 | 1,500 | 6,000 |
| eval exact recall | 250 | 250 | 250 | 250 | 1,000 |
| eval seen-family/new-template | 250 | 250 | 250 | 250 | 1,000 |
| eval held-out family | 250 | 250 | 250 | 250 | 1,000 |

The generated files are stored in:

```text
data/generated/h1/seed_30072026/
├── metadata.json
├── world.json
├── train.jsonl
└── test.jsonl   # all three evaluation slices, distinguished by `split`
```

Two additional held-out-family folds reuse the same world, selected people,
facts, template direction, and training protocol:

| Dataset directory suffix | Train families | Held-out family |
| --- | --- | --- |
| `seed_30072026` | direct question, imperative, profile field | nominal attribute |
| `seed_30072026_heldout_direct_question` | imperative, nominal attribute, profile field | direct question |
| `seed_30072026_heldout_profile_field` | direct question, imperative, nominal attribute | profile field |

`scripts/validate_h1_family_folds.py` checks that all folds contain the same
world and the same complete fact definitions in every evaluation slice. The
only permitted scientific axis is the template-family assignment.

## Reproduction and validation

Generate:

```bash
python3 scripts/generate_h1_dataset.py \
  --seed 30072026 \
  --people 500 \
  --output-dir data/generated/h1/seed_30072026
```

Validate:

```bash
python3 scripts/validate_h1_dataset.py \
  data/generated/h1/seed_30072026
```

For a non-default fold, pass its manifest explicitly:

```bash
python3 scripts/validate_h1_dataset.py \
  data/generated/h1/seed_30072026_heldout_direct_question \
  --split-manifest data/splits/h1_heldout_direct_question.json

python3 scripts/validate_h1_family_folds.py \
  data/generated/h1/seed_30072026 \
  data/generated/h1/seed_30072026_heldout_direct_question \
  data/generated/h1/seed_30072026_heldout_profile_field
```

Validation reconstructs the deterministic SFT and paired evaluation rows. It
checks record identity, consistent fact definitions, equal fact coverage across
the three evaluation slices, exact-recall membership in SFT, global exclusion
of seen-family eval template IDs from SFT, the held-out family, world
reproducibility, metadata counts, and hashes of every source file.
