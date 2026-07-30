# H1 train/test dataset

This dataset tests whether retrieval of already learned facts transfers to a
globally held-out template family.

## Controlled comparison

Train and test contain the same people and the same 2,000 atomic facts. The
only intended split axis is `template_family_id`:

- train: `direct_question`, `imperative`, `profile_field`;
- test: `nominal_attribute`.

Fact overlap is intentional for H1. A test fact absent from train would measure
generalization to a new fact or entity instead of retrieval through a new
surface form.

## Fact and example identity

An atomic fact is identified within a world by:

```text
person_id + relation_id
```

Its complete definition is:

```text
person_id + relation_id + fact_value
```

For example:

```text
person_0001 + residence_city + Saint-Tula
```

`fact_id` remains the same when this fact is rendered with different
templates. `example_id` additionally contains `world_id` and `template_id`, so
every fact/template realization is distinct.

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
| train | 2,500 | 2,500 | 2,000 | 2,000 | 9,000 |
| test | 1,000 | 1,000 | 1,000 | 1,000 | 4,000 |

The generated files are stored in:

```text
data/generated/h1/seed_30072026/
├── metadata.json
├── world.json
├── train.jsonl
└── test.jsonl
```

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

Validation checks the exact Cartesian product, record schema, unique example
IDs, consistent fact definitions, equal fact coverage between train and test,
the global family holdout, world reproducibility, metadata counts, and hashes
of every source file.
