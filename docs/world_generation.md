# Micro-world generation

The first world schema models a person and four ground-truth relations:

```text
Person --birth_date-----> Date
       --occupation-----> Occupation
       --birth_city-----> City
       --residence_city-> City
```

`Person` is the convenient typed view of one entity. `Fact` is the graph-edge
view used by later dataset generation. `World` owns both the generation seed
and the generated people, and exposes the corresponding facts.

## Determinism

`Seed` makes keyed choices rather than consuming one global pseudo-random
stream. Each choice has a stable key such as `person_0001.birth_city`. This
means adding an unrelated field later will not reshuffle existing attributes.

The same seed, source files, and number of people produce byte-for-byte
identical JSON. Person IDs are assigned as `person_0001`, `person_0002`, and so
on. Names are selected without replacement, so the current maximum world size
is the number of unique values in `data/strange_names.txt`.

## Residence rule

Each person independently moves with probability `1/3`. A person who does not
move has `residence_city == birth_city`. A person who moves is assigned a
different city. The realized fraction of movers in one world can differ from
exactly one third, especially for small worlds.

## Command line

Print a world to standard output:

```bash
python3 scripts/generate_world.py --seed 30072026 --people 100
```

Write it to a file:

```bash
python3 scripts/generate_world.py \
  --seed experiment-01 \
  --people 100 \
  --output artifacts/world.json
```

Both integer-looking and textual seeds are accepted and stored as strings.

## Deliberate first-version boundaries

- Cities, dates, and professions may be reused by multiple people.
- Names are unique within a world.
- The source lists are treated as curated inputs and must be non-empty and
  duplicate-free.
- Family relations and person-to-person edges are not generated yet.
- Dataset questions and train/evaluation splits are a later layer built from
  `World.facts`; they are not part of world generation.
