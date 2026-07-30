# Query template registry

Each `.jsonl` file contains one JSON object per line:

```json
{"operation_id":"GET_BIRTH_CITY","template_family_id":"nominal_attribute","template_id":"birth_city_nominal_001","answer_format_id":"canonical_value","template":"What is {person}'s city of birth?"}
```

Files and answer types:

- `born_in_city.jsonl` → city
- `lives_in_city.jsonl` → city
- `born_in_date.jsonl` → ISO date
- `profession.jsonl` → profession

Replace `{person}` with a value from `data/strange_names.txt`.

Templates within one file implement the same query operation, so semantic
equivalence is intentional. A template family counts as independent only when
it changes an interpretable linguistic or presentation property.

Use these curation rules:

- prefer two individual templates per family; one is sufficient for the
  initial H1 family-level holdout, while a future seen-family/new-template
  evaluation requires two;
- keep at most one family for a syntactic, lexical, discourse, or presentation
  construction;
- reject pairs that differ only by preposition placement, possessive/genitive
  alternation, or a wrapper such as `Can you say ...` versus `Tell me ...` when
  they are proposed as different families;
- require more than a single-word substitution between families;
- keep the requested answer type unambiguous;
- hold out `template_family_id` globally across all operations, not only within
  one operation.

The initial registry uses four deliberately different families per operation:

- a direct question;
- a nominal attribute query;
- an imperative;
- a profile-field completion.

Why these families are retained:

| Family | Independent change |
| --- | --- |
| `direct_question` | A finite wh-question asks through the event or state predicate. |
| `nominal_attribute` | A copular or attribute-focused question names the stored field. |
| `imperative` | A command requests the same value through a different discourse form. |
| `profile_field` | A structured record-completion task replaces ordinary question syntax. |

These are holistic surface-form families, not a factorial decomposition of
syntax, lexicon, and discourse. A held-out-family result therefore supports a
claim about transfer across the whole realization family. It must not be
reported as a pure syntax effect or a pure lexical effect.

Operation-specific semantic choices:

- birth-city templates always request a `city`; `town`, `hometown`, “where
  from,” and an unconstrained `birthplace` answer are excluded;
- birth-date templates request the full date, never only its year, month, or
  day;
- residence templates specify both `city` and current residence; “calls home”
  is excluded because it may express affiliation rather than residence;
- occupation templates ask for an occupation/profession or what the person
  works as; job title and organizational role are excluded.

The following contrasts are intentionally not separate families:

- pied-piping versus preposition stranding (`In which city ...?` versus
  `Which city ... in?`);
- possessive versus `of`-genitive (`{person}'s city of birth` versus
  `the city of {person}'s birth`);
- request-verb substitutions (`Name ...`, `Give ...`, `Tell me ...`);
- an unchanged core question inside a cleft or conversational wrapper;
- punctuation, politeness, or single-word synonym changes.

Marked constructions such as echo questions and topic-comment fragments are
also excluded from the initial registry. They are linguistically distinct, but
their unusual pragmatics would add a second difficulty beyond paraphrasing and
make a transfer failure harder to interpret.

`template_family_id` is global across operations and identifies the family used
for train/evaluation splitting. Structurally parallel forms cannot leak through
another operation under a different family ID. A held-out family must be absent
from SFT for every operation.

`template_id` identifies an individual wording within a family. The two close
wordings inside one family are not independent evidence. They exist so that SFT
can use one wording and seen-family evaluation can use the other without
repeating the exact prompt string. For example, pied-piping and preposition
stranding are both members of `direct_question`, not two families.

The `_001` and `_002` suffixes do not prescribe train/evaluation direction.
Their assignment must be counterbalanced across worlds or runs. Generated data
must keep exact-string recall, a new template from a seen family, and a globally
held-out family as three separate evaluation slices.

`operation_id` states which fact is requested. `answer_format_id` is
`canonical_value` for every template: a city, an ISO date, or a profession
value only. Profile-field templates state “value only” explicitly so their
presentation form does not silently change the answer contract.

`data/synonyms.json` is a source of candidate wording during manual curation,
not an automatic expansion list. Lexical substitutions alone must never create
new template families.

Run the registry validator before using the templates:

```bash
python3 scripts/validate_templates.py --strict
```

The validator checks the JSONL schema, operation and answer-format invariants,
identifiers, `{person}` slots, exact family membership, exact duplicates, and
high lexical or sequence similarity across different families within an
operation. Similarity inside one family is expected and is not counted as
independent evidence. Every retained family still requires semantic review
against the research charter.

This validator checks the registry only. Once generated split files exist, a
separate split validator must enforce global family holdout and the three
evaluation-slice contracts defined in the research charter.

## Initial H1 split

The first H1 experiment uses the family assignment in
`data/splits/h1_template_families.json`. `nominal_attribute` is held out
globally for test. `direct_question`, `imperative`, and `profile_field` are
assigned to train.

The manifest is the source of truth: downstream dataset generation filters the
registry by its family lists rather than maintaining copied train and test
template files. Validate it with:

```bash
python3 scripts/validate_h1_split.py
```

The validator asserts that:

- train and test family IDs are disjoint;
- every registry family is assigned exactly once;
- every template is covered by the split;
- the per-operation counts are 4/1 for five templates, 4/2 for six templates,
  and 5/2 for seven templates.
