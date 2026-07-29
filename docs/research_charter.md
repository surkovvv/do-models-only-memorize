# Research Charter: Does SFT Learn Facts or Retrieval Forms?

**Status:** Initial project definition  
**Scope of this document:** Research objective, conceptual framework, micro-world design, and data/evaluation requirements. Training recipes, model selection, hyperparameters, experiment schedules, infrastructure, and run tracking are intentionally deferred.

## 1. Project Objective

This project will build a controlled synthetic micro-world in order to study how factual knowledge acquired through supervised fine-tuning (SFT) can be retrieved.

The central question is:

> When a language model is fine-tuned on a new fact expressed through a particular question–answer form, does it acquire access to the underlying fact, or does successful retrieval remain tied to the linguistic forms observed during SFT?

The project will:

1. define a synthetic world whose factual assignments do not correspond to real-world knowledge;
2. generate multiple controlled linguistic realizations of those facts;
3. construct train and evaluation splits that isolate distinct forms of generalization;
4. operationally distinguish surface-form-dependent retrieval from fact-level retrieval;
5. produce datasets and evaluation slices suitable for reproducible SFT studies.

The initial deliverable is not a claim that SFT generally “only memorizes formats.” The intended result is a precise behavioral measurement of **surface-form dependence in factual retrieval after SFT**.

## 2. Motivation

A model can answer a training-style question correctly for at least two different reasons:

- it has acquired a representation of the underlying fact that can be accessed in multiple ways; or
- it has learned a narrow association between a particular input form and a particular output.

Ordinary datasets make these explanations difficult to separate. Their facts may already be present in pretraining, paraphrases may leak across splits, and changes in question wording often coincide with changes in task semantics or answer format.

A synthetic micro-world provides explicit control over:

- which facts exist;
- which facts and query operations appear during SFT;
- how each fact is expressed;
- which linguistic families are held out;
- what information the answer must contain;
- which source of variation changes between train and evaluation.

This control makes it possible to ask a narrow, falsifiable question about what generalizes after SFT.

## 3. Conceptual Model

The project separates the data-generating process into four layers.

### 3.1 World

The world is a collection of entities and ground-truth attributes. An initial person record may contain:

```json
{
  "name": "Gogr",
  "birth_date": {
    "day": 1,
    "month": 6,
    "year": 2000
  },
  "birth_city": "Tor-Velk",
  "residence_city": "Mor-Dalen",
  "occupation": "architect"
}
```

Person and city names are synthetic, and their attribute assignments do not correspond to real-world facts. This makes recovery from pretraining implausible and directly testable with a base-model control. Occupations may remain ordinary English terms because their meanings are part of the query semantics rather than the novel knowledge being learned.

### 3.2 Facts

A fact is a ground-truth relation or structured attribute in the world:

```text
birth_city(Gogr) = Tor-Velk
residence_city(Gogr) = Mor-Dalen
occupation(Gogr) = architect
birth_date(Gogr) = 2000-06-01
```

A date is stored as a structured value rather than as an indivisible string so that its day, month, and year can be queried independently.

### 3.3 Query Operations

A query operation specifies which information must be recovered. It is independent of the wording used to request that information.

Atomic operations include:

```text
GET_BIRTH_CITY
GET_RESIDENCE_CITY
GET_OCCUPATION
GET_BIRTH_DATE
GET_BIRTH_DAY
GET_BIRTH_MONTH
GET_BIRTH_YEAR
```

Composite operations request multiple known attributes:

```text
GET_BIRTH_CITY_AND_OCCUPATION
GET_RESIDENCE_CITY_AND_OCCUPATION
GET_BIRTH_DATE_AND_OCCUPATION
GET_BIRTH_CITY_AND_BIRTH_DATE
```

Composite operations are queries over multiple base facts; they are not additional facts in the world.

### 3.4 Linguistic Realizations

A template family expresses a query operation through a particular syntactic, lexical, discourse, or presentation form.

For `GET_BIRTH_CITY`, examples of distinct families include:

```text
In which city was {person} born?
What is {person}'s city of birth?
Identify the city in which {person} came into the world.
Fill the blank with the value only: {person} — City of birth: ____.
```

Minor edits do not constitute independent families. For example, adding “please,” changing punctuation, or replacing a single word while retaining the same construction should not be treated as a new source of evidence.

## 4. Operational Definitions

This project does not attempt to determine directly how a fact is internally represented. It uses behavioral criteria.

### Fact-level retrieval

A model exhibits evidence of fact-level retrieval when it can return information entailed by a fact under a controlled transformation that was not demonstrated for that fact during SFT.

The weakest evidence is successful retrieval under a held-out paraphrase of the same operation. Stronger evidence comes from retrieving a component of a structured fact or combining independently learned facts.

### Surface-form-dependent retrieval

Retrieval is surface-form-dependent when the model succeeds on question forms represented during SFT but degrades substantially on held-out template families while the person, fact, operation, and answer format remain fixed.

This behavior is consistent with memorization of a formulation or a narrow retrieval procedure. It does not, by itself, prove verbatim string memorization.

### Memorization

“Memorization” will be used only as a behavioral description and must be qualified. A failure on a held-out form may reflect:

- dependence on a prompt or template family;
- failure to interpret a new query form;
- an inaccessible rather than absent factual representation;
- sensitivity to tokenization of synthetic identifiers;
- failure to perform a new operation over an otherwise stored fact.

The project will therefore report measured transfer gaps rather than infer a unique internal mechanism from accuracy alone.

## 5. Evaluation Modes

The evaluation is organized around three transformations. They must remain separate because each tests a different capability.

### 5.1 Paraphrase Invariance

The fact, query operation, and answer format remain fixed; only the linguistic realization changes.

```text
SFT:
In which city was Gogr born?
→ Tor-Velk

Evaluation:
What is Gogr's city of birth?
→ Tor-Velk
```

This is the primary test of surface-form dependence.

Evidence of narrow form memorization:

- high accuracy on seen template families;
- substantially lower accuracy on held-out template families for the same facts and operations.

Evidence of form-invariant retrieval:

- comparable accuracy across seen and held-out template families.

### 5.2 Projection

SFT provides a structured fact in full, while evaluation requests a component of that fact through an operation absent from SFT.

```text
SFT:
What is Gogr's date of birth?
→ 2000-06-01

Evaluation:
In what year was Gogr born?
→ 2000
```

Projection is not a paraphrase test: both the query operation and required answer change. It tests whether the model can manipulate and selectively retrieve information from an acquired structured fact.

### 5.3 Composition

SFT teaches multiple atomic facts independently, while evaluation requests them jointly.

```text
Known facts:
birth_city(Gogr) = Tor-Velk
occupation(Gogr) = architect

Evaluation:
Where was Gogr born, and what is Gogr's occupation?
→ Tor-Velk; architect
```

Composition tests whether independently retrievable facts can be combined under a new query operation. At least one other multi-field answer format must be represented during SFT so that failure cannot be attributed solely to unfamiliar output formatting.

## 6. Research Questions and Hypotheses

### Primary research question

> After SFT on novel synthetic facts, how strongly does factual retrieval depend on the linguistic template families used during training?

### Primary behavioral hypotheses

**H1 — Surface-form dependence.** Accuracy will be higher on seen template families than on held-out families that request the same facts through the same operations.

**H2 — Diversity improves invariance.** Exposure to multiple genuinely distinct linguistic families for the same operation should reduce the gap between seen-family and held-out-family retrieval, provided that the underlying facts and training budget are controlled.

### Secondary hypotheses

**H3 — Paraphrase transfer is weaker than recall but easier than manipulation.** Seen-form recall should emerge before held-out paraphrase transfer, while projection and composition may require stronger generalization than paraphrase invariance.

**H4 — Generalization is not unitary.** A model may generalize across paraphrases while failing at projection or composition. Success on one evaluation mode must not be reported as success on the others.

These hypotheses describe expected behavioral patterns. They do not assume a particular mechanism inside the model.

## 7. Data Generation Requirements

Every generated example must retain enough provenance to identify all relevant sources of variation. A record should include at least:

```json
{
  "world_id": "world_001",
  "person_id": "person_0042",
  "fact_ids": ["person_0042.birth_city"],
  "operation_id": "GET_BIRTH_CITY",
  "template_family_id": "nominal_attribute",
  "template_id": "birth_city_nominal_001",
  "rendered_question": "What is Gogr's city of birth?",
  "canonical_answer": "Tor-Velk",
  "answer_format_id": "canonical_value",
  "split": "eval_paraphrase"
}
```

The generator must preserve a clean distinction between:

- entity identity;
- fact value;
- relation or attribute type;
- query operation;
- template family;
- individual template;
- answer format;
- world seed and split assignment.

Rendered strings alone are insufficient for reliable splitting or analysis.

## 8. Split Design Principles

### Split by generator provenance

Train and evaluation data must be separated by template family or operation, not by randomly splitting rendered rows. Two strings produced by nearly identical constructions are not meaningfully independent.

Template-family identifiers are global across operations. If a family is held
out, structurally parallel templates from that family must be absent from SFT
for every operation. Individual template identifiers remain operation-specific.

The generated dataset must distinguish three retrieval slices:

```text
eval_exact_recall
eval_seen_family_new_template
eval_heldout_family
```

- `eval_exact_recall` reuses the SFT `template_id`;
- `eval_seen_family_new_template` uses a different `template_id` from the same
  `template_family_id`;
- `eval_heldout_family` uses a family absent from SFT for every operation.

Assignments of sibling templates to SFT and seen-family evaluation must be
counterbalanced across worlds or runs. Always training on `_001` and evaluating
on `_002` would confound family transfer with one fixed paraphrase direction.

Before training, a split validator must assert at least:

```text
held_out_family_ids ∩ sft_family_ids = ∅
seen_family_eval_template_ids ∩ sft_template_ids = ∅
exact_recall_template_ids ⊆ sft_template_ids
```

### Hold one axis fixed while changing another

Each evaluation slice must isolate a single intended transformation:

- paraphrase: change the template family only;
- projection: change the operation and answer projection;
- composition: change from atomic to joint retrieval.

### Evaluate the same facts

The central paraphrase evaluation uses people and facts already presented during SFT. Holding the fact fixed ensures that the test concerns retrieval form rather than learning about unseen entities.

### Use canonical short answers

The core dataset maps natural-language questions to short, canonical answers:

```text
In which city was Gogr born?
→ Tor-Velk
```

The answer format remains constant across seen and held-out question forms. Full-sentence answers would introduce an additional source of variation and belong in a separate study.

### Prevent answer shortcuts

Attribute values should be sufficiently varied and reasonably balanced. The model should not obtain high accuracy by predicting a dominant city, year, month, or occupation.

### Keep semantic distinctions explicit

Relations that are linguistically similar but semantically different must remain separate. For example:

- birthplace is not necessarily hometown;
- birthplace is not current residence;
- “where someone is from” is too ambiguous for a controlled `GET_BIRTH_CITY` query.

### Keep evaluation facts out of the prompt

At evaluation time, the queried fact must not be supplied in context. The task is to measure retrieval from parameters after SFT, not in-context reading comprehension.

## 9. Core Measurements

The primary measurements are:

- accuracy on seen template families;
- accuracy on held-out template families;
- the seen-to-held-out transfer gap;
- accuracy on held-out projection operations;
- accuracy on held-out composition operations.

Results must be reportable by:

- query operation;
- template family;
- entity;
- attribute value;
- world instance.

Aggregate accuracy alone is not sufficient: a result that depends on one easy template family or one generated world is not evidence of robust generalization.

## 10. Intended Project Artifacts

The project will ultimately produce:

1. a versioned micro-world schema;
2. deterministic world generation from recorded seeds;
3. a curated registry of query operations and template families;
4. provenance-rich train and evaluation datasets;
5. validation checks for leakage, semantic equivalence, balance, and split integrity;
6. evaluation slices for seen-form recall, paraphrase invariance, projection, and composition;
7. reproducible evidence supporting or rejecting the stated behavioral hypotheses.

## 11. Current Scope Boundary

This phase defines **what is being measured** and **what data structure is required to measure it**.

The following decisions are intentionally outside the scope of this document:

- base model and model scale;
- SFT implementation and training framework;
- dataset cardinalities and token budgets;
- optimization settings and checkpoint schedule;
- experiment matrix and ablations;
- hardware utilization;
- logging and experiment-tracking stack;
- statistical power and replication policy;
- final reporting format.

Those decisions should be made only after the world schema, template taxonomy, split semantics, and evaluation contracts are stable.

## 12. Working Standard

The project should prefer a smaller, auditable dataset over a larger but ambiguous one. Each reported comparison must make clear:

1. what information the model received during SFT;
2. which fact is being queried;
3. which query operation is required;
4. which linguistic family changed;
5. which other variables remained fixed;
6. what behavioral conclusion the result does and does not support.

The governing principle is:

> A valid evaluation changes one interpretable property of factual retrieval at a time.
