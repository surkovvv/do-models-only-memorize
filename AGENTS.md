# Project instructions

## Success criteria

This repository has two equally important outputs:

1. a scientifically valid, reproducible research artifact;
2. the user's ability to explain, test, debug, and extend that artifact.

When these goals compete, preserve scientific validity first, then optimize for
learning rather than delivery speed.

## Default collaboration mode: learning-first

Use learning-first mode for non-trivial research and implementation work unless
the user explicitly asks for `ship mode`, `debug-only mode`, or an immediate
finished implementation.

For each new vertical slice:

1. Name the call path and the smallest behavior under investigation.
2. Identify the concrete libraries and methods involved.
3. State the input/output contract, tensor shapes or data schema, mutated state,
   and explicit guards.
4. Give the user a short prediction checkpoint before revealing the observed
   result when the user is actively participating. Do not manufacture a blocker
   for a request that clearly asks for immediate execution.
5. Change one causal variable at a time and name the expected observable signal.
6. Explain important decisions as:

   ```text
   contract → why it exists → what changes without it → failure signal →
   distinguishing test
   ```

7. Let the user design or write at least one hypothesis, assertion, test, or
   analysis query when practical.
8. After implementation, offer a short defense question and one deliberate
   breakage to diagnose.

Do not equate recognition of code with understanding. A concept is considered
learned when the user can predict behavior, explain two failure modes, locate the
signal, and construct a test that separates the correct and incorrect behavior.

## Research loop

For investigation and debugging, keep the loop falsifiable:

```text
question
→ current mental model
→ primary hypothesis + alternative
→ smallest controlled intervention
→ expected observation
→ actual observation
→ updated model
→ regression test or durable artifact
```

Prefer hypotheses about observable state over guesses about fixes. Use a tiny
fixture, toy model, mock, or one paired fact before running a full model or full
dataset.

## Scientific failures are first-class failures

Treat the following as seriously as exceptions or failing tests:

- train/evaluation leakage;
- broken fact pairing;
- changing more than the intended experimental axis;
- target or loss-mask drift;
- selecting hyperparameters on a held-out scientific slice;
- missing provenance, raw predictions, resolved configuration, or hashes;
- aggregate metrics that hide per-fact transitions;
- shared mutable state that makes results depend on call order.

When a scientific failure is silent, add an explicit validator or regression
test whenever possible.

## Code-change hygiene

- Work in small vertical slices and preserve unrelated user changes.
- Before adding an abstraction or dependency, show which concrete problem it
  solves in the current experiment.
- Prefer executable assertions and focused tests over explanatory comments that
  can drift.
- Report both the happy-path contract and the exact place/signal of failure.
- Distinguish a deterministic teaching model from the behavior of the real
  PyTorch/Transformers runtime.
- `microworld-lab/` is a separate nested repository; do not add or modify it as
  part of this repository's work unless the user explicitly scopes it in.

## Durable project memory

- Scientific objective and operational definitions: `docs/research_charter.md`.
- Current verified state and experiment priorities: `docs/next_steps.md`.
- Environment and real run commands: `docs/environment.md`.
- Interactive PRIMM exercises and the user's research journal:
  `learning-lab/README.md` and `learning-lab/`.

Update these documents when a decision would otherwise be lost between
sessions. Do not turn `AGENTS.md` into a chronological changelog.

