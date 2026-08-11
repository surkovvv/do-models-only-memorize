const OPERATIONS = ["birth_city", "birth_date", "residence_city", "occupation"];
const PEOPLE = ["person_0001", "person_0002", "person_0003"];
const TRAIN_FAMILIES = ["direct_question", "imperative", "profile_field"];

function valueFor(personId, operation) {
  return `${personId}:${operation}:value`;
}

function makeRow(personId, operation, family, templateId, split) {
  const factId = `${personId}.${operation}`;
  const factValue = valueFor(personId, operation);
  return {
    exampleId: `${split}.${factId}.${templateId}`,
    personId,
    factId,
    relationId: operation,
    factValue,
    canonicalAnswer: factValue,
    templateFamilyId: family,
    templateId,
    split,
  };
}

export function simulateDataset({
  familyOverlap = false,
  seenTemplateLeak = false,
  missingPair = false,
  answerDrift = false,
} = {}) {
  const trainFamilies = familyOverlap
    ? [...TRAIN_FAMILIES, "nominal_attribute"]
    : [...TRAIN_FAMILIES];
  const train = PEOPLE.flatMap((personId) =>
    OPERATIONS.flatMap((operation) =>
      TRAIN_FAMILIES.map((family) =>
        makeRow(personId, operation, family, `${operation}.${family}.001`, "train"),
      ),
    ),
  );
  const evaluationFacts = PEOPLE.slice(0, 2).flatMap((personId) =>
    OPERATIONS.map((operation) => ({ personId, operation })),
  );
  const exact = evaluationFacts.map(({ personId, operation }) =>
    makeRow(
      personId,
      operation,
      "direct_question",
      `${operation}.direct_question.001`,
      "eval_exact_recall",
    ),
  );
  const seen = evaluationFacts.map(({ personId, operation }) =>
    makeRow(
      personId,
      operation,
      "direct_question",
      `${operation}.direct_question.002`,
      "eval_seen_family_new_template",
    ),
  );
  const heldout = evaluationFacts.map(({ personId, operation }) =>
    makeRow(
      personId,
      operation,
      "nominal_attribute",
      `${operation}.nominal_attribute.001`,
      "eval_heldout_family",
    ),
  );

  if (seenTemplateLeak) {
    train.push({ ...seen[0], split: "train", exampleId: `train.leak.${seen[0].factId}` });
  }
  if (missingPair) {
    heldout.pop();
  }
  if (answerDrift) {
    seen[0] = { ...seen[0], canonicalAnswer: "different-value" };
  }

  const test = [...exact, ...seen, ...heldout];
  const factSets = {
    exact: new Set(exact.map((row) => row.factId)),
    seen: new Set(seen.map((row) => row.factId)),
    heldout: new Set(heldout.map((row) => row.factId)),
  };
  const sameFacts =
    factSets.exact.size === factSets.seen.size &&
    factSets.seen.size === factSets.heldout.size &&
    [...factSets.exact].every(
      (factId) => factSets.seen.has(factId) && factSets.heldout.has(factId),
    );
  const trainTemplateIds = new Set(train.map((row) => row.templateId));
  const seenIds = new Set(seen.map((row) => row.templateId));
  const globalSeenTemplateIsolation = [...seenIds].every(
    (templateId) => !trainTemplateIds.has(templateId),
  );
  const stableAnswers = test.every(
    (row) => row.canonicalAnswer === row.factValue,
  );
  const issues = [];
  if (familyOverlap) {
    issues.push("train/test family sets пересекаются: nominal_attribute больше не held-out.");
  }
  if (!globalSeenTemplateIsolation) {
    issues.push("seen-family eval template ID обнаружен в SFT глобально.");
  }
  if (!sameFacts) {
    issues.push("три evaluation slices покрывают разные fact_id; paired gap не определён.");
  }
  if (!stableAnswers) {
    issues.push("canonical_answer расходится с fact_value для одного rendering.");
  }

  return {
    train,
    test,
    slices: { exact, seen, heldout },
    trainFamilies,
    testFamilies: ["nominal_attribute"],
    uniqueTrainFacts: new Set(train.map((row) => row.factId)).size,
    uniqueEvaluationFacts: new Set(test.map((row) => row.factId)).size,
    sameFacts,
    globalSeenTemplateIsolation,
    stableAnswers,
    issues,
    warnings: issues,
    invariantHolds: issues.length === 0,
  };
}
