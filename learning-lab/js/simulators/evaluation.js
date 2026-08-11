export const DEFAULT_EVALUATION_ROWS = [
  { factId: "person_0001.birth_city", exact: true, seen: true, heldout: true },
  { factId: "person_0002.birth_city", exact: true, seen: true, heldout: false },
  { factId: "person_0003.birth_date", exact: true, seen: false, heldout: false },
  { factId: "person_0004.occupation", exact: true, seen: true, heldout: true },
  { factId: "person_0005.residence_city", exact: false, seen: false, heldout: false },
  { factId: "person_0006.occupation", exact: true, seen: false, heldout: true },
];

const COMPARISONS = [
  ["exact_minus_seen", "exact", "seen"],
  ["seen_minus_heldout", "seen", "heldout"],
  ["exact_minus_heldout", "exact", "heldout"],
];

function mean(values) {
  return values.reduce((total, value) => total + Number(value), 0) / values.length;
}

function transition(left, right) {
  if (left && right) return "both_correct";
  if (left) return "left_only";
  if (right) return "right_only";
  return "neither_correct";
}

export function summarizeEvaluation(
  rows = DEFAULT_EVALUATION_ROWS,
  { heldoutLeak = false, pairedFacts = true } = {},
) {
  if (!Array.isArray(rows) || rows.length === 0) {
    throw new Error("evaluation rows must be non-empty");
  }
  const factIds = rows.map((row) => row.factId);
  const duplicateFacts = factIds.filter((factId, index) => factIds.indexOf(factId) !== index);
  const accuracies = {
    exact: mean(rows.map((row) => row.exact)),
    seen: mean(rows.map((row) => row.seen)),
    heldout: mean(rows.map((row) => row.heldout)),
  };
  const comparisons = Object.fromEntries(
    COMPARISONS.map(([name, leftKey, rightKey]) => {
      const counts = {
        both_correct: 0,
        left_only: 0,
        right_only: 0,
        neither_correct: 0,
      };
      const pairedRows = rows.map((row) => {
        const rowTransition = transition(row[leftKey], row[rightKey]);
        counts[rowTransition] += 1;
        return {
          factId: row.factId,
          left: row[leftKey],
          right: row[rightKey],
          transition: rowTransition,
          delta: Number(row[leftKey]) - Number(row[rightKey]),
        };
      });
      return [
        name,
        {
          leftKey,
          rightKey,
          gap: accuracies[leftKey] - accuracies[rightKey],
          counts,
          rows: pairedRows,
        },
      ];
    }),
  );

  const warnings = [];
  if (heldoutLeak) {
    warnings.push(
      "Held-out template ID присутствует в SFT: сравнение больше не изолирует перенос на новую family.",
    );
  }
  if (!pairedFacts) {
    warnings.push(
      "Slices содержат разные facts: разность средних смешивает surface-form effect и сложность фактов.",
    );
  }
  if (duplicateFacts.length) {
    warnings.push("Один fact встречается в paired table больше одного раза.");
  }
  if (accuracies.exact < 0.8) {
    warnings.push(
      "Exact recall низок: сначала нужно показать, что SFT надёжно выучил факты; held-out gap пока неоднозначен.",
    );
  }

  return {
    rows: rows.map((row) => ({ ...row })),
    accuracies,
    comparisons,
    heldoutLeak,
    pairedFacts,
    warnings,
    invariantHolds: !heldoutLeak && pairedFacts && duplicateFacts.length === 0,
  };
}

