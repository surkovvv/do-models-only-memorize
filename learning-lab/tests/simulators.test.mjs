import test from "node:test";
import assert from "node:assert/strict";

import { simulateDataset } from "../js/simulators/dataset.js";
import { IGNORE_INDEX, simulateCollator } from "../js/simulators/collator.js";
import { simulateOptimizer } from "../js/simulators/optimizer.js";
import { simulateGeneration } from "../js/simulators/generation.js";
import {
  DEFAULT_EVALUATION_ROWS,
  summarizeEvaluation,
} from "../js/simulators/evaluation.js";

test("dataset baseline preserves paired H1 contract", () => {
  const result = simulateDataset();

  assert.equal(result.train.length, 36);
  assert.equal(result.test.length, 24);
  assert.deepEqual(
    Object.values(result.slices).map((rows) => rows.length),
    [8, 8, 8],
  );
  assert.equal(result.uniqueTrainFacts, 12);
  assert.equal(result.uniqueEvaluationFacts, 8);
  assert.equal(result.invariantHolds, true);
});

test("dataset simulator makes silent scientific failures observable", () => {
  const familyLeak = simulateDataset({ familyOverlap: true });
  assert.match(familyLeak.issues[0], /пересекаются/);
  assert.equal(familyLeak.warnings, familyLeak.issues);
  assert.match(simulateDataset({ seenTemplateLeak: true }).issues[0], /template ID/);
  assert.match(simulateDataset({ missingPair: true }).issues[0], /разные fact_id/);
  assert.match(simulateDataset({ answerDrift: true }).issues[0], /canonical_answer/);
});

test("collator masks every prompt and padding position", () => {
  const result = simulateCollator();

  assert.equal(result.invariantHolds, true);
  assert.equal(result.targetTokens, result.answerTokens);
  for (const cell of result.rows.flatMap((row) => row.cells)) {
    if (cell.role === "prompt" || cell.role === "padding") {
      assert.equal(cell.label, IGNORE_INDEX);
    } else {
      assert.notEqual(cell.label, IGNORE_INDEX);
    }
  }
});

test("collator exposes prompt and padding leakage separately", () => {
  const promptLeak = simulateCollator({ maskPrompt: false });
  const paddingLeak = simulateCollator({ maskPadding: false });

  assert.ok(promptLeak.leakedPromptTokens > 0);
  assert.equal(promptLeak.leakedPaddingTokens, 0);
  assert.ok(paddingLeak.leakedPaddingTokens > 0);
  assert.equal(paddingLeak.leakedPromptTokens, 0);
});

test("optimizer uses one update and scheduler tick per accumulation window", () => {
  const result = simulateOptimizer({ accumulationSteps: 2 });

  assert.equal(result.optimizerSteps, 3);
  assert.equal(result.schedulerTicks, 3);
  assert.deepEqual(
    result.updates.map((update) => update.windowSize),
    [2, 2, 1],
  );
  assert.equal(result.invariantHolds, true);
});

test("short optimizer tail is divided by its actual window size", () => {
  const result = simulateOptimizer({
    gradients: [1, 1, 1],
    accumulationSteps: 2,
    maxGradNorm: null,
  });

  assert.deepEqual(
    result.updates.map((update) => update.gradientBeforeClip),
    [1, 1],
  );
});

test("optimizer simulator distinguishes common ordering bugs", () => {
  const microZero = simulateOptimizer({ zeroMode: "micro" });
  const fastScheduler = simulateOptimizer({ schedulerMode: "micro" });
  const undivided = simulateOptimizer({ divideLoss: false, maxGradNorm: null });

  assert.equal(microZero.invariantHolds, false);
  assert.equal(fastScheduler.schedulerTicks, fastScheduler.gradients.length);
  assert.ok(
    Math.abs(undivided.updates[0].gradientBeforeClip) >
      Math.abs(simulateOptimizer({ maxGradNorm: null }).updates[0].gradientBeforeClip),
  );
});

test("generation baseline left-pads and returns only continuations", () => {
  const result = simulateGeneration();

  assert.equal(result.exactMatches, 2);
  assert.equal(result.rows.every((row) => !row.continuesFromPad), true);
  assert.equal(result.invariantHolds, true);
});

test("generation makes right-padding and missing prompt slice visible", () => {
  const rightPadded = simulateGeneration({ paddingSide: "right" });
  const untrimmed = simulateGeneration({ trimPrompt: false });

  assert.equal(rightPadded.exactMatches, 1);
  assert.equal(rightPadded.rows[0].continuesFromPad, true);
  assert.equal(untrimmed.exactMatches, 0);
});

test("paired evaluation computes accuracies, gaps, and transitions", () => {
  const result = summarizeEvaluation(DEFAULT_EVALUATION_ROWS);

  assert.equal(result.accuracies.exact, 5 / 6);
  assert.equal(result.accuracies.seen, 3 / 6);
  assert.equal(result.accuracies.heldout, 3 / 6);
  assert.ok(Math.abs(result.comparisons.exact_minus_seen.gap - 2 / 6) < 1e-12);
  assert.equal(
    Object.values(result.comparisons.seen_minus_heldout.counts).reduce(
      (total, value) => total + value,
      0,
    ),
    DEFAULT_EVALUATION_ROWS.length,
  );
});

test("evaluation flags leakage and unpaired slices independently", () => {
  const result = summarizeEvaluation(DEFAULT_EVALUATION_ROWS, {
    heldoutLeak: true,
    pairedFacts: false,
  });

  assert.equal(result.invariantHolds, false);
  assert.equal(result.warnings.length >= 2, true);
});
