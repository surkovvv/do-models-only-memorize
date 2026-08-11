import test from "node:test";
import assert from "node:assert/strict";

import { LESSONS, STAGES } from "../js/lessons.js";
import {
  canOpenStage,
  completedLessonCount,
  createDefaultState,
  normalizeState,
  notebookToMarkdown,
} from "../js/state.js";
import { exportProgress, importProgress } from "../js/storage.js";

test("new lesson gates Run until prediction is locked", () => {
  const lessonState = createDefaultState().lessons[LESSONS[0].id];

  assert.equal(canOpenStage(lessonState, "predict"), true);
  assert.equal(canOpenStage(lessonState, "run"), false);
  lessonState.predictionLocked = true;
  assert.equal(canOpenStage(lessonState, "run"), true);
  assert.equal(canOpenStage(lessonState, "investigate"), false);
  lessonState.experimentRuns = 1;
  assert.equal(canOpenStage(lessonState, "investigate"), true);
});

test("normalization repairs corrupted and unknown state fields", () => {
  const normalized = normalizeState({
    activeLessonId: "missing",
    lessons: {
      dataset: {
        stage: "teleport",
        completedStages: ["predict", "unknown", "predict"],
        confidenceBefore: 99,
      },
    },
  });

  assert.equal(normalized.activeLessonId, LESSONS[0].id);
  assert.equal(normalized.lessons.dataset.stage, "predict");
  assert.deepEqual(normalized.lessons.dataset.completedStages, ["predict"]);
  assert.equal(normalized.lessons.dataset.confidenceBefore, 5);
});

test("progress export and import round-trip through the schema", () => {
  const state = createDefaultState();
  state.lessons.dataset.notebook.hypothesis = "If leakage, the validator must fail.";
  state.lessons.dataset.completedStages = [...STAGES];

  const restored = importProgress(exportProgress(state));

  assert.equal(restored.lessons.dataset.notebook.hypothesis, state.lessons.dataset.notebook.hypothesis);
  assert.equal(completedLessonCount(restored), 1);
});

test("markdown export preserves hypotheses and make drafts", () => {
  const state = createDefaultState();
  state.lessons.collator.notebook.hypothesis = "Prompt tokens must be -100.";
  state.lessons.collator.makeDraft = "assert labels[prompt_mask].eq(-100).all()";

  const markdown = notebookToMarkdown(state);

  assert.match(markdown, /Prompt tokens must be -100/);
  assert.match(markdown, /assert labels\[prompt_mask\]/);
});

