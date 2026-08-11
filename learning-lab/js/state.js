import { LESSONS, STAGES } from "./lessons.js";

export const SCHEMA_VERSION = 1;

function emptyNotebook() {
  return {
    question: "",
    mentalModel: "",
    hypothesis: "",
    alternative: "",
    intervention: "",
    expected: "",
    observed: "",
    revisedModel: "",
    regressionTest: "",
  };
}

export function emptyLessonState() {
  return {
    stage: "predict",
    completedStages: [],
    selectedPrediction: null,
    predictionLocked: false,
    experimentRuns: 0,
    experimentHistory: [],
    notebook: emptyNotebook(),
    makeDraft: "",
    rubricChecks: [],
    confidenceBefore: 2,
    confidenceAfter: 2,
  };
}

export function createDefaultState() {
  return {
    schemaVersion: SCHEMA_VERSION,
    activeLessonId: LESSONS[0].id,
    lessons: Object.fromEntries(LESSONS.map((lesson) => [lesson.id, emptyLessonState()])),
  };
}

function uniqueKnownStages(stages) {
  return [...new Set(Array.isArray(stages) ? stages : [])].filter((stage) =>
    STAGES.includes(stage),
  );
}

export function normalizeState(candidate) {
  const fallback = createDefaultState();
  if (!candidate || typeof candidate !== "object") return fallback;

  const state = {
    ...fallback,
    activeLessonId: LESSONS.some((lesson) => lesson.id === candidate.activeLessonId)
      ? candidate.activeLessonId
      : fallback.activeLessonId,
  };
  for (const lesson of LESSONS) {
    const raw = candidate.lessons?.[lesson.id];
    if (!raw || typeof raw !== "object") continue;
    const base = emptyLessonState();
    state.lessons[lesson.id] = {
      ...base,
      stage: STAGES.includes(raw.stage) ? raw.stage : base.stage,
      completedStages: uniqueKnownStages(raw.completedStages),
      selectedPrediction: Number.isInteger(raw.selectedPrediction)
        ? raw.selectedPrediction
        : null,
      predictionLocked: Boolean(raw.predictionLocked),
      experimentRuns: Number.isInteger(raw.experimentRuns)
        ? Math.max(0, raw.experimentRuns)
        : 0,
      experimentHistory: Array.isArray(raw.experimentHistory)
        ? raw.experimentHistory.slice(-20)
        : [],
      notebook: Object.fromEntries(
        Object.keys(base.notebook).map((key) => [
          key,
          typeof raw.notebook?.[key] === "string" ? raw.notebook[key] : "",
        ]),
      ),
      makeDraft: typeof raw.makeDraft === "string" ? raw.makeDraft : "",
      rubricChecks: Array.isArray(raw.rubricChecks)
        ? raw.rubricChecks.filter(Number.isInteger)
        : [],
      confidenceBefore: Number.isInteger(raw.confidenceBefore)
        ? Math.min(5, Math.max(1, raw.confidenceBefore))
        : 2,
      confidenceAfter: Number.isInteger(raw.confidenceAfter)
        ? Math.min(5, Math.max(1, raw.confidenceAfter))
        : 2,
    };
  }
  return state;
}

export function canOpenStage(lessonState, stage) {
  if (stage === "predict") return true;
  if (stage === "run") return lessonState.predictionLocked;
  if (stage === "investigate") return lessonState.experimentRuns > 0;
  if (stage === "modify") return lessonState.experimentRuns > 0;
  if (stage === "make") return lessonState.completedStages.includes("investigate");
  return false;
}

export function completedLessonCount(state) {
  return LESSONS.filter((lesson) =>
    STAGES.every((stage) => state.lessons[lesson.id].completedStages.includes(stage)),
  ).length;
}

export function lessonProgress(lessonState) {
  return STAGES.filter((stage) => lessonState.completedStages.includes(stage)).length;
}

function escapeMarkdown(value) {
  return value.trim() || "_не заполнено_";
}

export function notebookToMarkdown(state) {
  const sections = LESSONS.map((lesson) => {
    const lessonState = state.lessons[lesson.id];
    const notebook = lessonState.notebook;
    return [
      `## ${lesson.number}. ${lesson.title}`,
      "",
      `- Уверенность до: ${lessonState.confidenceBefore}/5`,
      `- Уверенность после: ${lessonState.confidenceAfter}/5`,
      `- Экспериментов: ${lessonState.experimentRuns}`,
      "",
      `### Вопрос\n\n${escapeMarkdown(notebook.question)}`,
      `### Моя модель\n\n${escapeMarkdown(notebook.mentalModel)}`,
      `### Гипотеза\n\n${escapeMarkdown(notebook.hypothesis)}`,
      `### Альтернатива\n\n${escapeMarkdown(notebook.alternative)}`,
      `### Вмешательство\n\n${escapeMarkdown(notebook.intervention)}`,
      `### Ожидаемое наблюдение\n\n${escapeMarkdown(notebook.expected)}`,
      `### Фактическое наблюдение\n\n${escapeMarkdown(notebook.observed)}`,
      `### Обновлённая модель\n\n${escapeMarkdown(notebook.revisedModel)}`,
      `### Regression test\n\n${escapeMarkdown(notebook.regressionTest)}`,
      `### Make-задание\n\n${escapeMarkdown(lessonState.makeDraft)}`,
    ].join("\n");
  });

  return [
    "# Learning Lab — исследовательский журнал",
    "",
    `Экспортировано: ${new Date().toISOString()}`,
    "",
    ...sections,
  ].join("\n\n");
}

