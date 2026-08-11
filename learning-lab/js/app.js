import { LESSONS, STAGES, lessonById } from "./lessons.js";
import {
  canOpenStage,
  completedLessonCount,
  createDefaultState,
  lessonProgress,
  notebookToMarkdown,
} from "./state.js";
import { clearProgress, loadProgress, saveProgress } from "./storage.js";
import { simulateDataset } from "./simulators/dataset.js";
import { simulateCollator } from "./simulators/collator.js";
import { simulateOptimizer } from "./simulators/optimizer.js";
import { simulateGeneration } from "./simulators/generation.js";
import {
  DEFAULT_EVALUATION_ROWS,
  summarizeEvaluation,
} from "./simulators/evaluation.js";

const STAGE_LABELS = {
  predict: "Предскажи",
  run: "Запусти",
  investigate: "Исследуй",
  modify: "Сломай",
  make: "Сделай",
};

const NOTEBOOK_FIELDS = [
  ["question", "Вопрос"],
  ["mentalModel", "Моя текущая модель"],
  ["hypothesis", "Гипотеза"],
  ["alternative", "Альтернативная гипотеза"],
  ["intervention", "Минимальное вмешательство"],
  ["expected", "Ожидаемое наблюдение"],
  ["observed", "Фактическое наблюдение"],
  ["revisedModel", "Обновлённая модель"],
  ["regressionTest", "Regression test"],
];

const app = document.getElementById("lab-app");
let state = loadProgress();
const experimentOptions = {
  dataset: {
    familyOverlap: false,
    seenTemplateLeak: false,
    missingPair: false,
    answerDrift: false,
  },
  collator: {
    maskPrompt: true,
    maskPadding: true,
    includeGenerationPrompt: true,
  },
  optimizer: {
    accumulationSteps: 2,
    divideLoss: true,
    zeroMode: "window",
    schedulerMode: "optimizer",
    maxGradNorm: 1,
  },
  generation: {
    paddingSide: "left",
    trimPrompt: true,
    evalMode: true,
    inferenceMode: true,
  },
  evaluation: {
    heldoutLeak: false,
    pairedFacts: true,
    rows: DEFAULT_EVALUATION_ROWS.map((row) => ({ ...row })),
  },
};

function escapeHtml(value) {
  return String(value)
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#039;");
}

function checked(value) {
  return value ? "checked" : "";
}

function selected(value, expected) {
  return value === expected ? "selected" : "";
}

function formatPercent(value) {
  return `${(value * 100).toFixed(1)}%`;
}

function formatSigned(value) {
  return `${value >= 0 ? "+" : ""}${value.toFixed(3)}`;
}

function currentLesson() {
  return lessonById(state.activeLessonId);
}

function currentLessonState() {
  return state.lessons[state.activeLessonId];
}

function persist() {
  const saved = saveProgress(state);
  const status = document.getElementById("save-status");
  if (status) {
    status.textContent = saved ? "Прогресс сохранён локально" : "Не удалось сохранить прогресс";
  }
}

function completeStage(lessonState, stage) {
  if (!lessonState.completedStages.includes(stage)) {
    lessonState.completedStages.push(stage);
  }
}

function stageAfter(stage) {
  const index = STAGES.indexOf(stage);
  return STAGES[Math.min(STAGES.length - 1, index + 1)];
}

function setHash(lessonId) {
  const target = `#/lesson/${lessonId}`;
  if (window.location.hash !== target) window.location.hash = target;
}

function syncFromHash() {
  const match = window.location.hash.match(/^#\/lesson\/([a-z-]+)$/);
  if (match && LESSONS.some((lesson) => lesson.id === match[1])) {
    state.activeLessonId = match[1];
  } else {
    setHash(state.activeLessonId);
  }
}

function renderShell() {
  const completed = completedLessonCount(state);
  const totalStages = LESSONS.length * STAGES.length;
  const completedStages = LESSONS.reduce(
    (total, lesson) => total + lessonProgress(state.lessons[lesson.id]),
    0,
  );
  const progress = Math.round((completedStages / totalStages) * 100);

  app.innerHTML = `
    <header class="topbar">
      <div class="brand-block">
        <span class="eyebrow">do-models-only-memorize</span>
        <a class="brand" href="#/lesson/${LESSONS[0].id}">Research Loop Lab</a>
      </div>
      <div class="topbar-progress" aria-label="Общий прогресс">
        <span>${completed}/${LESSONS.length} лабораторных</span>
        <progress max="100" value="${progress}">${progress}%</progress>
      </div>
      <div class="topbar-actions">
        <button class="button secondary" type="button" data-action="export-notebook">
          Экспорт журнала
        </button>
        <button class="button ghost danger-text" type="button" data-action="reset-progress">
          Сбросить
        </button>
      </div>
    </header>
    <div class="workspace">
      ${renderSidebar()}
      <main id="lesson-content" class="lesson-main" tabindex="-1">
        ${renderLesson()}
      </main>
    </div>
    <div id="toast" class="toast" role="status" aria-live="polite"></div>
  `;
}

function renderSidebar() {
  return `
    <aside class="sidebar" aria-label="Карта лабораторных">
      <div class="sidebar-intro">
        <p class="eyebrow">PRIMM-путь</p>
        <p>Сначала фиксируй прогноз. Результат — только после проверяемой гипотезы.</p>
      </div>
      <nav>
        <ol class="lesson-list">
          ${LESSONS.map((lesson) => {
            const lessonState = state.lessons[lesson.id];
            const count = lessonProgress(lessonState);
            const active = lesson.id === state.activeLessonId;
            return `
              <li>
                <a
                  class="lesson-link ${active ? "active" : ""}"
                  href="#/lesson/${lesson.id}"
                  ${active ? 'aria-current="page"' : ""}
                >
                  <span class="lesson-number">${lesson.number}</span>
                  <span class="lesson-link-copy">
                    <strong>${escapeHtml(lesson.shortTitle)}</strong>
                    <span>${count}/${STAGES.length} этапов</span>
                  </span>
                  <span class="lesson-status" aria-label="${count === STAGES.length ? "завершено" : "в процессе"}">
                    ${count === STAGES.length ? "✓" : `${count}`}
                  </span>
                </a>
              </li>
            `;
          }).join("")}
        </ol>
      </nav>
      <div class="sidebar-note">
        <span class="status-dot"></span>
        <span id="save-status">Прогресс сохранён локально</span>
      </div>
    </aside>
  `;
}

function renderLesson() {
  const lesson = currentLesson();
  const lessonState = currentLessonState();
  const correct = lessonState.predictionLocked
    ? lessonState.selectedPrediction === lesson.prediction.correct
    : null;

  return `
    <section class="lesson-header" aria-labelledby="lesson-title">
      <div class="lesson-kicker">
        <span>Лабораторная ${lesson.number}</span>
        <span class="divider">/</span>
        <a href="${lesson.sourceHref}" target="_blank" rel="noreferrer">
          ${escapeHtml(lesson.source)} → ${escapeHtml(lesson.symbol)}
        </a>
      </div>
      <h1 id="lesson-title">${escapeHtml(lesson.title)}</h1>
      <p class="lesson-subtitle">${escapeHtml(lesson.subtitle)}</p>
      <div class="contract-strip">
        <div>
          <span class="metadata-label">Контракт</span>
          <p>${escapeHtml(lesson.contract)}</p>
        </div>
        <div>
          <span class="metadata-label">Стек</span>
          <p>${escapeHtml(lesson.library)}</p>
        </div>
      </div>
    </section>

    <nav class="stage-nav" aria-label="Этапы лабораторной">
      ${STAGES.map((stage, index) => {
        const open = canOpenStage(lessonState, stage);
        const active = lessonState.stage === stage;
        const done = lessonState.completedStages.includes(stage);
        return `
          <button
            class="stage-button ${active ? "active" : ""} ${done ? "done" : ""}"
            type="button"
            data-action="open-stage"
            data-stage="${stage}"
            ${open ? "" : "disabled"}
            aria-current="${active ? "step" : "false"}"
          >
            <span class="stage-index">${done ? "✓" : index + 1}</span>
            <span>${STAGE_LABELS[stage]}</span>
          </button>
        `;
      }).join("")}
    </nav>

    <section class="stage-panel" aria-labelledby="stage-heading">
      ${renderStage(lesson, lessonState, correct)}
    </section>

    ${renderNotebook(lessonState)}
  `;
}

function renderStage(lesson, lessonState, predictionCorrect) {
  if (lessonState.stage === "predict") {
    return renderPredictStage(lesson, lessonState, predictionCorrect);
  }
  if (lessonState.stage === "run") return renderRunStage(lesson, lessonState);
  if (lessonState.stage === "investigate") return renderInvestigateStage(lesson, lessonState);
  if (lessonState.stage === "modify") return renderModifyStage(lesson, lessonState);
  return renderMakeStage(lesson, lessonState);
}

function renderPredictStage(lesson, lessonState, predictionCorrect) {
  return `
    <div class="stage-heading-row">
      <div>
        <span class="eyebrow">Predict</span>
        <h2 id="stage-heading">Зафиксируй прогноз до запуска</h2>
      </div>
      ${confidenceControl("confidenceBefore", lessonState.confidenceBefore, "Уверенность до")}
    </div>
    <fieldset class="prediction-fieldset" ${lessonState.predictionLocked ? "disabled" : ""}>
      <legend>${escapeHtml(lesson.prediction.question)}</legend>
      <div class="prediction-options">
        ${lesson.prediction.options.map((option, index) => `
          <label class="prediction-option ${lessonState.selectedPrediction === index ? "selected" : ""}">
            <input
              type="radio"
              name="prediction"
              value="${index}"
              data-action="select-prediction"
              ${lessonState.selectedPrediction === index ? "checked" : ""}
            />
            <span class="option-letter">${String.fromCharCode(65 + index)}</span>
            <span>${escapeHtml(option)}</span>
          </label>
        `).join("")}
      </div>
    </fieldset>
    ${lessonState.predictionLocked ? `
      <div class="feedback ${predictionCorrect ? "success" : "warning"}" role="status">
        <strong>${predictionCorrect ? "Прогноз подтвердился." : "Прогноз не подтвердился — это полезные данные."}</strong>
        <span>${escapeHtml(lesson.prediction.explanation)}</span>
      </div>
      <div class="stage-actions">
        <button class="button primary" type="button" data-action="advance-stage">Перейти к запуску</button>
      </div>
    ` : `
      <div class="stage-actions">
        <button
          class="button primary"
          type="button"
          data-action="lock-prediction"
          ${lessonState.selectedPrediction == null ? "disabled" : ""}
        >
          Зафиксировать прогноз
        </button>
        <span class="action-hint">После фиксации исходный ответ останется в журнале.</span>
      </div>
    `}
  `;
}

function renderRunStage(lesson, lessonState) {
  return `
    <div class="stage-heading-row">
      <div>
        <span class="eyebrow">Run</span>
        <h2 id="stage-heading">Запусти минимальный baseline</h2>
      </div>
      <span class="run-count">Запусков: ${lessonState.experimentRuns}</span>
    </div>
    <p class="stage-lead">
      Полигон моделирует контракт <code>${escapeHtml(lesson.symbol)}</code> без GPU. После наблюдения
      проверь тот же инвариант командой на production-коде.
    </p>
    <div class="api-row" aria-label="Используемые API">
      ${lesson.apis.map((api) => `<code>${escapeHtml(api)}</code>`).join("")}
    </div>
    <div class="baseline-card">
      <div>
        <span class="metadata-label">Настройка</span>
        <strong>Все инварианты включены</strong>
        <p>Один детерминированный пример, только наблюдаемые состояния.</p>
      </div>
      <button class="button primary" type="button" data-action="run-baseline">
        Запустить baseline
      </button>
    </div>
    <div class="command-block">
      <div>
        <span class="metadata-label">Проверка на настоящем коде</span>
        <code>${escapeHtml(lesson.command)}</code>
      </div>
      <button class="button secondary compact" type="button" data-action="copy-command">Копировать</button>
    </div>
  `;
}

function renderInvestigateStage(lesson, lessonState) {
  const result = runSimulation(lesson.id, baselineOptions(lesson.id));
  return `
    <div class="stage-heading-row">
      <div>
        <span class="eyebrow">Investigate</span>
        <h2 id="stage-heading">Сверь mental model с трассой</h2>
      </div>
      <span class="status-badge good">Инварианты соблюдены</span>
    </div>
    ${renderSimulation(lesson.id, result, false)}
    <div class="failure-list">
      <span class="metadata-label">На что смотреть при поломке</span>
      <ul>
        ${lesson.failureModes.map((mode) => `<li>${escapeHtml(mode)}</li>`).join("")}
      </ul>
    </div>
    <div class="stage-actions">
      <button class="button primary" type="button" data-action="complete-investigate">
        Я могу объяснить baseline
      </button>
      <span class="action-hint">Сначала заполни «Фактическое наблюдение» в журнале ниже.</span>
    </div>
  `;
}

function renderModifyStage(lesson) {
  const options = experimentOptions[lesson.id];
  const result = runSimulation(lesson.id, options);
  return `
    <div class="stage-heading-row">
      <div>
        <span class="eyebrow">Modify</span>
        <h2 id="stage-heading">Измени одну причину, предскажи один сигнал</h2>
      </div>
      <span class="status-badge ${result.invariantHolds ? "good" : "bad"}">
        ${result.invariantHolds ? "Контракт цел" : "Контракт нарушен"}
      </span>
    </div>
    <p class="stage-lead">Переключатель меняет только учебную модель. Production-файлы не редактируются.</p>
    ${renderControls(lesson.id, options)}
    ${renderSimulation(lesson.id, result, true)}
    ${renderWarnings(result.warnings)}
    <div class="stage-actions">
      <button class="button primary" type="button" data-action="record-experiment">
        Записать эксперимент и перейти к Make
      </button>
      <span class="action-hint">Перед нажатием запиши ожидаемый результат в журнал.</span>
    </div>
  `;
}

function renderMakeStage(lesson, lessonState) {
  const allChecked = lesson.rubric.every((_, index) => lessonState.rubricChecks.includes(index));
  const enoughText = lessonState.makeDraft.trim().length >= 40;
  return `
    <div class="stage-heading-row">
      <div>
        <span class="eyebrow">Make</span>
        <h2 id="stage-heading">Сделай различающий тест</h2>
      </div>
      ${confidenceControl("confidenceAfter", lessonState.confidenceAfter, "Уверенность после")}
    </div>
    <div class="make-task">
      <span class="metadata-label">Задание</span>
      <p>${escapeHtml(lesson.makeTask)}</p>
    </div>
    <label class="field-label" for="make-draft">Твой план, псевдокод или assertion</label>
    <textarea
      id="make-draft"
      class="text-area code-draft"
      rows="8"
      data-state-field="makeDraft"
      placeholder="Arrange → Act → Assert. Как тест отличит правильное поведение от похожего неправильного?"
    >${escapeHtml(lessonState.makeDraft)}</textarea>
    <fieldset class="rubric-fieldset">
      <legend>Self-check: тест действительно различающий?</legend>
      ${lesson.rubric.map((item, index) => `
        <label class="check-row">
          <input
            type="checkbox"
            data-action="toggle-rubric"
            value="${index}"
            ${lessonState.rubricChecks.includes(index) ? "checked" : ""}
          />
          <span>${escapeHtml(item)}</span>
        </label>
      `).join("")}
    </fieldset>
    <div class="stage-actions">
      <button
        class="button primary"
        type="button"
        data-action="complete-make"
        ${allChecked && enoughText ? "" : "disabled"}
      >
        Завершить лабораторную
      </button>
      <span class="action-hint">
        ${enoughText ? "Черновик есть." : "Нужно хотя бы 40 символов."}
        ${allChecked ? " Rubric пройден." : " Отметь все пункты rubric."}
      </span>
    </div>
  `;
}

function confidenceControl(field, value, label) {
  return `
    <label class="confidence-control">
      <span>${label}: <strong>${value}/5</strong></span>
      <input
        type="range"
        min="1"
        max="5"
        value="${value}"
        data-confidence-field="${field}"
        aria-label="${label}"
      />
    </label>
  `;
}

function renderNotebook(lessonState) {
  return `
    <details class="notebook" ${lessonState.stage === "investigate" || lessonState.stage === "modify" ? "open" : ""}>
      <summary>
        <span>
          <span class="eyebrow">Research log</span>
          <strong>Гипотеза → вмешательство → наблюдение</strong>
        </span>
        <span class="notebook-status">автосохранение</span>
      </summary>
      <div class="notebook-grid">
        ${NOTEBOOK_FIELDS.map(([field, label]) => `
          <label class="notebook-field ${field === "observed" || field === "revisedModel" ? "wide" : ""}">
            <span>${label}</span>
            <textarea
              rows="3"
              data-notebook-field="${field}"
              placeholder="${notebookPlaceholder(field)}"
            >${escapeHtml(lessonState.notebook[field])}</textarea>
          </label>
        `).join("")}
      </div>
    </details>
  `;
}

function notebookPlaceholder(field) {
  const placeholders = {
    question: "Какой конкретный контракт я проверяю?",
    mentalModel: "Что, по-моему, происходит по шагам?",
    hypothesis: "Если причина X, я увижу Y…",
    alternative: "Какое ещё объяснение даст похожий результат?",
    intervention: "Как изменить только одну переменную?",
    expected: "Где и какой сигнал должен появиться?",
    observed: "Что фактически изменилось?",
    revisedModel: "Как теперь я объясняю систему?",
    regressionTest: "Какой тест не даст этому сломаться тихо?",
  };
  return placeholders[field];
}

function baselineOptions(lessonId) {
  const baselines = {
    dataset: {
      familyOverlap: false,
      seenTemplateLeak: false,
      missingPair: false,
      answerDrift: false,
    },
    collator: {
      maskPrompt: true,
      maskPadding: true,
      includeGenerationPrompt: true,
    },
    optimizer: {
      accumulationSteps: 2,
      divideLoss: true,
      zeroMode: "window",
      schedulerMode: "optimizer",
      maxGradNorm: 1,
    },
    generation: {
      paddingSide: "left",
      trimPrompt: true,
      evalMode: true,
      inferenceMode: true,
    },
    evaluation: {
      heldoutLeak: false,
      pairedFacts: true,
      rows: DEFAULT_EVALUATION_ROWS.map((row) => ({ ...row })),
    },
  };
  return baselines[lessonId];
}

function runSimulation(lessonId, options) {
  if (lessonId === "dataset") return simulateDataset(options);
  if (lessonId === "collator") return simulateCollator(options);
  if (lessonId === "optimizer") return simulateOptimizer(options);
  if (lessonId === "generation") return simulateGeneration(options);
  return summarizeEvaluation(options.rows, options);
}

function renderControls(lessonId, options) {
  if (lessonId === "dataset") {
    return `<fieldset class="control-panel"><legend>Контролируемые мутации</legend>
      ${switchControl("familyOverlap", "Пересечь train/test families", options.familyOverlap)}
      ${switchControl("seenTemplateLeak", "Добавить seen template ID в train", options.seenTemplateLeak)}
      ${switchControl("missingPair", "Удалить один held-out fact", options.missingPair)}
      ${switchControl("answerDrift", "Изменить canonical answer", options.answerDrift)}
    </fieldset>`;
  }
  if (lessonId === "collator") {
    return `<fieldset class="control-panel"><legend>Masking contract</legend>
      ${switchControl("maskPrompt", "Маскировать prompt", options.maskPrompt)}
      ${switchControl("maskPadding", "Маскировать padding", options.maskPadding)}
      ${switchControl("includeGenerationPrompt", "Добавлять assistant boundary", options.includeGenerationPrompt)}
    </fieldset>`;
  }
  if (lessonId === "optimizer") {
    return `<fieldset class="control-panel"><legend>Порядок optimizer events</legend>
      <label class="control-field"><span>Accumulation steps</span>
        <select data-control="accumulationSteps">
          ${[1, 2, 3, 4].map((value) => `<option value="${value}" ${selected(options.accumulationSteps, value)}>${value}</option>`).join("")}
        </select>
      </label>
      ${switchControl("divideLoss", "Делить loss на window size", options.divideLoss)}
      <label class="control-field"><span>zero_grad</span>
        <select data-control="zeroMode">
          <option value="window" ${selected(options.zeroMode, "window")}>в начале окна</option>
          <option value="micro" ${selected(options.zeroMode, "micro")}>на каждом micro-step</option>
          <option value="never" ${selected(options.zeroMode, "never")}>никогда</option>
        </select>
      </label>
      <label class="control-field"><span>scheduler.step</span>
        <select data-control="schedulerMode">
          <option value="optimizer" ${selected(options.schedulerMode, "optimizer")}>на optimizer-step</option>
          <option value="micro" ${selected(options.schedulerMode, "micro")}>на micro-step</option>
        </select>
      </label>
    </fieldset>`;
  }
  if (lessonId === "generation") {
    return `<fieldset class="control-panel"><legend>Generation contract</legend>
      <label class="control-field"><span>Padding side</span>
        <select data-control="paddingSide">
          <option value="left" ${selected(options.paddingSide, "left")}>left</option>
          <option value="right" ${selected(options.paddingSide, "right")}>right</option>
        </select>
      </label>
      ${switchControl("trimPrompt", "Удалять prompt из output", options.trimPrompt)}
      ${switchControl("evalMode", "Вызывать model.eval()", options.evalMode)}
      ${switchControl("inferenceMode", "Использовать inference_mode()", options.inferenceMode)}
    </fieldset>`;
  }
  return `<fieldset class="control-panel"><legend>Научный контракт</legend>
    ${switchControl("heldoutLeak", "Разрешить held-out template leakage", options.heldoutLeak)}
    ${switchControl("pairedFacts", "Сохранять одинаковые fact sets", options.pairedFacts)}
    <p class="control-help">Нажимай ✓/× в таблице, чтобы менять correctness одного fact.</p>
  </fieldset>`;
}

function switchControl(name, label, value) {
  return `
    <label class="switch-control">
      <input type="checkbox" data-control="${name}" ${checked(value)} />
      <span class="switch-track" aria-hidden="true"></span>
      <span>${escapeHtml(label)}</span>
    </label>
  `;
}

function renderSimulation(lessonId, result, interactive) {
  if (lessonId === "dataset") return renderDataset(result);
  if (lessonId === "collator") return renderCollator(result);
  if (lessonId === "optimizer") return renderOptimizer(result);
  if (lessonId === "generation") return renderGeneration(result);
  return renderEvaluation(result, interactive);
}

function metricCard(label, value, detail = "") {
  return `<div class="metric-card"><span>${label}</span><strong>${value}</strong>${detail ? `<small>${detail}</small>` : ""}</div>`;
}

function renderDataset(result) {
  return `
    <div class="metric-grid">
      ${metricCard("Train rows", result.train.length, `${result.uniqueTrainFacts} facts`)}
      ${metricCard("Eval rows", result.test.length, `${result.uniqueEvaluationFacts} facts`)}
      ${metricCard("Fact pairing", result.sameFacts ? "совпадает" : "сломано")}
      ${metricCard("Template isolation", result.globalSeenTemplateIsolation ? "глобальная" : "leak")}
    </div>
    <div class="slice-flow" aria-label="Поток одного факта через slices">
      <div class="flow-node"><span>train</span><strong>direct_question.001</strong><small>тот же fact</small></div>
      <span class="flow-arrow">→</span>
      <div class="flow-node"><span>exact</span><strong>direct_question.001</strong><small>та же форма</small></div>
      <span class="flow-arrow">→</span>
      <div class="flow-node"><span>seen</span><strong>direct_question.002</strong><small>новый template</small></div>
      <span class="flow-arrow">→</span>
      <div class="flow-node"><span>held-out</span><strong>nominal_attribute.001</strong><small>новая family</small></div>
    </div>
    <div class="table-wrap">
      <table>
        <thead><tr><th>Slice</th><th>Rows</th><th>Unique facts</th><th>Family</th></tr></thead>
        <tbody>
          <tr><td>train</td><td>${result.train.length}</td><td>${result.uniqueTrainFacts}</td><td>${result.trainFamilies.join(", ")}</td></tr>
          <tr><td>exact</td><td>${result.slices.exact.length}</td><td>${new Set(result.slices.exact.map((row) => row.factId)).size}</td><td>direct_question</td></tr>
          <tr><td>seen</td><td>${result.slices.seen.length}</td><td>${new Set(result.slices.seen.map((row) => row.factId)).size}</td><td>direct_question</td></tr>
          <tr><td>heldout</td><td>${result.slices.heldout.length}</td><td>${new Set(result.slices.heldout.map((row) => row.factId)).size}</td><td>nominal_attribute</td></tr>
        </tbody>
      </table>
    </div>
  `;
}

function renderCollator(result) {
  return `
    <div class="metric-grid">
      ${metricCard("Batch shape", `2 × ${result.width}`)}
      ${metricCard("LOSS positions", result.targetTokens, `${result.answerTokens} answer tokens`)}
      ${metricCard("Prompt leak", result.leakedPromptTokens)}
      ${metricCard("Padding leak", result.leakedPaddingTokens)}
    </div>
    <div class="token-legend" aria-label="Легенда">
      <span><i class="legend-swatch mask"></i> MASK (label=-100)</span>
      <span><i class="legend-swatch loss"></i> LOSS</span>
      <span><i class="legend-swatch padding"></i> padding</span>
    </div>
    ${result.rows.map((row, rowIndex) => `
      <details class="token-row" ${rowIndex === 0 ? "open" : ""}>
        <summary>${escapeHtml(row.name)} · sequence=${row.sequenceLength} · prompt=${row.promptLength}</summary>
        <div class="token-grid">
          ${row.cells.map((cell) => `
            <div
              class="token-cell ${cell.role} ${cell.contributesToLoss ? "loss" : "mask"}"
              aria-label="позиция ${cell.index}, токен ${cell.token}, attention ${cell.attention}, label ${cell.label}"
            >
              <span class="token-index">${cell.index}</span>
              <strong>${escapeHtml(cell.token)}</strong>
              <small>a:${cell.attention} · y:${cell.label}</small>
              <span class="token-state">${cell.contributesToLoss ? "LOSS" : "MASK"}</span>
            </div>
          `).join("")}
        </div>
      </details>
    `).join("")}
  `;
}

function renderOptimizer(result) {
  return `
    <div class="metric-grid">
      ${metricCard("Micro-steps", result.gradients.length)}
      ${metricCard("Optimizer steps", result.optimizerSteps, `expected ${result.expectedOptimizerSteps}`)}
      ${metricCard("Scheduler ticks", result.schedulerTicks)}
      ${metricCard("Final parameter", result.finalParameter.toFixed(5), "start 1.00000")}
    </div>
    <div class="optimizer-timeline" aria-label="Трасса optimizer events">
      ${result.gradients.map((gradient, index) => {
        const microStep = index + 1;
        const events = result.events.filter((event) => event.microStep === microStep);
        return `<div class="timeline-column">
          <div class="timeline-head"><span>μ${microStep}</span><strong>g=${gradient}</strong></div>
          ${events.map((event) => `<span class="event-chip ${event.kind}">${eventLabel(event)}</span>`).join("")}
        </div>`;
      }).join("")}
    </div>
    <div class="table-wrap">
      <table>
        <thead><tr><th>Update</th><th>Window</th><th>Gradient до clip</th><th>Применён</th><th>LR</th><th>Δ parameter</th></tr></thead>
        <tbody>
          ${result.updates.map((update) => `<tr>
            <td>${update.optimizerStep}</td>
            <td>${update.windowSize}</td>
            <td>${update.gradientBeforeClip.toFixed(3)}</td>
            <td>${update.appliedGradient.toFixed(3)}</td>
            <td>${update.learningRate.toFixed(4)}</td>
            <td>${formatSigned(update.delta)}</td>
          </tr>`).join("")}
        </tbody>
      </table>
    </div>
  `;
}

function eventLabel(event) {
  if (event.kind === "zero") return "zero_grad";
  if (event.kind === "backward") return `backward +${event.contribution.toFixed(2)}`;
  if (event.kind === "optimizer") return `step Δ${event.delta.toFixed(3)}`;
  return `scheduler #${event.tick}`;
}

function renderGeneration(result) {
  return `
    <div class="metric-grid">
      ${metricCard("Padded width", result.promptWidth)}
      ${metricCard("Exact matches", `${result.exactMatches}/${result.rows.length}`)}
      ${metricCard("Padding", result.paddingSide)}
      ${metricCard("Prompt slice", result.trimPrompt ? "on" : "off")}
    </div>
    <div class="generation-rows">
      ${result.rows.map((row) => `
        <div class="generation-row ${row.exactMatch ? "correct" : "incorrect"}">
          <div class="generation-label">
            <strong>${escapeHtml(row.name)}</strong>
            <span>${row.exactMatch ? "✓ exact match" : "× mismatch"}</span>
          </div>
          <div class="token-sequence">
            ${row.paddedPrompt.map((token) => `<code class="sequence-token ${token === "<pad>" ? "pad" : ""}">${escapeHtml(token)}</code>`).join("")}
            <span class="generation-cursor">│</span>
            ${row.generated.map((token) => `<code class="sequence-token generated">${escapeHtml(token)}</code>`).join("")}
          </div>
          <div class="prediction-line"><span>prediction</span><code>${escapeHtml(row.prediction)}</code><span>target</span><code>${escapeHtml(row.target)}</code></div>
        </div>
      `).join("")}
    </div>
    <div class="research-clue">
      <strong>Продвинутый bug hunt:</strong>
      <span><code>generate_answers</code> меняет общий tokenizer на left padding, а collator задаёт right padding только в <code>__init__</code>. Проверь, какое состояние увидит ленивый DataLoader после baseline eval.</span>
    </div>
  `;
}

function renderEvaluation(result, interactive) {
  const familyGap = result.comparisons.seen_minus_heldout;
  return `
    <div class="metric-grid">
      ${metricCard("Exact recall", formatPercent(result.accuracies.exact))}
      ${metricCard("Seen template", formatPercent(result.accuracies.seen))}
      ${metricCard("Held-out family", formatPercent(result.accuracies.heldout))}
      ${metricCard("Family gap", formatSigned(familyGap.gap))}
    </div>
    <div class="table-wrap">
      <table class="evaluation-table">
        <thead><tr><th>fact_id</th><th>Exact</th><th>Seen</th><th>Held-out</th><th>Seen → held</th></tr></thead>
        <tbody>
          ${result.rows.map((row, rowIndex) => {
            const transition = familyGap.rows[rowIndex].transition;
            return `<tr>
              <td><code>${escapeHtml(row.factId)}</code></td>
              ${["exact", "seen", "heldout"].map((field) => `<td>
                ${interactive ? `<button class="correctness-toggle ${row[field] ? "yes" : "no"}" type="button" data-toggle-eval="${rowIndex}:${field}" aria-label="${field}: ${row[field] ? "верно" : "неверно"}">${row[field] ? "✓" : "×"}</button>` : `<span class="correctness-mark ${row[field] ? "yes" : "no"}">${row[field] ? "✓" : "×"}</span>`}
              </td>`).join("")}
              <td><span class="transition-label">${transition}</span></td>
            </tr>`;
          }).join("")}
        </tbody>
      </table>
    </div>
    <div class="transition-grid">
      ${Object.entries(familyGap.counts).map(([name, count]) => `<div><span>${name}</span><strong>${count}</strong></div>`).join("")}
    </div>
  `;
}

function renderWarnings(warnings) {
  warnings = warnings ?? [];
  if (!warnings.length) {
    return `<div class="feedback success"><strong>Нарушений не обнаружено.</strong><span>Теперь измени одну настройку и заранее назови сигнал.</span></div>`;
  }
  return `<div class="feedback warning" role="alert"><strong>Наблюдаемые последствия</strong><ul>${warnings.map((warning) => `<li>${escapeHtml(warning)}</li>`).join("")}</ul></div>`;
}

function summarizeRun(lessonId, result) {
  const summaries = {
    dataset: () => `${result.train.length} train / ${result.test.length} eval; issues=${result.issues.length}`,
    collator: () => `targets=${result.targetTokens}; prompt_leak=${result.leakedPromptTokens}; pad_leak=${result.leakedPaddingTokens}`,
    optimizer: () => `optimizer_steps=${result.optimizerSteps}; scheduler_ticks=${result.schedulerTicks}; parameter=${result.finalParameter.toFixed(5)}`,
    generation: () => `exact_matches=${result.exactMatches}/${result.rows.length}; warnings=${result.warnings.length}`,
    evaluation: () => `EM=${Object.values(result.accuracies).map((value) => value.toFixed(3)).join("/")}; gap=${result.comparisons.seen_minus_heldout.gap.toFixed(3)}`,
  };
  return summaries[lessonId]();
}

function showToast(message) {
  const toast = document.getElementById("toast");
  if (!toast) return;
  toast.textContent = message;
  toast.classList.add("visible");
  window.setTimeout(() => toast.classList.remove("visible"), 2200);
}

function downloadText(filename, contents, type = "text/plain") {
  const blob = new Blob([contents], { type });
  const url = URL.createObjectURL(blob);
  const link = document.createElement("a");
  link.href = url;
  link.download = filename;
  link.click();
  URL.revokeObjectURL(url);
}

function rerenderMain() {
  const main = document.getElementById("lesson-content");
  const sidebar = document.querySelector(".sidebar");
  if (main) main.innerHTML = renderLesson();
  if (sidebar) sidebar.outerHTML = renderSidebar();
}

app.addEventListener("click", async (event) => {
  const button = event.target.closest("button");
  if (!button) return;
  const action = button.dataset.action;
  const lesson = currentLesson();
  const lessonState = currentLessonState();

  if (action === "open-stage") {
    const stage = button.dataset.stage;
    if (canOpenStage(lessonState, stage)) {
      lessonState.stage = stage;
      persist();
      rerenderMain();
    }
    return;
  }
  if (action === "lock-prediction") {
    if (lessonState.selectedPrediction == null) return;
    lessonState.predictionLocked = true;
    completeStage(lessonState, "predict");
    persist();
    rerenderMain();
    return;
  }
  if (action === "advance-stage") {
    lessonState.stage = "run";
    persist();
    rerenderMain();
    return;
  }
  if (action === "run-baseline") {
    const result = runSimulation(lesson.id, baselineOptions(lesson.id));
    lessonState.experimentRuns += 1;
    lessonState.experimentHistory.push({
      kind: "baseline",
      summary: summarizeRun(lesson.id, result),
      at: new Date().toISOString(),
    });
    completeStage(lessonState, "run");
    lessonState.stage = "investigate";
    persist();
    rerenderMain();
    return;
  }
  if (action === "complete-investigate") {
    completeStage(lessonState, "investigate");
    lessonState.stage = "modify";
    persist();
    rerenderMain();
    return;
  }
  if (action === "record-experiment") {
    const result = runSimulation(lesson.id, experimentOptions[lesson.id]);
    lessonState.experimentRuns += 1;
    lessonState.experimentHistory.push({
      kind: "mutation",
      summary: summarizeRun(lesson.id, result),
      options: JSON.parse(JSON.stringify(experimentOptions[lesson.id])),
      at: new Date().toISOString(),
    });
    completeStage(lessonState, "modify");
    lessonState.stage = "make";
    persist();
    rerenderMain();
    return;
  }
  if (action === "complete-make") {
    completeStage(lessonState, "make");
    persist();
    const nextIndex = LESSONS.findIndex((candidate) => candidate.id === lesson.id) + 1;
    if (nextIndex < LESSONS.length) {
      state.activeLessonId = LESSONS[nextIndex].id;
      setHash(state.activeLessonId);
    } else {
      showToast("Все лабораторные завершены. Экспортируй журнал для защиты.");
      rerenderMain();
    }
    return;
  }
  if (action === "toggle-rubric") {
    const index = Number(button.value);
    if (!Number.isInteger(index)) return;
    lessonState.rubricChecks = button.checked
      ? [...new Set([...lessonState.rubricChecks, index])]
      : lessonState.rubricChecks.filter((value) => value !== index);
    persist();
    rerenderMain();
    return;
  }
  if (button.dataset.toggleEval) {
    const [rowIndexText, field] = button.dataset.toggleEval.split(":");
    const rowIndex = Number(rowIndexText);
    const row = experimentOptions.evaluation.rows[rowIndex];
    if (row && ["exact", "seen", "heldout"].includes(field)) {
      row[field] = !row[field];
      rerenderMain();
    }
    return;
  }
  if (action === "copy-command") {
    try {
      await navigator.clipboard.writeText(lesson.command);
      showToast("Команда скопирована");
    } catch {
      showToast("Не удалось скопировать; выдели команду вручную");
    }
    return;
  }
  if (action === "export-notebook") {
    downloadText("research-loop-notebook.md", notebookToMarkdown(state), "text/markdown");
    showToast("Журнал экспортирован");
    return;
  }
  if (action === "reset-progress") {
    if (window.confirm("Сбросить весь прогресс и исследовательский журнал?")) {
      clearProgress();
      state = createDefaultState();
      setHash(state.activeLessonId);
      renderShell();
    }
  }
});

app.addEventListener("change", (event) => {
  const target = event.target;
  const lessonState = currentLessonState();
  if (target.matches('[data-action="select-prediction"]')) {
    lessonState.selectedPrediction = Number(target.value);
    persist();
    rerenderMain();
    return;
  }
  if (target.dataset.control) {
    const lessonId = state.activeLessonId;
    const name = target.dataset.control;
    const value =
      target.type === "checkbox"
        ? target.checked
        : name === "accumulationSteps"
          ? Number(target.value)
          : target.value;
    experimentOptions[lessonId][name] = value;
    rerenderMain();
    return;
  }
  if (target.dataset.confidenceField) {
    lessonState[target.dataset.confidenceField] = Number(target.value);
    persist();
    rerenderMain();
    return;
  }
  if (target.dataset.action === "toggle-rubric") {
    const index = Number(target.value);
    if (!Number.isInteger(index)) return;
    lessonState.rubricChecks = target.checked
      ? [...new Set([...lessonState.rubricChecks, index])]
      : lessonState.rubricChecks.filter((value) => value !== index);
    persist();
    updateMakeCompletionState();
  }
});

app.addEventListener("input", (event) => {
  const target = event.target;
  const lessonState = currentLessonState();
  if (target.dataset.notebookField) {
    lessonState.notebook[target.dataset.notebookField] = target.value;
    persist();
  }
  if (target.dataset.stateField === "makeDraft") {
    lessonState.makeDraft = target.value;
    persist();
    updateMakeCompletionState();
  }
});

function updateMakeCompletionState() {
  const lesson = currentLesson();
  const lessonState = currentLessonState();
  const button = document.querySelector('[data-action="complete-make"]');
  if (!button) return;
  const allChecked = lesson.rubric.every((_, index) =>
    lessonState.rubricChecks.includes(index),
  );
  button.disabled = !(allChecked && lessonState.makeDraft.trim().length >= 40);
}

window.addEventListener("hashchange", () => {
  syncFromHash();
  persist();
  renderShell();
  document.getElementById("lesson-content")?.focus();
});

syncFromHash();
renderShell();
