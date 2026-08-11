export const STAGES = ["predict", "run", "investigate", "modify", "make"];

export const LESSONS = [
  {
    id: "dataset",
    number: "01",
    title: "Dataset validity",
    shortTitle: "Данные",
    subtitle: "Когда код работает, а эксперимент уже не отвечает на вопрос",
    source: "dataset.py",
    symbol: "generate_h1_examples",
    sourceHref: "../dataset.py",
    library: "Python stdlib + dataclasses",
    apis: ["set", "tuple", "hashlib.sha256", "json"],
    contract:
      "Один и тот же fact проходит через exact, seen-template и held-out-family renderings; меняется только контролируемая языковая ось.",
    prediction: {
      question:
        "Мини-мир: 3 человека × 4 операции × 3 train families. Сколько train rows и сколько rows в каждом eval slice при 2 eval people?",
      options: [
        "36 train; по 8 в каждом eval slice",
        "12 train; по 24 в каждом eval slice",
        "36 train; по 6 в каждом eval slice",
      ],
      correct: 0,
      explanation:
        "Train: 3 × 4 × 3 = 36. Каждый eval slice: 2 × 4 = 8; всего evaluation rows 24.",
    },
    failureModes: [
      "Family overlap не обязан вызвать runtime exception, но уничтожает held-out contrast.",
      "Неполное pairing смешивает сложность фактов с эффектом формулировки.",
      "Изменённый canonical_answer превращает один fact в несколько несовместимых определений.",
    ],
    makeTask:
      "Напиши assertion или тест: множества fact_id во всех трёх eval slices равны, а seen template IDs не встречаются в train.",
    rubric: [
      "Проверяется равенство всех трёх множеств, а не только их размеры.",
      "Проверка template leakage глобальная, не только для того же fact.",
      "Сообщение ошибки называет нарушенный научный контракт.",
    ],
    command:
      "uv run pytest scripts/test_dataset.py::DatasetGenerationTests::test_generates_sft_and_three_paired_evaluation_slices -vv",
  },
  {
    id: "collator",
    number: "02",
    title: "Token & mask microscope",
    shortTitle: "Collator",
    subtitle: "Какие токены действительно участвуют в loss",
    source: "scripts/sft_smoke.py",
    symbol: "SFTCollator.__call__",
    sourceHref: "../scripts/sft_smoke.py",
    library: "PyTorch + Transformers",
    apis: ["DataLoader(collate_fn=...)", "apply_chat_template", "Tensor.clone", "labels == -100"],
    contract:
      "input_ids, attention_mask и labels имеют одну форму; prompt и padding замаскированы -100, assistant span остаётся target.",
    prediction: {
      question:
        "В двух строках 6 assistant-токенов суммарно. Что станет с target_tokens, если выключить только prompt masking?",
      options: [
        "Останется 6",
        "Увеличится: prompt тоже начнёт участвовать в loss",
        "Станет 0",
      ],
      correct: 1,
      explanation:
        "labels изначально копируют input_ids. Только запись -100 исключает позицию из loss.",
    },
    failureModes: [
      "Без prompt mask модель оптимизирует воспроизведение system/user текста, а не только ответа.",
      "Без padding mask objective зависит от длины соседних строк в батче.",
      "Разные chat-template boundaries нарушают проверку prompt-prefix/full-conversation.",
    ],
    makeTask:
      "Спроектируй assert_sft_batch_contract(batch): формы совпадают, padding и prompt имеют -100, target в каждой строке непуст.",
    rubric: [
      "Assertion проверяет каждую строку, а не только весь batch.",
      "Отдельно проверяются attention padding и prompt span.",
      "Есть тест, который намеренно оставляет один prompt token немаскированным.",
    ],
    command:
      "uv run python scripts/sft_smoke.py runtime.debug=true training.batch_size=2 training.smoke_steps=1 runtime.save_model=false",
  },
  {
    id: "optimizer",
    number: "03",
    title: "Optimizer step simulator",
    shortTitle: "Train step",
    subtitle: "Micro-step, accumulation window и настоящий update",
    source: "scripts/sft_smoke.py",
    symbol: "train_step / train",
    sourceHref: "../scripts/sft_smoke.py",
    library: "PyTorch",
    apis: ["loss.backward", "optimizer.zero_grad", "AdamW.step", "clip_grad_norm_", "LambdaLR.step"],
    contract:
      "Backward идёт на каждом micro-batch; zero_grad — в начале окна; optimizer и scheduler — один раз в конце окна.",
    prediction: {
      question:
        "Пять micro-batches, accumulation_steps=2. Сколько optimizer steps и какой размер последнего окна?",
      options: [
        "2 шага, хвост отброшен",
        "3 шага, последнее окно размера 1",
        "5 шагов, последнее окно размера 1",
      ],
      correct: 1,
      explanation:
        "ceil(5/2)=3. Код вычисляет фактический window_size, поэтому короткий хвост не недомасштабируется.",
    },
    failureModes: [
      "zero_grad на каждом micro-step стирает предыдущие contributions.",
      "Loss без деления меняет масштаб gradient вместе с accumulation_steps.",
      "Scheduler на каждом micro-step заканчивает schedule раньше optimizer.",
    ],
    makeTask:
      "Напиши tiny-model тест, доказывающий, что два micro-batches дают один optimizer update и один scheduler tick.",
    rubric: [
      "Тест считает вызовы zero_grad/backward/step, а не только final loss.",
      "Отдельно проверяется короткое последнее окно.",
      "Есть отрицательный вариант с zero_grad на каждом micro-step.",
    ],
    command:
      "uv run pytest scripts/test_sft_smoke.py::SmokeTrainingTests::test_full_training_evaluates_baseline_before_accumulated_optimizer_steps -vv",
  },
  {
    id: "generation",
    number: "04",
    title: "Generation debugger",
    shortTitle: "Generation",
    subtitle: "Тихие ошибки padding, mode и prompt slicing",
    source: "scripts/sft_smoke.py",
    symbol: "generate_answers",
    sourceHref: "../scripts/sft_smoke.py",
    library: "Transformers + PyTorch",
    apis: ["apply_chat_template", "model.eval", "torch.inference_mode", "model.generate", "batch_decode"],
    contract:
      "Decoder-only prompts pad слева; generation начинается после assistant boundary; возвращённый prompt удаляется по padded width.",
    prediction: {
      question:
        "Почему output_ids режется по общей prompt_width, а не по attention_mask.sum() каждой строки?",
      options: [
        "Потому что generate возвращает padded input одинаковой ширины перед continuation",
        "Чтобы случайно удалить первый answer token",
        "Разницы для left padding нет",
      ],
      correct: 0,
      explanation:
        "При left padding индивидуальная non-pad длина меньше общей ширины. Slice по ней оставит часть prompt в prediction.",
    },
    failureModes: [
      "Right padding заставляет короткую строку продолжаться с pad-позиции.",
      "Без prompt slice exact match сравнивает target со всей беседой.",
      "eval() и inference_mode() независимы: первый меняет поведение модулей, второй отключает autograd bookkeeping.",
    ],
    makeTask:
      "Напиши regression test с двумя prompts разной длины: predictions сохраняют порядок и не содержат prompt.",
    rubric: [
      "В fixture действительно разные длины prompt.",
      "Проверяется padding_side во время вызова generate.",
      "Проверяется точный continuation, а не только количество predictions.",
    ],
    command:
      "uv run pytest scripts/test_sft_smoke.py::SmokeTrainingTests::test_generated_eval_uses_bounded_batches -vv",
  },
  {
    id: "evaluation",
    number: "05",
    title: "Paired evaluation bench",
    shortTitle: "Evaluation",
    subtitle: "Как traces превращаются в интерпретируемый gap",
    source: "scripts/sft_smoke.py",
    symbol: "evaluate_splits",
    sourceHref: "../scripts/sft_smoke.py",
    library: "Python collections + JSONL artifacts",
    apis: ["zip(strict=True)", "asdict", "grouped exact match", "paired transition"],
    contract:
      "Каждый fact присутствует во всех трёх slices; raw prediction сохраняется; gaps считаются для тех же facts.",
    prediction: {
      question:
        "Если exact=true, seen=false для одного fact, как называется paired transition и чему равен correctness delta?",
      options: ["left_only и +1", "right_only и -1", "neither_correct и 0"],
      correct: 0,
      explanation:
        "Левая сторона верна, правая нет: left_only; Number(true)-Number(false)=+1.",
    },
    failureModes: [
      "Разные fact sets превращают paired contrast в сравнение разных задач.",
      "Held-out template leakage делает family gap неподходящим доказательством transfer.",
      "Низкий exact recall означает, что held-out failure ещё нельзя приписывать формулировке.",
    ],
    makeTask:
      "Из traces восстанови overall paired gap и transition counts; затем сломай один fact_id и потребуй явную ошибку pairing.",
    rubric: [
      "Pair key включает world_id и fact_id.",
      "Проверяется совпадение provenance, а не только ключа.",
      "Raw и normalized predictions не смешиваются.",
    ],
    command:
      "uv run pytest scripts/test_sft_smoke.py::SmokeTrainingTests::test_eval_persists_full_traces_aggregates_and_paired_gaps -vv",
  },
];

export function lessonById(id) {
  return LESSONS.find((lesson) => lesson.id === id) ?? LESSONS[0];
}

