# Research Loop Lab

Интерактивный учебный полигон для репозитория `do-models-only-memorize`.
Он превращает уже написанный код в пять PRIMM-лабораторных:

1. dataset validity;
2. tokenization и loss masking;
3. gradient accumulation и optimizer ordering;
4. decoder-only generation;
5. paired scientific evaluation.

Полигон не запускает Qwen и не эмулирует весь PyTorch. Каждый симулятор —
детерминированная модель конкретного контракта production-кода. Рядом с ним
показаны настоящий symbol, используемые API и команда для проверки на реальном
тесте.

## Запуск

Из корня репозитория:

```bash
python3 -m http.server 8000
```

Открыть:

```text
http://localhost:8000/learning-lab/
```

Сервер нужно запускать именно из корня: тогда ссылки из лабораторных могут
открывать актуальные `dataset.py` и `scripts/sft_smoke.py`.

## Как проходить

Каждая лабораторная открывает этапы последовательно:

```text
Predict → Run → Investigate → Modify → Make
```

- До `Run` нужно зафиксировать прогноз.
- В `Investigate` сравнивается mental model с наблюдаемой трассой.
- В `Modify` меняется одна причина и показывается её failure signal.
- В `Make` нужно спроектировать различающий regression test.
- Исследовательский журнал сохраняется в `localStorage` и экспортируется в
  Markdown кнопкой в шапке.

Свободный текст не оценивается поиском «правильных слов»: завершение Make
использует self-check rubric. Это намеренно — объяснение должно быть твоим, а
не подобранным под скрытый шаблон.

## Проверки полигона

Зависимости не нужны; используется встроенный Node test runner:

```bash
node --test learning-lab/tests/*.test.mjs
node --check learning-lab/js/app.js
```

Production-тесты репозитория по-прежнему запускаются отдельно:

```bash
uv run pytest
```

## Где хранится состояние

Прогресс и заметки остаются только в браузерном `localStorage` под ключом
`do-models-only-memorize.learning-lab.v1`. Кнопка «Сбросить» удаляет их после
подтверждения. Экспортированный Markdown не содержит model weights, dataset
records или иных внешних данных — только заполненный исследовательский журнал.

