const EXAMPLES = [
  {
    name: "short prompt",
    prompt: ["<user>", "Job", "of", "Riln", "?", "<assistant>"],
    answer: ["cartographer", "<eos>"],
    target: "cartographer",
  },
  {
    name: "long prompt",
    prompt: [
      "<user>",
      "In",
      "which",
      "city",
      "was",
      "Gogr",
      "born",
      "?",
      "<assistant>",
    ],
    answer: ["Tor-Velk", "<eos>"],
    target: "Tor-Velk",
  },
];

function decode(tokens) {
  return tokens
    .filter((token) => !new Set(["<pad>", "<eos>"]).has(token))
    .join(" ")
    .replace(/\s+([?.!,])/g, "$1");
}

export function simulateGeneration({
  paddingSide = "left",
  trimPrompt = true,
  evalMode = true,
  inferenceMode = true,
} = {}) {
  if (!new Set(["left", "right"]).has(paddingSide)) {
    throw new Error("paddingSide must be left or right");
  }
  const promptWidth = Math.max(...EXAMPLES.map((example) => example.prompt.length));
  const rows = EXAMPLES.map((example) => {
    const padCount = promptWidth - example.prompt.length;
    const paddedPrompt =
      paddingSide === "left"
        ? [...Array(padCount).fill("<pad>"), ...example.prompt]
        : [...example.prompt, ...Array(padCount).fill("<pad>")];
    const continuesFromPad = paddedPrompt.at(-1) === "<pad>";
    const generated = continuesFromPad ? ["<invalid-continuation>", "<eos>"] : example.answer;
    const output = [...paddedPrompt, ...generated];
    const predictionTokens = trimPrompt ? output.slice(promptWidth) : output;
    const prediction = decode(predictionTokens);

    return {
      name: example.name,
      target: example.target,
      paddedPrompt,
      generated,
      output,
      prediction,
      continuesFromPad,
      exactMatch: prediction === example.target,
    };
  });

  const warnings = [];
  if (rows.some((row) => row.continuesFromPad)) {
    warnings.push(
      "Короткий decoder-only prompt заканчивается pad-токеном, и generation продолжает не тот контекст.",
    );
  }
  if (!trimPrompt) {
    warnings.push(
      "generate() возвращает prompt вместе с продолжением; без slice exact match сравнивает target со всей беседой.",
    );
  }
  if (!evalMode) {
    warnings.push(
      "model.eval() не вызван: training-dependent слои могут сделать evaluation нестабильным.",
    );
  }
  if (!inferenceMode) {
    warnings.push(
      "inference_mode() не включён: ответ может быть тем же, но autograd создаёт лишнее состояние и расходует память.",
    );
  }

  return {
    promptWidth,
    paddingSide,
    trimPrompt,
    evalMode,
    inferenceMode,
    rows,
    exactMatches: rows.filter((row) => row.exactMatch).length,
    warnings,
    invariantHolds:
      paddingSide === "left" && trimPrompt && evalMode && inferenceMode,
  };
}
