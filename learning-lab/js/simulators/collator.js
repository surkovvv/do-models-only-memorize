export const IGNORE_INDEX = -100;

const RAW_EXAMPLES = [
  {
    name: "birth city",
    promptLength: 14,
    tokens: [
      "<|system|>",
      "Answer",
      "with",
      "the",
      "value",
      "only",
      ".",
      "<|user|>",
      "Where",
      "was",
      "Gogr",
      "born",
      "?",
      "<|assistant|>",
      "Tor",
      "-",
      "Velk",
      "<|end|>",
    ],
  },
  {
    name: "occupation",
    promptLength: 13,
    tokens: [
      "<|system|>",
      "Answer",
      "with",
      "the",
      "value",
      "only",
      ".",
      "<|user|>",
      "Job",
      "of",
      "Riln",
      "?",
      "<|assistant|>",
      "cartographer",
      "<|end|>",
    ],
  },
];

function tokenId(token, index) {
  if (token === "<|pad|>") return 0;
  return 100 + index;
}

export function simulateCollator({
  maskPrompt = true,
  maskPadding = true,
  includeGenerationPrompt = true,
} = {}) {
  const width = Math.max(...RAW_EXAMPLES.map((example) => example.tokens.length));
  const rows = RAW_EXAMPLES.map((example, rowIndex) => {
    const missingGenerationPrompt = includeGenerationPrompt ? 0 : 1;
    const effectivePromptLength = Math.max(0, example.promptLength - missingGenerationPrompt);
    const paddingLength = width - example.tokens.length;
    const tokens = [...example.tokens, ...Array(paddingLength).fill("<|pad|>")];
    const attentionMask = tokens.map((token) => (token === "<|pad|>" ? 0 : 1));
    const inputIds = tokens.map((token, index) => tokenId(token, index + rowIndex * width));
    const labels = inputIds.map((id, index) => {
      if (maskPadding && attentionMask[index] === 0) return IGNORE_INDEX;
      if (maskPrompt && index < effectivePromptLength) return IGNORE_INDEX;
      return id;
    });

    return {
      name: example.name,
      tokens,
      inputIds,
      attentionMask,
      labels,
      promptLength: effectivePromptLength,
      sequenceLength: example.tokens.length,
      cells: tokens.map((token, index) => ({
        index,
        token,
        inputId: inputIds[index],
        attention: attentionMask[index],
        label: labels[index],
        role:
          attentionMask[index] === 0
            ? "padding"
            : index < effectivePromptLength
              ? "prompt"
              : "answer",
        contributesToLoss: labels[index] !== IGNORE_INDEX,
      })),
    };
  });

  const allCells = rows.flatMap((row) => row.cells);
  const targetTokens = allCells.filter((cell) => cell.contributesToLoss).length;
  const leakedPromptTokens = allCells.filter(
    (cell) => cell.role === "prompt" && cell.contributesToLoss,
  ).length;
  const leakedPaddingTokens = allCells.filter(
    (cell) => cell.role === "padding" && cell.contributesToLoss,
  ).length;
  const answerTokens = allCells.filter((cell) => cell.role === "answer").length;
  const missingAssistantBoundary = !includeGenerationPrompt;
  const warnings = [];

  if (leakedPromptTokens) {
    warnings.push(
      `Loss включает ${leakedPromptTokens} prompt-токенов: objective уже не равен «учить только ответ».`,
    );
  }
  if (leakedPaddingTokens) {
    warnings.push(
      `Loss включает ${leakedPaddingTokens} pad-токенов: длина батча влияет на objective.`,
    );
  }
  if (missingAssistantBoundary) {
    warnings.push(
      "Граница начала assistant-ответа отсутствует: prompt-only и full tokenization больше не имеют одного контракта.",
    );
  }

  return {
    width,
    rows,
    targetTokens,
    answerTokens,
    leakedPromptTokens,
    leakedPaddingTokens,
    missingAssistantBoundary,
    invariantHolds:
      leakedPromptTokens === 0 &&
      leakedPaddingTokens === 0 &&
      !missingAssistantBoundary &&
      targetTokens === answerTokens,
    warnings,
  };
}

