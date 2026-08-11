const DEFAULT_GRADIENTS = [1.2, 0.8, 1.6, 0.4, 0.7];

function schedulerMultiplier(tick) {
  return Math.max(0.1, 1 - tick * 0.15);
}

function clip(value, maxNorm) {
  if (maxNorm == null || Math.abs(value) <= maxNorm) return value;
  return Math.sign(value) * maxNorm;
}

export function simulateOptimizer({
  gradients = DEFAULT_GRADIENTS,
  accumulationSteps = 2,
  divideLoss = true,
  zeroMode = "window",
  schedulerMode = "optimizer",
  maxGradNorm = 1,
  baseLearningRate = 0.01,
} = {}) {
  if (!Number.isInteger(accumulationSteps) || accumulationSteps <= 0) {
    throw new Error("accumulationSteps must be a positive integer");
  }
  if (!Array.isArray(gradients) || gradients.length === 0) {
    throw new Error("gradients must be a non-empty array");
  }
  if (!new Set(["window", "micro", "never"]).has(zeroMode)) {
    throw new Error("unknown zeroMode");
  }
  if (!new Set(["optimizer", "micro"]).has(schedulerMode)) {
    throw new Error("unknown schedulerMode");
  }

  let gradientBuffer = 0;
  let parameter = 1;
  let optimizerSteps = 0;
  let schedulerTicks = 0;
  const events = [];
  const updates = [];

  for (let index = 0; index < gradients.length; index += 1) {
    const windowStart = Math.floor(index / accumulationSteps) * accumulationSteps;
    const windowEnd = Math.min(windowStart + accumulationSteps, gradients.length);
    const windowSize = windowEnd - windowStart;
    const isWindowStart = index === windowStart;
    const isWindowEnd = index + 1 === windowEnd;

    if (zeroMode === "micro" || (zeroMode === "window" && isWindowStart)) {
      gradientBuffer = 0;
      events.push({ microStep: index + 1, kind: "zero", value: 0 });
    }

    const contribution = gradients[index] / (divideLoss ? windowSize : 1);
    gradientBuffer += contribution;
    events.push({
      microStep: index + 1,
      kind: "backward",
      rawGradient: gradients[index],
      contribution,
      accumulatedGradient: gradientBuffer,
    });

    if (schedulerMode === "micro") {
      schedulerTicks += 1;
      events.push({
        microStep: index + 1,
        kind: "scheduler",
        tick: schedulerTicks,
        multiplier: schedulerMultiplier(schedulerTicks),
      });
    }

    if (isWindowEnd) {
      const gradientBeforeClip = gradientBuffer;
      const appliedGradient = clip(gradientBeforeClip, maxGradNorm);
      const multiplier = schedulerMultiplier(schedulerTicks);
      const learningRate = baseLearningRate * multiplier;
      const delta = -learningRate * appliedGradient;
      parameter += delta;
      optimizerSteps += 1;
      updates.push({
        optimizerStep: optimizerSteps,
        microStep: index + 1,
        windowSize,
        gradientBeforeClip,
        appliedGradient,
        learningRate,
        delta,
        parameter,
      });
      events.push({
        microStep: index + 1,
        kind: "optimizer",
        optimizerStep: optimizerSteps,
        gradientBeforeClip,
        appliedGradient,
        delta,
      });

      if (schedulerMode === "optimizer") {
        schedulerTicks += 1;
        events.push({
          microStep: index + 1,
          kind: "scheduler",
          tick: schedulerTicks,
          multiplier: schedulerMultiplier(schedulerTicks),
        });
      }
    }
  }

  const expectedOptimizerSteps = Math.ceil(gradients.length / accumulationSteps);
  const warnings = [];
  if (!divideLoss) {
    warnings.push(
      "Loss не делится на размер accumulation-window: gradient растёт вместе с числом micro-batch.",
    );
  }
  if (zeroMode === "micro") {
    warnings.push(
      "zero_grad вызывается на каждом micro-step: предыдущие micro-gradient стираются.",
    );
  }
  if (zeroMode === "never" && expectedOptimizerSteps > 1) {
    warnings.push(
      "Gradient не очищается между optimizer steps: новое окно содержит историю предыдущего.",
    );
  }
  if (schedulerMode === "micro") {
    warnings.push(
      `Scheduler сделал ${schedulerTicks} тиков вместо ${optimizerSteps}: LR schedule идёт слишком быстро.`,
    );
  }

  return {
    gradients: [...gradients],
    accumulationSteps,
    optimizerSteps,
    expectedOptimizerSteps,
    schedulerTicks,
    events,
    updates,
    finalParameter: parameter,
    warnings,
    invariantHolds:
      divideLoss &&
      zeroMode === "window" &&
      schedulerMode === "optimizer" &&
      optimizerSteps === expectedOptimizerSteps,
  };
}

