import { createDefaultState, normalizeState } from "./state.js";

const STORAGE_KEY = "do-models-only-memorize.learning-lab.v1";

export function loadProgress(storage = globalThis.localStorage) {
  try {
    const raw = storage?.getItem(STORAGE_KEY);
    return raw ? normalizeState(JSON.parse(raw)) : createDefaultState();
  } catch {
    return createDefaultState();
  }
}

export function saveProgress(state, storage = globalThis.localStorage) {
  try {
    storage?.setItem(STORAGE_KEY, JSON.stringify(state));
    return true;
  } catch {
    return false;
  }
}

export function clearProgress(storage = globalThis.localStorage) {
  try {
    storage?.removeItem(STORAGE_KEY);
    return true;
  } catch {
    return false;
  }
}

export function exportProgress(state) {
  return JSON.stringify(normalizeState(state), null, 2);
}

export function importProgress(serialized) {
  return normalizeState(JSON.parse(serialized));
}

export { STORAGE_KEY };

