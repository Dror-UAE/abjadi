import { existsSync, readFileSync } from "node:fs";
import { dirname, join, resolve } from "node:path";
import { fileURLToPath } from "node:url";

const here = dirname(fileURLToPath(import.meta.url));

/** Monorepo root: apps/api/src/lib → ../../../../ */
export const REPO_ROOT = resolve(here, "../../../../");

export const MODEL_ROOT = join(REPO_ROOT, "models", "musnad-ocr-model");

export const MODEL_PYTHON = join(MODEL_ROOT, ".venv", "bin", "python");

export const PREDICT_CLI = join(MODEL_ROOT, "predict_cli.py");

export const LABELS_PATH = join(MODEL_ROOT, "config", "labels.json");

export function isModelReady(): boolean {
  return (
    existsSync(MODEL_PYTHON) &&
    existsSync(PREDICT_CLI) &&
    existsSync(join(MODEL_ROOT, "model", "musnad_final.pth"))
  );
}

type LabelEntry = {
  character?: string;
  name?: string;
  arabic_name?: string;
};

type LabelsFile = {
  entries?: LabelEntry[];
};

let arabicNameByChar: Map<string, string> | null = null;

export function getArabicName(character: string): string | undefined {
  if (!arabicNameByChar) {
    arabicNameByChar = new Map();
    try {
      const raw = JSON.parse(readFileSync(LABELS_PATH, "utf8")) as LabelsFile;
      for (const entry of raw.entries ?? []) {
        if (entry.character && entry.arabic_name) {
          arabicNameByChar.set(entry.character, entry.arabic_name);
        }
      }
    } catch {
      // Labels are optional enrichment; OCR still works without them.
    }
  }
  return arabicNameByChar.get(character);
}
