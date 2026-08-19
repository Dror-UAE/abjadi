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

export const VERSION_PATH = join(MODEL_ROOT, "VERSION.json");

export type OcrMode = "paper" | "stone";

/** Abjadi scans stone inscriptions by default. */
export const DEFAULT_OCR_MODE: OcrMode = "stone";

const PAPER_WEIGHTS = ["model/musnad_final.pth"] as const;

const STONE_WEIGHTS = [
  "model/musnad_final.pth",
  "model/class_prototypes.pt",
  "model/shape_bank.pt",
  "model/letter_boundary_v2.pth",
] as const;

export type ModelInfo = {
  ready: boolean;
  version?: string;
  syncedAt?: string;
  weightFiles?: string[];
  stoneReady?: boolean;
  paperReady?: boolean;
};

function hasWeights(weights: readonly string[]): boolean {
  return weights.every((rel) => existsSync(join(MODEL_ROOT, rel)));
}

export function isModelReady(mode: OcrMode = DEFAULT_OCR_MODE): boolean {
  if (
    !existsSync(MODEL_PYTHON) ||
    !existsSync(PREDICT_CLI)
  ) {
    return false;
  }

  return hasWeights(mode === "stone" ? STONE_WEIGHTS : PAPER_WEIGHTS);
}

type VersionFile = {
  version?: string;
  synced_at_utc?: string;
  files?: Record<string, unknown>;
};

export function getModelInfo(): ModelInfo {
  const paperReady = isModelReady("paper");
  const stoneReady = isModelReady("stone");
  const ready = isModelReady(DEFAULT_OCR_MODE);

  if (!existsSync(VERSION_PATH)) {
    return { ready, paperReady, stoneReady };
  }

  try {
    const raw = JSON.parse(readFileSync(VERSION_PATH, "utf8")) as VersionFile;
    return {
      ready,
      paperReady,
      stoneReady,
      version: raw.version,
      syncedAt: raw.synced_at_utc,
      weightFiles: raw.files ? Object.keys(raw.files) : undefined,
    };
  } catch {
    return { ready, paperReady, stoneReady };
  }
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

export function parseOcrMode(raw: string | undefined | null): OcrMode {
  const value = raw?.trim().toLowerCase();
  if (value === "paper") return "paper";
  return "stone";
}
