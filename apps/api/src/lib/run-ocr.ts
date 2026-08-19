import { spawn } from "node:child_process";
import { randomUUID } from "node:crypto";
import { readFile, mkdir, unlink, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";

import {
  DEFAULT_OCR_MODE,
  getArabicName,
  isModelReady,
  MODEL_PYTHON,
  MODEL_ROOT,
  PREDICT_CLI,
  type OcrMode,
} from "./model-paths.js";
import { enrichOcrWithArabic } from "./musnad-translate.js";

export type { OcrMode };
export { DEFAULT_OCR_MODE };

export type OcrGlyph = {
  character: string;
  display?: string;
  name?: string;
  arabicName?: string;
  arabicLetter?: string;
  confidence: number;
  trusted?: boolean;
  isSeparator?: boolean;
};

export type OcrLine = {
  text: string;
  glyphs: OcrGlyph[];
};

export type OcrResult = {
  ok: true;
  text: string;
  arabicText: string;
  arabicLines: string[];
  nLines: number;
  nGlyphs: number;
  lines: OcrLine[];
  glyphs: OcrGlyph[];
  device: string;
  mode: OcrMode;
  overlayBase64?: string;
};

export type OcrError = {
  ok: false;
  error: string;
  detail?: string;
};

type RawGlyph = {
  character?: string;
  display?: string;
  name?: string;
  confidence?: number;
  trusted?: boolean;
  is_separator?: boolean;
};

type RawLine = {
  text?: string;
  glyphs?: RawGlyph[];
};

type RawOcrPayload = {
  ok?: boolean;
  text?: string;
  n_lines?: number;
  n_glyphs?: number;
  lines?: RawLine[];
  glyphs?: RawGlyph[];
  device?: string;
  overlay_path?: string;
  mode?: string;
};

function mapGlyph(raw: RawGlyph): OcrGlyph {
  const character = raw.character ?? "?";
  return {
    character,
    display: raw.display,
    name: raw.name,
    arabicName: getArabicName(character),
    confidence: Number(raw.confidence ?? 0),
    trusted: raw.trusted,
    isSeparator: Boolean(raw.is_separator),
  };
}

function flattenGlyphs(lines: OcrLine[], topLevel: OcrGlyph[]): OcrGlyph[] {
  if (topLevel.length > 0) return topLevel;
  return lines.flatMap((line) => line.glyphs);
}

function normalizeOcrResult(
  raw: RawOcrPayload,
  mode: OcrMode,
  overlayBase64?: string
): OcrResult {
  const lines = (raw.lines ?? []).map((line) => ({
    text: line.text ?? "",
    glyphs: (line.glyphs ?? []).map(mapGlyph),
  }));
  const glyphs = flattenGlyphs(lines, (raw.glyphs ?? []).map(mapGlyph));

  return enrichOcrWithArabic({
    ok: true,
    text: raw.text ?? lines.map((l) => l.text).join("\n"),
    nLines: raw.n_lines ?? lines.length,
    nGlyphs: raw.n_glyphs ?? glyphs.filter((g) => !g.isSeparator).length,
    lines,
    glyphs,
    device: raw.device ?? "cpu",
    mode,
    ...(overlayBase64 ? { overlayBase64 } : {}),
  });
}

function runPredictCli(
  imagePath: string,
  outDir: string,
  mode: OcrMode
): Promise<RawOcrPayload> {
  const cliArgs = [
    PREDICT_CLI,
    mode === "stone" ? "--stone" : "--paper",
    "--json",
    "--out-dir",
    outDir,
    imagePath,
  ];

  return new Promise((resolve, reject) => {
    const child = spawn(MODEL_PYTHON, cliArgs, {
      cwd: MODEL_ROOT,
      env: { ...process.env, PYTHONUNBUFFERED: "1" },
    });

    let stdout = "";
    let stderr = "";

    child.stdout.on("data", (chunk: Buffer) => {
      stdout += chunk.toString("utf8");
    });
    child.stderr.on("data", (chunk: Buffer) => {
      stderr += chunk.toString("utf8");
    });

    child.on("error", (err) => reject(err));

    child.on("close", (code) => {
      if (code !== 0) {
        reject(
          new Error(
            stderr.trim() || stdout.trim() || `OCR process exited with code ${code}`
          )
        );
        return;
      }

      const trimmed = stdout.trim();
      const start = trimmed.indexOf("{");
      const end = trimmed.lastIndexOf("}");
      if (start < 0 || end < start) {
        reject(new Error(`OCR returned no JSON. stderr: ${stderr.slice(0, 500)}`));
        return;
      }

      try {
        resolve(JSON.parse(trimmed.slice(start, end + 1)) as RawOcrPayload);
      } catch (err) {
        reject(
          new Error(
            `Failed to parse OCR JSON: ${err instanceof Error ? err.message : String(err)}`
          )
        );
      }
    });
  });
}

async function readOverlayBase64(
  raw: RawOcrPayload,
  outDir: string
): Promise<string | undefined> {
  const candidates = [
    raw.overlay_path,
    join(outDir, "overlay.png"),
    join(outDir, "detect", "overlay.png"),
  ].filter(Boolean) as string[];

  for (const overlayPath of candidates) {
    try {
      const overlayBytes = await readFile(overlayPath);
      if (overlayBytes.byteLength > 0) {
        return overlayBytes.toString("base64");
      }
    } catch {
      // try next candidate
    }
  }

  return undefined;
}

export async function runOcr(
  imageBytes: Buffer,
  filename: string,
  mode: OcrMode = DEFAULT_OCR_MODE
): Promise<OcrResult | OcrError> {
  if (!isModelReady(mode)) {
    return {
      ok: false,
      error: "model_not_ready",
      detail:
        mode === "stone"
          ? "Stone OCR weights missing. Run `pnpm setup:model` from the repo root."
          : "Run `pnpm setup:model` from the repo root (requires Python 3.11 + torch).",
    };
  }

  const id = randomUUID();
  const workDir = join(tmpdir(), "abjadi-ocr", id);
  await mkdir(workDir, { recursive: true });

  const ext = filename.toLowerCase().endsWith(".png")
    ? ".png"
    : filename.toLowerCase().endsWith(".webp")
      ? ".webp"
      : ".jpg";
  const imagePath = join(workDir, `input${ext}`);
  const outDir = join(workDir, "out");

  try {
    await writeFile(imagePath, imageBytes);
    const raw = await runPredictCli(imagePath, outDir, mode);
    const overlayBase64 = await readOverlayBase64(raw, outDir);
    return normalizeOcrResult(raw, mode, overlayBase64);
  } catch (err) {
    return {
      ok: false,
      error: "ocr_failed",
      detail: err instanceof Error ? err.message : String(err),
    };
  } finally {
    try {
      await unlink(imagePath);
    } catch {
      // ignore
    }
  }
}

/** @deprecated Use runOcr(..., "paper") */
export async function runPaperOcr(
  imageBytes: Buffer,
  filename: string
): Promise<OcrResult | OcrError> {
  return runOcr(imageBytes, filename, "paper");
}
