import { randomUUID } from "node:crypto";
import { readFile, mkdir, unlink, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import sharp from "sharp";

import {
  DEFAULT_OCR_MODE,
  getArabicName,
  isModelReady,
  type OcrMode,
} from "./model-paths.js";
import { enrichOcrWithArabic } from "./musnad-translate.js";
import { runOcrWorker } from "./ocr-worker.js";

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

async function readOverlayBase64(
  overlayPath: string | undefined,
  outDir: string
): Promise<string | undefined> {
  const candidates = [
    overlayPath,
    join(outDir, "overlay.png"),
    join(outDir, "detect", "overlay.png"),
  ].filter(Boolean) as string[];

  for (const p of candidates) {
    try {
      const bytes = await readFile(p);
      if (bytes.byteLength > 0) return bytes.toString("base64");
    } catch {
      // try next
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
          : "Run `pnpm setup:model` from the repo root (requires Python 3.x + torch).",
    };
  }

  const id = randomUUID();
  const workDir = join(tmpdir(), "abjadi-ocr", id);
  await mkdir(workDir, { recursive: true });

  // Always write as PNG — OpenCV on the worker reads PNG reliably across all
  // camera/gallery sources. JPEG from mobile can have sub-encodings that
  // cv2.imread silently fails on, returning None and raising FileNotFoundError.
  const imagePath = join(workDir, "input.png");
  const outDir = join(workDir, "out");

  try {
    // Convert to PNG so OpenCV (cv2.imread) reads it reliably regardless of
    // source encoding (mobile JPEG sub-formats, WEBP, etc.)
    const pngBytes = await sharp(imageBytes).png().toBuffer();
    await writeFile(imagePath, pngBytes);

    const response = await runOcrWorker(imagePath, outDir, mode);

    if (!response.ok) {
      return {
        ok: false,
        error: "ocr_failed",
        detail: response.error ?? "Unknown OCR error",
      };
    }

    const raw = response as unknown as RawOcrPayload;
    const overlayBase64 = await readOverlayBase64(raw.overlay_path, outDir);
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
