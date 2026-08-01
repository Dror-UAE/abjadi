import { randomUUID } from "node:crypto";

import type { OcrResult } from "./run-ocr.js";
import { getSupabase, isSupabaseConfigured, makePublicId } from "./supabase.js";

export type PersistedScan = {
  scanId: string;
  publicId: string;
};

function avgConfidence(result: OcrResult): number {
  const glyphs = result.glyphs.filter((g) => !g.isSeparator && g.character !== "?");
  if (!glyphs.length) return 0;
  const sum = glyphs.reduce((acc, g) => acc + Math.max(0, Math.min(1, g.confidence)), 0);
  return Math.round((sum / glyphs.length) * 100);
}

function mimeFromFilename(filename: string): string {
  const lower = filename.toLowerCase();
  if (lower.endsWith(".png")) return "image/png";
  if (lower.endsWith(".webp")) return "image/webp";
  return "image/jpeg";
}

function extFromFilename(filename: string): string {
  const lower = filename.toLowerCase();
  if (lower.endsWith(".png")) return "png";
  if (lower.endsWith(".webp")) return "webp";
  return "jpg";
}

/**
 * Persist OCR output + images to Supabase.
 * No-ops (returns null) when Supabase env is missing — OCR still succeeds.
 */
export async function persistScanAndOcr(
  imageBytes: Buffer,
  filename: string,
  result: OcrResult
): Promise<PersistedScan | null> {
  if (!isSupabaseConfigured()) return null;

  const supabase = getSupabase();
  if (!supabase) return null;

  const publicId = makePublicId();
  const scanId = randomUUID();
  const folder = `anonymous/${scanId}`;
  const sourceExt = extFromFilename(filename);
  const sourcePath = `${folder}/source.${sourceExt}`;
  let overlayPath: string | null = null;

  const { error: sourceErr } = await supabase.storage
    .from("scan-images")
    .upload(sourcePath, imageBytes, {
      contentType: mimeFromFilename(filename),
      upsert: false,
    });

  if (sourceErr) {
    console.error("[supabase] source upload failed:", sourceErr.message);
    throw new Error(`Storage upload failed: ${sourceErr.message}`);
  }

  if (result.overlayBase64) {
    overlayPath = `${folder}/overlay.png`;
    const overlayBytes = Buffer.from(result.overlayBase64, "base64");
    const { error: overlayErr } = await supabase.storage
      .from("scan-images")
      .upload(overlayPath, overlayBytes, {
        contentType: "image/png",
        upsert: false,
      });
    if (overlayErr) {
      console.error("[supabase] overlay upload failed:", overlayErr.message);
      overlayPath = null;
    }
  }

  const { error: scanErr } = await supabase.from("scans").insert({
    id: scanId,
    public_id: publicId,
    source_image_path: sourcePath,
    overlay_image_path: overlayPath,
    status: "analyzed",
    avg_confidence: avgConfidence(result),
    api_device: result.device,
  });

  if (scanErr) {
    console.error("[supabase] scan insert failed:", scanErr.message);
    throw new Error(`Scan insert failed: ${scanErr.message}`);
  }

  const payloadForDb = {
    ...result,
    overlayBase64: result.overlayBase64 ? "[stored-in-storage]" : undefined,
  };

  const { error: ocrErr } = await supabase.from("ocr_results").insert({
    scan_id: scanId,
    raw_text: result.text,
    n_lines: result.nLines,
    n_glyphs: result.nGlyphs,
    lines: result.lines,
    glyphs: result.glyphs,
    mode: result.mode,
    device: result.device,
    raw_payload: payloadForDb,
  });

  if (ocrErr) {
    console.error("[supabase] ocr_results insert failed:", ocrErr.message);
    throw new Error(`OCR insert failed: ${ocrErr.message}`);
  }

  return { scanId, publicId };
}

export type DocumentationInput = {
  scanId: string;
  title: string;
  scriptType?: string;
  language?: string;
  region?: string;
  country?: string;
  description?: string;
  condition?: string;
  imageSource?: string;
  ocrTextEdited?: string;
  notes?: string;
  confidence?: number;
  /** Extra images as base64 (optional) */
  extraImages?: Array<{ base64: string; filename?: string; mimeType?: string }>;
  /** Extra documents (PDF, etc.) as base64 (optional) */
  extraDocuments?: Array<{ base64: string; filename?: string; mimeType?: string }>;
};

export type DocumentationResult = {
  id: string;
  publicId: string;
  status: string;
};

function extensionForAttachment(mime: string, filename?: string): string {
  const fromName = filename?.split(".").pop()?.toLowerCase();
  if (fromName && /^[a-z0-9]{1,8}$/.test(fromName)) return fromName;

  const lower = mime.toLowerCase();
  if (lower.includes("pdf")) return "pdf";
  if (lower.includes("png")) return "png";
  if (lower.includes("webp")) return "webp";
  if (lower.includes("jpeg") || lower.includes("jpg")) return "jpg";
  if (lower.includes("msword")) return "doc";
  if (lower.includes("wordprocessingml")) return "docx";
  if (lower.includes("ms-excel") || lower.includes("spreadsheetml")) return "xlsx";
  if (lower.includes("text/plain")) return "txt";
  return "bin";
}

export async function persistDocumentation(
  input: DocumentationInput
): Promise<DocumentationResult> {
  if (!isSupabaseConfigured()) {
    throw new Error("Supabase is not configured");
  }

  const supabase = getSupabase();
  if (!supabase) {
    throw new Error("Supabase is not configured");
  }

  const { data: scan, error: scanLookupErr } = await supabase
    .from("scans")
    .select("id, public_id, source_image_path")
    .eq("id", input.scanId)
    .maybeSingle();

  if (scanLookupErr || !scan) {
    throw new Error("Scan not found");
  }

  const docId = randomUUID();
  const publicId = makePublicId("DOC");

  const { error: docErr } = await supabase.from("documentations").insert({
    id: docId,
    scan_id: input.scanId,
    public_id: publicId,
    title: input.title.trim() || "نقش بدون عنوان",
    script_type: input.scriptType?.trim() ?? "",
    language: input.language?.trim() ?? "",
    region: input.region?.trim() ?? "",
    country: input.country?.trim() ?? "",
    description: input.description?.trim() ?? "",
    condition: input.condition?.trim() ?? "",
    image_source: input.imageSource?.trim() ?? "",
    ocr_text_edited: input.ocrTextEdited?.trim() ?? "",
    notes: input.notes?.trim() ?? "",
    confidence: input.confidence ?? null,
    status: "under_review",
  });

  if (docErr) {
    throw new Error(`Documentation insert failed: ${docErr.message}`);
  }

  const imageRows: Array<{
    documentation_id: string;
    storage_path: string;
    is_primary: boolean;
    sort_order: number;
  }> = [];

  if (scan.source_image_path) {
    imageRows.push({
      documentation_id: docId,
      storage_path: scan.source_image_path,
      is_primary: true,
      sort_order: 0,
    });
  }

  let order = 1;

  const extras = [...(input.extraImages ?? []), ...(input.extraDocuments ?? [])];
  for (const extra of extras) {
    const raw = extra.base64.includes(",")
      ? extra.base64.slice(extra.base64.indexOf(",") + 1)
      : extra.base64;
    const bytes = Buffer.from(raw, "base64");
    if (!bytes.byteLength) continue;

    const mime = extra.mimeType || "image/jpeg";
    const ext = extensionForAttachment(mime, extra.filename);
    const path = `anonymous/${docId}/extra-${String(order).padStart(2, "0")}.${ext}`;

    const { error: upErr } = await supabase.storage
      .from("documentation-images")
      .upload(path, bytes, { contentType: mime, upsert: false });

    if (upErr) {
      console.error("[supabase] extra attachment upload failed:", upErr.message);
      continue;
    }

    imageRows.push({
      documentation_id: docId,
      storage_path: path,
      is_primary: false,
      sort_order: order,
    });
    order += 1;
  }

  if (imageRows.length) {
    const { error: imgErr } = await supabase.from("documentation_images").insert(imageRows);
    if (imgErr) {
      console.error("[supabase] documentation_images insert failed:", imgErr.message);
    }
  }

  await supabase.from("scans").update({ status: "documented" }).eq("id", input.scanId);

  return { id: docId, publicId, status: "under_review" };
}
