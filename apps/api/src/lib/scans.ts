import { getSupabase, isSupabaseConfigured } from "./supabase.js";

const SIGNED_URL_TTL_SEC = 60 * 60;

type OcrRow = {
  raw_text: string;
  n_lines: number;
  n_glyphs: number;
  lines: unknown;
  glyphs: unknown;
  mode: string;
  device: string | null;
};

type DocumentationRow = {
  title: string;
};

type ScanRow = {
  id: string;
  public_id: string;
  status: string;
  avg_confidence: number | null;
  source_image_path: string | null;
  overlay_image_path: string | null;
  created_at: string;
  ocr_results: OcrRow | OcrRow[] | null;
  documentations: DocumentationRow | DocumentationRow[] | null;
};

export type ScanSummary = {
  id: string;
  publicId: string;
  status: string;
  avgConfidence: number | null;
  createdAt: string;
  previewText: string;
  documentationTitle?: string;
  sourceImageUrl?: string;
  overlayImageUrl?: string;
};

export type ScanDetail = ScanSummary & {
  result: {
    ok: true;
    text: string;
    nLines: number;
    nGlyphs: number;
    lines: unknown;
    glyphs: unknown;
    device: string;
    mode: "paper";
  };
};

function firstOcr(row: ScanRow): OcrRow | null {
  if (!row.ocr_results) return null;
  return Array.isArray(row.ocr_results) ? row.ocr_results[0] ?? null : row.ocr_results;
}

function firstDocumentationTitle(row: ScanRow): string | undefined {
  if (!row.documentations) return undefined;
  const doc = Array.isArray(row.documentations)
    ? row.documentations[0]
    : row.documentations;
  const title = doc?.title?.trim();
  return title || undefined;
}

async function signedUrl(
  bucket: string,
  path: string | null | undefined
): Promise<string | undefined> {
  if (!path) return undefined;
  const supabase = getSupabase();
  if (!supabase) return undefined;

  const { data, error } = await supabase.storage
    .from(bucket)
    .createSignedUrl(path, SIGNED_URL_TTL_SEC);

  if (error || !data?.signedUrl) {
    console.error("[scans] signed url failed:", error?.message);
    return undefined;
  }

  return data.signedUrl;
}

function mapSummary(row: ScanRow, urls: { source?: string; overlay?: string }): ScanSummary {
  const ocr = firstOcr(row);
  const preview = ocr?.raw_text?.trim().slice(0, 80) || "—";

  return {
    id: row.id,
    publicId: row.public_id,
    status: row.status,
    avgConfidence: row.avg_confidence,
    createdAt: row.created_at,
    previewText: preview,
    documentationTitle: firstDocumentationTitle(row),
    sourceImageUrl: urls.source,
    overlayImageUrl: urls.overlay,
  };
}

export async function listScans(limit = 50): Promise<ScanSummary[]> {
  if (!isSupabaseConfigured()) {
    throw new Error("Supabase is not configured");
  }

  const supabase = getSupabase();
  if (!supabase) throw new Error("Supabase is not configured");

  const { data, error } = await supabase
    .from("scans")
    .select(
      `
      id,
      public_id,
      status,
      avg_confidence,
      source_image_path,
      overlay_image_path,
      created_at,
      ocr_results (
        raw_text,
        n_lines,
        n_glyphs,
        lines,
        glyphs,
        mode,
        device
      ),
      documentations (
        title
      )
    `
    )
    .order("created_at", { ascending: false })
    .limit(limit);

  if (error) throw new Error(error.message);

  const rows = (data ?? []) as ScanRow[];
  const summaries: ScanSummary[] = [];

  for (const row of rows) {
    const [source, overlay] = await Promise.all([
      signedUrl("scan-images", row.source_image_path),
      signedUrl("scan-images", row.overlay_image_path),
    ]);
    summaries.push(mapSummary(row, { source, overlay }));
  }

  return summaries;
}

export async function getScanById(scanId: string): Promise<ScanDetail | null> {
  if (!isSupabaseConfigured()) {
    throw new Error("Supabase is not configured");
  }

  const supabase = getSupabase();
  if (!supabase) throw new Error("Supabase is not configured");

  const { data, error } = await supabase
    .from("scans")
    .select(
      `
      id,
      public_id,
      status,
      avg_confidence,
      source_image_path,
      overlay_image_path,
      created_at,
      ocr_results (
        raw_text,
        n_lines,
        n_glyphs,
        lines,
        glyphs,
        mode,
        device
      ),
      documentations (
        title
      )
    `
    )
    .eq("id", scanId)
    .maybeSingle();

  if (error) throw new Error(error.message);
  if (!data) return null;

  const row = data as ScanRow;
  const ocr = firstOcr(row);
  if (!ocr) return null;

  const [source, overlay] = await Promise.all([
    signedUrl("scan-images", row.source_image_path),
    signedUrl("scan-images", row.overlay_image_path),
  ]);

  const summary = mapSummary(row, { source, overlay });

  return {
    ...summary,
    result: {
      ok: true,
      text: ocr.raw_text,
      nLines: ocr.n_lines,
      nGlyphs: ocr.n_glyphs,
      lines: ocr.lines,
      glyphs: ocr.glyphs,
      device: ocr.device ?? "unknown",
      mode: "paper",
    },
  };
}

export type DeleteScanResult =
  | { ok: true; id: string }
  | { ok: false; error: "not_found" | "delete_failed"; detail?: string };

/**
 * Delete a scan row (cascades OCR + documentation rows) and best-effort
 * remove related storage objects.
 */
export async function deleteScanById(scanId: string): Promise<DeleteScanResult> {
  if (!isSupabaseConfigured()) {
    throw new Error("Supabase is not configured");
  }

  const supabase = getSupabase();
  if (!supabase) throw new Error("Supabase is not configured");

  const { data: scan, error: lookupErr } = await supabase
    .from("scans")
    .select(
      `
      id,
      source_image_path,
      overlay_image_path,
      documentations (
        id,
        documentation_images (
          storage_path
        )
      )
    `
    )
    .eq("id", scanId)
    .maybeSingle();

  if (lookupErr) {
    return { ok: false, error: "delete_failed", detail: lookupErr.message };
  }
  if (!scan) {
    return { ok: false, error: "not_found" };
  }

  const scanPaths = [scan.source_image_path, scan.overlay_image_path].filter(
    (p): p is string => Boolean(p)
  );

  type DocImage = { storage_path?: string | null };
  type DocNode = {
    id?: string;
    documentation_images?: DocImage | DocImage[] | null;
  };

  const docsRaw = (scan as { documentations?: DocNode | DocNode[] | null }).documentations;
  const docs = docsRaw ? (Array.isArray(docsRaw) ? docsRaw : [docsRaw]) : [];
  const docPaths: string[] = [];
  for (const doc of docs) {
    const images = doc.documentation_images
      ? Array.isArray(doc.documentation_images)
        ? doc.documentation_images
        : [doc.documentation_images]
      : [];
    for (const img of images) {
      if (img.storage_path) docPaths.push(img.storage_path);
    }
  }

  if (scanPaths.length) {
    const { error: scanStorageErr } = await supabase.storage
      .from("scan-images")
      .remove(scanPaths);
    if (scanStorageErr) {
      console.error("[scans] storage cleanup (scan-images) failed:", scanStorageErr.message);
    }
  }

  if (docPaths.length) {
    const { error: docStorageErr } = await supabase.storage
      .from("documentation-images")
      .remove(docPaths);
    if (docStorageErr) {
      console.error(
        "[scans] storage cleanup (documentation-images) failed:",
        docStorageErr.message
      );
    }
  }

  const { error: deleteErr } = await supabase.from("scans").delete().eq("id", scanId);
  if (deleteErr) {
    return { ok: false, error: "delete_failed", detail: deleteErr.message };
  }

  return { ok: true, id: scanId };
}
