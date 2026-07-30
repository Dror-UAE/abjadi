import "./load-env.js";

import { serve } from "@hono/node-server";
import { Hono, type Context } from "hono";
import { cors } from "hono/cors";

import { isModelReady } from "./lib/model-paths.js";
import { createOcrJob, getOcrJob, runOcrAndPersist, warmOcrModel } from "./lib/ocr-jobs.js";
import { persistDocumentation } from "./lib/persist.js";
import { getScanById, listScans } from "./lib/scans.js";
import { isSupabaseConfigured } from "./lib/supabase.js";

const app = new Hono();
const MAX_BYTES = 12 * 1024 * 1024;

app.use("*", cors());

app.get("/", (c) => {
  return c.json({
    name: "abjadi-api",
    status: "ok",
    modelReady: isModelReady(),
    supabaseConfigured: isSupabaseConfigured(),
  });
});

app.get("/health", (c) => {
  const modelReady = isModelReady();
  return c.json({
    status: modelReady ? "healthy" : "degraded",
    modelReady,
    supabaseConfigured: isSupabaseConfigured(),
  });
});

type JsonOcrBody = {
  imageBase64?: string;
  filename?: string;
  mimeType?: string;
};

type ImageParseResult =
  | { bytes: Buffer; filename: string }
  | { error: string; detail?: string; status: 400 | 413 };

async function readImageFromRequest(c: Context): Promise<ImageParseResult> {
  const contentType = c.req.header("content-type") ?? "";

  if (contentType.includes("application/json")) {
    const body = await c.req.json<JsonOcrBody>();
    const imageBase64 = body.imageBase64?.trim();
    if (!imageBase64) {
      return {
        error: "missing_image",
        detail: "Expected JSON field `imageBase64`.",
        status: 400,
      };
    }

    const raw = imageBase64.includes(",")
      ? imageBase64.slice(imageBase64.indexOf(",") + 1)
      : imageBase64;

    let bytes: Buffer;
    try {
      bytes = Buffer.from(raw, "base64");
    } catch {
      return { error: "invalid_base64", detail: "Could not decode imageBase64.", status: 400 };
    }

    if (bytes.byteLength === 0) {
      return { error: "empty_image", status: 400 };
    }
    if (bytes.byteLength > MAX_BYTES) {
      return {
        error: "image_too_large",
        detail: `Max size is ${MAX_BYTES} bytes.`,
        status: 413,
      };
    }

    return {
      bytes,
      filename: body.filename || "scan.jpg",
    };
  }

  const body = await c.req.parseBody({ all: true });
  const file = body.image ?? body.file;

  if (!file || typeof file === "string") {
    return {
      error: "missing_image",
      detail: "Expected JSON `imageBase64` or multipart field `image`.",
      status: 400,
    };
  }

  const upload = file as File;
  const arrayBuffer = await upload.arrayBuffer();
  const bytes = Buffer.from(arrayBuffer);

  if (bytes.byteLength === 0) {
    return { error: "empty_image", status: 400 };
  }
  if (bytes.byteLength > MAX_BYTES) {
    return {
      error: "image_too_large",
      detail: `Max size is ${MAX_BYTES} bytes.`,
      status: 413,
    };
  }

  return { bytes, filename: upload.name || "scan.jpg" };
}

app.post("/ocr", async (c) => {
  const parsed = await readImageFromRequest(c);
  if ("error" in parsed) {
    return c.json(
      { ok: false, error: parsed.error, detail: parsed.detail },
      parsed.status
    );
  }

  const asyncMode =
    c.req.query("async") === "1" ||
    c.req.header("x-abjadi-ocr-async") === "1";

  if (asyncMode) {
    try {
      const job = await createOcrJob(parsed.bytes, parsed.filename);
      return c.json(
        {
          ok: true,
          async: true,
          jobId: job.id,
          status: job.status,
          pollUrl: `/ocr/jobs/${job.id}`,
        },
        202
      );
    } catch (err) {
      const detail = err instanceof Error ? err.message : String(err);
      return c.json(
        { ok: false, error: "job_create_failed", detail },
        500
      );
    }
  }

  const result = await runOcrAndPersist(parsed.bytes, parsed.filename);

  if (!result.ok) {
    const status = result.error === "model_not_ready" ? 503 : 500;
    return c.json(result, status);
  }

  return c.json(result);
});

app.get("/ocr/jobs/:id", async (c) => {
  const id = c.req.param("id")?.trim();
  if (!id) {
    return c.json({ ok: false, error: "missing_job_id" }, 400);
  }

  const job = await getOcrJob(id);
  if (!job) {
    return c.json({ ok: false, error: "job_not_found" }, 404);
  }

  if (job.status === "succeeded" && job.result) {
    return c.json({
      ...job.result,
      jobId: job.id,
      status: job.status,
    });
  }

  if (job.status === "failed") {
    return c.json(
      {
        ok: false,
        jobId: job.id,
        status: job.status,
        error: job.error ?? "ocr_failed",
        detail: job.detail,
      },
      500
    );
  }

  return c.json({
    ok: true,
    jobId: job.id,
    status: job.status,
    pending: true,
  });
});

app.get("/scans", async (c) => {
  if (!isSupabaseConfigured()) {
    return c.json(
      {
        ok: false,
        error: "supabase_not_configured",
        detail: "Set SUPABASE_URL and SUPABASE_SECRET_KEY (or SUPABASE_SERVICE_ROLE_KEY) on the API.",
      },
      503
    );
  }

  const limit = Math.min(Number(c.req.query("limit") ?? 50), 100);

  try {
    const scans = await listScans(limit);
    return c.json({ ok: true, scans });
  } catch (err) {
    const message = err instanceof Error ? err.message : String(err);
    return c.json({ ok: false, error: "list_scans_failed", detail: message }, 500);
  }
});

app.get("/scans/:id", async (c) => {
  if (!isSupabaseConfigured()) {
    return c.json(
      {
        ok: false,
        error: "supabase_not_configured",
        detail: "Set SUPABASE_URL and SUPABASE_SECRET_KEY (or SUPABASE_SERVICE_ROLE_KEY) on the API.",
      },
      503
    );
  }

  const id = c.req.param("id")?.trim();
  if (!id) {
    return c.json({ ok: false, error: "missing_scan_id" }, 400);
  }

  try {
    const scan = await getScanById(id);
    if (!scan) {
      return c.json({ ok: false, error: "scan_not_found" }, 404);
    }
    return c.json({ ok: true, scan });
  } catch (err) {
    const message = err instanceof Error ? err.message : String(err);
    return c.json({ ok: false, error: "get_scan_failed", detail: message }, 500);
  }
});

type DocumentationBody = {
  scanId?: string;
  title?: string;
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
  extraImages?: Array<{ base64: string; filename?: string; mimeType?: string }>;
};

app.post("/documentations", async (c) => {
  if (!isSupabaseConfigured()) {
    return c.json(
      {
        ok: false,
        error: "supabase_not_configured",
        detail: "Set SUPABASE_URL and SUPABASE_SECRET_KEY (or SUPABASE_SERVICE_ROLE_KEY) on the API.",
      },
      503
    );
  }

  const body = await c.req.json<DocumentationBody>();
  if (!body.scanId?.trim()) {
    return c.json({ ok: false, error: "missing_scan_id" }, 400);
  }

  try {
    const saved = await persistDocumentation({
      scanId: body.scanId.trim(),
      title: body.title ?? "",
      scriptType: body.scriptType,
      language: body.language,
      region: body.region,
      country: body.country,
      description: body.description,
      condition: body.condition,
      imageSource: body.imageSource,
      ocrTextEdited: body.ocrTextEdited,
      notes: body.notes,
      confidence: body.confidence,
      extraImages: body.extraImages,
    });

    return c.json({ ok: true, ...saved });
  } catch (err) {
    const message = err instanceof Error ? err.message : String(err);
    const status = message.includes("not found") ? 404 : 500;
    return c.json({ ok: false, error: "documentation_failed", detail: message }, status);
  }
});

const port = Number(process.env.PORT ?? 3500);
const hostname = process.env.HOST ?? "0.0.0.0";

serve({ fetch: app.fetch, port, hostname }, () => {
  console.log(`API listening on http://${hostname}:${port}`);
  console.log(`Model ready: ${isModelReady()}`);
  console.log(`Supabase: ${isSupabaseConfigured() ? "configured" : "not configured"}`);
  warmOcrModel();
});
