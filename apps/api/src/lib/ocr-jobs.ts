import { randomUUID } from "node:crypto";

import { persistScanAndOcr } from "./persist.js";
import { runPaperOcr, type OcrError, type OcrResult } from "./run-ocr.js";
import { isModelReady, MODEL_PYTHON, MODEL_ROOT } from "./model-paths.js";
import { isSupabaseConfigured } from "./supabase.js";
import { spawn } from "node:child_process";

export type OcrJobStatus = "queued" | "running" | "succeeded" | "failed";

export type OcrJobSuccess = OcrResult & {
  scanId?: string;
  publicId?: string;
  persisted?: boolean;
  persistError?: string;
};

export type OcrJobRecord = {
  id: string;
  status: OcrJobStatus;
  createdAt: number;
  updatedAt: number;
  error?: string;
  detail?: string;
  result?: OcrJobSuccess;
};

const jobs = new Map<string, OcrJobRecord>();
const MAX_JOBS = 100;
const JOB_TTL_MS = 30 * 60 * 1000;

function pruneJobs(): void {
  const cutoff = Date.now() - JOB_TTL_MS;
  for (const [id, job] of jobs) {
    if (job.updatedAt < cutoff) jobs.delete(id);
  }
  while (jobs.size > MAX_JOBS) {
    const oldest = [...jobs.entries()].sort((a, b) => a[1].createdAt - b[1].createdAt)[0];
    if (!oldest) break;
    jobs.delete(oldest[0]);
  }
}

function touch(job: OcrJobRecord): void {
  job.updatedAt = Date.now();
  jobs.set(job.id, job);
}

async function finalizeOcr(
  imageBytes: Buffer,
  filename: string,
  result: OcrResult
): Promise<OcrJobSuccess> {
  let scanId: string | undefined;
  let publicId: string | undefined;
  let persistError: string | undefined;

  if (isSupabaseConfigured()) {
    try {
      const saved = await persistScanAndOcr(imageBytes, filename, result);
      if (saved) {
        scanId = saved.scanId;
        publicId = saved.publicId;
      }
    } catch (err) {
      persistError = err instanceof Error ? err.message : String(err);
      console.error("[ocr-job] persist failed:", persistError);
    }
  }

  return {
    ...result,
    scanId,
    publicId,
    persisted: Boolean(scanId),
    ...(persistError ? { persistError } : {}),
  };
}

async function runJob(jobId: string, imageBytes: Buffer, filename: string): Promise<void> {
  const job = jobs.get(jobId);
  if (!job) return;

  job.status = "running";
  touch(job);

  try {
    const result = await runPaperOcr(imageBytes, filename);
    if (!result.ok) {
      job.status = "failed";
      job.error = result.error;
      job.detail = result.detail;
      touch(job);
      return;
    }

    job.result = await finalizeOcr(imageBytes, filename, result);
    job.status = "succeeded";
    touch(job);
  } catch (err) {
    job.status = "failed";
    job.error = "ocr_failed";
    job.detail = err instanceof Error ? err.message : String(err);
    touch(job);
  }
}

export function createOcrJob(imageBytes: Buffer, filename: string): OcrJobRecord {
  pruneJobs();
  const id = randomUUID();
  const now = Date.now();
  const job: OcrJobRecord = {
    id,
    status: "queued",
    createdAt: now,
    updatedAt: now,
  };
  jobs.set(id, job);

  // Fire-and-forget; client polls GET /ocr/jobs/:id
  void runJob(id, imageBytes, filename);

  return job;
}

export function getOcrJob(id: string): OcrJobRecord | undefined {
  pruneJobs();
  return jobs.get(id);
}

/** Import torch/model in a short-lived process so first OCR is warmer. */
export function warmOcrModel(): void {
  if (!isModelReady()) {
    console.warn("[ocr] skip warmup — model not ready");
    return;
  }

  const child = spawn(
    MODEL_PYTHON,
    [
      "-c",
      "import torch; print('torch', torch.__version__); from inference.paper_ocr import PaperOcrEngine; print('paper_ocr_import_ok')",
    ],
    {
      cwd: MODEL_ROOT,
      env: { ...process.env, PYTHONUNBUFFERED: "1" },
    }
  );

  child.stdout.on("data", (chunk: Buffer) => {
    console.log("[ocr-warmup]", chunk.toString("utf8").trim());
  });
  child.stderr.on("data", (chunk: Buffer) => {
    console.warn("[ocr-warmup]", chunk.toString("utf8").trim());
  });
  child.on("close", (code) => {
    console.log(`[ocr-warmup] finished code=${code}`);
  });
}

export async function runOcrAndPersist(
  imageBytes: Buffer,
  filename: string
): Promise<OcrJobSuccess | OcrError> {
  const result = await runPaperOcr(imageBytes, filename);
  if (!result.ok) return result;
  return finalizeOcr(imageBytes, filename, result);
}
