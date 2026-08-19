import { randomUUID } from "node:crypto";

import { persistScanAndOcr } from "./persist.js";
import {
  DEFAULT_OCR_MODE,
  isModelReady,
  type OcrMode,
} from "./model-paths.js";
import { runOcr, type OcrError, type OcrResult } from "./run-ocr.js";
import { warmWorker } from "./ocr-worker.js";
import { getSupabase, isSupabaseConfigured } from "./supabase.js";

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
  mode: OcrMode;
  createdAt: number;
  updatedAt: number;
  error?: string;
  detail?: string;
  result?: OcrJobSuccess;
};

const jobs = new Map<string, OcrJobRecord>();
const MAX_JOBS = 100;
const JOB_TTL_MS = 30 * 60 * 1000;
const JOB_BUCKET = "scan-images";
const JOB_PREFIX = "_ocr-jobs";

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

async function saveJob(job: OcrJobRecord): Promise<void> {
  job.updatedAt = Date.now();
  jobs.set(job.id, job);

  const supabase = getSupabase();
  if (!supabase) return;

  const path = `${JOB_PREFIX}/${job.id}.json`;
  const body = Buffer.from(JSON.stringify(job), "utf8");
  const { error } = await supabase.storage
    .from(JOB_BUCKET)
    .upload(path, body, {
      contentType: "application/json",
      upsert: true,
    });

  if (error) {
    console.error(`[ocr-job] failed to persist ${job.id}:`, error.message);
    throw new Error(`Could not persist OCR job: ${error.message}`);
  }
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

async function runJob(
  jobId: string,
  imageBytes: Buffer,
  filename: string,
  mode: OcrMode
): Promise<void> {
  const job = jobs.get(jobId);
  if (!job) return;

  job.status = "running";
  await saveJob(job);

  try {
    const result = await runOcr(imageBytes, filename, mode);
    if (!result.ok) {
      job.status = "failed";
      job.error = result.error;
      job.detail = result.detail;
      await saveJob(job);
      return;
    }

    job.result = await finalizeOcr(imageBytes, filename, result);
    job.status = "succeeded";
    await saveJob(job);
  } catch (err) {
    job.status = "failed";
    job.error = "ocr_failed";
    job.detail = err instanceof Error ? err.message : String(err);
    try {
      await saveJob(job);
    } catch (persistErr) {
      console.error("[ocr-job] failed to save error state:", persistErr);
    }
  }
}

export async function createOcrJob(
  imageBytes: Buffer,
  filename: string,
  mode: OcrMode = DEFAULT_OCR_MODE
): Promise<OcrJobRecord> {
  pruneJobs();
  const id = randomUUID();
  const now = Date.now();
  const job: OcrJobRecord = {
    id,
    status: "queued",
    mode,
    createdAt: now,
    updatedAt: now,
  };

  await saveJob(job);
  void runJob(id, imageBytes, filename, mode);
  return job;
}

export async function getOcrJob(id: string): Promise<OcrJobRecord | undefined> {
  pruneJobs();
  const local = jobs.get(id);

  const supabase = getSupabase();
  if (!supabase) return local;

  const path = `${JOB_PREFIX}/${id}.json`;
  const { data, error } = await supabase.storage.from(JOB_BUCKET).download(path);
  if (error || !data) {
    if (error && !error.message.toLowerCase().includes("not found")) {
      console.error(`[ocr-job] failed to load ${id}:`, error.message);
    }
    return local;
  }

  try {
    const job = JSON.parse(await data.text()) as OcrJobRecord;
    jobs.set(job.id, job);
    return job;
  } catch (err) {
    console.error(`[ocr-job] invalid persisted job ${id}:`, err);
    return undefined;
  }
}

/** Start the persistent Python worker so the first real request pays no cold-start cost. */
export function warmOcrModel(): void {
  if (!isModelReady(DEFAULT_OCR_MODE)) {
    console.warn("[ocr] skip warmup — model not ready");
    return;
  }
  warmWorker();
}

export async function runOcrAndPersist(
  imageBytes: Buffer,
  filename: string,
  mode: OcrMode = DEFAULT_OCR_MODE
): Promise<OcrJobSuccess | OcrError> {
  const result = await runOcr(imageBytes, filename, mode);
  if (!result.ok) return result;
  return finalizeOcr(imageBytes, filename, result);
}
