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

/** Strip huge overlay payload before writing job JSON to Supabase storage. */
function jobForStorage(job: OcrJobRecord): OcrJobRecord {
  if (!job.result?.overlayBase64) return job;
  return {
    ...job,
    result: {
      ...job.result,
      overlayBase64: "[omitted]",
    },
  };
}

async function saveJob(job: OcrJobRecord): Promise<void> {
  job.updatedAt = Date.now();
  jobs.set(job.id, job);

  const supabase = getSupabase();
  if (!supabase) return;

  const path = `${JOB_PREFIX}/${job.id}.json`;
  const body = Buffer.from(JSON.stringify(jobForStorage(job)), "utf8");
  const { error } = await supabase.storage.from(JOB_BUCKET).upload(path, body, {
    contentType: "application/json",
    upsert: true,
  });

  if (error) {
    // Log but don't throw — Supabase storage is a best-effort backup.
    // The in-memory record is the authoritative source while the server runs.
    console.error(`[ocr-job] failed to persist ${job.id}:`, error.message);
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
  // Don't await Supabase — mobile needs to see "running" from in-memory ASAP.
  void saveJob(job);

  try {
    const result = await runOcr(imageBytes, filename, mode);
    if (!result.ok) {
      job.status = "failed";
      job.error = result.error;
      job.detail = result.detail;
      void saveJob(job);
      return;
    }

    // Mark succeeded IMMEDIATELY so mobile polls get the OCR result without
    // waiting for Supabase image/overlay uploads (which can take 30–90s and
    // made every other user think the server was hung).
    job.result = {
      ...result,
      persisted: false,
    };
    job.status = "succeeded";
    void saveJob(job);

    // Persist scan + images in the background; update scanId when done.
    void (async () => {
      try {
        const finalized = await finalizeOcr(imageBytes, filename, result);
        const current = jobs.get(jobId);
        if (!current || current.status !== "succeeded") return;
        current.result = finalized;
        void saveJob(current);
      } catch (err) {
        console.error("[ocr-job] background persist failed:", err);
      }
    })();
  } catch (err) {
    job.status = "failed";
    job.error = "ocr_failed";
    job.detail = err instanceof Error ? err.message : String(err);
    void saveJob(job);
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

  // Store in-memory immediately so the 202 response goes out without waiting
  // for a Supabase round-trip (which can block for seconds and trigger Fly's
  // 60s HTTP timeout before the client even gets the jobId).
  jobs.set(id, job);
  void saveJob(job); // fire-and-forget to Supabase
  void runJob(id, imageBytes, filename, mode);
  return job;
}

export async function getOcrJob(id: string): Promise<OcrJobRecord | undefined> {
  pruneJobs();

  // Always prefer the in-memory record — it is the authoritative source while
  // the server is running. Supabase storage writes are async and may lag behind
  // the in-memory state, causing polls to see a stale "running" status even
  // after the job has already succeeded.
  const local = jobs.get(id);
  if (local) return local;

  // Only fall back to Supabase when the job isn't in memory (e.g. after a
  // server restart or when a different instance handled the job).
  const supabase = getSupabase();
  if (!supabase) return undefined;

  const path = `${JOB_PREFIX}/${id}.json`;
  const { data, error } = await supabase.storage.from(JOB_BUCKET).download(path);
  if (error || !data) {
    if (error && !error.message.toLowerCase().includes("not found")) {
      console.error(`[ocr-job] failed to load ${id}:`, error.message);
    }
    return undefined;
  }

  try {
    const job = JSON.parse(await data.text()) as OcrJobRecord;
    // Don't restore omitted overlay placeholder into memory as real data.
    if (job.result?.overlayBase64 === "[omitted]") {
      delete job.result.overlayBase64;
    }
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
