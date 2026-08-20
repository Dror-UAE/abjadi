/**
 * Persistent Python OCR worker.
 *
 * Spawns `ocr_worker.py` once, loads all model weights, then routes every
 * OCR request through stdin/stdout (one JSON line each way).  A single
 * concurrency lock serialises requests so only one runs at a time, preventing
 * OOM on the 2 GB shared Fly VM.
 *
 * The worker is started lazily on the first request and restarted automatically
 * if it crashes.
 */

import { spawn, type ChildProcessWithoutNullStreams } from "node:child_process";
import { createInterface } from "node:readline";
import { randomUUID } from "node:crypto";
import { join } from "node:path";

import { MODEL_PYTHON, MODEL_ROOT, isModelReady, type OcrMode } from "./model-paths.js";

const WORKER_SCRIPT = join(MODEL_ROOT, "ocr_worker.py");
const READY_TIMEOUT_MS = 120_000; // time allowed for cold boot + weight loading
const REQUEST_TIMEOUT_MS = 120_000; // hard kill if a single inference hangs
const QUEUE_TIMEOUT_MS = 90_000; // fail fast if another scan is already running
const RECYCLE_AFTER_JOBS = 3; // restart Python worker to reclaim RAM on 2GB VM

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

type WorkerRequest = {
  id: string;
  mode: OcrMode;
  image_path: string;
  out_dir: string;
};

type WorkerResponse = {
  id: string;
  ok: boolean;
  error?: string;
  [key: string]: unknown;
};

type PendingRequest = {
  resolve: (res: WorkerResponse) => void;
  reject: (err: Error) => void;
  timer: NodeJS.Timeout;
};

type QueueWaiter = {
  resolve: () => void;
  reject: (err: Error) => void;
  timer: NodeJS.Timeout;
};

// ---------------------------------------------------------------------------
// Worker state
// ---------------------------------------------------------------------------

let proc: ChildProcessWithoutNullStreams | null = null;
let workerReady = false;
let restartScheduled = false;
let jobsSinceRecycle = 0;
const pending = new Map<string, PendingRequest>();

// Concurrency lock — only one OCR job runs at a time.
let running = false;
const queue: QueueWaiter[] = [];

function acquireLock(): Promise<void> {
  return new Promise((resolve, reject) => {
    if (!running) {
      running = true;
      resolve();
      return;
    }

    const waiter: QueueWaiter = {
      resolve: () => resolve(),
      reject,
      timer: setTimeout(() => {
        const idx = queue.indexOf(waiter);
        if (idx >= 0) queue.splice(idx, 1);
        reject(
          new Error(
            "OCR queue timeout — another scan is still running. Please try again in a moment."
          )
        );
      }, QUEUE_TIMEOUT_MS),
    };
    queue.push(waiter);
  });
}

function releaseLock(): void {
  const next = queue.shift();
  if (next) {
    clearTimeout(next.timer);
    // Keep running=true — ownership transfers to the next waiter.
    next.resolve();
  } else {
    running = false;
  }
}

// ---------------------------------------------------------------------------
// Worker lifecycle
// ---------------------------------------------------------------------------

function startWorker(): void {
  if (proc) return;

  console.log("[ocr-worker] starting Python worker...");

  proc = spawn(MODEL_PYTHON, [WORKER_SCRIPT], {
    cwd: MODEL_ROOT,
    env: {
      ...process.env,
      PYTHONUNBUFFERED: "1",
      // Leave CPU headroom for Node so mobile polling stays responsive.
      OMP_NUM_THREADS: "1",
      MKL_NUM_THREADS: "1",
      OPENBLAS_NUM_THREADS: "1",
      NUMEXPR_NUM_THREADS: "1",
    },
    stdio: ["pipe", "pipe", "pipe"],
  });

  workerReady = false;

  const rl = createInterface({ input: proc.stdout, crlfDelay: Infinity });

  rl.on("line", (line) => {
    const trimmed = line.trim();
    if (!trimmed) return;

    let msg: WorkerResponse;
    try {
      msg = JSON.parse(trimmed) as WorkerResponse;
    } catch {
      console.warn("[ocr-worker] unparseable stdout:", trimmed.slice(0, 200));
      return;
    }

    // Ready signal from boot
    if (msg.id === "__ready__") {
      workerReady = true;
      console.log("[ocr-worker] ready");
      return;
    }

    const entry = pending.get(msg.id);
    if (!entry) return;

    clearTimeout(entry.timer);
    pending.delete(msg.id);
    entry.resolve(msg);
  });

  proc.stderr.on("data", (chunk: Buffer) => {
    console.log("[ocr-worker]", chunk.toString("utf8").trimEnd());
  });

  proc.on("error", (err) => {
    console.error("[ocr-worker] process error:", err.message);
    handleCrash(new Error(`Worker process error: ${err.message}`));
  });

  proc.on("close", (code) => {
    console.warn(`[ocr-worker] exited with code=${code}`);
    handleCrash(new Error(`Worker exited unexpectedly (code ${code})`));
  });
}

function handleCrash(err: Error): void {
  workerReady = false;
  proc = null;

  // Reject in-flight inference requests only.
  // Do NOT touch the concurrency lock here — runOcrWorker's `finally`
  // releases it. Touching the lock here caused double-release races that
  // let two jobs run at once and hung/OOM'd the VM.
  for (const [id, entry] of pending) {
    clearTimeout(entry.timer);
    pending.delete(id);
    entry.reject(err);
  }

  // Schedule a restart after a short delay to avoid spin-loops
  if (!restartScheduled) {
    restartScheduled = true;
    setTimeout(() => {
      restartScheduled = false;
      if (!proc) {
        console.log("[ocr-worker] restarting...");
        startWorker();
      }
    }, 3_000);
  }
}

function stopWorker(): void {
  if (!proc) return;
  console.log("[ocr-worker] stopping");
  proc.stdin.end();
  proc.kill("SIGTERM");
  proc = null;
  workerReady = false;
}

// Graceful shutdown on process exit
process.on("exit", stopWorker);
process.on("SIGTERM", () => {
  stopWorker();
  process.exit(0);
});
process.on("SIGINT", () => {
  stopWorker();
  process.exit(0);
});

// ---------------------------------------------------------------------------
// Wait for the worker to be ready (used after startWorker)
// ---------------------------------------------------------------------------

function waitForReady(): Promise<void> {
  if (workerReady) return Promise.resolve();

  return new Promise((resolve, reject) => {
    const deadline = Date.now() + READY_TIMEOUT_MS;

    const poll = setInterval(() => {
      if (workerReady) {
        clearInterval(poll);
        resolve();
        return;
      }
      if (!proc || Date.now() > deadline) {
        clearInterval(poll);
        reject(new Error("OCR worker failed to become ready within timeout"));
      }
    }, 200);
  });
}

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

/**
 * Send one OCR job to the persistent worker.
 * Acquires the concurrency lock so only one job runs at a time.
 */
export async function runOcrWorker(
  imagePath: string,
  outDir: string,
  mode: OcrMode
): Promise<WorkerResponse> {
  if (!isModelReady(mode)) {
    return {
      id: "?",
      ok: false,
      error:
        mode === "stone"
          ? "Stone OCR weights missing. Run `pnpm setup:model` from the repo root."
          : "Run `pnpm setup:model` from the repo root (requires Python 3.x + torch).",
    };
  }

  // Start on first use
  if (!proc) startWorker();

  // Wait for boot/weight-loading to complete
  await waitForReady();

  // Acquire single-job lock (fails after QUEUE_TIMEOUT_MS so other users
  // don't hang forever behind a stuck scan)
  await acquireLock();

  try {
    const response = await sendRequest(imagePath, outDir, mode);
    jobsSinceRecycle += 1;
    if (jobsSinceRecycle >= RECYCLE_AFTER_JOBS && proc) {
      console.log(
        `[ocr-worker] recycling Python worker after ${jobsSinceRecycle} jobs to free memory`
      );
      jobsSinceRecycle = 0;
      const old = proc;
      proc = null;
      workerReady = false;
      old.kill("SIGTERM");
      // Next request will startWorker() fresh.
    }
    return response;
  } finally {
    releaseLock();
  }
}

function sendRequest(
  imagePath: string,
  outDir: string,
  mode: OcrMode
): Promise<WorkerResponse> {
  return new Promise((resolve, reject) => {
    if (!proc) {
      reject(new Error("OCR worker is not running"));
      return;
    }

    const id = randomUUID();

    const timer = setTimeout(() => {
      pending.delete(id);
      // Kill the worker — it's stuck. handleCrash will restart it.
      console.error(
        `[ocr-worker] request ${id} timed out after ${REQUEST_TIMEOUT_MS / 1000}s — killing worker`
      );
      if (proc) {
        proc.kill("SIGKILL");
      }
      reject(new Error(`OCR request timed out after ${REQUEST_TIMEOUT_MS / 1000}s`));
    }, REQUEST_TIMEOUT_MS);

    pending.set(id, { resolve, reject, timer });

    const req: WorkerRequest = { id, mode, image_path: imagePath, out_dir: outDir };
    const line = JSON.stringify(req) + "\n";

    proc.stdin.write(line, (err) => {
      if (err) {
        clearTimeout(timer);
        pending.delete(id);
        reject(new Error(`Failed to write to OCR worker: ${err.message}`));
      }
    });
  });
}

/**
 * Pre-warm the worker so the first real request doesn't pay the cold-start cost.
 * Called once at API startup.
 */
export function warmWorker(): void {
  if (!isModelReady()) {
    console.warn("[ocr-worker] skip warm — model not ready");
    return;
  }
  if (proc) return;
  startWorker();
  // waitForReady resolves in the background; errors are logged, not thrown
  waitForReady()
    .then(() => console.log("[ocr-worker] warm complete"))
    .catch((err: unknown) => console.warn("[ocr-worker] warm failed:", err));
}
