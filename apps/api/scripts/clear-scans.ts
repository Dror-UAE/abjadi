/**
 * Clear old scans from the database and storage.
 *
 * Usage (from repo root):
 *   pnpm --filter @abjadi/api clear-scans                # dry-run, all scans older than 30 days
 *   pnpm --filter @abjadi/api clear-scans -- --days 7    # dry-run, older than 7 days
 *   pnpm --filter @abjadi/api clear-scans -- --all       # dry-run, every scan
 *   pnpm --filter @abjadi/api clear-scans -- --confirm   # actually delete (default: 30-day cutoff)
 *   pnpm --filter @abjadi/api clear-scans -- --all --confirm
 *
 * Requires SUPABASE_URL + SUPABASE_SECRET_KEY (or SUPABASE_SERVICE_ROLE_KEY) in apps/api/.env
 */

import "../src/load-env.js";

import { createClient } from "@supabase/supabase-js";

// ---------------------------------------------------------------------------
// Config
// ---------------------------------------------------------------------------

const args = process.argv.slice(2);
const DRY_RUN = !args.includes("--confirm");
const ALL_SCANS = args.includes("--all");
const DAYS_IDX = args.indexOf("--days");
const OLDER_THAN_DAYS = DAYS_IDX >= 0 ? Number(args[DAYS_IDX + 1] ?? 30) : 30;
const BATCH = 50;

if (Number.isNaN(OLDER_THAN_DAYS) || OLDER_THAN_DAYS <= 0) {
  console.error("--days must be a positive number.");
  process.exit(1);
}

const SUPABASE_URL = process.env.SUPABASE_URL?.trim();
const SUPABASE_KEY =
  process.env.SUPABASE_SECRET_KEY?.trim() ??
  process.env.SUPABASE_SERVICE_ROLE_KEY?.trim();

if (!SUPABASE_URL || !SUPABASE_KEY) {
  console.error(
    "Missing SUPABASE_URL / SUPABASE_SECRET_KEY. Set them in apps/api/.env"
  );
  process.exit(1);
}

const supabase = createClient(SUPABASE_URL, SUPABASE_KEY, {
  auth: { persistSession: false, autoRefreshToken: false },
});

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

type DocImage = { storage_path?: string | null };
type DocNode = {
  id?: string;
  documentation_images?: DocImage | DocImage[] | null;
};

type ScanRow = {
  id: string;
  created_at: string;
  source_image_path: string | null;
  overlay_image_path: string | null;
  documentations?: DocNode | DocNode[] | null;
};

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

function cutoffDate(): string {
  const d = new Date();
  d.setUTCDate(d.getUTCDate() - OLDER_THAN_DAYS);
  return d.toISOString();
}

function docStoragePaths(scan: ScanRow): string[] {
  const docsRaw = scan.documentations;
  const docs = docsRaw
    ? Array.isArray(docsRaw)
      ? docsRaw
      : [docsRaw]
    : [];
  const paths: string[] = [];
  for (const doc of docs) {
    const imgs = doc.documentation_images
      ? Array.isArray(doc.documentation_images)
        ? doc.documentation_images
        : [doc.documentation_images]
      : [];
    for (const img of imgs) {
      if (img.storage_path) paths.push(img.storage_path);
    }
  }
  return paths;
}

// ---------------------------------------------------------------------------
// Fetch page of scans to delete
// ---------------------------------------------------------------------------

async function fetchPage(cursor: string | null): Promise<ScanRow[]> {
  let query = supabase
    .from("scans")
    .select(
      `
      id,
      created_at,
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
    .order("created_at", { ascending: true })
    .limit(BATCH);

  if (!ALL_SCANS) {
    query = query.lt("created_at", cutoffDate());
  }
  if (cursor) {
    query = query.gt("created_at", cursor);
  }

  const { data, error } = await query;
  if (error) throw new Error(`Fetch failed: ${error.message}`);
  return (data ?? []) as ScanRow[];
}

// ---------------------------------------------------------------------------
// Delete one scan (storage + row)
// ---------------------------------------------------------------------------

async function deleteScan(scan: ScanRow): Promise<void> {
  const scanPaths = [scan.source_image_path, scan.overlay_image_path].filter(
    (p): p is string => Boolean(p)
  );
  const docPaths = docStoragePaths(scan);

  if (scanPaths.length) {
    const { error } = await supabase.storage
      .from("scan-images")
      .remove(scanPaths);
    if (error) {
      console.warn(
        `  [warn] scan-images cleanup for ${scan.id}: ${error.message}`
      );
    }
  }

  if (docPaths.length) {
    const { error } = await supabase.storage
      .from("documentation-images")
      .remove(docPaths);
    if (error) {
      console.warn(
        `  [warn] documentation-images cleanup for ${scan.id}: ${error.message}`
      );
    }
  }

  const { error } = await supabase.from("scans").delete().eq("id", scan.id);
  if (error) throw new Error(`Row delete failed for ${scan.id}: ${error.message}`);
}

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------

async function main(): Promise<void> {
  const label = ALL_SCANS
    ? "all scans"
    : `scans older than ${OLDER_THAN_DAYS} day(s) (before ${cutoffDate()})`;

  console.log(`\nAbjadi — clear-scans`);
  console.log(`Target : ${label}`);
  console.log(`Mode   : ${DRY_RUN ? "DRY RUN (pass --confirm to delete)" : "LIVE DELETE"}`);
  console.log("");

  let cursor: string | null = null;
  let total = 0;
  let deleted = 0;
  let failed = 0;

  while (true) {
    const page = await fetchPage(cursor);
    if (page.length === 0) break;

    total += page.length;

    for (const scan of page) {
      const created = scan.created_at.slice(0, 10);
      if (DRY_RUN) {
        console.log(`  [dry-run] would delete scan ${scan.id}  (created ${created})`);
        deleted++;
      } else {
        try {
          await deleteScan(scan);
          console.log(`  [deleted] ${scan.id}  (created ${created})`);
          deleted++;
        } catch (err) {
          console.error(
            `  [error]   ${scan.id}: ${err instanceof Error ? err.message : String(err)}`
          );
          failed++;
        }
      }
    }

    // advance cursor using the last row's created_at to avoid re-fetching
    cursor = page[page.length - 1]!.created_at;

    // if the page wasn't full we've reached the end
    if (page.length < BATCH) break;
  }

  console.log("");
  console.log(
    `Done. Found ${total} scan(s). ${DRY_RUN ? "Would delete" : "Deleted"}: ${deleted}. Failed: ${failed}.`
  );
  if (DRY_RUN && total > 0) {
    console.log("Run with --confirm to actually delete.");
  }
}

main().catch((err) => {
  console.error("Unexpected error:", err instanceof Error ? err.message : err);
  process.exit(1);
});
