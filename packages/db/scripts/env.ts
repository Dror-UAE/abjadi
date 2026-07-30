import { config } from "dotenv";
import { dirname, resolve } from "node:path";
import { fileURLToPath } from "node:url";

import postgres, { type Sql } from "postgres";

const pkgRoot = resolve(dirname(fileURLToPath(import.meta.url)), "..");

export function loadEnv(): void {
  config({ path: resolve(pkgRoot, "../../apps/api/.env") });
}

export function getDatabaseUrl(): string {
  loadEnv();
  const url = process.env.DATABASE_URL?.trim();
  if (!url) {
    console.error("[db] DATABASE_URL is missing in apps/api/.env");
    console.error("  Supabase → Settings → Database → Connection string → URI");
    console.error("  Direct (recommended for migrate):");
    console.error("    postgresql://postgres:PASSWORD@db.PROJECT_REF.supabase.co:5432/postgres");
    process.exit(1);
  }
  if (url.includes("sslmode=")) return url;
  return url.includes("?") ? `${url}&sslmode=require` : `${url}?sslmode=require`;
}

export function createPgClient(url: string): Sql {
  return postgres(url, { max: 1, ssl: "require", connect_timeout: 15 });
}

export function printConnectionHint(err: unknown): void {
  const msg = err instanceof Error ? err.message : String(err);
  if (
    msg.includes("not found") ||
    msg.includes("ENOTFOUND") ||
    msg.includes("Tenant or user not found")
  ) {
    console.error("\n[db] Connection hint:");
    console.error("  The pooler host/region may be wrong. In Supabase Dashboard → Database,");
    console.error("  use **Direct connection** (port 5432), not Session pooler:");
    console.error("    postgresql://postgres:PASSWORD@db.PROJECT_REF.supabase.co:5432/postgres");
  }
}
