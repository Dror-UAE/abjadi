import { readFileSync } from "node:fs";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";

import { createPgClient, getDatabaseUrl, printConnectionHint } from "./env.js";

const here = dirname(fileURLToPath(import.meta.url));
const sqlPath = join(here, "..", "sql", "extras.sql");
const extras = readFileSync(sqlPath, "utf8");

const db = createPgClient(getDatabaseUrl());

try {
  await db.unsafe(extras);
  console.log("[db] applied sql/extras.sql (RLS, storage, triggers)");
} catch (err) {
  console.error("[db] extras failed:", err instanceof Error ? err.message : err);
  printConnectionHint(err);
  process.exit(1);
} finally {
  await db.end();
}
