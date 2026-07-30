import { dirname, resolve } from "node:path";
import { fileURLToPath } from "node:url";

import { drizzle } from "drizzle-orm/postgres-js";
import { migrate } from "drizzle-orm/postgres-js/migrator";

import { createPgClient, getDatabaseUrl, printConnectionHint } from "./env.js";

const migrationsFolder = resolve(dirname(fileURLToPath(import.meta.url)), "../drizzle");
const url = getDatabaseUrl();
const connection = createPgClient(url);
const db = drizzle(connection);

try {
  console.log("[db] applying drizzle migrations...");
  await migrate(db, { migrationsFolder });
  console.log("[db] drizzle migrations applied");
} catch (err) {
  console.error("[db] migration failed:", err instanceof Error ? err.message : err);
  printConnectionHint(err);
  process.exit(1);
} finally {
  await connection.end();
}
