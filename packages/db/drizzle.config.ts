import { config } from "dotenv";
import { defineConfig } from "drizzle-kit";
import { dirname, resolve } from "node:path";
import { fileURLToPath } from "node:url";

const root = dirname(fileURLToPath(import.meta.url));
config({ path: resolve(root, "../../apps/api/.env") });

const raw = process.env.DATABASE_URL?.trim();
const url =
  raw && !raw.includes("sslmode=")
    ? raw.includes("?")
      ? `${raw}&sslmode=require`
      : `${raw}?sslmode=require`
    : raw;

if (!url) {
  console.warn(
    "[drizzle] DATABASE_URL is not set. Copy apps/api/.env or set DATABASE_URL before db:generate / db:migrate."
  );
}

export default defineConfig({
  schema: "./src/schema.ts",
  out: "./drizzle",
  dialect: "postgresql",
  dbCredentials: {
    url: url ?? "postgresql://postgres:postgres@localhost:5432/postgres",
    ssl: "require",
  },
  strict: true,
  verbose: true,
});
