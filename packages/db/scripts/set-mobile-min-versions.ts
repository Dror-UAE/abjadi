import { createPgClient, getDatabaseUrl, printConnectionHint } from "./env.js";

type Mode = "force" | "reset";

const mode = process.argv[2];

if (mode !== "force" && mode !== "reset") {
  console.error("Usage: bun run ./scripts/set-mobile-min-versions.ts <force|reset>");
  process.exit(1);
}

const minimum = mode === "force" ? "9.0.0" : "1.0.0";
const url = getDatabaseUrl();
const sql = createPgClient(url);

try {
  const rows = await sql<
    {
      ios_minimum_supported_version: string;
      android_minimum_supported_version: string;
    }[]
  >`
    update public.mobile_version_policies
    set
      ios_minimum_supported_version = ${minimum},
      android_minimum_supported_version = ${minimum}
    where is_active = true
    returning
      ios_minimum_supported_version,
      android_minimum_supported_version
  `;

  if (rows.length === 0) {
    console.error(
      "[db] No active mobile_version_policies row found. Run `pnpm db:migrate` first."
    );
    process.exit(1);
  }

  if (rows.length > 1) {
    console.error(
      `[db] Expected one active policy row, found ${rows.length}. Aborting without further changes.`
    );
    process.exit(1);
  }

  const row = rows[0]!;

  if (mode === "force") {
    console.log("Force update test enabled");
  } else {
    console.log("Force update test disabled");
  }
  console.log(`iOS minimum: ${row.ios_minimum_supported_version}`);
  console.log(`Android minimum: ${row.android_minimum_supported_version}`);
} catch (err) {
  console.error(
    "[db] Failed to update mobile version policy:",
    err instanceof Error ? err.message : err
  );
  printConnectionHint(err);
  process.exit(1);
} finally {
  await sql.end();
}
