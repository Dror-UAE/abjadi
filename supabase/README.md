# Supabase setup (Abjadi)

## 1. Create project

1. Open [supabase.com](https://supabase.com) → New project.
2. Copy **Project URL** and **service_role** key (Settings → API).
3. Copy **Database connection string** (Settings → Database → URI, pooler `6543`).

## 2. API env

In `apps/api/.env`:

```bash
SUPABASE_URL=https://YOUR_PROJECT.supabase.co
SUPABASE_SERVICE_ROLE_KEY=eyJ...
DATABASE_URL=postgresql://postgres.YOUR_PROJECT:YOUR_PASSWORD@aws-0-REGION.pooler.supabase.com:6543/postgres
```

## 3. Run migrations (Drizzle)

From the repo root:

```bash
pnpm install

# After changing packages/db/src/schema.ts
bun db:generate

# Apply migrations + RLS/storage/triggers
bun db:migrate
```

- `db:generate` — diff schema → SQL in `packages/db/drizzle/`
- `db:migrate` — runs Drizzle migrations, then `packages/db/sql/extras.sql` (RLS, buckets, triggers)

Optional: `bun db:studio` opens Drizzle Studio against `DATABASE_URL`.

Restart `pnpm dev:api` / `bun dev` after env changes.

Without Supabase env vars, OCR still works but **nothing is saved** to the database.

## 4. Flow

- `POST /ocr` → Storage (source + overlay) + `scans` + `ocr_results`
- `POST /documentations` → `documentations` + optional extra images
- Mobile submits documentation after Result → returns `public_id` (`ABJ-…`)

Auth is optional for v1: API uses the **service role**. Add Supabase Auth later and set `user_id` from JWT.

## Legacy SQL

`supabase/migrations/20260729120000_init_scans_docs.sql` is kept for reference. Prefer `bun db:migrate` for new environments.
