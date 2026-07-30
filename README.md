# Abjadi

pnpm + Turborepo monorepo.

## Structure

```
apps/
  web/     # Next.js (Vercel)
  api/     # Hono API
  mobile/  # Expo (iOS / Android)
packages/
  ui/                 # Shared React components
  typescript-config/  # Shared TSConfigs
  eslint-config/      # Shared ESLint configs
models/
  musnad-ocr-model/   # Musnad OCR inference (Python)
```

## Commands

```bash
pnpm install          # install all workspace deps
pnpm setup:model      # create Python venv + install model deps
pnpm dev              # run all apps in dev mode
pnpm dev:web          # Next.js only
pnpm dev:api          # API only
pnpm dev:mobile       # Expo only
pnpm build            # build everything
pnpm lint             # lint all packages
pnpm typecheck        # typecheck all packages
```

- Web: http://localhost:3000
- API: http://localhost:3001
- Mobile: Expo Dev Tools (from `pnpm dev:mobile`)
- Model: `models/musnad-ocr-model` (Python; see package README)

## Mobile ↔ OCR loop

1. `pnpm setup:model` (once)
2. `pnpm dev:api` — serves `POST /ocr` on port 3001 (binds `0.0.0.0`)
3. Point the app at the API:
   - Simulator: `EXPO_PUBLIC_API_URL=http://localhost:3001`
   - Physical device: `EXPO_PUBLIC_API_URL=http://<your-lan-ip>:3001`
4. `pnpm dev:mobile` — capture/crop → Analyzing uploads → Result shows model text

Smoke-test the API:

```bash
curl -s http://localhost:3001/health
curl -s -F "image=@models/musnad-ocr-model/test.png" http://localhost:3001/ocr | head
```

## Supabase (scans + documentation)

1. Create a Supabase project.
2. Run SQL in `supabase/migrations/20260729120000_init_scans_docs.sql` (see `supabase/README.md`).
3. Set in `apps/api/.env`:

```bash
SUPABASE_URL=https://YOUR_PROJECT.supabase.co
SUPABASE_SERVICE_ROLE_KEY=...
```

4. Restart the API. Successful OCR writes to `scans` + `ocr_results` + Storage.
5. Mobile **وثّق النقش** → `POST /documentations` → `documentations` table.
## Deploy web to Vercel

1. Commit and push this repo to GitHub.
2. In [vercel.com](https://vercel.com) → **Add New Project** → import the repo.
3. Set **Root Directory** to `apps/web`.
4. Keep Framework as **Next.js**. Build/output can stay default.
5. Deploy.

CLI alternative (from repo root after `vercel login`):

```bash
cd apps/web
vercel
```

Locales will be live at `/ar` and `/en`.

## Deploy API to Fly.io

This deploys `apps/api` together with the Python OCR model.

1. Install Fly CLI and login:

```bash
brew install flyctl
fly auth login
```

2. Edit `apps/api/fly.toml`:
   - Set `app = "your-unique-app-name"`
   - Optionally change `primary_region`

3. Create Fly app (from repo root):

```bash
fly launch --no-deploy -c apps/api/fly.toml
```

4. Set required secrets:

```bash
fly secrets set -c apps/api/fly.toml \
  SUPABASE_URL="https://YOUR_PROJECT.supabase.co" \
  SUPABASE_SECRET_KEY="sb_secret_..." \
  SUPABASE_SERVICE_ROLE_KEY="eyJ..." \
  SUPABASE_PUBLISHABLE_KEY="sb_publishable_..."
```

5. Deploy:

```bash
fly deploy -c apps/api/fly.toml
```

6. Verify:

```bash
curl https://YOUR_APP_NAME.fly.dev/health
```

7. Point mobile app to Fly API:

```bash
EXPO_PUBLIC_API_URL=https://YOUR_APP_NAME.fly.dev
```
