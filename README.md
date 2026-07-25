# Abjadi

pnpm + Turborepo monorepo.

## Structure

```
apps/
  web/   # Next.js (Vercel)
  api/   # Hono API
packages/
  ui/                 # Shared React components
  typescript-config/  # Shared TSConfigs
  eslint-config/      # Shared ESLint configs
```

## Commands

```bash
pnpm install          # install all workspace deps
pnpm dev              # run all apps in dev mode
pnpm build            # build everything
pnpm lint             # lint all packages
pnpm typecheck        # typecheck all packages
```

- Web: http://localhost:3000
- API: http://localhost:3001

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
