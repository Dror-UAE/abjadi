FROM node:22-bookworm-slim

ENV PNPM_HOME="/pnpm"
ENV PATH="$PNPM_HOME:$PATH"

WORKDIR /app

# System deps for Python OCR (torch + opencv-headless runtime)
RUN apt-get update && apt-get install -y --no-install-recommends \
  python3 \
  python3-venv \
  python3-pip \
  libglib2.0-0 \
  libgomp1 \
  && rm -rf /var/lib/apt/lists/*

RUN corepack enable

# Copy only required workspace manifests first for better build caching.
COPY package.json pnpm-lock.yaml pnpm-workspace.yaml turbo.json ./
COPY apps/api/package.json apps/api/package.json
COPY packages/typescript-config/package.json packages/typescript-config/package.json
COPY packages/eslint-config/package.json packages/eslint-config/package.json

# Install only API workspace deps (+ its local workspace deps).
RUN pnpm install --frozen-lockfile --filter @abjadi/api...

# Copy sources needed at runtime/build.
COPY apps/api apps/api
COPY packages/typescript-config packages/typescript-config
COPY models/musnad-ocr-model models/musnad-ocr-model

# Create model venv in the exact path expected by model-paths.ts
RUN python3 -m venv /app/models/musnad-ocr-model/.venv \
  && /app/models/musnad-ocr-model/.venv/bin/pip install --no-cache-dir -r /app/models/musnad-ocr-model/requirements.txt

# Build API (TypeScript -> dist)
RUN pnpm --filter @abjadi/api build

WORKDIR /app/apps/api

ENV NODE_ENV=production
ENV HOST=0.0.0.0
ENV PORT=3001

EXPOSE 3001

CMD ["node", "dist/index.js"]
