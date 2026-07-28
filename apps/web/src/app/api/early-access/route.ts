import { mkdir, readFile, writeFile } from "node:fs/promises";
import path from "node:path";
import { NextResponse } from "next/server";

export const runtime = "nodejs";

type EarlyAccessSubmission = {
  name: string | null;
  email: string;
  userType: string;
  organizationName: string | null;
  createdAt: string;
};

type EarlyAccessPayload = {
  name?: unknown;
  email?: unknown;
  userType?: unknown;
  organizationName?: unknown;
};

const DATA_DIR = path.join(process.cwd(), "src", "data");
const DATA_FILE = path.join(DATA_DIR, "early-access-requests.json");
const EMAIL_REGEX = /^[^\s@]+@[^\s@]+\.[^\s@]+$/;
const ORG_REQUIRED_TYPES = new Set([
  "Museum",
  "Government Organization",
  "University / Research Institution",
]);

function normalizeOptionalString(value: unknown): string | null {
  if (typeof value !== "string") {
    return null;
  }

  const trimmed = value.trim();
  return trimmed.length > 0 ? trimmed : null;
}

function normalizeRequiredString(value: unknown): string | null {
  const normalized = normalizeOptionalString(value);
  return normalized && normalized.length > 0 ? normalized : null;
}

async function readSubmissions(): Promise<EarlyAccessSubmission[]> {
  try {
    const existingRaw = await readFile(DATA_FILE, "utf-8");
    const existing = JSON.parse(existingRaw) as unknown;
    return Array.isArray(existing) ? (existing as EarlyAccessSubmission[]) : [];
  } catch {
    return [];
  }
}

export async function POST(request: Request) {
  try {
    const payload = (await request.json()) as EarlyAccessPayload;
    const name = normalizeOptionalString(payload.name);
    const email = normalizeRequiredString(payload.email);
    const userType = normalizeRequiredString(payload.userType);
    const organizationName = normalizeOptionalString(payload.organizationName);

    if (!email || !EMAIL_REGEX.test(email)) {
      return NextResponse.json(
        { error: "A valid email address is required." },
        { status: 400 },
      );
    }

    if (!userType) {
      return NextResponse.json(
        { error: "User type is required." },
        { status: 400 },
      );
    }

    if (ORG_REQUIRED_TYPES.has(userType) && !organizationName) {
      return NextResponse.json(
        { error: "Organization name is required for this user type." },
        { status: 400 },
      );
    }

    const submission: EarlyAccessSubmission = {
      name,
      email,
      userType,
      organizationName,
      createdAt: new Date().toISOString(),
    };

    const existing = await readSubmissions();
    const next = [...existing, submission];

    await mkdir(DATA_DIR, { recursive: true });
    await writeFile(DATA_FILE, JSON.stringify(next, null, 2), "utf-8");

    return NextResponse.json({ ok: true }, { status: 201 });
  } catch {
    return NextResponse.json(
      { error: "Unable to process request." },
      { status: 500 },
    );
  }
}
