import { mkdir, readFile, writeFile } from "node:fs/promises";
import path from "node:path";
import { NextResponse } from "next/server";
import nodemailer from "nodemailer";

export const runtime = "nodejs";

function createTransporter() {
  return nodemailer.createTransport({
    host: process.env.SMTP_HOST || "smtp.hostinger.com",
    port: Number(process.env.SMTP_PORT || 465),
    secure: process.env.SMTP_SECURE !== "false",
    auth: {
      user: process.env.SMTP_USER || "info@abjadi.ai",
      pass: process.env.SMTP_PASS,
    },
  });
}

async function sendNotificationEmail(submission: EarlyAccessSubmission): Promise<void> {
  const to = process.env.SMTP_TO || process.env.SMTP_USER || "info@abjadi.ai";
  const transporter = createTransporter();

  const lines = [
    `الاسم: ${submission.name || "—"}`,
    `البريد الإلكتروني: ${submission.email}`,
    `نوع المستخدم: ${submission.userType}`,
    submission.organizationName ? `المؤسسة: ${submission.organizationName}` : null,
    `التاريخ: ${new Date(submission.createdAt).toLocaleString("ar-SA", { timeZone: "Asia/Riyadh" })}`,
  ].filter(Boolean).join("\n");

  await transporter.sendMail({
    from: `"أبجدي" <${process.env.SMTP_USER || "info@abjadi.ai"}>`,
    to,
    subject: `طلب وصول مبكر جديد — ${submission.email}`,
    text: `طلب وصول مبكر جديد:\n\n${lines}`,
    html: `<div dir="rtl" style="font-family:Arial,sans-serif;line-height:1.7">
<h2>طلب وصول مبكر جديد</h2>
<table style="border-collapse:collapse">
<tr><td style="padding:4px 12px;font-weight:bold">الاسم</td><td>${submission.name || "—"}</td></tr>
<tr><td style="padding:4px 12px;font-weight:bold">البريد الإلكتروني</td><td>${submission.email}</td></tr>
<tr><td style="padding:4px 12px;font-weight:bold">نوع المستخدم</td><td>${submission.userType}</td></tr>
${submission.organizationName ? `<tr><td style="padding:4px 12px;font-weight:bold">المؤسسة</td><td>${submission.organizationName}</td></tr>` : ""}
<tr><td style="padding:4px 12px;font-weight:bold">التاريخ</td><td>${new Date(submission.createdAt).toLocaleString("ar-SA", { timeZone: "Asia/Riyadh" })}</td></tr>
</table>
</div>`,
  });
}

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

    // Send email notification — log but don't fail the request if it errors.
    try {
      await sendNotificationEmail(submission);
    } catch (emailErr) {
      console.error("[early-access] email send failed:", emailErr);
    }

    return NextResponse.json({ ok: true }, { status: 201 });
  } catch {
    return NextResponse.json(
      { error: "Unable to process request." },
      { status: 500 },
    );
  }
}
