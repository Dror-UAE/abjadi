import { NextResponse } from "next/server";
import nodemailer from "nodemailer";

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

const EMAIL_REGEX = /^[^\s@]+@[^\s@]+\.[^\s@]+$/;
const ORG_REQUIRED_TYPES = new Set([
  "Museum",
  "Government Organization",
  "University / Research Institution",
  "متحف",
  "منظمة حكومية",
  "جامعة / مؤسسة بحثية",
]);

function normalizeOptionalString(value: unknown): string | null {
  if (typeof value !== "string") return null;
  const trimmed = value.trim();
  return trimmed.length > 0 ? trimmed : null;
}

function normalizeRequiredString(value: unknown): string | null {
  const n = normalizeOptionalString(value);
  return n && n.length > 0 ? n : null;
}

async function sendNotificationEmail(s: EarlyAccessSubmission): Promise<void> {
  const smtpPass = process.env.SMTP_PASS;
  if (!smtpPass) {
    console.warn("[early-access] SMTP_PASS not set — skipping email");
    return;
  }

  const transporter = nodemailer.createTransport({
    host: process.env.SMTP_HOST || "smtp.hostinger.com",
    port: Number(process.env.SMTP_PORT || 465),
    secure: process.env.SMTP_SECURE !== "false",
    auth: {
      user: process.env.SMTP_USER || "info@abjadi.ai",
      pass: smtpPass,
    },
  });

  const to = process.env.SMTP_TO || process.env.SMTP_USER || "info@abjadi.ai";
  const date = new Date(s.createdAt).toLocaleString("ar-SA", { timeZone: "Asia/Riyadh" });

  await transporter.sendMail({
    from: `"أبجدي" <${process.env.SMTP_USER || "info@abjadi.ai"}>`,
    to,
    subject: `طلب وصول مبكر جديد — ${s.email}`,
    text: [
      "طلب وصول مبكر جديد:",
      `الاسم: ${s.name || "—"}`,
      `البريد الإلكتروني: ${s.email}`,
      `نوع المستخدم: ${s.userType}`,
      s.organizationName ? `المؤسسة: ${s.organizationName}` : null,
      `التاريخ: ${date}`,
    ]
      .filter(Boolean)
      .join("\n"),
    html: `<div dir="rtl" style="font-family:Arial,sans-serif;line-height:1.7">
<h2>طلب وصول مبكر جديد</h2>
<table style="border-collapse:collapse">
<tr><td style="padding:4px 16px;font-weight:bold">الاسم</td><td>${s.name || "—"}</td></tr>
<tr><td style="padding:4px 16px;font-weight:bold">البريد الإلكتروني</td><td>${s.email}</td></tr>
<tr><td style="padding:4px 16px;font-weight:bold">نوع المستخدم</td><td>${s.userType}</td></tr>
${s.organizationName ? `<tr><td style="padding:4px 16px;font-weight:bold">المؤسسة</td><td>${s.organizationName}</td></tr>` : ""}
<tr><td style="padding:4px 16px;font-weight:bold">التاريخ</td><td>${date}</td></tr>
</table>
</div>`,
  });
}

export async function POST(request: Request) {
  let payload: EarlyAccessPayload;
  try {
    payload = (await request.json()) as EarlyAccessPayload;
  } catch {
    return NextResponse.json({ error: "Invalid JSON body." }, { status: 400 });
  }

  const name = normalizeOptionalString(payload.name);
  const email = normalizeRequiredString(payload.email);
  const userType = normalizeRequiredString(payload.userType);
  const organizationName = normalizeOptionalString(payload.organizationName);

  if (!email || !EMAIL_REGEX.test(email)) {
    return NextResponse.json({ error: "A valid email address is required." }, { status: 400 });
  }
  if (!userType) {
    return NextResponse.json({ error: "User type is required." }, { status: 400 });
  }
  if (ORG_REQUIRED_TYPES.has(userType) && !organizationName) {
    return NextResponse.json(
      { error: "Organization name is required for this user type." },
      { status: 400 }
    );
  }

  const submission: EarlyAccessSubmission = {
    name,
    email,
    userType,
    organizationName,
    createdAt: new Date().toISOString(),
  };

  try {
    await sendNotificationEmail(submission);
  } catch (err) {
    console.error("[early-access] email send failed:", err);
    return NextResponse.json({ error: "Unable to send email." }, { status: 500 });
  }

  return NextResponse.json({ ok: true }, { status: 201 });
}
