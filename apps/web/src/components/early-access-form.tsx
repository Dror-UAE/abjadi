"use client";

import { FormEvent, useMemo, useState } from "react";
import { useTranslations } from "next-intl";

type BaseUserType =
  | "individual-researcher"
  | "student"
  | "archaeologist"
  | "historian"
  | "museum"
  | "government-organization"
  | "university-research-institution"
  | "other";

const USER_TYPES: BaseUserType[] = [
  "individual-researcher",
  "student",
  "archaeologist",
  "historian",
  "museum",
  "government-organization",
  "university-research-institution",
  "other",
];

const ORG_USER_TYPES = new Set<BaseUserType>([
  "museum",
  "government-organization",
  "university-research-institution",
]);

const EMAIL_REGEX = /^[^\s@]+@[^\s@]+\.[^\s@]+$/;

export function EarlyAccessForm() {
  const t = useTranslations("EarlyAccess");
  const [name, setName] = useState("");
  const [email, setEmail] = useState("");
  const [userType, setUserType] = useState<BaseUserType | "">("");
  const [otherType, setOtherType] = useState("");
  const [organizationName, setOrganizationName] = useState("");
  const [error, setError] = useState<string | null>(null);
  const [success, setSuccess] = useState(false);
  const [isSubmitting, setIsSubmitting] = useState(false);

  const shouldShowOrganizationField = useMemo(
    () => (userType ? ORG_USER_TYPES.has(userType) : false),
    [userType],
  );

  async function handleSubmit(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    setError(null);
    setSuccess(false);

    const trimmedName = name.trim();
    const trimmedEmail = email.trim();
    const trimmedOtherType = otherType.trim();
    const trimmedOrganization = organizationName.trim();

    if (!trimmedEmail || !EMAIL_REGEX.test(trimmedEmail)) {
      setError(t("errors.email"));
      return;
    }

    if (!userType) {
      setError(t("errors.userType"));
      return;
    }

    if (userType === "other" && !trimmedOtherType) {
      setError(t("errors.otherType"));
      return;
    }

    if (shouldShowOrganizationField && !trimmedOrganization) {
      setError(t("errors.organizationName"));
      return;
    }

    const resolvedUserType =
      userType === "other" ? trimmedOtherType : t(`userTypes.${userType}`);

    if (!resolvedUserType) {
      setError(t("errors.userType"));
      return;
    }

    setIsSubmitting(true);

    try {
      const response = await fetch("/api/early-access", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({
          name: trimmedName || null,
          email: trimmedEmail,
          userType: resolvedUserType,
          organizationName: shouldShowOrganizationField ? trimmedOrganization : null,
        }),
      });

      if (!response.ok) {
        throw new Error("Request failed");
      }

      setSuccess(true);
      setName("");
      setEmail("");
      setUserType("");
      setOtherType("");
      setOrganizationName("");
    } catch {
      setError(t("errors.request"));
    } finally {
      setIsSubmitting(false);
    }
  }

  return (
    <form className="early-access-form" onSubmit={handleSubmit} noValidate>
      <label className="early-access-form__field">
        <span>{t("fields.name")}</span>
        <input
          type="text"
          name="name"
          value={name}
          onChange={(event) => setName(event.target.value)}
          autoComplete="name"
        />
      </label>

      <label className="early-access-form__field">
        <span>{t("fields.email")}</span>
        <input
          type="email"
          name="email"
          value={email}
          onChange={(event) => setEmail(event.target.value)}
          autoComplete="email"
          required
          aria-invalid={error?.toLowerCase().includes("email") ?? false}
        />
      </label>

      <label className="early-access-form__field">
        <span>{t("fields.userType")}</span>
        <select
          name="userType"
          value={userType}
          onChange={(event) => setUserType(event.target.value as BaseUserType)}
          required
        >
          <option value="">{t("fields.selectOption")}</option>
          {USER_TYPES.map((type) => (
            <option key={type} value={type}>
              {t(`userTypes.${type}`)}
            </option>
          ))}
        </select>
      </label>

      {userType === "other" ? (
        <label className="early-access-form__field">
          <span>{t("fields.otherType")}</span>
          <input
            type="text"
            name="otherType"
            value={otherType}
            onChange={(event) => setOtherType(event.target.value)}
            required
          />
        </label>
      ) : null}

      {shouldShowOrganizationField ? (
        <label className="early-access-form__field">
          <span>{t("fields.organizationName")}</span>
          <input
            type="text"
            name="organizationName"
            value={organizationName}
            onChange={(event) => setOrganizationName(event.target.value)}
            required
          />
        </label>
      ) : null}

      {error ? <p className="early-access-form__message early-access-form__message--error">{error}</p> : null}
      {success ? (
        <p className="early-access-form__message early-access-form__message--success">
          {t("success")}
        </p>
      ) : null}

      <button className="btn btn--solid early-access-form__submit" type="submit" disabled={isSubmitting}>
        {isSubmitting ? t("loading") : t("submit")}
      </button>
    </form>
  );
}
