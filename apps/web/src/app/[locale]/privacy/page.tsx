import type { Metadata } from "next";
import { getTranslations, setRequestLocale } from "next-intl/server";
import { SiteHeader } from "../../../components/site-header";
import { Link } from "../../../i18n/navigation";
import { isAppLocale, routing } from "../../../i18n/routing";

type Props = {
  params: Promise<{ locale: string }>;
};

export async function generateMetadata({ params }: Props): Promise<Metadata> {
  const { locale: raw } = await params;
  const locale = isAppLocale(raw) ? raw : routing.defaultLocale;
  const t = await getTranslations({ locale, namespace: "Privacy" });

  return {
    title: t("metaTitle"),
    description: t("metaDescription"),
  };
}

export default async function PrivacyPage({ params }: Props) {
  const { locale: raw } = await params;
  const locale = isAppLocale(raw) ? raw : routing.defaultLocale;
  setRequestLocale(locale);

  const t = await getTranslations("Privacy");
  const tFooter = await getTranslations("Footer");
  const email = t("email");

  const sections = [
    ["collectTitle", "collectBody"],
    ["useTitle", "useBody"],
    ["storageTitle", "storageBody"],
    ["shareTitle", "shareBody"],
    ["retentionTitle", "retentionBody"],
    ["rightsTitle", "rightsBody"],
    ["childrenTitle", "childrenBody"],
    ["changesTitle", "changesBody"],
  ] as const;

  return (
    <>
      <SiteHeader solid />

      <main className="legal-page">
        <article className="legal">
          <p className="legal__eyebrow">{t("updated")}</p>
          <h1>{t("title")}</h1>
          <p className="legal__intro">{t("intro")}</p>

          {sections.map(([titleKey, bodyKey]) => (
            <section key={titleKey}>
              <h2>{t(`sections.${titleKey}`)}</h2>
              <p>{t(`sections.${bodyKey}`)}</p>
            </section>
          ))}

          <section>
            <h2>{t("sections.contactTitle")}</h2>
            <p>
              {t.rich("sections.contactBody", {
                email: () => <a href={`mailto:${email}`}>{email}</a>,
              })}
            </p>
          </section>
        </article>
      </main>

      <footer className="site-footer">
        <nav className="site-footer__links" aria-label={tFooter("privacy")}>
          <Link href="/privacy">{tFooter("privacy")}</Link>
          <Link href="/support">{tFooter("support")}</Link>
        </nav>
        <p>{tFooter("rights", { year: new Date().getFullYear() })}</p>
      </footer>
    </>
  );
}
