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
  const t = await getTranslations({ locale, namespace: "Support" });

  return {
    title: t("metaTitle"),
    description: t("metaDescription"),
  };
}

export default async function SupportPage({ params }: Props) {
  const { locale: raw } = await params;
  const locale = isAppLocale(raw) ? raw : routing.defaultLocale;
  setRequestLocale(locale);

  const t = await getTranslations("Support");
  const tFooter = await getTranslations("Footer");
  const email = t("email");

  return (
    <>
      <SiteHeader solid />

      <main className="legal-page">
        <article className="legal">
          <h1>{t("title")}</h1>
          <p className="legal__intro">{t("lede")}</p>

          <section className="legal__panel">
            <h2>{t("contactTitle")}</h2>
            <p>
              {t.rich("contactCopy", {
                email: () => <a href={`mailto:${email}`}>{email}</a>,
              })}
            </p>
            <a className="btn btn--primary" href={`mailto:${email}`}>
              {t("emailCta")}
            </a>
          </section>

          <h2>{t("topicsTitle")}</h2>

          <section>
            <h3>{t("topics.scanTitle")}</h3>
            <p>{t("topics.scanBody")}</p>
          </section>

          <section>
            <h3>{t("topics.historyTitle")}</h3>
            <p>{t("topics.historyBody")}</p>
          </section>

          <section>
            <h3>{t("topics.accessTitle")}</h3>
            <p>{t("topics.accessBody")}</p>
          </section>

          <section>
            <h3>{t("topics.privacyTitle")}</h3>
            <p>{t("topics.privacyBody")}</p>
            <p>
              <Link href="/privacy">{t("topics.privacyLink")}</Link>
            </p>
          </section>
        </article>
      </main>

      <footer className="site-footer">
        <nav className="site-footer__links" aria-label={tFooter("support")}>
          <Link href="/privacy">{tFooter("privacy")}</Link>
          <Link href="/support">{tFooter("support")}</Link>
        </nav>
        <p>{tFooter("rights", { year: new Date().getFullYear() })}</p>
      </footer>
    </>
  );
}
