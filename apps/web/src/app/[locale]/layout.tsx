import type { Metadata } from "next";
import localFont from "next/font/local";
import { NextIntlClientProvider } from "next-intl";
import { getMessages, getTranslations, setRequestLocale } from "next-intl/server";
import { notFound } from "next/navigation";
import {
  isAppLocale,
  localeDirection,
  routing,
  type AppLocale,
} from "../../i18n/routing";
import "../globals.css";

const uthmanNaskh = localFont({
  src: "../../fonts/KFGQPC_uthman_taha_naskh_regular.ttf",
  variable: "--font-arabic",
  display: "swap",
  weight: "400",
});

type Props = {
  children: React.ReactNode;
  params: Promise<{ locale: string }>;
};

export function generateStaticParams() {
  return routing.locales.map((locale) => ({ locale }));
}

export async function generateMetadata({ params }: Props): Promise<Metadata> {
  const { locale: raw } = await params;
  const locale = isAppLocale(raw) ? raw : routing.defaultLocale;
  const t = await getTranslations({ locale, namespace: "Meta" });

  return {
    title: t("title"),
    description: t("description"),
    icons: {
      icon: "/favicon.png",
      apple: "/logo.png",
    },
    openGraph: {
      title: t("ogTitle"),
      description: t("ogDescription"),
      images: ["/logo.png"],
      locale: locale === "ar" ? "ar_AE" : "en_US",
    },
  };
}

export default async function LocaleLayout({ children, params }: Props) {
  const { locale: raw } = await params;
  if (!isAppLocale(raw)) notFound();
  const locale: AppLocale = raw;

  setRequestLocale(locale);
  const messages = await getMessages();
  const dir = localeDirection(locale);

  return (
    <html lang={locale} dir={dir} className={uthmanNaskh.variable}>
      <body suppressHydrationWarning>
        <NextIntlClientProvider messages={messages}>
          {children}
        </NextIntlClientProvider>
      </body>
    </html>
  );
}
