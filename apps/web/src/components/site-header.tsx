"use client";

import Image from "next/image";
import { useLocale, useTranslations } from "next-intl";
import { useEffect, useState } from "react";
import { Link, usePathname } from "../i18n/navigation";
import type { AppLocale } from "../i18n/routing";

type Props = {
  solid?: boolean;
};

export function SiteHeader({ solid = false }: Props) {
  const t = useTranslations("Header");
  const tBrand = useTranslations("Brand");
  const locale = useLocale() as AppLocale;
  const pathname = usePathname();
  const isHome = pathname === "/";
  const [scrolled, setScrolled] = useState(solid || !isHome);

  const nextLocale: AppLocale = locale === "ar" ? "en" : "ar";

  useEffect(() => {
    if (solid || !isHome) {
      setScrolled(true);
      return;
    }
    const onScroll = () => setScrolled(window.scrollY > 24);
    onScroll();
    window.addEventListener("scroll", onScroll, { passive: true });
    return () => window.removeEventListener("scroll", onScroll);
  }, [solid, isHome]);

  const links = [
    { href: isHome ? "#how" : "/#how", label: t("how") },
    { href: isHome ? "#demo" : "/#demo", label: t("demo") },
    { href: isHome ? "#read" : "/#read", label: t("read") },
    { href: isHome ? "#archive" : "/#archive", label: t("archive") },
  ];

  const earlyAccessHref = isHome ? "#early-access" : "/#early-access";

  return (
    <header className={`site-header${scrolled ? " site-header--solid" : ""}`}>
      <div className="site-header__inner">
        <Link className="site-header__brand" href="/">
          <Image src="/logo.png" alt="" width={36} height={36} priority />
          <span>{tBrand("name")}</span>
        </Link>

        <nav className="site-header__nav" aria-label={t("navLabel")}>
          {links.map((link) => (
            <a key={link.href} href={link.href}>
              {link.label}
            </a>
          ))}
        </nav>

        <div className="site-header__actions">
          <Link
            className="site-header__locale"
            href={pathname}
            locale={nextLocale}
            hrefLang={nextLocale}
          >
            {t("locale")}
          </Link>
          <a className="site-header__cta" href={earlyAccessHref}>
            {t("cta")}
          </a>
        </div>
      </div>
    </header>
  );
}
