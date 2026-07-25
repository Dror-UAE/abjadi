import Image from "next/image";
import { getTranslations, setRequestLocale } from "next-intl/server";
import { DemoVideo } from "../../components/demo-video";
import { SiteHeader } from "../../components/site-header";
import { isAppLocale, routing } from "../../i18n/routing";

type Props = {
  params: Promise<{ locale: string }>;
};

export default async function HomePage({ params }: Props) {
  const { locale: raw } = await params;
  const locale = isAppLocale(raw) ? raw : routing.defaultLocale;
  setRequestLocale(locale);

  const tBrand = await getTranslations("Brand");
  const tHero = await getTranslations("Hero");
  const tHow = await getTranslations("How");
  const tDemo = await getTranslations("Demo");
  const tProcess = await getTranslations("Process");
  const tDownload = await getTranslations("Download");
  const tFooter = await getTranslations("Footer");

  return (
    <>
      <SiteHeader />

      <section className="hero">
        <div className="hero__media" aria-hidden="true">
          <Image
            src="/hero.png"
            alt=""
            fill
            priority
            sizes="100vw"
            quality={90}
          />
          <div className="hero__shade" />
        </div>

        <div className="hero__content">
          <Image
            className="hero__mark"
            src="/logo.png"
            alt=""
            width={88}
            height={88}
            priority
          />

          <h1 className="hero__brand">{tBrand("name")}</h1>

          <p className="hero__headline">{tHero("headline")}</p>
          <p className="hero__lede">{tHero("lede")}</p>

          <div className="hero__actions">
            <a className="btn btn--primary" href="#download">
              {tHero("start")}
            </a>
            <a className="btn btn--ghost" href="#how">
              {tHero("how")}
            </a>
          </div>
        </div>
      </section>

      <main>
        <section className="section" id="how" aria-labelledby="how-title">
          <div className="section__intro">
            <p className="section__eyebrow">{tHow("eyebrow")}</p>
            <h2 className="section__title" id="how-title">
              {tHow("title")}
            </h2>
            <p className="section__copy">{tHow("copy")}</p>
          </div>

          <div className="steps">
            <article className="step">
              <div className="step__phone">
                <Image
                  src="/mobile-mock-2.png"
                  alt={tHow("captureAlt")}
                  width={220}
                  height={476}
                  sizes="220px"
                />
              </div>
              <h3>{tHow("captureTitle")}</h3>
              <p>{tHow("captureCopy")}</p>
            </article>

            <article className="step" id="scan">
              <div className="step__phone">
                <Image
                  src="/mobile-mock-4.png"
                  alt={tHow("scanAlt")}
                  width={220}
                  height={476}
                  sizes="220px"
                />
              </div>
              <h3>{tHow("scanTitle")}</h3>
              <p>{tHow("scanCopy")}</p>
            </article>

            <article className="step" id="read">
              <div className="step__phone">
                <Image
                  src="/mobile-mock-5.png"
                  alt={tHow("readAlt")}
                  width={220}
                  height={476}
                  sizes="220px"
                />
              </div>
              <h3>{tHow("readTitle")}</h3>
              <p>{tHow("readCopy")}</p>
            </article>

            <article className="step" id="archive">
              <div className="step__phone">
                <Image
                  src="/mobile-mock-6.png"
                  alt={tHow("archiveAlt")}
                  width={220}
                  height={476}
                  sizes="220px"
                />
              </div>
              <h3>{tHow("archiveTitle")}</h3>
              <p>{tHow("archiveCopy")}</p>
            </article>
          </div>
        </section>

        <section className="section demo" id="demo" aria-labelledby="demo-title">
          <div className="section__intro">
            <p className="section__eyebrow">{tDemo("eyebrow")}</p>
            <h2 className="section__title" id="demo-title">
              {tDemo("title")}
            </h2>
            <p className="section__copy">{tDemo("copy")}</p>
          </div>

          <DemoVideo
            title={tDemo("videoTitle")}
            poster="/mobile-mock-6.png"
          />
        </section>

        <section className="band" aria-labelledby="process-title">
          <div className="band__inner">
            <div className="band__copy">
              <h2 id="process-title">{tProcess("title")}</h2>
              <p>{tProcess("copy")}</p>
              <a className="btn btn--primary" href="#download">
                {tProcess("cta")}
              </a>
            </div>
            <div className="band__visual">
              <Image
                src="/mobile-mock-4.png"
                alt={tProcess("analysisAlt")}
                width={200}
                height={432}
                sizes="200px"
              />
              <Image
                src="/mobile-mock-1.png"
                alt={tProcess("scanAlt")}
                width={200}
                height={432}
                sizes="200px"
              />
            </div>
          </div>
        </section>

        <section
          className="closing"
          id="download"
          aria-labelledby="download-title"
        >
          <Image
            className="closing__logo"
            src="/logo.png"
            alt=""
            width={72}
            height={72}
          />
          <h2 id="download-title">{tDownload("title")}</h2>
          <p>{tDownload("copy")}</p>
          <div className="closing__actions">
            <a
              className="btn btn--solid"
              href="#"
              aria-label={tDownload("appStoreSoon")}
            >
              App Store{" "}
              <Image
                src="/App_Store.svg.webp"
                alt=""
                width={16}
                height={16}
              />
            </a>
            <a
              className="btn btn--solid"
              href="#"
              aria-label={tDownload("playStoreSoon")}
            >
              Google Play{" "}
              <Image
                src="/Google_Play_icon.svg.webp"
                alt=""
                width={16}
                height={16}
              />
            </a>
          </div>
          <p className="closing__access">
            {tDownload("access")}{" "}
            <a href="mailto:info@abjadi.ai">{tDownload("accessEmail")}</a>
          </p>
        </section>
      </main>

      <footer className="site-footer">
        <p>{tFooter("rights", { year: new Date().getFullYear() })}</p>
      </footer>
    </>
  );
}
