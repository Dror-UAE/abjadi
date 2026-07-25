import { getTranslations } from "next-intl/server";
import { Link } from "../../i18n/navigation";

export default async function NotFound() {
  const t = await getTranslations("NotFound");
  const tBrand = await getTranslations("Brand");

  return (
    <main className="closing">
      <h1>{t("title")}</h1>
      <p>{t("copy")}</p>
      <Link className="btn btn--solid" href="/">
        {tBrand("name")}
      </Link>
    </main>
  );
}
