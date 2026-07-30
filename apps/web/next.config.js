import createNextIntlPlugin from "next-intl/plugin";

const withNextIntl = createNextIntlPlugin("./src/i18n/request.ts");

/** @type {import('next').NextConfig} */
const nextConfig = {
  transpilePackages: ["@abjadi/ui"],
  images: {
    unoptimized: true,
  },
  async redirects() {
    return [
      {
        source: "/privacy",
        destination: "/ar/privacy",
        permanent: false,
      },
      {
        source: "/support",
        destination: "/ar/support",
        permanent: false,
      },
    ];
  },
};

export default withNextIntl(nextConfig);
