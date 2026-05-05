/** @type {import('next').NextConfig} */
const nextConfig = {
  output: "export",
  trailingSlash: true,
  basePath: "/PepSeqPred",
  assetPrefix: "/PepSeqPred",
  images: {
    unoptimized: true
  }
};

export default nextConfig;
