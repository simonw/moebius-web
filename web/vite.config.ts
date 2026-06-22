import { defineConfig, type Plugin } from "vite";
import fs from "node:fs";
import path from "node:path";

const COI = {
  "Cross-Origin-Opener-Policy": "same-origin",
  "Cross-Origin-Embedder-Policy": "require-corp",
};

// Source of truth for the onnxruntime-web runtime assets (.wasm / glue .mjs).
const ORT_SRC = path.resolve(__dirname, "node_modules/onnxruntime-web/dist");
const ortAssets = () =>
  fs.readdirSync(ORT_SRC).filter((f) => f.endsWith(".wasm") || f.endsWith(".mjs"));

// Serve/copy the ORT runtime at /ort/*. They must NOT go through /public, because ORT
// loads its glue via dynamic import() and Vite would try to module-transform a /public .mjs.
function ortRuntime(): Plugin {
  const mime: Record<string, string> = {
    ".wasm": "application/wasm",
    ".mjs": "text/javascript",
    ".js": "text/javascript",
  };
  const handler = (req: any, res: any, next: any) => {
    const url: string = req.url || "";
    if (!url.startsWith("/ort/")) return next();
    const name = path.basename(url.split("?")[0]);
    const fp = path.join(ORT_SRC, name);
    if (!fs.existsSync(fp)) return next();
    res.setHeader("Content-Type", mime[path.extname(fp)] || "application/octet-stream");
    for (const [k, v] of Object.entries(COI)) res.setHeader(k, v);
    fs.createReadStream(fp).pipe(res);
  };
  return {
    name: "ort-runtime",
    configureServer(server) {
      server.middlewares.use(handler);
    },
    configurePreviewServer(server) {
      server.middlewares.use(handler);
    },
    // copy the assets into the build output at dist/ort/*
    closeBundle() {
      const outDir = path.resolve(__dirname, "dist/ort");
      fs.mkdirSync(outDir, { recursive: true });
      for (const f of ortAssets()) fs.copyFileSync(path.join(ORT_SRC, f), path.join(outDir, f));
    },
  };
}

export default defineConfig({
  // GitHub Pages project site is served under /<repo>/. Set BASE_PATH=/moebius-web/ in CI.
  base: process.env.BASE_PATH || "/",
  plugins: [ortRuntime()],
  server: { headers: COI, fs: { allow: [".."] } },
  preview: { headers: COI },
  optimizeDeps: { exclude: ["onnxruntime-web"] },
  build: { target: "es2022" },
});
