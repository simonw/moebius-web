import { defineConfig } from "vite";

// ORT-Web ships .wasm/.mjs assets that must be served as-is and not pre-bundled.
// We also need cross-origin isolation headers for multi-threaded WASM (SharedArrayBuffer).
export default defineConfig({
  server: {
    headers: {
      "Cross-Origin-Opener-Policy": "same-origin",
      "Cross-Origin-Embedder-Policy": "require-corp",
    },
    fs: { allow: [".."] },
  },
  preview: {
    headers: {
      "Cross-Origin-Opener-Policy": "same-origin",
      "Cross-Origin-Embedder-Policy": "require-corp",
    },
  },
  optimizeDeps: { exclude: ["onnxruntime-web"] },
  build: { target: "es2022" },
});
