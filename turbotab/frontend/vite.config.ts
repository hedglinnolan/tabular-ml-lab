/// <reference types="vitest/config" />
import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";
import { dirname, resolve } from "node:path";
import { fileURLToPath } from "node:url";

const here = dirname(fileURLToPath(import.meta.url));
const fonts = resolve(here, "../../static/fonts");

// `npm run dev` proxies /api to the Python server on 8787.
// `npm run dev:mock` (mode "mock") serves the MSW worker from mock-public/ instead,
// so the production build never ships a service worker.
export default defineConfig(({ mode }) => ({
  plugins: [react()],
  publicDir: mode === "mock" ? "mock-public" : false,
  server: {
    // tokens.css loads the fonts vendored at the repository root (static/fonts).
    fs: { allow: [here, fonts] },
    proxy:
      mode === "mock"
        ? undefined
        : {
            "/api": {
              target: "http://127.0.0.1:8787",
              changeOrigin: false,
            },
          },
  },
  build: {
    outDir: "dist",
    emptyOutDir: true,
    sourcemap: true,
    // ~160 kB gzipped, served from the user's own machine; the lab route is split off.
    chunkSizeWarningLimit: 600,
  },
  test: {
    environment: "jsdom",
    globals: true,
    setupFiles: ["./src/test/setup.ts"],
    include: ["src/**/*.test.{ts,tsx}"],
  },
}));
