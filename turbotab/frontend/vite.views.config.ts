/**
 * The static build of /lab/views: `npm run build:views`.
 *
 * views.html → <VIEWS_OUT>/index.html (default build/views-dist beside this file, which git
 * ignores): every exhibit view kind on its fixtures in both themes, no server and no mock
 * worker. The folder runs from any static host or from disk: relative URLs (base "./"), one script inlined into index.html (a browser refuses module scripts
 * from file://), the stylesheet beside it with the font inlined (a browser refuses fonts from
 * file:// too).
 */
import { readdirSync, readFileSync, renameSync, rmSync, writeFileSync } from "node:fs";
import { dirname, resolve } from "node:path";
import { fileURLToPath } from "node:url";
import react from "@vitejs/plugin-react";
import { defineConfig, type Plugin } from "vite";

const here = dirname(fileURLToPath(import.meta.url));
const outDir = resolve(process.env.VIEWS_OUT ?? resolve(here, "build", "views-dist"));

/** After the bundle is written: inline the one script, drop `crossorigin`, name the page index.html. */
function staticFolder(): Plugin {
  return {
    name: "turbotab-views-static-folder",
    apply: "build",
    closeBundle() {
      const page = resolve(outDir, "views.html");
      let html = readFileSync(page, "utf8");
      html = html.replace(
        /<script type="module" crossorigin src="\.\/(assets\/[^"]+\.js)"><\/script>/,
        (_m, file: string) => {
          const code = readFileSync(resolve(outDir, file), "utf8").replace(/<\/script/gi, "<\\/script");
          rmSync(resolve(outDir, file));
          return `<script type="module">${code}</script>`;
        },
      );
      html = html.replace(/ crossorigin(?=[ >])/g, "");
      if (/<script[^>]+src=/.test(html)) throw new Error("views build: a script was left external");
      writeFileSync(page, html);
      renameSync(page, resolve(outDir, "index.html"));
      for (const f of readdirSync(resolve(outDir, "assets"))) {
        if (f.endsWith(".js")) throw new Error(`views build: ${f} is a second chunk`);
      }
    },
  };
}

export default defineConfig({
  plugins: [react(), staticFolder()],
  base: "./",
  publicDir: false,
  build: {
    outDir,
    emptyOutDir: true,
    sourcemap: false,
    modulePreload: false,
    // The fixtures are captured engine output; this build is for review, not speed.
    chunkSizeWarningLimit: 12000,
    assetsInlineLimit: (file) => (file.endsWith(".woff2") ? true : undefined),
    rolldownOptions: {
      input: resolve(here, "views.html"),
      output: { codeSplitting: false },
    },
  },
});
