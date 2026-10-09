/**
 * /lab/views (dev:mock only): every exhibit view kind on its fixtures, light and dark side by side,
 * so a reviewer judges each view once, in both themes, before any exhibit draws it (FOUNDATION §5
 * rule 9). Each view folder lists its entries in a `*.lab.tsx` file; this page gathers them.
 */
import "../../../explore/calm-kit/tokens.css";
import "../../../explore/calm-kit/base.css";
import tokens from "../../../explore/calm-kit/tokens.css?raw";
import { QUESTIONS } from "../../stage/purposes";
import type { LabEntry } from "./entry";
import l from "./lab.module.css";

const MODULES = import.meta.glob<LabEntry[]>("../**/*.lab.tsx", { import: "entries", eager: true });

export const LAB_ENTRIES: LabEntry[] = Object.keys(MODULES)
  .sort()
  .flatMap((k) => MODULES[k]!);

/**
 * The calm tokens scoped to a theme: the light block of tokens.css on [data-lab-theme="light"], its
 * dark block on [data-lab-theme="dark"], so both themes render on one page from the same file.
 */
export function scopedTokens(css: string): string {
  const light = /:root\s*\{([^}]*)\}/.exec(css)?.[1] ?? "";
  const dark = /:root\[data-theme="dark"\]\s*\{([^}]*)\}/.exec(css)?.[1] ?? "";
  return `[data-lab-theme="light"]{${light};color-scheme:light}[data-lab-theme="dark"]{${dark}}`;
}

const SCOPED = scopedTokens(tokens);

export function LabViews() {
  const kinds = [...new Set(LAB_ENTRIES.map((e) => e.kind))];
  return (
    <main className={l.page}>
      <style>{SCOPED}</style>
      <header className={l.header}>
        <h1>Exhibit views</h1>
        <p>
          Each view kind on its fixtures, light and dark. {LAB_ENTRIES.length} entries: {kinds.join(", ")}.
        </p>
        <nav className={l.nav} aria-label="Views">
          {LAB_ENTRIES.map((e) => (
            <a key={e.id} href={`#${e.id}`}>
              {e.title}
            </a>
          ))}
        </nav>
      </header>
      {LAB_ENTRIES.map((e) => (
        <section key={e.id} id={e.id} className={l.entry} aria-labelledby={`${e.id}-h`}>
          <h2 id={`${e.id}-h`}>{e.title}</h2>
          <p className={l.meta}>
            {QUESTIONS[e.purpose.question]} It shows {e.purpose.answer}. <span>{e.source}.</span>
          </p>
          <div className={l.pair}>
            {(["light", "dark"] as const).map((theme) => (
              <div key={theme} className={l.theme} data-lab-theme={theme}>
                <span className={l.themeName}>{theme === "light" ? "Light" : "Dark"}</span>
                {e.render()}
              </div>
            ))}
          </div>
        </section>
      ))}
    </main>
  );
}
