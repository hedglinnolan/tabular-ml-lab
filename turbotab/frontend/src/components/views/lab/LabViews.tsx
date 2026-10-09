/**
 * /lab/views (dev:mock only): every exhibit view with its fixture, in the light and the dark theme
 * side by side. Each wave of views adds a `*.lab.tsx` beside this file exporting `entries`; the
 * page collects them, so adding a view kind never edits this page.
 */
import type { ReactNode } from "react";
import "../../../explore/calm-kit/tokens.css";
import "../../../explore/calm-kit/base.css";
import tokens from "../../../explore/calm-kit/tokens.css?raw";

export interface LabEntry {
  /** the view kind (curve, calibration, decision curve, …) */
  kind: string;
  /** which fixture, and in what state */
  name: string;
  /** where the fixture comes from, in one line */
  source: string;
  render: () => ReactNode;
}

const modules = import.meta.glob<{ entries: LabEntry[] }>("./*.lab.tsx", { eager: true });
const ENTRIES: LabEntry[] = Object.keys(modules)
  .sort()
  .flatMap((k) => modules[k]!.entries);

/** The token blocks of tokens.css re-scoped to an element, so both themes render on one page. */
export function scopedTokens(css: string): string {
  const light = /:root\s*\{([^}]*)\}/.exec(css)?.[1] ?? "";
  const dark = /:root\[data-theme="dark"\]\s*\{([^}]*)\}/.exec(css)?.[1] ?? "";
  return `[data-views-theme="light"]{${light};color-scheme:light}[data-views-theme="dark"]{${dark}}`;
}

const page: React.CSSProperties = { maxWidth: 1480, margin: "0 auto", padding: "24px 24px 64px" };
const pane: React.CSSProperties = { background: "var(--canvas)", color: "var(--canvas-ink)", borderRadius: 10, padding: 18, minWidth: 0, fontFamily: "var(--font)" };

export function LabViews({ entries = ENTRIES }: { entries?: LabEntry[] }) {
  const kinds = [...new Set(entries.map((e) => e.kind))];
  return (
    <main style={page}>
      <style>{scopedTokens(tokens)}</style>
      <h1 style={{ margin: "0 0 4px", fontSize: 26 }}>Views</h1>
      <p style={{ margin: "0 0 24px", color: "var(--muted)" }}>Every exhibit view kind with its fixtures, light and dark (FOUNDATION §5, rule 9).</p>
      {kinds.map((kind) => (
        <section key={kind} style={{ marginBottom: 40 }} aria-labelledby={`kind-${kind}`}>
          <h2 id={`kind-${kind}`} style={{ fontSize: 20, margin: "0 0 12px" }}>
            {kind}
          </h2>
          {entries
            .filter((e) => e.kind === kind)
            .map((e) => (
              <article key={e.name} style={{ marginBottom: 28 }} data-lab-entry={`${kind}: ${e.name}`}>
                <h3 style={{ fontSize: 16, margin: "0 0 2px" }}>{e.name}</h3>
                <p style={{ margin: "0 0 10px", color: "var(--muted)", fontSize: 14 }}>{e.source}</p>
                <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(420px, 1fr))", gap: 16 }}>
                  {(["light", "dark"] as const).map((t) => (
                    <div key={t} data-views-theme={t} style={pane}>
                      {e.render()}
                    </div>
                  ))}
                </div>
              </article>
            ))}
        </section>
      ))}
    </main>
  );
}
