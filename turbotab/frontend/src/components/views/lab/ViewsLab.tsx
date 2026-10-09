/**
 * /lab/views (dev:mock only): every exhibit view kind (FOUNDATION §5 rule 9) with its fixture and
 * its states, each drawn in the light and the dark theme side by side. The kinds come from the
 * `*.lab.tsx` files under src/components/views, so each family's file adds itself.
 */
import "../../../explore/calm-kit/tokens.css";
import "../../../explore/calm-kit/base.css";
import tokens from "../../../explore/calm-kit/tokens.css?raw";
import { QUESTIONS } from "../../stage/purposes";
import type { LabEntry } from "./entry";
import { scopedThemes } from "./themes";
import l from "./lab.module.css";

const MODULES = import.meta.glob<{ entries: LabEntry[] }>("../**/*.lab.tsx", { eager: true });

export const ENTRIES: LabEntry[] = Object.values(MODULES)
  .flatMap((m) => m.entries)
  .sort((a, b) => a.order - b.order);

// Vitest serves CSS as empty text; the browser gets the file.
const THEMES = tokens ? scopedThemes(tokens) : "";

/** A kind's anchor on the page ("decision curve" → "decision-curve"). */
export const anchorOf = (kind: string) => kind.replace(/\s+/g, "-");

export function ViewsLab({ entries = ENTRIES }: { entries?: LabEntry[] }) {
  return (
    <div className={l.lab}>
      <style>{THEMES}</style>
      <header className={l.head}>
        <h1>View kinds</h1>
        <p>Each exhibit view kind with its fixture and its states, in the light and the dark theme.</p>
        <nav className={l.toc} aria-label="View kinds">
          {entries.map((e) => (
            <a key={e.kind} href={`#${anchorOf(e.kind)}`}>
              {e.kind}
            </a>
          ))}
        </nav>
      </header>
      {entries.map((e) => (
        <section key={e.kind} id={anchorOf(e.kind)} className={l.kind} aria-labelledby={`${anchorOf(e.kind)}-h`}>
          <h2 id={`${anchorOf(e.kind)}-h`}>{e.kind}</h2>
          <p className={l.purpose}>
            Answers “{QUESTIONS[e.purpose.question]}”: {e.purpose.answer}.
          </p>
          {e.samples.map((s) => (
            <div key={s.label} className={l.sample}>
              <h3>{s.label}</h3>
              {s.source ? <p className={l.source}>{s.source}</p> : null}
              <div className={s.wide ? l.stack : l.pair}>
                {(["light", "dark"] as const).map((t) => (
                  <div key={t} className={l.frame} data-lab-theme={t}>
                    <span className={l.theme}>{t}</span>
                    {s.render()}
                  </div>
                ))}
              </div>
            </div>
          ))}
        </section>
      ))}
    </div>
  );
}

