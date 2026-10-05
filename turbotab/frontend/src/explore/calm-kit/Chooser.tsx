/**
 * The calm chooser (#/): the four structures, each a plain card, all built on the one kit and the
 * one scenario, so the only difference between them is structure (FOUNDATION §6).
 */
import { ThemeSwitch, kit as k } from "./index";

export type StructureId = "qa" | "paper" | "quest" | "map";

export const STRUCTURES: { id: StructureId; title: string; line: string }[] = [
  { id: "qa", title: "The Q&A card", line: "One question at a time beside the canvas; the manuscript is one click away." },
  { id: "paper", title: "The paper", line: "The methods section is the page: its blanks are the questions, its phrases change answers." },
  { id: "quest", title: "The quest log", line: "The open questions as a list of objectives, walked one at a time." },
  { id: "map", title: "The map", line: "The analysis as a map of its stages; each stage opens its questions." },
];

export function CalmChooser({
  hrefs,
  built = { qa: true, paper: true, quest: true, map: true },
}: {
  hrefs: Record<StructureId | "kit", string>;
  built?: Record<StructureId, boolean>;
}) {
  const pending = STRUCTURES.filter((s) => !built[s.id]);
  return (
    <div className={k.page}>
      <header className={k.top}>
        <span className={k.brand}>TurboTab</span>
        <div className={k.topright}>
          <a className={k.linkish} href={hrefs.kit}>
            The kit
          </a>
          <ThemeSwitch />
        </div>
      </header>
      <main className={k.chooser}>
        <h1>Four structures for one analysis</h1>
        <p>
          The same NHANES question, the same engine answers and the same parts, organized four ways. Each walks from the first draft
          to the locked Table 2.
        </p>
        {pending.length > 0 && (
          <p className={k.chooserNote} data-testid="chooser-pending">
            {pending.length === STRUCTURES.length ? "None of the four structures is built yet." : `${pending.length} of the four structures are not built yet.`}{" "}
            Only the shared parts exist so far: you can <a href={hrefs.qa}>walk the shared reference questions</a> or{" "}
            <a href={hrefs.kit}>see every canvas layout in the kit</a>.
          </p>
        )}
        <div className={k.cards}>
          {STRUCTURES.map((s) =>
            built[s.id] ? (
              <a key={s.id} className={k.pcard} href={hrefs[s.id]} data-testid={`structure-${s.id}`}>
                <h2>{s.title}</h2>
                <p>{s.line}</p>
                <span>Open</span>
              </a>
            ) : (
              <div key={s.id} className={k.pcard} data-off="true" aria-disabled="true" data-testid={`structure-${s.id}`}>
                <h2>{s.title}</h2>
                <p>{s.line}</p>
                <span>Not built yet</span>
              </div>
            ),
          )}
        </div>
      </main>
    </div>
  );
}
