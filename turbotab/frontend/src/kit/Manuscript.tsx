/**
 * The manuscript, props-driven (FOUNDATION §3, "The manuscript rail"; the kit's ManuscriptBody,
 * Rail and Overlay with their CSS, rewritten from the fixture's WalkApi to the methods in their
 * sections, P8: GET /methods/sections). A slim rail at the far left with the count of lines; opened,
 * the manuscript lies over the card column, never over the tapestry. The newest line is marked for
 * a moment. A recorded line can be clicked to change it: it opens the stage that asks it (the
 * methods text changes only through its decisions).
 */
import { useEffect, useState, type ReactNode } from "react";
import type { MethodsSection, MethodsSections } from "../api/m3-types";
import { Plain } from "./parts";
import k from "./kit.module.css";

type Entry = MethodsSection["entries"][number];

/** The newest recorded line: the highest record sequence number among the sections' lines. */
export function newestSeq(sections: readonly MethodsSection[]): number | null {
  let best: number | null = null;
  for (const sec of sections)
    for (const e of sec.entries) if (e.seq !== null && (best === null || e.seq > best)) best = e.seq;
  return best;
}

export function ManuscriptBody({
  methods,
  onOpen,
  title = "Manuscript",
  action,
}: {
  methods: MethodsSections;
  /** Open the card that asks the decision a line records (its decision kind). */
  onOpen?: (kind: string) => void;
  title?: string;
  action?: ReactNode;
}) {
  const newest = newestSeq(methods.sections);
  return (
    <div data-testid="manuscript">
      <div className={k.msHead}>
        <h2>{title}</h2>
        {action}
      </div>
      {methods.sections.map((sec) => (
        <section key={sec.key} className={k.msSection} data-testid={`ms-section-${sec.key}`}>
          <h3>
            {sec.title} <small>{sec.item}</small>
          </h3>
          <ul className={k.msList}>
            {sec.entries.map((e, i) => (
              <li
                key={`${e.record_id ?? e.source}-${i}`}
                className={k.msEntry}
                data-kind={e.source === "record" ? "recorded" : "stated"}
                data-newest={(newest !== null && e.seq === newest) || undefined}
                data-testid={e.record_id ? `ms-${e.record_id}` : undefined}
              >
                <Line entry={e} onOpen={onOpen} />
                {e.post_seal ? (
                  <span className={k.msAfter}>Recorded after the held-out rows were opened.</span>
                ) : e.after_estimates ? (
                  <span className={k.msAfter}>Recorded after the estimates were seen.</span>
                ) : null}
              </li>
            ))}
          </ul>
        </section>
      ))}
    </div>
  );
}

function Line({ entry, onOpen }: { entry: Entry; onOpen?: ((kind: string) => void) | undefined }) {
  if (entry.source === "record" && entry.kind && onOpen) {
    const kind = entry.kind;
    return (
      <button type="button" className={k.msPhrase} onClick={() => onOpen(kind)} title="Change this answer">
        <Plain text={entry.text} />
      </button>
    );
  }
  return <Plain text={entry.text} />;
}

/** The rail's open state; Escape closes it. */
export function useManuscriptRail() {
  const [open, setOpen] = useState(false);
  useEffect(() => {
    if (!open) return;
    const esc = (e: KeyboardEvent) => e.key === "Escape" && setOpen(false);
    window.addEventListener("keydown", esc);
    return () => window.removeEventListener("keydown", esc);
  }, [open]);
  return { open, setOpen };
}

export function ManuscriptRail({
  count,
  open,
  setOpen,
}: {
  count: number | null;
  open: boolean;
  setOpen: (b: boolean) => void;
}) {
  return (
    <button
      type="button"
      className={k.rail}
      aria-expanded={open}
      onClick={() => setOpen(!open)}
      data-testid="manuscript-rail"
    >
      Manuscript {count !== null ? <span className={k.railCount}>{count}</span> : null}
    </button>
  );
}

export function ManuscriptOverlay({
  methods,
  open,
  setOpen,
  onOpen,
}: {
  methods: MethodsSections | undefined;
  open: boolean;
  setOpen: (b: boolean) => void;
  onOpen: (kind: string) => void;
}) {
  if (!open) return null;
  return (
    <aside className={k.overlay} aria-label="Manuscript" data-testid="manuscript-overlay" data-kit>
      <span style={{ float: "right" }}>
        <button type="button" className={k.linkish} onClick={() => setOpen(false)}>
          Close
        </button>
      </span>
      {methods ? (
        <ManuscriptBody
          methods={methods}
          onOpen={(kind) => {
            onOpen(kind);
            setOpen(false);
          }}
        />
      ) : (
        <p className={k.note}>Reading the methods…</p>
      )}
    </aside>
  );
}
