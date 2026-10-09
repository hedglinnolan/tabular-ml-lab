/**
 * The page view kind (FOUNDATION §5 rule 9, designed once here): a manuscript page preview, with
 * the exhibit placed in Results, Discussion or the Supplement, or left out, and its caption.
 *
 * Design decisions, recorded:
 * - Two sheets in the app's own paper (surface and ink, not the tapestry's): the main text, with
 *   Results then Discussion, and the Supplement, which always ends with "Analyses left out" (the
 *   methods floor: every analysis run is listed, whatever its placement, FOUNDATION §8).
 * - The exhibit is drawn where it sits now, with its number and caption and, when given, a small
 *   copy of the exhibit itself; the paper's other exhibits are quiet caption-only blocks, so the
 *   page reads as the paper without competing.
 * - Pointing at a placement option (`preview`) moves the exhibit there in the choice's data color,
 *   indigo, and leaves a gray outline where it sits now: gray is now, indigo is what the choice
 *   touches (§4). Nothing else on the page changes color.
 * - An exhibit the floor fixes (the locked primary, the held-out score) does not move; pointing
 *   elsewhere says why in one line (rule 7).
 * - A section with nothing drafted says so in one quiet line rather than leaving a blank.
 * - The table alternative lists every exhibit with where it sits and its caption.
 */
import type { ReactNode } from "react";
import type { Purpose } from "../stage/purposes";
import { ExhibitTable } from "./ExhibitTable";
import { TableAlternative, useTip } from "./shared";
import type { PageData, PageExhibit, Placement, TableData } from "./types";
import v from "./views.module.css";

export const PAGE_PURPOSE: Purpose = {
  question: "provenance",
  answer: "where the exhibit sits in the paper, with its caption, and what moving it changes",
};

export const PLACEMENT_LABEL: Record<Placement, string> = {
  results: "Results",
  discussion: "Discussion",
  supplement: "Supplement",
  left_out: "Left out of the paper",
};

export interface PagePreviewProps {
  data: PageData;
  /** The placement option pointed at on the card, drawn in the choice's data color. */
  preview?: Placement | null;
  /** A small copy of the exhibit itself, shown in its slot. */
  thumb?: ReactNode;
}

/** Where the exhibit is drawn: the pointed placement when the floor allows it, else its own. */
export function shownPlacement(ex: PageExhibit, preview?: Placement | null): { at: Placement; was: Placement | null; held: boolean } {
  if (!preview || preview === ex.placement) return { at: ex.placement, was: null, held: false };
  if (ex.fixed) return { at: ex.placement, was: null, held: true };
  return { at: preview, was: ex.placement, held: false };
}

/** Tables before figures, each by number: "Table 2" before "Table 10". */
export function paperOrder(a: PageExhibit, b: PageExhibit): number {
  if (a.kind !== b.kind) return a.kind === "table" ? -1 : 1;
  return a.number.localeCompare(b.number, "en", { numeric: true });
}

export function pageTable(data: PageData, preview?: Placement | null): TableData {
  const { at, was } = shownPlacement(data.exhibit, preview);
  const all = [data.exhibit, ...(data.others ?? [])];
  return {
    number: "Exhibits",
    title: "Where each exhibit sits in the paper",
    stub: "Exhibit",
    columns: [
      { key: "where", label: "Placement" },
      { key: "caption", label: "Caption" },
    ],
    rows: all.map((e) => ({
      kind: "row" as const,
      key: e.key,
      label: e.number,
      primary: e.key === data.exhibit.key,
      cells: {
        where: {
          kind: "text" as const,
          text:
            e.key === data.exhibit.key && was
              ? `${PLACEMENT_LABEL[at]} (pointed at; now ${PLACEMENT_LABEL[was]})`
              : PLACEMENT_LABEL[e.key === data.exhibit.key ? at : e.placement],
        },
        caption: { kind: "text" as const, text: e.caption },
      },
    })),
    footnotes: [],
  };
}

export function PagePreview({ data, preview, thumb }: PagePreviewProps) {
  const tip = useTip();
  const ex = data.exhibit;
  const { at, was, held } = shownPlacement(ex, preview);
  const others = data.others ?? [];

  const slot = (e: PageExhibit, mode: "this" | "touched" | "was" | "other") => (
    <div
      key={`${e.key}-${mode}`}
      className={v.slot}
      data-this={mode === "this" ? "true" : undefined}
      data-touched={mode === "touched" ? "true" : undefined}
      data-was={mode === "was" ? "true" : undefined}
      data-other={mode === "other" ? "true" : undefined}
      data-slot={e.key}
      tabIndex={mode === "other" ? undefined : 0}
      {...tip.on(e.number, mode === "was" ? `now in ${PLACEMENT_LABEL[e.placement]}` : `in ${PLACEMENT_LABEL[mode === "touched" ? at : e.placement]}`)}
    >
      <p className={v.slotCaption}>
        <b>{e.number}.</b> {mode === "was" ? "Sits here now." : e.caption}
      </p>
      {(mode === "this" || mode === "touched") && thumb ? <div className={v.thumb}>{thumb}</div> : null}
    </div>
  );

  /** Every exhibit block in one place, in number order: this one (moved or not) and the others. */
  const slots = (place: Placement) => {
    const out: { e: PageExhibit; mode: "this" | "touched" | "was" | "other" }[] = others.filter((o) => o.placement === place).map((e) => ({ e, mode: "other" as const }));
    if (at === place) out.push({ e: ex, mode: was ? "touched" : "this" });
    if (was === place) out.push({ e: ex, mode: "was" });
    return out.sort((a, b) => paperOrder(a.e, b.e)).map(({ e, mode }) => slot(e, mode));
  };

  const paragraphs = (lines: string[]) =>
    lines.length ? (
      lines.map((t, i) => <p key={i}>{t}</p>)
    ) : (
      <p className={v.quiet}>Nothing drafted here yet.</p>
    );

  const supplement = slots("supplement");
  const leftOut = slots("left_out");
  return (
    <div className={v.view} data-exhibit-view="page" data-placement={at}>
      {held ? (
        <p className={v.fixed} role="note">
          {ex.number} stays in {PLACEMENT_LABEL[ex.placement]}: {ex.fixed}
        </p>
      ) : null}
      <div className={v.pages}>
        <section aria-label="Main text">
          <p className={v.pageLabel}>Main text</p>
          <div className={v.page}>
            <h4>Results</h4>
            {paragraphs(data.text.results)}
            {slots("results")}
            <h4>Discussion</h4>
            {paragraphs(data.text.discussion)}
            {slots("discussion")}
          </div>
        </section>
        <section aria-label="Supplement">
          <p className={v.pageLabel}>Supplement</p>
          <div className={v.page}>
            {supplement.length ? supplement : <p className={v.quiet}>No exhibit in the supplement yet.</p>}
            <h4>Analyses left out</h4>
            {leftOut.length ? leftOut : <p className={v.quiet}>None: every analysis run has a place in the paper.</p>}
          </div>
        </section>
      </div>
      {tip.node}
      <TableAlternative>
        <ExhibitTable data={pageTable(data, preview)} />
      </TableAlternative>
    </div>
  );
}
