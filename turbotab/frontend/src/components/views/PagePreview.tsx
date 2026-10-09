/**
 * The page view kind (FOUNDATION §5 rule 9, designed once here): a manuscript page preview, with
 * the exhibit placed in Results, Discussion or the Supplement, or left out, and its caption.
 *
 * Design decisions, recorded:
 * - Two sheets in the app's own paper (surface and ink, not the tapestry's): the main text, with
 *   Results then Discussion, and the Supplement, which always ends with "Analyses left out" (the
 *   methods floor: every analysis run is listed, whatever its placement, FOUNDATION §8).
 * - The exhibit is drawn where it sits now as a block of its number and caption, at the page's
 *   own type size; the paper's other exhibits are quieter blocks of the same kind, so the page
 *   reads as the paper without competing. No shrunken copy of the exhibit: scaled down, its axis
 *   and table text fall below a readable size, and the exhibit itself is on the tapestry beside.
 * - Pointing at a placement option (`preview`) moves the exhibit there in the choice's data color,
 *   indigo, and leaves a gray outline where it sits now: gray is now, indigo is what the choice
 *   touches (§4). Nothing else on the page changes color.
 * - Numbers follow placement order (Table 1, Figure S1; none when left out), and a preview
 *   renumbers the page as the move would, the drafted sentences' references included.
 * - An exhibit the floor fixes (the locked primary, the held-out score) does not move, nor does
 *   one pointed at a placement the floor does not allow it; either way one line says why and
 *   where it can go (rule 7).
 * - Under a closed gate (rule 6) the drafted sentences, which carry the estimates, are not shown;
 *   one line says when they open, and the placements and captions stay. The gate has no default.
 * - A section with nothing drafted says so in one quiet line rather than leaving a blank.
 * - The table alternative lists every exhibit with where it sits and its caption.
 */
import type { Purpose } from "../stage/purposes";
import { numberInPlacementOrder } from "./adapters";
import { ExhibitTable } from "./ExhibitTable";
import { TableAlternative, useTip } from "./common/parts";
import type { Gate, PageData, PageExhibit, Placement, TableData } from "./types";
import v from "./exhibit.module.css";

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

/** A placement inside a sentence: "the Discussion", "Results". */
const IN: Record<Placement, string> = {
  results: "Results",
  discussion: "the Discussion",
  supplement: "the Supplement",
  left_out: "left out of the paper",
};

export interface PagePreviewProps {
  data: PageData;
  /** Rule 6: the line said instead of the drafted sentences while the gate is closed; null when
   *  open. */
  gate: Gate;
  /** The placement option pointed at on the card, drawn in the choice's data color. */
  preview?: Placement | null;
}

/** The exhibit the preview is about. */
export const focusOf = (data: PageData): PageExhibit => data.exhibits.find((e) => e.key === data.focus)!;

const nameOf = (e: PageExhibit) => e.number ?? "This analysis";

/** "Results or the Supplement, or leaving it out". */
function allowedWords(allowed: Placement[]): string {
  const places = allowed.filter((p) => p !== "left_out").map((p) => IN[p]);
  const list = places.length > 1 ? `${places.slice(0, -1).join(", ")} or ${places.at(-1)}` : (places[0] ?? "");
  if (!allowed.includes("left_out")) return list;
  return list ? `${list}, or leaving it out` : "leaving it out";
}

/** Where the exhibit is drawn: the pointed placement when the floor allows it, else its own, with
 *  the one line saying why it stays. */
export function shownPlacement(ex: PageExhibit, preview?: Placement | null): { at: Placement; was: Placement | null; held: string | null } {
  if (!preview || preview === ex.placement) return { at: ex.placement, was: null, held: null };
  if (ex.fixed) return { at: ex.placement, was: null, held: `${nameOf(ex)} stays in ${PLACEMENT_LABEL[ex.placement]}: ${ex.fixed}` };
  if (ex.allowed && !ex.allowed.includes(preview))
    return {
      at: ex.placement,
      was: null,
      held: `${nameOf(ex)} cannot be ${preview === "left_out" ? "left out" : `placed in ${IN[preview]}`}: the methods floor allows ${allowedWords(ex.allowed)}.`,
    };
  return { at: preview, was: ex.placement, held: null };
}

/** Each exhibit's number on the page drawn: its own at rest, and as the move would renumber the
 *  paper when a placement is previewed. */
export function shownNumbers(data: PageData, at: Placement, moved: boolean): Map<string, string | null> {
  if (!moved) return new Map(data.exhibits.map((e) => [e.key, e.number]));
  return numberInPlacementOrder(data.exhibits.map((e) => (e.key === data.focus ? { ...e, placement: at } : e)));
}

/** The drafted sentences with their exhibit references renumbered as the move would: "(Table 2)"
 *  reads "(Table 1)" when Table 1 is pointed out of the paper. */
export function renumberText(lines: string[], from: (string | null)[], to: (string | null)[]): string[] {
  const map = new Map<string, string>();
  from.forEach((f, i) => {
    if (f && to[i] && f !== to[i]) map.set(f, to[i]!);
  });
  if (!map.size) return lines;
  const re = new RegExp(`\\b(${[...map.keys()].map((k) => k.replace(/[.*+?^${}()|[\]\\]/g, "\\$&")).join("|")})\\b`, "g");
  return lines.map((t) => t.replace(re, (m) => map.get(m)!));
}

export function pageTable(data: PageData, preview?: Placement | null): TableData {
  const ex = focusOf(data);
  const { at, was } = shownPlacement(ex, preview);
  const numbers = shownNumbers(data, at, !!was);
  return {
    number: "Exhibits",
    title: "Where each exhibit sits in the paper",
    stub: "Exhibit",
    columns: [
      { key: "where", label: "Placement" },
      { key: "caption", label: "Caption" },
    ],
    rows: data.exhibits.map((e) => ({
      kind: "row" as const,
      key: e.key,
      label: numbers.get(e.key) ?? "Not numbered",
      primary: e.key === data.focus,
      cells: {
        where: {
          kind: "text" as const,
          text:
            e.key === data.focus && was
              ? `${PLACEMENT_LABEL[at]} (pointed at; now ${PLACEMENT_LABEL[was]})`
              : PLACEMENT_LABEL[e.key === data.focus ? at : e.placement],
        },
        caption: { kind: "text" as const, text: e.caption },
      },
    })),
    footnotes: [],
  };
}

type Mode = "this" | "touched" | "was" | "other";

export function PagePreview({ data, gate, preview }: PagePreviewProps) {
  const tip = useTip();
  const ex = focusOf(data);
  const { at, was, held } = shownPlacement(ex, preview);
  const numbers = shownNumbers(data, at, !!was);
  const text = (lines: string[]) =>
    renumberText(
      lines,
      data.exhibits.map((e) => e.number),
      data.exhibits.map((e) => numbers.get(e.key) ?? null),
    );

  const slot = (e: PageExhibit, mode: Mode) => {
    // The outline where the exhibit sits now carries no number: the move may give its number to
    // another exhibit, and two blocks would read as one.
    const number = mode === "was" ? null : numbers.get(e.key);
    return (
      <div
        key={`${e.key}-${mode}`}
        className={v.slot}
        data-this={mode === "this" ? "true" : undefined}
        data-touched={mode === "touched" ? "true" : undefined}
        data-was={mode === "was" ? "true" : undefined}
        data-other={mode === "other" ? "true" : undefined}
        data-slot={e.key}
        tabIndex={mode === "other" ? undefined : 0}
        {...tip.on(number ?? e.caption, mode === "was" ? `now ${IN[e.placement]}` : `${mode === "touched" ? "would be " : ""}${IN[mode === "touched" ? at : e.placement]}`)}
      >
        <p className={v.slotCaption}>
          {number ? <b>{number}. </b> : null}
          {mode === "was" ? "Where it sits now." : e.caption}
        </p>
      </div>
    );
  };

  /** Every exhibit block of a section, in the exhibit model's order: this one (moved or not) and
   *  the others. */
  const slots = (place: Placement) =>
    data.exhibits.flatMap((e) => {
      if (e.key !== data.focus) return e.placement === place ? [slot(e, "other")] : [];
      const out = [];
      if (at === place) out.push(slot(e, was ? "touched" : "this"));
      if (was === place) out.push(slot(e, "was"));
      return out;
    });

  const paragraphs = (lines: string[]) =>
    gate !== null ? null : lines.length ? (
      lines.map((t, i) => <p key={i}>{t}</p>)
    ) : (
      <p className={v.quiet}>Nothing drafted here yet.</p>
    );

  const supplement = slots("supplement");
  const leftOut = slots("left_out");
  return (
    <div className={v.view} data-exhibit-view="page" data-placement={at}>
      {gate !== null ? (
        <p className={v.fixed} role="note" data-testid="page-gate">
          {gate}
        </p>
      ) : null}
      {held ? (
        <p className={v.fixed} role="note">
          {held}
        </p>
      ) : null}
      <div className={v.pages}>
        <section aria-label="Main text">
          <p className={v.pageLabel}>Main text</p>
          <div className={v.page}>
            <h4>Results</h4>
            {paragraphs(text(data.text.results))}
            {slots("results")}
            <h4>Discussion</h4>
            {paragraphs(text(data.text.discussion))}
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
        <ExhibitTable data={pageTable(data, preview)} gate={null} />
      </TableAlternative>
    </div>
  );
}
