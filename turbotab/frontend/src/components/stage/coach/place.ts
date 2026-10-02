/**
 * Where the coach's notes sit (M2_CONTRACT §6; Nolan, 2026-10-01): at most two notes per view, in
 * the coach's amber, each on its own line of a band above the picture, with a straight leader down
 * to what it is about — a column, a range on the value axis, some points, a row-flow step. Pure, so
 * the one rule that matters is tested without a browser: a note never leaves its view, and its
 * leader starts under the note it belongs to.
 *
 * The view resolves each anchor to a span in its own pixels (`resolve` in each view); notes whose
 * anchor the view cannot draw still sit on the band, without a leader.
 */

/** One line of the band: a note is one line of serif text (≤ 12 words). */
export const NOTE_H = 22;
/** Space between the band and the picture below it. */
export const BAND_GAP = 6;
/** A note starts this far left of its leader, so the leader drops from near its start. */
export const LEAD_IN = 14;
/** At most two notes per view (M2_CONTRACT §6). */
export const MAX_NOTES = 2;

/** What a leader points at, in the view's own pixels (the band's height already included). */
export interface Span {
  x0: number;
  x1: number;
  /** Where the leader ends. */
  y: number;
  /** bracket: a stretch of an axis · tick: one place · ring: points · none: the leader alone. */
  mark: "bracket" | "tick" | "ring" | "none";
}

export interface Placed {
  /** The note's line (0 or 1) after ordering to keep leaders clear of the other note. */
  line: number;
  left: number;
  top: number;
  width: number;
  /** null: the anchor is not drawn in this view; the note stands on the band alone. */
  leaderX: number | null;
}

/** A secondary card at least this wide (a lone thumbnail spans the stage) keeps its notes. */
export const NOTES_MIN_W = 520;

/**
 * The notes a view draws: all of them (at most two) on a primary card, and on a secondary card
 * wide enough to hold them; none on a narrow thumbnail, where they would crowd the picture out.
 * Every view kind answers the same way, so a note never disappears for one kind and not another.
 */
export function notesFor<T>(coach: readonly T[] | null | undefined, compact: boolean, width: number): T[] {
  if (!coach?.length) return [];
  return !compact || width >= NOTES_MIN_W ? coach.slice(0, MAX_NOTES) : [];
}

export function bandHeight(n: number): number {
  const k = Math.min(MAX_NOTES, Math.max(0, n));
  return k ? k * NOTE_H + BAND_GAP : 0;
}

const clamp = (v: number, lo: number, hi: number) => Math.max(lo, Math.min(hi, v));

/** Where a span's leader drops: the middle of the part of it inside the view. */
export function leaderOf(span: Span | null, box: number): number | null {
  if (!span) return null;
  const a = clamp(Math.min(span.x0, span.x1), 0, box);
  const b = clamp(Math.max(span.x0, span.x1), 0, box);
  return Math.round(((a + b) / 2) * 2) / 2;
}

function placeOne(width: number, lx: number | null, box: number, line: number): Placed {
  const w = Math.max(0, Math.min(width, box));
  const left = lx === null ? 0 : clamp(lx - LEAD_IN, 0, Math.max(0, box - w));
  return { line, left, top: line * NOTE_H, width: w, leaderX: lx };
}

/** A leader drops from its note's line through every line below it: does it cross that note? */
function crossings(placed: Placed[]): number {
  let n = 0;
  for (const p of placed) {
    if (p.leaderX === null) continue;
    for (const q of placed) {
      if (q === p || q.line <= p.line) continue;
      if (p.leaderX > q.left - 2 && p.leaderX < q.left + q.width + 2) n++;
    }
  }
  return n;
}

/**
 * Place up to two notes of the given (measured) widths over a view `box` pixels wide. Notes keep
 * their order unless swapping their lines keeps a leader from passing through the other note.
 */
export function placeNotes(widths: number[], spans: (Span | null)[], box: number): Placed[] {
  const n = Math.min(MAX_NOTES, widths.length);
  const lx = Array.from({ length: n }, (_, i) => leaderOf(spans[i] ?? null, box));
  const inOrder = Array.from({ length: n }, (_, i) => placeOne(widths[i]!, lx[i]!, box, i));
  if (n < 2 || crossings(inOrder) === 0) return inOrder;
  const swapped = Array.from({ length: n }, (_, i) => placeOne(widths[i]!, lx[i]!, box, n - 1 - i));
  return crossings(swapped) < crossings(inOrder) ? swapped : inOrder;
}

/** An estimate of a note's width before it is measured: serif 12.5 px, plus padding. */
export function estimateWidth(text: string): number {
  return Math.round(text.replace(/`/g, "").length * 6.3 + 18);
}
