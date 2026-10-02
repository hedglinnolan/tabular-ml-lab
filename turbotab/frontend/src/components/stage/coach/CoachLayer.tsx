/**
 * The coach's notes on a view (M2_CONTRACT §6): amber, at most two, each on a line of the band at
 * the top of the view with a straight leader to what it is about. They point at the picture and
 * never pre-select anything: a note is a fact about the user's rows, and the user decides.
 *
 * The layer sits over its view (absolute, inset 0, no pointer events); the view reserves the band's
 * height at its top (`bandHeight`) and resolves each note's anchor to a span in its own pixels.
 * Notes are measured once drawn, so a long note on a narrow card stays inside it.
 */
import { useLayoutEffect, useMemo, useRef, useState } from "react";
import type { CoachNote } from "../../../api/m2-stage-types";
import { Rich } from "../text";
import { estimateWidth, MAX_NOTES, NOTE_H, placeNotes, type Span } from "./place";
import s from "./coach.module.css";

interface Props {
  notes: CoachNote[];
  spans: (Span | null)[];
  /** The view's width, px. */
  width: number;
}

export function CoachLayer({ notes, spans, width }: Props) {
  const shown = notes.slice(0, MAX_NOTES);
  const refs = useRef<(HTMLParagraphElement | null)[]>([]);
  const key = shown.map((n) => n.text).join("|");
  const [measured, setMeasured] = useState<{ key: string; widths: number[] } | null>(null);
  const widths =
    measured && measured.key === key ? measured.widths : shown.map((n) => estimateWidth(n.text));
  useLayoutEffect(() => {
    const next = shown.map((n, i) => {
      const el = refs.current[i];
      // scrollWidth: the note's own one-line width, even while its box is clamped to the view.
      return el && el.scrollWidth > 0 ? Math.ceil(el.scrollWidth) + 2 : estimateWidth(n.text);
    });
    setMeasured((m) =>
      m && m.key === key && m.widths.every((w, i) => Math.abs(w - next[i]!) < 1) ? m : { key, widths: next },
    );
    // Measured per set of notes and per width; `shown` is derived from `key`.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [key, width]);
  const placed = useMemo(() => placeNotes(widths, spans, width), [widths, spans, width]);
  if (!shown.length || width <= 0) return null;
  return (
    <div className={s.layer} aria-hidden={false} data-coach={shown.length}>
      <svg className={s.leaders} aria-hidden="true">
        {placed.map((p, i) => {
          const span = spans[i];
          if (p.leaderX === null || !span) return null;
          const top = p.top + NOTE_H - 2;
          const x = p.leaderX;
          const a = Math.max(0, Math.min(span.x0, span.x1));
          const b = Math.min(width, Math.max(span.x0, span.x1));
          return (
            <g key={i} data-anchor-mark={span.mark}>
              <line x1={x} x2={x} y1={top} y2={span.y} className={s.leader} />
              {span.mark === "bracket" ? (
                <path d={`M${a + 0.5},${span.y - 4}V${span.y}H${b - 0.5}V${span.y - 4}`} className={s.bracket} />
              ) : span.mark === "ring" ? (
                <circle cx={x} cy={span.y} r={6} className={s.bracket} />
              ) : span.mark === "tick" ? (
                <circle cx={x} cy={span.y} r={2.4} className={s.dot} />
              ) : null}
            </g>
          );
        })}
      </svg>
      {shown.map((n, i) => {
        const p = placed[i]!;
        return (
          <p
            key={n.text}
            ref={(el) => {
              refs.current[i] = el;
            }}
            className={s.note}
            style={{ top: p.top, left: p.left, maxWidth: width }}
            data-testid="coach-note"
            data-purpose="coach_note"
            data-anchor={n.anchor.kind}
          >
            <Rich text={n.text} />
          </p>
        );
      })}
    </div>
  );
}

/** The note texts as one accessible line, for views that draw them inline instead (the row flow). */
export function coachLabel(notes: CoachNote[]): string {
  return notes.map((n) => n.text.replace(/`/g, "")).join(" · ");
}
