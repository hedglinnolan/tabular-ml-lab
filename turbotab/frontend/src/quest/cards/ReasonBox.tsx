/**
 * The reason for an answer (G14; the owner's Q11, 2026-11-04: "an optional one-line box beside each
 * recorded answer, collapsed by default"). The server keeps a decision's reason as its record's
 * note (`POST …/decisions?note=`, SWARM_ENGINE_PLAN E4): in the decision log, which the view serves
 * and the export carries as `decisions.jsonl`, and never in the methods.
 *
 * Collapsed, it is one quiet link; opened, one line of text, focused, which says it is kept as the
 * decision's note, the server's word for it. What is typed goes with the next decision the card
 * records (cards/predict/card.tsx `useCard`), and the box is collapsed and empty again once that
 * decision is recorded; a refused press keeps it, and a note the server refuses as too long is
 * answered here, under the text to shorten. It never starts at the reason on record, since a new
 * answer's reason is its own. The reason on the record that answers what the card shows as
 * recorded is shown above it, in the person's own words, as recorded.
 */
import { useEffect, useId, useRef, type ReactNode } from "react";
import c from "./ReasonBox.module.css";

/** The reason being written for the next record: whether the box is open, and its text. */
export interface Reason {
  open: boolean;
  text: string;
}

export const NO_REASON: Reason = { open: false, text: "" };

/** The note a reason sends: its text without the blanks around it (as the server keeps it,
 *  decisions.py `check_note`), or none where nothing is left. */
export function noteOf(reason: Reason): string | null {
  const text = reason.text.trim();
  return text === "" ? null : text;
}

export function ReasonBox({
  reason,
  onChange,
  onRecord,
  answer = null,
}: {
  reason: Reason;
  onChange: (reason: Reason) => void;
  /** The note on the record that answers what the card shows as recorded, or null. */
  onRecord: string | null;
  /** The server's refusal of the note sent, in its words, or nothing. */
  answer?: ReactNode;
}) {
  const id = useId();
  const input = useRef<HTMLInputElement>(null);
  const asked = useRef(false);
  useEffect(() => {
    if (reason.open && asked.current) input.current?.focus();
    asked.current = false;
  }, [reason.open]);
  return (
    <div className={c.box} data-testid="reason-box">
      {onRecord !== null ? (
        <p className={c.onRecord} data-testid="reason-on-record">
          <span className={c.onRecordLabel}>Reason on record:</span> {onRecord}
        </p>
      ) : null}
      {reason.open ? (
        <div className={c.field}>
          <label className={c.label} htmlFor={id}>
            Your reason
          </label>
          <input
            ref={input}
            id={id}
            type="text"
            className={c.input}
            value={reason.text}
            onChange={(e) => onChange({ open: true, text: e.target.value })}
            aria-describedby={`${id}-hint`}
            data-testid="reason-input"
          />
          <p id={`${id}-hint`} className={c.hint}>
            Recorded with the answer as the decision's note, in the decision log; not written into the methods.
          </p>
        </div>
      ) : (
        <button
          type="button"
          className={c.add}
          onClick={() => {
            asked.current = true;
            onChange({ ...reason, open: true });
          }}
          data-testid="reason-add"
        >
          Add a reason
        </button>
      )}
      {answer}
    </div>
  );
}
