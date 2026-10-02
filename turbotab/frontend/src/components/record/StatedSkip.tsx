/**
 * A question the Router did not ask because the data already answers it (DESIGN_LANGUAGE §09: a
 * rendered skip is a muted neutral row, never green): "Not asked:", the reading with its evidence,
 * and "Ask me anyway", which reopens the question. M2 states two this way: the repeats (from date
 * spacing or a visit label) and the grain, when a recognized identifier is unique on every row
 * ("Not asked: every `SEQN` appears once, so each person is one row").
 *
 * The server's reason may already begin with "Not asked:"; the row says it once.
 */
import type { ReactNode } from "react";
import { Prose } from "../Prose";
import { SkipRow } from "./blocks";
import s from "./Record.module.css";

const LEAD = /^\s*not asked\s*[:—–-]\s*/i;

/** The reason without its own "Not asked:" (the row adds it), first letter as written. */
export function skipReason(reason: string | null | undefined): string {
  return (reason ?? "").replace(LEAD, "").trim();
}

export function StatedSkip({
  qkey,
  reason,
  lead,
  onAsk,
}: {
  qkey: string;
  reason: string | null | undefined;
  /** Data said before the reason (the task's reading: `glucose` read as regression). */
  lead?: ReactNode;
  onAsk: () => void;
}) {
  const text = skipReason(reason);
  return (
    <SkipRow layoutId={`q-${qkey}`} onAsk={onAsk} testId={`skip-${qkey}`}>
      <span className={s.notAsked}>Not asked:</span> {lead}
      {text ? <Prose text={text} /> : null}
    </SkipRow>
  );
}
