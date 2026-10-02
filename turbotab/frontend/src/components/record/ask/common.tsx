/** What every question's answer area shares. */
import type { ReactNode } from "react";
import type { QuestionKey, TeachingEntry } from "../../../api/m1-types";
import type { Decision } from "../../../api/schema";
import c from "./ask.module.css";

export interface AskProps {
  entry: TeachingEntry | undefined;
  /** A decision is being recorded. */
  pending: boolean;
  /** Record a decision; `at` names the option a refusal should answer under. */
  record: (decision: Decision, at?: string) => void;
  /** Reopened with an answer on record: close and keep it. Absent otherwise. */
  keep?: (() => void) | undefined;
  /** The server's answer to a press that did not record (a refusal), at its option. */
  answerAt: { key: string; node: ReactNode } | null;
  /** Shell props the Record passes straight to <Question>. */
  shell: {
    qkey: QuestionKey;
    now: boolean;
    arriving: boolean;
    onArrived: () => void;
    onOpenDrawer: () => void;
    reopened: boolean;
    /** At most one coach line per card (M2_CONTRACT §10). */
    coach?: ReactNode;
    /** Findings held for this question, resurfacing inside it (M2_CONTRACT §4). */
    resurfaced?: ReactNode;
  };
}

/** The teaching option for a value: its label (≤ 4 words) and consequence (≤ 16 words). */
export function taught(entry: TeachingEntry | undefined, value: string) {
  return entry?.options.find((o) => o.value === value);
}

export function Keep({
  keep,
  label = "Keep the recorded answer",
}: {
  keep?: (() => void) | undefined;
  label?: string;
}) {
  if (!keep) return null;
  return (
    <button
      type="button"
      className={c.ghost}
      onClick={keep}
      title="Closes the question. Nothing is recorded, and nothing on record changes."
      data-testid="keep"
    >
      {label}
    </button>
  );
}

export function Actions({ children }: { children: ReactNode }) {
  return <div className={c.actions}>{children}</div>;
}

/** The answer area's own record button, for questions that are answered as a set. */
export function RecordButton({
  children,
  disabled,
  onClick,
  title,
  testId,
}: {
  children: ReactNode;
  disabled?: boolean;
  onClick: () => void;
  title: string;
  testId?: string;
}) {
  return (
    <button
      type="button"
      className={c.primary}
      disabled={disabled}
      onClick={onClick}
      title={title}
      data-testid={testId}
    >
      {children}
    </button>
  );
}

export const fmtCount = (n: number) => Math.round(n).toLocaleString("en-US");
