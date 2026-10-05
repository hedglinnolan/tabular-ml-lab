/**
 * The Record's three silhouettes for a FACT-style question (DESIGN_LANGUAGE §04, §09):
 *
 *   QuestionBlock     open: serif question, one line of rationale, answers
 *   DecisionSentence  recorded: one serif sentence, mono values, green keyline, "change"
 *   SkipRow           not asked: a muted neutral row with its provenance — never green
 *
 * All three take the same `layoutId`, so answering a question SETTLES: the same
 * element morphs from question to sentence instead of being swapped out.
 */
import { useId, type ReactNode } from "react";
import { motion } from "motion/react";
import { useTransitions } from "../../motion/prefs";
import styles from "./blocks.module.css";

interface QuestionProps {
  layoutId: string;
  kicker: string;
  title: ReactNode;
  why: ReactNode;
  consumer?: ReactNode;
  /** Grow in from the cause above (first appearance only). */
  arrive?: boolean;
  children: ReactNode;
  testId?: string;
}

export function QuestionBlock({
  layoutId,
  kicker,
  title,
  why,
  consumer,
  arrive = false,
  children,
  testId,
}: QuestionProps) {
  const t = useTransitions();
  const headingId = useId();
  return (
    <motion.section
      layoutId={layoutId}
      layout
      initial={arrive ? { opacity: 0, y: 6 } : false}
      animate={{ opacity: 1, y: 0 }}
      transition={{ ...t.arrive, layout: t.settle }}
      className={styles.question}
      style={{ borderRadius: 14 }}
      aria-labelledby={headingId}
      data-testid={testId}
      data-block="question"
    >
      <motion.div layout="position" transition={{ layout: t.settle }} className={styles.inner}>
        <div className={styles.kicker}>{kicker}</div>
        <h2 id={headingId} className={styles.title}>
          {title}
        </h2>
        <p className={styles.why}>{why}</p>
        {children}
        {consumer ? (
          <details className={styles.consumer}>
            <summary>Why we ask</summary>
            <p>{consumer}</p>
          </details>
        ) : null}
      </motion.div>
    </motion.section>
  );
}

interface SentenceProps {
  layoutId: string;
  children: ReactNode;
  /** What "change" reopens, for its accessible name. */
  subject: string;
  onChange?: () => void;
  /** Takes the answer back (the engine's `revert`): the slot returns to its previous answer. */
  onUndo?: () => void;
  meta?: ReactNode;
  /** What is true of the answer now that the sentence cannot say (it did not run, its counts
   *  predate a later answer): under the sentence, in the coach's voice. */
  note?: ReactNode;
  testId?: string;
}

export function DecisionSentence({
  layoutId,
  children,
  subject,
  onChange,
  onUndo,
  meta,
  note,
  testId,
}: SentenceProps) {
  const t = useTransitions();
  return (
    <motion.div
      layoutId={layoutId}
      layout
      initial={false}
      transition={{ layout: t.settle }}
      className={styles.decision}
      style={{ borderRadius: 10 }}
      data-testid={testId}
      data-block="decision"
    >
      <div className={styles.sentenceBody}>
        <motion.p layout="position" transition={{ layout: t.settle }} className={styles.prose}>
          {children}
        </motion.p>
        {note ? (
          <div className={styles.sentenceNote} data-testid={testId ? `${testId}-note` : undefined}>
            {note}
          </div>
        ) : null}
      </div>
      <motion.div layout="position" transition={{ layout: t.settle }} className={styles.aside}>
        {meta ? <span className={styles.meta}>{meta}</span> : null}
        {onChange ? (
          <button
            type="button"
            className={styles.change}
            onClick={onChange}
            aria-label={`Change ${subject}`}
            title={`Reopens the question. A new answer is added to the record; this one stays in its history.`}
          >
            change
          </button>
        ) : null}
        {onUndo ? (
          <button
            type="button"
            className={styles.change}
            onClick={onUndo}
            aria-label={`Undo ${subject}`}
            title="Takes this answer back: the answer before it returns, or the question opens again. Both stay in the history."
            data-testid={testId ? `${testId}-undo` : undefined}
          >
            undo
          </button>
        ) : null}
      </motion.div>
    </motion.div>
  );
}

interface SkipProps {
  layoutId: string;
  children: ReactNode;
  onAsk: () => void;
  testId?: string;
}

export function SkipRow({ layoutId, children, onAsk, testId }: SkipProps) {
  const t = useTransitions();
  return (
    <motion.div
      layoutId={layoutId}
      layout
      initial={false}
      transition={{ layout: t.settle }}
      className={styles.skip}
      style={{ borderRadius: 10 }}
      data-testid={testId}
      data-block="skip"
    >
      <motion.p layout="position" transition={{ layout: t.settle }} className={styles.skipText}>
        {children}
      </motion.p>
      <motion.button
        layout="position"
        transition={{ layout: t.settle }}
        type="button"
        className={styles.change}
        onClick={onAsk}
      >
        Ask me anyway
      </motion.button>
    </motion.div>
  );
}

/** Earlier answers to the same question: kept, muted, never deleted. */
export function History({
  items,
}: {
  /** `tag` says why an answer is not in force; "superseded" unless given. */
  items: { id: string; seq: number; when: string; sentence: ReactNode; tag?: string }[];
}) {
  if (items.length === 0) return null;
  return (
    <ol className={styles.history} aria-label="Earlier answers">
      {items.map((h) => (
        <li key={h.id} className={styles.historyItem}>
          <span className={styles.historyMeta}>
            #{h.seq} · {h.when}
          </span>
          <span className={styles.historyText}>{h.sentence}</span>
          <span className={styles.superseded}>{h.tag ?? "superseded"}</span>
        </li>
      ))}
    </ol>
  );
}

/** A pending row while the answer a question depends on is still being computed. */
export function Pending({ children, testId }: { children: ReactNode; testId?: string }) {
  return (
    <div className={styles.pending} data-testid={testId} role="status">
      {children}
    </div>
  );
}
