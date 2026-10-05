/**
 * An open question, in its grammar's silhouette (DESIGN_LANGUAGE §09):
 *
 *   FACT    interrogative, about the table: the lightest object on screen — no border, no
 *           tint — with the teal marker only while it is the question being asked
 *   CHOICE  a modeling decision: a bordered card, neutral while open
 *
 * Teaching (BLUEPRINT §11.5): the question and its one line (layer 0) are all that is needed
 * to answer; each option's consequence is layer 1; "why?" opens in place (layer 2); the
 * concept drawer is one press further (layer 3) and never required.
 *
 * Settle and arrive: the block shares its layoutId with the decision sentence it settles
 * into. A question arriving because of that settle waits for it to finish, then takes the
 * keyboard focus, so the two never overlap and focus never falls to the page.
 */
import { useEffect, useId, useRef, useState, type ReactNode } from "react";
import { motion } from "motion/react";
import type { QuestionKey, TeachingEntry } from "../../api/m1-types";
import { DUR, useTransitions } from "../../motion/prefs";
import { cx } from "../../util/format";
import { EvidenceBadge, Taught, TermsProvider } from "./teach";
import s from "./Question.module.css";

export type Grammar = "fact" | "choice";

/** Which grammar each question speaks (§09): facts about the table, choices about the model. */
export const GRAMMAR: Record<QuestionKey, Grammar> = {
  lens: "fact",
  orientation: "fact",
  target: "fact",
  event: "choice",
  task: "fact",
  follow_up: "fact",
  purpose: "fact",
  grain: "fact",
  repeat_kind: "fact",
  unit: "choice",
  aggregation: "choice",
  temporal: "fact",
  roles: "fact",
  clusters: "fact",
  survey: "choice",
  estimand: "choice",
  adjustment: "fact",
  time_varying: "choice",
  exclusions: "choice",
  missing: "choice",
  split: "choice",
  energy_adjustment: "choice",
  form: "choice",
  modification: "choice",
  causal: "choice",
  models: "choice",
  substitution: "choice",
  open_seal: "choice",
};

/**
 * After an answer settles, the arriving question should be readable with its cause — the
 * sentence just recorded — still in sight above it (arrive grows from its cause). Nothing
 * moves when the question is already in view.
 */
function bringIntoView(heading: HTMLElement | null, reduced: boolean) {
  const slot = heading?.closest<HTMLElement>("[data-slot]");
  if (!slot) return;
  // The Record scrolls in its own column on wide screens; on narrow ones, the page scrolls.
  const main = slot.closest<HTMLElement>("main");
  const own = main && main.scrollHeight > main.clientHeight + 1 ? main : null;
  const view = own
    ? own.getBoundingClientRect()
    : { top: 0, bottom: window.innerHeight, height: window.innerHeight };
  const box = slot.getBoundingClientRect();
  if (box.top >= view.top && box.bottom <= view.bottom) return;
  const cause = slot.previousElementSibling as HTMLElement | null;
  const anchor = cause && box.height + cause.offsetHeight < view.height ? cause : slot;
  anchor.scrollIntoView({ block: "start", behavior: reduced ? "auto" : "smooth" });
}

interface Props {
  qkey: QuestionKey;
  entry: TeachingEntry | undefined;
  /** Overrides the entry's question (e.g. to name the column it asks about). */
  title?: ReactNode;
  /** Their data, before theory (§11.6): a sentence about their own columns. */
  data?: ReactNode;
  /** The question being asked: wears the teal marker. */
  now: boolean;
  /** Arriving because an answer just settled: waits for the settle, then takes focus. */
  arriving?: boolean;
  onArrived?: () => void;
  onOpenDrawer?: () => void;
  /** Reopened with "change": says so. */
  reopened?: boolean;
  /** At most one coach line per card (M2_CONTRACT §10): data-grounded, amber, never choosing. */
  coach?: ReactNode;
  /** Findings held for this question, resurfacing inside it, attributed (M2_CONTRACT §4). */
  resurfaced?: ReactNode;
  children: ReactNode;
}

export function Question({
  qkey,
  entry,
  title,
  data,
  now,
  arriving = false,
  onArrived,
  onOpenDrawer,
  reopened = false,
  coach,
  resurfaced,
  children,
}: Props) {
  const t = useTransitions();
  const headingId = useId();
  const whyId = useId();
  const heading = useRef<HTMLHeadingElement>(null);
  const [why, setWhy] = useState(false);
  const grammar = GRAMMAR[qkey];

  useEffect(() => {
    if (!arriving) return;
    const wait = t.reduced ? 0 : DUR.settle * 1000;
    const timer = window.setTimeout(() => {
      const active = document.activeElement;
      // Only take focus the settle left behind on the page; never steal it from a control.
      if (!active || active === document.body || !document.body.contains(active)) {
        heading.current?.focus({ preventScroll: true });
      }
      bringIntoView(heading.current, t.reduced);
      onArrived?.();
    }, wait);
    return () => window.clearTimeout(timer);
  }, [arriving, onArrived, t.reduced]);

  return (
    <TermsProvider terms={entry?.terms}>
      <motion.section
        layoutId={`q-${qkey}`}
        layout
        initial={arriving ? { opacity: 0, y: 6 } : false}
        animate={{ opacity: 1, y: 0 }}
        transition={{
          ...t.arrive,
          delay: arriving && !t.reduced ? DUR.settle : 0,
          layout: t.settle,
        }}
        className={cx(s.question, grammar === "fact" ? s.fact : s.choice, now && s.now)}
        style={{ borderRadius: grammar === "fact" ? 4 : 14 }}
        aria-labelledby={headingId}
        data-testid={`question-${qkey}`}
        data-block="question"
        data-grammar={grammar}
        data-now={now || undefined}
      >
        <motion.div layout="position" transition={{ layout: t.settle }} className={s.inner}>
          <div className={s.kickerRow}>
            <span className={s.kicker}>{entry?.title ?? qkey.replace(/_/g, " ")}</span>
            {entry?.evidence ? <EvidenceBadge {...entry.evidence} /> : null}
            {reopened ? (
              <span className={s.reopened}>reopened · nothing changes until you record</span>
            ) : null}
          </div>
          <h2 id={headingId} ref={heading} tabIndex={-1} className={s.title}>
            {title ?? (entry ? <Taught text={entry.question} /> : null)}
          </h2>
          {entry ? (
            <p className={s.oneLiner}>
              <Taught text={entry.one_liner} />{" "}
              <button
                type="button"
                className={s.whyButton}
                aria-expanded={why}
                aria-controls={whyId}
                onClick={() => setWhy((w) => !w)}
                data-testid={`why-${qkey}`}
              >
                why?
              </button>
            </p>
          ) : null}
          {why && entry ? (
            <div id={whyId} className={s.why} data-testid={`why-panel-${qkey}`}>
              <p className={s.whyText}>
                <Taught text={entry.why} />
              </p>
              <p className={s.consumer}>
                <span className={s.consumerLabel}>Who reads the answer</span> {entry.consumer}
              </p>
              {entry.drawer && onOpenDrawer ? (
                <button
                  type="button"
                  className={s.drawerButton}
                  onClick={onOpenDrawer}
                  data-testid={`drawer-${qkey}`}
                >
                  The concept, in depth: {entry.drawer.sections.length} short sections →
                </button>
              ) : null}
            </div>
          ) : null}
          {data ? <div className={s.data}>{data}</div> : null}
          {coach ? (
            <p className={s.coach} data-testid={`coach-${qkey}`}>
              {coach}
            </p>
          ) : null}
          {resurfaced}
          <div className={s.body}>{children}</div>
        </motion.div>
      </motion.section>
    </TermsProvider>
  );
}
