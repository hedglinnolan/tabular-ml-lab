/**
 * Repairs on findings (M2_CONTRACT §4, §10; OPENING_SEQUENCE §01: "structural diagnosis, repairs,
 * impossibility pass", before the outcome). Not a question: the findings that carry repair options,
 * in the Record's flow before the outcome question, each one claim and its options.
 *
 *   an option     focusing it (hover, keyboard, a tap) previews the repair on the stage — the
 *                 changed cells and the column's distribution; Enter or a click applies it
 *   Ask me at …   holds the finding for the question it targets, where it resurfaces, pre-checked
 *   Keep as is    dismisses it; nothing changes, and the record keeps that it was seen
 *
 * Once acted on, a finding settles into its recorded sentence (green, with "change"). A finding a
 * later answer or another repair already settled folds into "answered by #N".
 *
 * <Resurfaced> is the other half of deferral: inside the question a finding was held for, it comes
 * back attributed ("You set this aside at #5") and pre-checked with its first repair. Recording the
 * question then applies the checked repair, or dismisses the finding the user unchecked.
 */
import { useRef, useState, type ReactNode } from "react";
import type { QuestionKey, TeachingEntry } from "../../api/m1-types";
import type { Decision, DecisionRecord, Finding } from "../../api/schema";
import { useStageFocus, type StageFocus } from "../../state/focus";
import { cx } from "../../util/format";
import { Prose } from "../Prose";
import { DecisionSentence } from "./blocks";
import { cardsOf, restLine, type FindingCard } from "./Findings";
import {
  findingState,
  recordedRepair,
  repairId,
  type DeferredChoice,
  type FindingState,
} from "./findingState";
import { Options, type OptionItem } from "./Options";
import { Taught, TermsProvider } from "./teach";
import f from "./Findings.module.css";
import r from "./Repairs.module.css";

const PUSHED = 3;

/** An answer to a press that did not record, at the control it answers. */
export interface RepairAnswer {
  /** `${findingId}|${optionId}` or `${findingId}|actions`. */
  key: string;
  node: ReactNode;
}

interface RepairsProps {
  findings: readonly Finding[];
  decisions: readonly DecisionRecord[];
  entry: TeachingEntry | undefined;
  /** The question a finding can be held for, when it is still ahead; else null. */
  deferTarget: (f: Finding) => QuestionKey | null;
  /** "the exclusions": a question's name in running text. */
  subject: (k: QuestionKey) => string;
  record: (decision: Decision, at: string) => void;
  pending: boolean;
  answerAt: RepairAnswer | null;
  /** The post-seal mark for a recorded sentence, when it was made after the seal was opened. */
  marks?: (rec: DecisionRecord) => ReactNode;
}

const origin = (x: Finding) =>
  x.source === "pack"
    ? `${x.lens ?? "domain"} pack`
    : x.source === "structural"
      ? "structure"
      : "profile";

function optionsOf(x: Finding, subjectRow: string): OptionItem[] {
  return x.repairs.map((o, i) => ({
    key: repairId(o, i),
    label: <Prose text={o.label} />,
    previewLabel: `${o.label.replace(/`/g, "")} · ${subjectRow}`,
    line: o.consequence,
    decision: o.decision as Decision,
    tags: o.row_local ? undefined : [{ text: "in each fold", tone: "badge" as const }],
  }));
}

export function RepairsSection({
  findings,
  decisions,
  entry,
  deferTarget,
  subject,
  record,
  pending,
  answerAt,
  marks,
}: RepairsProps) {
  const [reopened, setReopened] = useState<ReadonlySet<string>>(new Set());
  const [open, setOpen] = useState(false);
  const [pages, setPages] = useState<Record<string, number>>({});
  const [why, setWhy] = useState(false);
  const { setFocus, endPreview } = useStageFocus();
  const listRef = useRef<HTMLUListElement>(null);

  const withRepairs = findings.filter((x) => x.repairs.length > 0);
  if (withRepairs.length === 0) return null;
  const states = new Map<string, FindingState>(
    withRepairs.map((x) => [x.id, findingState(x, decisions)]),
  );
  const stateOf = (x: Finding) => states.get(x.id)!;
  const acting = withRepairs.filter((x) => stateOf(x).kind === "open" || reopened.has(x.id));
  const settled = withRepairs.filter(
    (x) => !reopened.has(x.id) && ["applied", "dismissed", "deferred"].includes(stateOf(x).kind),
  );
  const answered = withRepairs.filter((x) => !reopened.has(x.id) && stateOf(x).kind === "answered");
  const cards = cardsOf(acting);
  const shown = open ? cards : cards.slice(0, PUSHED);
  const rest = cards.slice(PUSHED);
  const recordOf = (id: string) => decisions.find((d) => d.id === id);

  const close = (id: string) =>
    setReopened((s) => {
      const next = new Set(s);
      next.delete(id);
      return next;
    });
  const act = (x: Finding, d: Decision, at: string) => {
    close(x.id);
    record(d, `${x.id}|${at}`);
  };
  const turn = (card: FindingCard, to: number) => {
    const p = Math.max(0, Math.min(card.pages.length - 1, to));
    setPages((s) => ({ ...s, [card.id]: p }));
    setFocus({ kind: "finding", findingId: card.pages[p]!.id });
  };

  return (
    <TermsProvider terms={entry?.terms}>
      <section className={r.section} aria-labelledby="repairs-heading" data-testid="repairs">
        <div className={r.head}>
          <h2 id="repairs-heading" className={r.kicker}>
            {entry?.title ?? "Repairs before the outcome"}
          </h2>
          {acting.length ? (
            <span className={r.count} data-testid="repairs-open-count">
              {acting.length} to decide
            </span>
          ) : null}
        </div>
        {entry && acting.length ? (
          <p className={r.oneLiner}>
            <Taught text={entry.one_liner} />{" "}
            <button
              type="button"
              className={r.whyButton}
              aria-expanded={why}
              onClick={() => setWhy((w) => !w)}
              data-testid="why-repairs"
            >
              why?
            </button>
          </p>
        ) : null}
        {why && entry ? (
          <p className={r.why}>
            <Taught text={entry.why} />
          </p>
        ) : null}
        {shown.length ? (
          <ul
            className={f.cards}
            ref={listRef}
            onPointerLeave={endPreview}
            data-testid="repair-cards"
          >
            {shown.map((card) => {
              const p = Math.min(pages[card.id] ?? 0, card.pages.length - 1);
              const x = card.pages[p]!;
              const st = stateOf(x);
              const n = card.pages.length;
              const to = deferTarget(x);
              const isReopened = reopened.has(x.id);
              const recorded = recordedRepair(x, st, decisions);
              const answerFor = (key: string) =>
                answerAt && answerAt.key.startsWith(`${x.id}|`) && answerAt.key === `${x.id}|${key}`
                  ? answerAt.node
                  : null;
              const optionAnswer =
                answerAt && answerAt.key.startsWith(`${x.id}|`)
                  ? { key: answerAt.key.slice(x.id.length + 1), node: answerAt.node }
                  : null;
              return (
                <li
                  key={card.id}
                  className={cx(f.card, r.card)}
                  data-severity={card.severity}
                  data-testid={`repair-${x.id}`}
                  aria-label={`Finding${n > 1 ? `, ${p + 1} of ${n}` : ""}: ${x.summary.replace(/`/g, "")}`}
                >
                  <div className={f.top}>
                    <span className={f.origin}>{origin(x)}</span>
                    {x.evidence ? (
                      <span className={f.badge} title={x.evidence.source}>
                        {x.evidence.status.toUpperCase()}
                      </span>
                    ) : null}
                    {isReopened ? (
                      <span className={r.reopened}>
                        reopened · nothing changes until you choose
                      </span>
                    ) : null}
                    {n > 1 ? (
                      <span className={f.pager}>
                        <button
                          type="button"
                          className={f.pageBtn}
                          aria-label="Previous finding of this kind"
                          disabled={p === 0}
                          onClick={() => turn(card, p - 1)}
                        >
                          ‹
                        </button>
                        <span className={f.pageText} aria-live="polite">
                          {p + 1} of {n}
                        </span>
                        <button
                          type="button"
                          className={f.pageBtn}
                          aria-label="Next finding of this kind"
                          disabled={p === n - 1}
                          onClick={() => turn(card, p + 1)}
                        >
                          ›
                        </button>
                      </span>
                    ) : null}
                  </div>
                  <p
                    className={cx(f.claim, r.claim)}
                    tabIndex={0}
                    onFocus={() => setFocus({ kind: "finding", findingId: x.id })}
                    title="Shows why this was raised, on the stage"
                    data-testid="repair-claim"
                  >
                    <Prose text={x.summary} />
                  </p>
                  <Options
                    items={optionsOf(x, x.affected_columns.slice(0, 2).join(", "))}
                    mode="single"
                    onRecord={(o) => o.decision && act(x, o.decision, o.key)}
                    recordedKey={recorded}
                    pending={pending}
                    answerAt={optionAnswer}
                    label="Repairs — preview with the arrow keys, Enter to apply"
                    testId={`repair-options-${x.id}`}
                    hint={false}
                  />
                  <div className={r.actions}>
                    {to ? (
                      <button
                        type="button"
                        className={r.secondary}
                        disabled={pending}
                        onClick={() =>
                          act(x, { kind: "defer_finding", finding_id: x.id, to }, "actions")
                        }
                        title={
                          entry?.options.find((o) => o.value === "defer")?.consequence ??
                          "The finding waits inside the question it belongs to, checked and attributed."
                        }
                        data-testid="repair-defer"
                      >
                        Ask me at {subject(to)}
                      </button>
                    ) : null}
                    <button
                      type="button"
                      className={r.secondary}
                      disabled={pending}
                      onClick={() =>
                        act(
                          x,
                          { kind: "dismiss_finding", finding_id: x.id, reason: null },
                          "actions",
                        )
                      }
                      title={
                        entry?.options.find((o) => o.value === "dismiss")?.consequence ??
                        "Nothing changes; the record keeps that it was seen and set aside."
                      }
                      data-testid="repair-dismiss"
                    >
                      Keep as is
                    </button>
                    {isReopened ? (
                      <button
                        type="button"
                        className={r.link}
                        onClick={() => close(x.id)}
                        title="Closes the finding. Nothing is recorded, and nothing on record changes."
                        data-testid="repair-keep"
                      >
                        Keep the recorded answer
                      </button>
                    ) : null}
                    <span className={r.keys} aria-hidden="true">
                      <kbd>↑</kbd>
                      <kbd>↓</kbd> preview <kbd>Enter</kbd> apply
                    </span>
                  </div>
                  {answerFor("actions") ? (
                    <div className={r.answer}>{answerFor("actions")}</div>
                  ) : null}
                </li>
              );
            })}
          </ul>
        ) : null}
        {rest.length > 0 ? (
          <div className={cx(f.rest, open && f.restOpen)}>
            <span className={f.restText}>
              {open ? "Every finding with a repair is shown." : restLine(rest)}
            </span>
            <button
              type="button"
              className={f.restButton}
              onClick={() => setOpen((o) => !o)}
              aria-expanded={open}
              data-testid="repairs-more"
            >
              {open ? "Show three" : "Show"}
            </button>
          </div>
        ) : null}
        {settled.length ? (
          <div className={r.settled} data-testid="repairs-settled">
            {settled.map((x) => {
              const st = stateOf(x) as Exclude<FindingState, { kind: "open" | "answered" }>;
              const rec = recordOf(st.recordId);
              const said =
                rec?.sentence ??
                (st.kind === "deferred"
                  ? `Set aside for ${subject(st.to)}, where it will be raised again.`
                  : st.kind === "dismissed"
                    ? "Dismissed; it stays in the record."
                    : "The repair was applied.");
              return (
                <DecisionSentence
                  key={x.id}
                  layoutId={`repair-${x.id}`}
                  subject={`what was done about ${x.affected_columns[0] ?? "this finding"}`}
                  onChange={() => setReopened((s) => new Set(s).add(x.id))}
                  meta={
                    <>
                      {rec && marks ? marks(rec) : null}#{st.seq}
                    </>
                  }
                  testId={`repair-settled-${x.id}`}
                >
                  <Prose text={said} />
                </DecisionSentence>
              );
            })}
          </div>
        ) : null}
        {answered.length ? (
          <p className={r.answeredLine} data-testid="repairs-answered">
            {answered.length === 1 ? "One more was" : `${answered.length} more were`} answered by{" "}
            {[...new Set(answered.map((x) => `#${(stateOf(x) as { seq: number }).seq}`))].join(
              ", ",
            )}
            :{" "}
            {answered
              .map((x) => x.affected_columns[0])
              .filter(Boolean)
              .slice(0, 4)
              .map((c, i) => (
                <span key={c}>
                  {i > 0 ? ", " : ""}
                  <code className="v">{c}</code>
                </span>
              ))}
            .
          </p>
        ) : null}
      </section>
    </TermsProvider>
  );
}

// ── deferral, the other half: inside the question it was held for ───────────

export function Resurfaced({
  findings,
  decisions,
  choices,
  onChoice,
}: {
  findings: readonly Finding[];
  decisions: readonly DecisionRecord[];
  choices: Readonly<Record<string, DeferredChoice | undefined>>;
  onChoice: (findingId: string, choice: DeferredChoice) => void;
}) {
  const { setFocus, preview, endPreview } = useStageFocus();
  if (findings.length === 0) return null;
  return (
    <div className={r.resurfaced} data-testid="resurfaced" onPointerLeave={endPreview}>
      <div className={r.resurfacedHead}>Set aside for this question</div>
      {findings.map((x) => {
        const st = findingState(x, decisions);
        const seq = st.kind === "deferred" ? st.seq : null;
        const choice =
          choices[x.id] ??
          (x.repairs.length
            ? { checked: true as const, repair: repairId(x.repairs[0]!, 0) }
            : null);
        const chosenIndex =
          choice && choice.checked
            ? Math.max(
                0,
                x.repairs.findIndex((o, i) => repairId(o, i) === choice.repair),
              )
            : 0;
        const chosen = x.repairs[chosenIndex];
        const focusOf = (i: number): StageFocus => ({
          kind: "option",
          decision: x.repairs[i]!.decision as Decision,
          label: x.repairs[i]!.label.replace(/`/g, ""),
        });
        return (
          <div key={x.id} className={r.held} data-testid={`held-${x.id}`}>
            <p className={r.heldBy}>
              You set this aside{seq !== null ? <> at #{seq}</> : null}; this is the question that
              acts on it.
            </p>
            <p className={r.heldClaim}>
              <Prose text={x.summary} />
            </p>
            {choice ? (
              <div className={r.heldChoice}>
                <label className={r.heldCheck}>
                  <input
                    type="checkbox"
                    checked={choice.checked}
                    onChange={(e) =>
                      onChoice(
                        x.id,
                        e.target.checked
                          ? {
                              checked: true,
                              repair: repairId(x.repairs[chosenIndex]!, chosenIndex),
                            }
                          : { checked: false },
                      )
                    }
                    data-testid={`held-check-${x.id}`}
                  />
                  {choice.checked
                    ? "Applied when you record this answer:"
                    : "Not applied; dismissed when you record this answer."}
                </label>
                {choice.checked && x.repairs.length > 1 ? (
                  <span className={r.heldOptions} role="radiogroup" aria-label="Which repair">
                    {x.repairs.map((o, i) => (
                      <button
                        key={repairId(o, i)}
                        type="button"
                        role="radio"
                        aria-checked={i === chosenIndex}
                        className={r.heldOption}
                        onClick={() => onChoice(x.id, { checked: true, repair: repairId(o, i) })}
                        onFocus={() => setFocus(focusOf(i))}
                        onPointerMove={() => preview(focusOf(i))}
                        data-testid={`held-option-${repairId(o, i)}`}
                      >
                        <Prose text={o.label} />
                      </button>
                    ))}
                  </span>
                ) : null}
                {choice.checked && chosen ? (
                  <span className={r.heldLine}>
                    <Taught text={chosen.consequence} />
                  </span>
                ) : null}
              </div>
            ) : null}
          </div>
        );
      })}
    </div>
  );
}
