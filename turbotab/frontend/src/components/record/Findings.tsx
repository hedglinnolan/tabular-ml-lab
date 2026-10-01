/**
 * What the lenses noticed, within the doctrine (BLUEPRINT §11.7; the prototype's S3):
 * a finding is a one-line claim plus its lever. At most three cards are pushed, same-kind
 * findings share one paged card, and the rest are counted and typed, one press away.
 *
 * Focusing a card (hover, keyboard) puts its evidence on the stage. A lever is a navigation
 * the user asked for: it takes the Record to the question that acts on the claim.
 *
 * A finding whose question has been answered is settled: it stops being pushed and its lever
 * stops pressing for an answer the Record already holds (DRIVE_RUBRIC §2.7 "Didn't we just
 * settle…?"). It folds into one green line, "Answered by #6", its evidence still a press away,
 * and the three pushed cards are drawn from the findings still open.
 */
import { useRef, useState, type KeyboardEvent } from "react";
import type { Finding, FindingsArtifact, Severity } from "../../api/schema";
import type { QuestionKey } from "../../api/m1-types";
import { useStageFocus, type StageFocus } from "../../state/focus";
import { cx } from "../../util/format";
import { Prose } from "../Prose";
import s from "./Findings.module.css";

export const PUSHED = 3;

export interface FindingCard {
  id: string;
  severity: Severity;
  pages: Finding[];
}

const RANK: Record<Severity, number> = { critical: 0, warning: 1, info: 2 };

/** Same-kind findings share one card (its pages); cards are ranked by their gravest page. */
export function cardsOf(findings: Finding[]): FindingCard[] {
  const cards: FindingCard[] = [];
  const byGroup = new Map<string, FindingCard>();
  for (const f of findings) {
    const existing = f.group ? byGroup.get(f.group) : undefined;
    if (existing) {
      existing.pages.push(f);
      if (RANK[f.severity] < RANK[existing.severity]) existing.severity = f.severity;
      continue;
    }
    const card: FindingCard = {
      id: f.group ? `group:${f.group}` : f.id,
      severity: f.severity,
      pages: [f],
    };
    if (f.group) byGroup.set(f.group, card);
    cards.push(card);
  }
  return cards
    .map((c, i) => ({ c, i }))
    .sort((a, b) => RANK[a.c.severity] - RANK[b.c.severity] || a.i - b.i)
    .map(({ c }) => c);
}

const SEVERITY_WORD: Record<Severity, [string, string]> = {
  critical: ["critical", "critical"],
  warning: ["warning", "warnings"],
  info: ["note", "notes"],
};

/** "7 more — 3 warnings, 4 notes": what is folded away, counted and typed. */
export function restLine(rest: FindingCard[]): string {
  const counts: Record<Severity, number> = { critical: 0, warning: 0, info: 0 };
  let n = 0;
  for (const card of rest) {
    for (const f of card.pages) {
      counts[f.severity] += 1;
      n += 1;
    }
  }
  const parts = (["critical", "warning", "info"] as Severity[])
    .filter((sv) => counts[sv] > 0)
    .map((sv) => `${counts[sv]} ${SEVERITY_WORD[sv][counts[sv] === 1 ? 0 : 1]}`);
  return `${n} more — ${parts.join(", ")}`;
}

function origin(f: Finding): string {
  if (f.source === "pack") return `${f.lens ?? "domain"} pack`;
  if (f.source === "structural") return "structure";
  return "profile";
}

const asFocus = (f: Finding): StageFocus => ({ kind: "finding", findingId: f.id });

/** The recorded answer that settles a finding: its number, and its sentence's opening. */
export interface Settlement {
  seq: number;
  said: string;
}

interface Props {
  artifact: FindingsArtifact;
  /** Take the Record to the question that acts on a finding. */
  onRoute: (to: QuestionKey) => void;
  /** The recorded answer to the question a finding routes to, if it has one. */
  answeredBy?: (f: Finding) => Settlement | null;
}

export function FindingsCards({ artifact, onRoute, answeredBy }: Props) {
  const settledOf = (f: Finding) => answeredBy?.(f) ?? null;
  const openFindings = artifact.findings.filter((f) => !settledOf(f));
  const settled = artifact.findings.filter((f) => settledOf(f));
  const cards = cardsOf(openFindings);
  const [open, setOpen] = useState(false);
  const [showSettled, setShowSettled] = useState(false);
  const [pages, setPages] = useState<Record<string, number>>({});
  const refs = useRef<(HTMLLIElement | null)[]>([]);
  const { focus, preview, endPreview, setFocus, reset } = useStageFocus();
  const pushed = cards.slice(0, PUSHED);
  const rest = cards.slice(PUSHED);
  const shown = open ? cards : pushed;
  const shownId = focus.kind === "finding" ? focus.findingId : null;

  const pageOf = (card: FindingCard) => Math.min(pages[card.id] ?? 0, card.pages.length - 1);
  const turn = (card: FindingCard, to: number) => {
    const p = Math.max(0, Math.min(card.pages.length - 1, to));
    setPages((x) => ({ ...x, [card.id]: p }));
    // The stage follows the page the user turned to.
    setFocus(asFocus(card.pages[p]!));
  };

  const onKey = (i: number, card: FindingCard) => (e: KeyboardEvent<HTMLLIElement>) => {
    if (e.target !== e.currentTarget) return;
    if (e.key === "ArrowDown") {
      e.preventDefault();
      refs.current[Math.min(shown.length - 1, i + 1)]?.focus();
    } else if (e.key === "ArrowUp") {
      e.preventDefault();
      refs.current[Math.max(0, i - 1)]?.focus();
    } else if (e.key === "ArrowRight" && card.pages.length > 1) {
      e.preventDefault();
      turn(card, pageOf(card) + 1);
    } else if (e.key === "ArrowLeft" && card.pages.length > 1) {
      e.preventDefault();
      turn(card, pageOf(card) - 1);
    } else if (e.key === "Escape") {
      e.preventDefault();
      reset();
    }
  };

  if (artifact.findings.length === 0) {
    return <p className={s.none}>Nothing to report under the chosen lenses.</p>;
  }

  // Settled findings, grouped by the answer that settled them (newest answer first).
  const bySeq = new Map<number, { said: string; findings: Finding[] }>();
  for (const f of settled) {
    const by = settledOf(f)!;
    const group = bySeq.get(by.seq) ?? { said: by.said, findings: [] };
    group.findings.push(f);
    bySeq.set(by.seq, group);
  }
  const settledGroups = [...bySeq.entries()].sort((a, b) => b[0] - a[0]);

  return (
    <>
      {cards.length === 0 ? (
        <p className={s.none}>Every finding here has been answered in the record.</p>
      ) : null}
      <ul className={s.cards} onPointerLeave={endPreview} data-testid="finding-cards">
        {shown.map((card, i) => {
          const p = pageOf(card);
          const f = card.pages[p]!;
          const n = card.pages.length;
          return (
            <li
              key={card.id}
              ref={(el) => {
                refs.current[i] = el;
              }}
              className={s.card}
              tabIndex={0}
              data-severity={card.severity}
              data-shown={card.pages.some((pg) => pg.id === shownId) || undefined}
              data-testid={`finding-${card.id}`}
              aria-label={`Finding${n > 1 ? `, ${p + 1} of ${n}` : ""}: ${f.summary}`}
              onFocus={(e) => e.target === e.currentTarget && setFocus(asFocus(f))}
              onPointerMove={() => preview(asFocus(f))}
              onKeyDown={onKey(i, card)}
            >
              <div className={s.top}>
                <span className={s.origin}>{origin(f)}</span>
                {f.evidence ? (
                  <span className={s.badge} title={f.evidence.source}>
                    {f.evidence.status.toUpperCase()}
                  </span>
                ) : null}
                {n > 1 ? (
                  <span className={s.pager}>
                    <button
                      type="button"
                      className={s.pageBtn}
                      aria-label="Previous finding of this kind"
                      disabled={p === 0}
                      onClick={() => turn(card, p - 1)}
                    >
                      ‹
                    </button>
                    <span className={s.pageText} aria-live="polite">
                      {p + 1} of {n}
                    </span>
                    <button
                      type="button"
                      className={s.pageBtn}
                      aria-label="Next finding of this kind"
                      disabled={p === n - 1}
                      onClick={() => turn(card, p + 1)}
                    >
                      ›
                    </button>
                  </span>
                ) : null}
              </div>
              <p className={s.claim}>
                <Prose text={f.summary} />
              </p>
              {f.routes_to && f.lever_label ? (
                <div className={s.levers}>
                  <button
                    type="button"
                    className={s.lever}
                    onClick={() => onRoute(f.routes_to!)}
                    data-testid="lever"
                  >
                    {f.lever_label} <span aria-hidden="true">→</span>
                  </button>
                </div>
              ) : null}
            </li>
          );
        })}
      </ul>
      {rest.length > 0 ? (
        <div className={cx(s.rest, open && s.restOpen)}>
          <span className={s.restText}>{open ? "Every finding is shown." : restLine(rest)}</span>
          <button
            type="button"
            className={s.restButton}
            onClick={() => setOpen((o) => !o)}
            aria-expanded={open}
            data-testid="findings-more"
          >
            {open ? "Show three" : "Show"}
          </button>
        </div>
      ) : null}
      {settled.length > 0 ? (
        <div className={s.settled} data-testid="findings-settled">
          <div className={s.settledHead}>
            <span className={s.settledText}>
              {settled.length} answered in the record{" "}
              {settledGroups.map(([seq, g], i) => (
                <span key={seq}>
                  {i > 0 ? ", " : "— "}
                  {g.findings.length} by #{seq}
                </span>
              ))}
            </span>
            <button
              type="button"
              className={s.restButton}
              onClick={() => setShowSettled((v) => !v)}
              aria-expanded={showSettled}
              data-testid="findings-settled-toggle"
            >
              {showSettled ? "Hide" : "Show"}
            </button>
          </div>
          {showSettled ? (
            <ul className={s.settledList} onPointerLeave={endPreview}>
              {settledGroups.flatMap(([seq, g]) =>
                g.findings.map((f) => (
                  <li
                    key={f.id}
                    className={s.settledItem}
                    tabIndex={0}
                    data-shown={shownId === f.id || undefined}
                    data-testid={`settled-${f.id}`}
                    onFocus={(e) => e.target === e.currentTarget && setFocus(asFocus(f))}
                    onPointerMove={() => preview(asFocus(f))}
                    onKeyDown={(e) => {
                      if (e.key === "Escape") {
                        e.preventDefault();
                        reset();
                      }
                    }}
                  >
                    <span className={s.settledBy}>
                      Answered by #{seq}: <Prose text={g.said} />
                    </span>
                    <span className={s.settledClaim}>
                      <Prose text={f.summary} />
                    </span>
                  </li>
                )),
              )}
            </ul>
          ) : null}
        </div>
      ) : null}
      <p className={s.keys} aria-hidden="true">
        <kbd>↑</kbd>
        <kbd>↓</kbd> evidence <kbd>←</kbd>
        <kbd>→</kbd> page <kbd>Esc</kbd> your data now
      </p>
      <p className={s.basis}>{artifact.basis}</p>
    </>
  );
}
