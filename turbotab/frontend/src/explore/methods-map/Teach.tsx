/**
 * Progressive disclosure on the video-game tutorial standard (BLUEPRINT §11.4): a concept is
 * taught at the first decision that uses it, in full, from the teaching pack; everywhere after,
 * it is a dotted term that opens its one-sentence card on hover or focus. The fading is per
 * concept and automatic — no setting. A card's one line is said in full the first time the card
 * opens, and the *why?* opens in place.
 */
import { useState, type ReactNode } from "react";
import { Rich } from "../../components/stage/text";
import { FX } from "./fixture";
import type { NodeId } from "./model";
import c from "./screen.module.css";

export function definition(teach: string, term: string): string | null {
  return FX.teaching[teach]?.terms.find((t) => t.term === term)?.definition ?? null;
}

/** A term met before: the word, dotted, with its card on hover or focus. */
export function Term({ teach, term, children }: { teach: string; term: string; children?: ReactNode }) {
  const def = definition(teach, term);
  return (
    <span className={c.term} tabIndex={0} data-testid={`term-${term}`}>
      {children ?? term}
      {def ? (
        <span role="tooltip" className={c.termCard}>
          <b>{term}</b> — <Rich text={def} />
        </span>
      ) : null}
    </span>
  );
}

export interface TermUse {
  teach: string;
  term: string;
}

/** The concepts a card uses: in full where first met, condensed to chips after. */
export function Concepts({
  node,
  terms,
  firstMet,
}: {
  node: NodeId;
  terms: TermUse[];
  firstMet: Record<string, NodeId>;
}) {
  if (!terms.length) return null;
  const fresh = terms.filter((t) => (firstMet[t.term] ?? node) === node);
  const known = terms.filter((t) => (firstMet[t.term] ?? node) !== node);
  return (
    <div className={c.concepts}>
      {fresh.length ? (
        <dl className={c.fresh} data-testid="concepts-new">
          {fresh.map((t) => (
            <div key={t.term} className={c.freshRow}>
              <dt>{t.term}</dt>
              <dd>
                <Rich text={definition(t.teach, t.term) ?? ""} />
              </dd>
            </div>
          ))}
        </dl>
      ) : null}
      {known.length ? (
        <p className={c.known} data-testid="concepts-known">
          {known.map((t, i) => (
            <span key={t.term}>
              {i ? " · " : null}
              <Term teach={t.teach} term={t.term} />
            </span>
          ))}
        </p>
      ) : null}
    </div>
  );
}

/** The card's question, its one line (in full on a first visit), and the *why?* in place. */
export function Ask({
  teach,
  first,
  question,
}: {
  teach: string | undefined;
  first: boolean;
  /** Overrides the teaching entry's question (a slot the pack names differently). */
  question?: string;
}) {
  const [why, setWhy] = useState(false);
  const entry = teach ? FX.teaching[teach] : undefined;
  if (!entry && !question) return null;
  return (
    <div className={c.ask}>
      <h2 className={c.question}>{question ?? entry!.question}</h2>
      {entry && first ? (
        <p className={c.oneLiner}>
          <Rich text={entry.one_liner} />{" "}
          <button type="button" className={c.why} aria-expanded={why} onClick={() => setWhy((v) => !v)}>
            why?
          </button>
        </p>
      ) : entry ? (
        <button type="button" className={c.why} aria-expanded={why} onClick={() => setWhy((v) => !v)}>
          why?
        </button>
      ) : null}
      {why && entry ? (
        <p className={c.whyText}>
          <Rich text={entry.why} />
        </p>
      ) : null}
    </div>
  );
}
