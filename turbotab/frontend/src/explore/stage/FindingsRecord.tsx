/**
 * S3 — findings as one-line claims with their levers. Three are pushed; same-kind
 * findings share one paged card; the rest are counted and typed, one press away
 * (instant: disclosure is not a consequence, §05.2). Focusing a card puts its evidence
 * on the stage; paging a card morphs the evidence (the flag's bars grow and shrink).
 */
import { useRef, useState, type KeyboardEvent } from "react";
import type { FindingCard } from "./findings";
import type { ScenarioId } from "./scenarios";
import { Rich } from "./text";
import s from "./StageScreen.module.css";

interface Props {
  pushed: FindingCard[];
  rest: FindingCard[];
  restLine: string;
  total: number;
  basis: string;
  activeCard: string | null;
  pages: Record<string, number>;
  onHover: (card: string) => void;
  onFocusCard: (card: string) => void;
  onKeyboard: () => void;
  onEscape: () => void;
  onPage: (card: string, page: number) => void;
  onRoute: (to: ScenarioId) => void;
}

function Card({
  card,
  page,
  active,
  onHover,
  onFocusCard,
  onPage,
  onRoute,
  onKeyDown,
  cardRef,
}: {
  card: FindingCard;
  page: number;
  active: boolean;
  onHover: () => void;
  onFocusCard: () => void;
  onPage: (p: number) => void;
  onRoute: (to: ScenarioId) => void;
  onKeyDown: (e: KeyboardEvent<HTMLLIElement>) => void;
  cardRef: (el: HTMLLIElement | null) => void;
}) {
  const [receipt, setReceipt] = useState(false);
  const p = card.pages[page]!;
  const n = card.pages.length;
  return (
    <li
      ref={cardRef}
      className={s.fcard}
      tabIndex={0}
      data-active={active || undefined}
      data-card={card.id}
      aria-label={`Finding${n > 1 ? `, ${page + 1} of ${n}` : ""}`}
      onFocus={onFocusCard}
      onPointerEnter={onHover}
      onKeyDown={onKeyDown}
    >
      <div className={s.ftop}>
        <span className={s.forigin}>{card.origin}</span>
        {card.badge ? <span className={s.fbadge}>{card.badge}</span> : null}
        {n > 1 ? (
          <span className={s.pager}>
            <button
              type="button"
              className={s.pageBtn}
              aria-label="Previous"
              disabled={page === 0}
              onClick={() => onPage(page - 1)}
            >
              ‹
            </button>
            <span className={s.pageText}>
              {page + 1} of {n}
            </span>
            <button
              type="button"
              className={s.pageBtn}
              aria-label="Next"
              disabled={page === n - 1}
              onClick={() => onPage(page + 1)}
            >
              ›
            </button>
          </span>
        ) : null}
      </div>
      <p className={s.fclaim}>
        <Rich text={p.claim} />
      </p>
      <div className={s.flever}>
        {card.lever && "to" in card.lever ? (
          <button type="button" className={s.lever} onClick={() => onRoute((card.lever as { to: ScenarioId }).to)}>
            {card.lever.label} <span aria-hidden="true">→</span>
          </button>
        ) : card.lever ? (
          receipt ? (
            <span className={s.receipt}>
              <Rich text={(card.lever as { receipt: string }).receipt} />
            </span>
          ) : (
            <button type="button" className={s.lever} onClick={() => setReceipt(true)}>
              {card.lever.label}
            </button>
          )
        ) : null}
      </div>
    </li>
  );
}

export function FindingsRecord({
  pushed,
  rest,
  restLine,
  total,
  basis,
  activeCard,
  pages,
  onHover,
  onFocusCard,
  onKeyboard,
  onEscape,
  onPage,
  onRoute,
}: Props) {
  const [open, setOpen] = useState(false);
  const shown = open ? [...pushed, ...rest] : pushed;
  const refs = useRef<(HTMLLIElement | null)[]>([]);

  const keyFor = (i: number, card: FindingCard) => (e: KeyboardEvent<HTMLLIElement>) => {
    if (e.target !== e.currentTarget) return;
    const page = pages[card.id] ?? 0;
    if (e.key === "ArrowDown") {
      e.preventDefault();
      onKeyboard();
      refs.current[Math.min(shown.length - 1, i + 1)]?.focus();
    } else if (e.key === "ArrowUp") {
      e.preventDefault();
      onKeyboard();
      refs.current[Math.max(0, i - 1)]?.focus();
    } else if (e.key === "ArrowRight" && card.pages.length > 1) {
      e.preventDefault();
      onKeyboard();
      onPage(card.id, Math.min(card.pages.length - 1, page + 1));
    } else if (e.key === "ArrowLeft" && card.pages.length > 1) {
      e.preventDefault();
      onKeyboard();
      onPage(card.id, Math.max(0, page - 1));
    } else if (e.key === "Escape") {
      e.preventDefault();
      onEscape();
    }
  };

  return (
    <section className={s.findings} aria-labelledby="findings-title">
      <h2 id="findings-title" className={s.findingsTitle}>
        Noticed in this table <span className={s.count}>{total}</span>
      </h2>
      <ul className={s.fcards} data-testid="findings">
        {shown.map((c, i) => (
          <Card
            key={c.id}
            card={c}
            page={pages[c.id] ?? 0}
            active={activeCard === c.id}
            onHover={() => onHover(c.id)}
            onFocusCard={() => onFocusCard(c.id)}
            onPage={(p) => onPage(c.id, p)}
            onRoute={onRoute}
            onKeyDown={keyFor(i, c)}
            cardRef={(el) => {
              refs.current[i] = el;
            }}
          />
        ))}
      </ul>
      {!open ? (
        <div className={s.rest}>
          <span className={s.restText}>
            <Rich text={restLine} />
          </span>
          <button type="button" className={s.restButton} onClick={() => setOpen(true)} aria-expanded={false}>
            Show
          </button>
        </div>
      ) : null}
      <p className={s.keys} aria-hidden="true">
        <kbd>↑</kbd>
        <kbd>↓</kbd> evidence <kbd>←</kbd>
        <kbd>→</kbd> page <kbd>Esc</kbd> pipeline
      </p>
      <p className={s.fbasis}>{basis}</p>
    </section>
  );
}
