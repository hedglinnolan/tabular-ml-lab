/**
 * A CHOICE question whose options are small multiples of their own consequence.
 *
 *   ┌ question · one line of why ───────────────────────────────┐
 *   │ [card][card][card][card][card][card]   ← every option's    │
 *   │    ▲                                     picture, at once  │
 *   │ ┌ stage: the focused option, enlarged ─────────────────┐   │
 *   │ └──────────────────────────────────────────────────────┘   │
 *   │ [Use the residual method]     ← → preview · P pin · ↵       │
 *   └────────────────────────────────────────────────────────────┘
 *
 * The cards never move or resize (nothing reflows under the cursor, DRIVE_RUBRIC §2.4); the
 * stage has a fixed height and its caret sits under the focused card. Focus is preview: arrow
 * keys and hover move it, nothing is recorded until the button (or Enter) is pressed. "P" pins
 * the focused option, and the stage then shows the pinned and the focused option side by side.
 */
import { useEffect, useId, useRef, type KeyboardEvent, type ReactNode } from "react";
import { motion } from "motion/react";
import { Prose } from "../../components/Prose";
import { useTransitions } from "../../motion/prefs";
import type { OptionBase } from "./fixture";
import { cx } from "./util";
import s from "./inline.module.css";

export interface QuestionProps<O extends OptionBase> {
  layoutId: string;
  kicker: string;
  question: string;
  why: string;
  options: O[];
  focusKey: string | null;
  pinKey: string | null;
  onFocus: (key: string | null) => void;
  onPin: (key: string | null) => void;
  onChoose: (key: string) => void;
  spark: (o: O) => ReactNode;
  stat: (o: O) => ReactNode;
  stage: (focus: O | null, pinned: O | null) => ReactNode;
  stageHeight: number;
  /** Extra text for the choose button when the focused option is a variant (partition). */
  chooseLabel?: (o: O) => string | null;
  canChoose?: (o: O) => boolean;
  testId?: string;
}

const DWELL_MS = 70;

export function Question<O extends OptionBase>({
  layoutId,
  kicker,
  question,
  why,
  options,
  focusKey,
  pinKey,
  onFocus,
  onPin,
  onChoose,
  spark,
  stat,
  stage,
  stageHeight,
  chooseLabel,
  canChoose,
  testId,
}: QuestionProps<O>) {
  const t = useTransitions();
  const headingId = useId();
  const stageId = useId();
  const cards = useRef<(HTMLDivElement | null)[]>([]);
  const dwell = useRef<number | null>(null);
  const focusIdx = options.findIndex((o) => o.key === focusKey);
  const focus = focusIdx >= 0 ? options[focusIdx]! : null;
  const pinned = options.find((o) => o.key === pinKey && o.key !== focusKey) ?? null;
  const pinIdx = pinned ? options.indexOf(pinned) : -1;
  const n = options.length;

  useEffect(
    () => () => {
      if (dwell.current) window.clearTimeout(dwell.current);
    },
    [],
  );

  const choosable = (o: O | null): o is O => !!o && (canChoose ? canChoose(o) : !o.refused);

  const move = (to: number) => {
    const i = (to + n) % n;
    const el = cards.current[i];
    el?.focus();
    onFocus(options[i]!.key);
  };

  const onKey = (e: KeyboardEvent<HTMLDivElement>) => {
    const at = focusIdx >= 0 ? focusIdx : 0;
    switch (e.key) {
      case "ArrowRight":
      case "ArrowDown":
        e.preventDefault();
        move(focusIdx >= 0 ? at + 1 : 0);
        break;
      case "ArrowLeft":
      case "ArrowUp":
        e.preventDefault();
        move(focusIdx >= 0 ? at - 1 : n - 1);
        break;
      case "Home":
        e.preventDefault();
        move(0);
        break;
      case "End":
        e.preventDefault();
        move(n - 1);
        break;
      case "p":
      case "P":
        if (focus) {
          e.preventDefault();
          onPin(pinKey === focus.key ? null : focus.key);
        }
        break;
      case "Escape":
        e.preventDefault();
        if (pinKey) onPin(null);
        else onFocus(null);
        break;
      case "Enter":
      case " ":
        e.preventDefault();
        if (choosable(focus)) onChoose(focus.key);
        break;
    }
  };

  // Up to six options share the width; more keep their size and the strip scrolls sideways
  // (focus scrolls the focused card into view), so a long shelf never shrinks its pictures.
  const cols = n <= 6 ? `repeat(${n}, minmax(0, 1fr))` : `repeat(${n}, 132px)`;
  const label = focus ? (chooseLabel?.(focus) ?? focus.choose) : null;

  return (
    <motion.section
      layoutId={layoutId}
      layout
      transition={{ layout: t.settle }}
      className={s.question}
      style={{ borderRadius: 16 }}
      aria-labelledby={headingId}
      data-testid={testId}
    >
      <motion.div layout="position" transition={{ layout: t.settle }} className={s.qInner}>
        <div className={s.qKicker}>{kicker}</div>
        <h2 id={headingId} className={s.qTitle}>
          <Prose text={question} />
        </h2>
        <p className={s.qWhy}>
          <Prose text={why} />
        </p>

        <div className={s.stripScroll}>
          <div
            className={s.strip}
            style={{ gridTemplateColumns: cols }}
            role="radiogroup"
            aria-labelledby={headingId}
            aria-describedby={stageId}
            onKeyDown={onKey}
            onPointerLeave={() => {
              if (dwell.current) window.clearTimeout(dwell.current);
            }}
          >
            {options.map((o, i) => {
              const isFocus = o.key === focusKey;
              const isPin = o.key === pinKey;
              return (
                <div key={o.key} className={s.cell}>
                  <div
                    ref={(el) => {
                      cards.current[i] = el;
                    }}
                    role="radio"
                    aria-checked={isFocus}
                    aria-disabled={o.refused ? true : undefined}
                    tabIndex={isFocus || (focusIdx < 0 && i === 0) ? 0 : -1}
                    className={cx(s.card, isFocus && s.cardFocus, isPin && s.cardPin)}
                    data-refused={!!o.refused}
                    data-option={o.key}
                    onFocus={() => {
                      if (!isFocus) onFocus(o.key);
                    }}
                    onClick={() => {
                      cards.current[i]?.focus();
                      onFocus(o.key);
                    }}
                    onPointerEnter={() => {
                      if (dwell.current) window.clearTimeout(dwell.current);
                      dwell.current = window.setTimeout(() => onFocus(o.key), DWELL_MS);
                    }}
                  >
                    <span className={s.cardHead}>
                      <span className={s.cardLabel}>{o.label}</span>
                      {o.usual ? <span className={s.usual}>usual</span> : null}
                    </span>
                    <span className={s.cardSpark}>{spark(o)}</span>
                    <span className={s.cardStat}>{stat(o)}</span>
                  </div>
                  <button
                    type="button"
                    tabIndex={-1}
                    className={cx(s.pin, isPin && s.pinOn)}
                    aria-pressed={isPin}
                    aria-label={isPin ? `Unpin ${o.label}` : `Pin ${o.label} to compare`}
                    title={isPin ? "Unpin" : "Pin to compare (P)"}
                    onClick={() => onPin(isPin ? null : o.key)}
                    disabled={!!o.refused}
                  >
                    <PinGlyph />
                  </button>
                </div>
              );
            })}
          </div>

          <div className={s.carets} style={{ gridTemplateColumns: cols }} aria-hidden="true">
            {pinIdx >= 0 ? (
              <span className={s.caret} data-tone="c2" style={{ gridColumn: pinIdx + 1 }} />
            ) : null}
            {focusIdx >= 0 ? (
              <span className={s.caret} data-tone="c1" style={{ gridColumn: focusIdx + 1 }} />
            ) : null}
          </div>
        </div>
        <div
          id={stageId}
          className={s.stage}
          data-stage
          data-state={focus ? (pinned ? "compare" : "focus") : "idle"}
          style={{ minHeight: stageHeight }}
        >
          {stage(focus, pinned)}
        </div>
        <p className="visually-hidden" aria-live="polite">
          {focus
            ? `Previewing ${focus.label}: ${(focus.refused ?? focus.consequence).replace(/`/g, "")} Nothing is recorded.`
            : ""}
        </p>

        <div className={s.actions}>
          <button
            type="button"
            className={s.choose}
            disabled={!choosable(focus)}
            onClick={() => focus && onChoose(focus.key)}
          >
            {focus
              ? choosable(focus)
                ? label
                : "Not available here"
              : "Preview an option to choose it"}
          </button>
          <span className={s.keys} aria-hidden="true">
            <kbd>←</kbd>
            <kbd>→</kbd> preview <kbd>P</kbd> compare <kbd>↵</kbd> choose
          </span>
        </div>
      </motion.div>
    </motion.section>
  );
}

function PinGlyph() {
  return (
    <svg width="12" height="12" viewBox="0 0 12 12" aria-hidden="true">
      <path
        d="M4.2 1.2h3.6l-.5 3.3 1.9 1.8v.9H2.8v-.9l1.9-1.8zM6 7.2v3.6"
        fill="currentColor"
        stroke="currentColor"
        strokeWidth="0.9"
        strokeLinejoin="round"
        strokeLinecap="round"
      />
    </svg>
  );
}
