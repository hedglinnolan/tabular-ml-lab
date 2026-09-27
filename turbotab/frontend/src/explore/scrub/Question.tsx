/**
 * The open question and its shelf of options — a keyboard listbox that drives the scrub.
 *
 *   ↑ / ↓        focus the next option; the stage morphs to what it would do (a preview)
 *   ← / →        flip the stage to your data now / back to the choice
 *   Enter        record the focused option (never a not-applicable one)
 *
 * Options carry measured columns (e.g. "r with kcal", "kcal") instead of prose, so the shelf can be
 * compared at a glance and stays one row per option however many there are. A not-applicable
 * option stays on the shelf, disabled, with its reason and its exit (§0: never shortened).
 */
import { useEffect, useId, useRef, useState, type CSSProperties, type KeyboardEvent, type ReactNode } from "react";
import { LayoutGroup } from "motion/react";
import { Prose } from "../../components/Prose";
import { DecisionSentence, QuestionBlock } from "../../components/record/blocks";
import { cx } from "../../util/format";
import { useScrub } from "./engine/scrub";
import s from "./Question.module.css";

export interface ShelfOption {
  key: string;
  label: string;
  cols: string[];
  consequence: string;
  tag?: string;
  badge?: string;
  why?: ReactNode;
  caveats?: string[];
  /** Other preview states that belong to this row (e.g. partition on three nutrients). */
  alias?: string[];
  disabled?: { reason: string; exit?: { key: string; label: string } };
}

interface Props {
  id: string;
  kicker: string;
  question: string;
  why: string;
  columns: string[];
  options: ShelfOption[];
  recorded: { key: string; sentence: string } | null;
  onRecord: (key: string) => void;
  onChange: () => void;
  /** Which preview states may be recorded (never a refused one). */
  recordable: (key: string) => boolean;
  /** CSS grid widths of the measured columns. */
  colWidths?: string;
}

export function Question({
  id,
  kicker,
  question,
  why,
  columns,
  options,
  recorded,
  onRecord,
  onChange,
  recordable,
  colWidths,
}: Props) {
  const { active, focus, flip } = useScrub();
  const listId = useId();
  const box = useRef<HTMLDivElement>(null);
  const hover = useRef<number | null>(null);
  const [whyOpen, setWhyOpen] = useState<string | null>(null);
  const rowOf = (key: string | null) =>
    key ? (options.find((o) => o.key === key || o.alias?.includes(key)) ?? null) : null;
  const current = rowOf(active);

  useEffect(() => () => {
    if (hover.current) window.clearTimeout(hover.current);
  }, []);

  const canRecord = (key: string | null) => !!key && recordable(key);

  const move = (delta: number) => {
    const i = current ? options.indexOf(current) : -1;
    const next = options[Math.max(0, Math.min(options.length - 1, i < 0 ? 0 : i + delta))];
    if (next) focus(next.key);
  };

  const onKey = (e: KeyboardEvent) => {
    switch (e.key) {
      case "ArrowDown":
        e.preventDefault();
        move(1);
        break;
      case "ArrowUp":
        e.preventDefault();
        move(-1);
        break;
      case "Home":
        e.preventDefault();
        if (options[0]) focus(options[0].key);
        break;
      case "End":
        e.preventDefault();
        if (options.length) focus(options[options.length - 1]!.key);
        break;
      case "ArrowLeft":
        e.preventDefault();
        flip("now");
        break;
      case "ArrowRight":
        e.preventDefault();
        if (active) flip("after");
        break;
      case "Enter":
        if (canRecord(active)) {
          e.preventDefault();
          onRecord(active!);
        }
        break;
      case "Escape":
        e.preventDefault();
        focus(null);
        break;
    }
  };

  if (recorded) {
    const row = rowOf(recorded.key);
    return (
      <LayoutGroup id={id}>
        <DecisionSentence layoutId={id} subject={kicker.toLowerCase()} onChange={onChange} meta={row?.label}>
          <Prose text={recorded.sentence} />
        </DecisionSentence>
      </LayoutGroup>
    );
  }

  return (
    <LayoutGroup id={id}>
      <QuestionBlock layoutId={id} kicker={kicker} title={<Prose text={question} />} why={<Prose text={why} />}>
        <div style={colWidths ? ({ "--cols": colWidths } as CSSProperties) : undefined}>
        <div className={s.head} aria-hidden="true">
          <span />
          {columns.map((c) => (
            <span key={c} className={s.colHead}>
              <Prose text={c} />
            </span>
          ))}
        </div>
        <div
          ref={box}
          className={s.list}
          role="listbox"
          tabIndex={0}
          aria-label={kicker}
          aria-activedescendant={current ? `${listId}-${current.key}` : undefined}
          onKeyDown={onKey}
          data-testid={`${id}-options`}
        >
          {options.map((o) => {
            const on = current?.key === o.key;
            return (
              <div
                key={o.key}
                id={`${listId}-${o.key}`}
                role="option"
                aria-selected={on}
                aria-disabled={o.disabled ? true : undefined}
                className={cx(s.option, on && s.on, o.disabled && s.disabled)}
                data-key={o.key}
                onClick={() => {
                  focus(o.key);
                  box.current?.focus();
                }}
                onPointerEnter={() => {
                  if (hover.current) window.clearTimeout(hover.current);
                  hover.current = window.setTimeout(() => {
                    if (!on) focus(o.key);
                  }, 180);
                }}
                onPointerLeave={() => {
                  if (hover.current) window.clearTimeout(hover.current);
                }}
              >
                <div className={s.row}>
                  <span className={s.mark} aria-hidden="true" />
                  <span className={s.label}>
                    {o.label}
                    {o.tag ? <span className={s.tag}>{o.tag}</span> : null}
                  </span>
                  {o.cols.map((c, i) => (
                    <span key={i} className={s.col}>
                      {c}
                    </span>
                  ))}
                </div>
                {o.disabled ? (
                  <div className={s.reason}>
                    <Prose text={o.disabled.reason} />
                    {o.disabled.exit ? (
                      <button
                        type="button"
                        className={s.exit}
                        data-on={active === o.disabled.exit.key || undefined}
                        onClick={(e) => {
                          e.stopPropagation();
                          focus(o.disabled!.exit!.key);
                          box.current?.focus();
                        }}
                      >
                        <Prose text={o.disabled.exit.label} />
                      </button>
                    ) : null}
                  </div>
                ) : on ? (
                  <div className={s.detail}>
                    <p className={s.consequence}>
                      <Prose text={o.consequence} />
                    </p>
                    {o.why || o.badge ? (
                      <div className={s.more}>
                        {o.why ? (
                          <button
                            type="button"
                            className={s.whyButton}
                            aria-expanded={whyOpen === o.key}
                            onClick={(e) => {
                              e.stopPropagation();
                              setWhyOpen(whyOpen === o.key ? null : o.key);
                            }}
                          >
                            why?
                          </button>
                        ) : null}
                        {o.badge ? <span className={s.badge}>{o.badge}</span> : null}
                      </div>
                    ) : null}
                    {whyOpen === o.key && o.why ? (
                      <div className={s.whyBody}>
                        <p>{o.why}</p>
                        {o.caveats?.map((c) => (
                          <p key={c} className={s.caveat}>
                            {c}
                          </p>
                        ))}
                      </div>
                    ) : null}
                  </div>
                ) : null}
              </div>
            );
          })}
        </div>
        </div>
        <p className={s.keys} aria-hidden="true">
          <kbd>↑</kbd>
          <kbd>↓</kbd> choices · <kbd>←</kbd>
          <kbd>→</kbd> before / after · <kbd>Enter</kbd> records
        </p>
      </QuestionBlock>
    </LayoutGroup>
  );
}
