/**
 * The Record's options: a name and one consequence line each, nothing more.
 *
 * Hover or focus previews an option on the stage; ↑ ↓ move through them so the user
 * learns by contrast (flip, watch, flip back); Enter or a click records. An option that
 * cannot be used is still on the shelf — focusable, previewable, and saying why — and
 * pressing it answers with its exit instead of doing nothing.
 */
import { useId, useRef, useState, type KeyboardEvent } from "react";
import type { OptionModel } from "./scenarios";
import { Rich } from "./text";
import s from "./StageScreen.module.css";

interface Props {
  options: OptionModel[];
  previewKey: string | null;
  recordedKey: string | null;
  onFocusOption: (key: string) => void;
  onHoverOption: (key: string) => void;
  onKeyboard: () => void;
  onEscape: () => void;
  onRecord: (key: string) => void;
  terms?: Record<string, string>;
}

export function OptionList({
  options,
  previewKey,
  recordedKey,
  onFocusOption,
  onHoverOption,
  onKeyboard,
  onEscape,
  onRecord,
  terms,
}: Props) {
  const base = useId();
  const refs = useRef<(HTMLLIElement | null)[]>([]);
  const initial = Math.max(0, options.findIndex((o) => o.key === recordedKey));
  const [focus, setFocus] = useState(initial);
  const [refusedAt, setRefusedAt] = useState<string | null>(null);
  const [whyOpen, setWhyOpen] = useState<string | null>(null);

  const move = (i: number) => {
    const j = Math.max(0, Math.min(options.length - 1, i));
    setFocus(j);
    refs.current[j]?.focus();
  };

  const press = (o: OptionModel) => {
    if (o.refused) {
      setRefusedAt(o.key); // a refusal answers at the control, with its exit
      return;
    }
    onRecord(o.key);
  };

  const onKeyDown = (e: KeyboardEvent<HTMLUListElement>) => {
    const i = focus;
    if (e.key === "ArrowDown" || e.key === "ArrowRight") {
      e.preventDefault();
      onKeyboard();
      move(i + 1);
    } else if (e.key === "ArrowUp" || e.key === "ArrowLeft") {
      e.preventDefault();
      onKeyboard();
      move(i - 1);
    } else if (e.key === "Home") {
      e.preventDefault();
      onKeyboard();
      move(0);
    } else if (e.key === "End") {
      e.preventDefault();
      onKeyboard();
      move(options.length - 1);
    } else if (e.key === "Enter" || e.key === " ") {
      e.preventDefault();
      press(options[i]!);
    } else if (e.key === "Escape") {
      e.preventDefault();
      setWhyOpen(null);
      onEscape();
    } else if (e.key === "?") {
      e.preventDefault();
      const o = options[i]!;
      if (o.why) setWhyOpen(whyOpen === o.key ? null : o.key);
    }
  };

  return (
    <>
      <ul
        className={s.options}
        role="listbox"
        aria-label="Options — hover or use the arrow keys to preview, Enter to record"
        onKeyDown={onKeyDown}
        data-testid="options"
      >
        {options.map((o, i) => {
          const active = previewKey === o.key;
          const recorded = recordedKey === o.key;
          return (
            <li
              key={o.key}
              id={`${base}-${o.key}`}
              ref={(el) => {
                refs.current[i] = el;
              }}
              role="option"
              aria-selected={recorded}
              aria-disabled={o.refused ? true : undefined}
              tabIndex={i === focus ? 0 : -1}
              className={s.option}
              data-active={active || undefined}
              data-recorded={recorded || undefined}
              data-refused={o.refused ? true : undefined}
              data-key={o.key}
              onFocus={() => {
                setFocus(i);
                onFocusOption(o.key);
              }}
              onPointerEnter={() => onHoverOption(o.key)}
              onClick={() => press(o)}
            >
              <span className={s.pip} aria-hidden="true" />
              <span className={s.optHead}>
                <span className={s.optLabel}>{o.label}</span>
                {o.tag ? <span className={s.optTag}>{o.tag}</span> : null}
                {o.refused ? <span className={s.optNa}>not applicable</span> : null}
                {recorded ? <span className={s.optRecorded}>recorded</span> : null}
                {o.why ? (
                  <button
                    type="button"
                    className={s.whyButton}
                    tabIndex={-1}
                    aria-expanded={whyOpen === o.key}
                    onClick={(e) => {
                      e.stopPropagation();
                      setWhyOpen(whyOpen === o.key ? null : o.key);
                    }}
                  >
                    why?
                  </button>
                ) : null}
              </span>
              <span className={s.optLine}>
                <Rich text={o.line} terms={terms} />
              </span>
              {o.why && whyOpen === o.key ? (
                <span className={s.optWhy}>
                  {o.badge ? <span className={s.optBadge}>{o.badge}</span> : null}
                  <Rich text={o.why} terms={terms} />
                </span>
              ) : null}
              {o.refused ? (
                <span className={s.optExit}>
                  <button
                    type="button"
                    className={s.exitButton}
                    onClick={(e) => {
                      e.stopPropagation();
                      onRecord(o.refused!.exitKey);
                    }}
                    onKeyDown={(e) => e.stopPropagation()}
                    data-flash={refusedAt === o.key || undefined}
                  >
                    {o.refused.exitLabel} instead
                  </button>
                </span>
              ) : null}
            </li>
          );
        })}
      </ul>
      <p className={s.keys} aria-hidden="true">
        <kbd>↑</kbd>
        <kbd>↓</kbd> preview <kbd>?</kbd> why <kbd>Enter</kbd> record <kbd>Esc</kbd> pipeline
      </p>
    </>
  );
}
