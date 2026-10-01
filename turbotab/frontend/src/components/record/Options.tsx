/**
 * A question's options: a name and one consequence line each (BLUEPRINT §11, layer 1).
 *
 * Hover, keyboard focus and the arrow keys put an option's consequence on the stage — the
 * user learns by contrast, one key per option, and the stage morphs. Enter (or a click)
 * records. On touch, the first tap previews and a second tap records. An option that does
 * not apply is still on the shelf: focusable, previewable, saying why — and pressing it asks
 * the server, whose refusal answers at the option with its exits.
 */
import { useId, useRef, useState, type KeyboardEvent, type ReactNode } from "react";
import type { Decision } from "../../api/schema";
import { focusKey, useStageFocus, type StageFocus } from "../../state/focus";
import { cx } from "../../util/format";
import { keyAction, nextIndex } from "./optionNav";
import { Taught } from "./teach";
import s from "./Options.module.css";

export interface OptionTag {
  text: string;
  tone?: "usual" | "recorded" | "na" | "detected" | "suggested" | "fit" | "badge";
}

export interface OptionItem {
  key: string;
  label: ReactNode;
  /** The consequence, in the app's voice (backticks mark data). */
  line: string;
  /** What pressing records, and what the stage previews. Null: nothing to record yet. */
  decision: Decision | null;
  /** How the stage names the preview; defaults to the label when it is a string. */
  previewLabel?: string;
  /** What the stage's record button records, when it differs from `decision` (multi-select). */
  record?: Decision;
  recordLabel?: string;
  tags?: OptionTag[];
  /** Data beside the label, e.g. the rows a rule removes. */
  data?: ReactNode;
  /** Not applicable here: the reason, said in place of nothing happening. */
  na?: string;
  /** A second line under the consequence (a hint, a concern). */
  note?: ReactNode;
  /** Inline controls the option needs (a custom range). */
  extra?: ReactNode;
}

interface Props {
  items: OptionItem[];
  mode: "single" | "multi";
  /** Multi: which options are chosen. */
  selected?: ReadonlySet<string>;
  onToggle?: (key: string) => void;
  /** Single: record this option. Multi: Enter records the selection (the item is the active one). */
  onRecord: (item: OptionItem) => void;
  /** The option whose answer is on record, marked as such. */
  recordedKey?: string | null;
  pending?: boolean;
  /** Shown under the option it answers (a refusal with its exits). */
  answerAt?: { key: string; node: ReactNode } | null;
  label: string;
  testId?: string;
}

function previewOf(item: OptionItem): StageFocus | null {
  if (!item.decision) return null;
  const label =
    item.previewLabel ?? (typeof item.label === "string" ? item.label : String(item.key));
  return {
    kind: "option",
    decision: item.decision,
    label,
    ...(item.record ? { record: item.record } : {}),
    ...(item.recordLabel ? { recordLabel: item.recordLabel } : {}),
  };
}

export function Options({
  items,
  mode,
  selected,
  onToggle,
  onRecord,
  recordedKey = null,
  pending = false,
  answerAt = null,
  label,
  testId,
}: Props) {
  const base = useId();
  const refs = useRef<(HTMLLIElement | null)[]>([]);
  const touch = useRef(false);
  const { focus, preview, endPreview, setFocus, reset } = useStageFocus();
  const initial = Math.max(
    0,
    items.findIndex((o) => o.key === recordedKey),
  );
  const [activeRaw, setActive] = useState(initial);
  const active = Math.min(activeRaw, Math.max(0, items.length - 1));
  // Touch: the option a first tap previewed; a second tap on it records.
  const [armed, setArmed] = useState<string | null>(null);
  const shownKey = focus.kind === "option" ? focusKey(focus) : null;

  const move = (to: number) => {
    setActive(to);
    refs.current[to]?.focus();
  };

  const press = (item: OptionItem) => {
    if (pending) return;
    if (mode === "multi") onToggle?.(item.key);
    else onRecord(item);
  };

  const onKeyDown = (e: KeyboardEvent<HTMLUListElement>) => {
    // Keys typed into an option's own controls (a custom range) stay theirs.
    if ((e.target as HTMLElement).closest("input, select, textarea, button")) return;
    const to = nextIndex(e.key, active, items.length);
    if (to !== null) {
      e.preventDefault();
      move(to);
      return;
    }
    const action = keyAction(e.key, mode);
    const item = items[active];
    if (!action || !item) return;
    e.preventDefault();
    if (action === "escape") reset();
    else if (action === "toggle") onToggle?.(item.key);
    else if (!pending) onRecord(item); // multi: the chosen set, or the shown option if none is

  };

  return (
    <>
      <ul
        className={s.options}
        role="listbox"
        aria-label={label}
        aria-multiselectable={mode === "multi" || undefined}
        aria-busy={pending || undefined}
        onKeyDown={onKeyDown}
        onPointerLeave={endPreview}
        data-testid={testId}
      >
        {items.map((item, i) => {
          const f = previewOf(item);
          const shown = f !== null && shownKey === focusKey(f);
          const chosen = mode === "multi" ? (selected?.has(item.key) ?? false) : false;
          const recorded = recordedKey === item.key;
          const answer = answerAt?.key === item.key ? answerAt.node : null;
          return [
            <li
              key={item.key}
              id={`${base}-${item.key}`}
              ref={(el) => {
                refs.current[i] = el;
              }}
              role="option"
              aria-selected={mode === "multi" ? chosen : recorded}
              aria-disabled={item.na ? true : undefined}
              tabIndex={i === active ? 0 : -1}
              className={s.option}
              data-mode={mode}
              data-shown={shown || undefined}
              data-chosen={chosen || undefined}
              data-recorded={recorded || undefined}
              data-na={item.na ? true : undefined}
              data-key={item.key}
              data-testid={`option-${item.key}`}
              onFocus={(e) => {
                if (e.target !== e.currentTarget) return;
                setActive(i);
                if (f) setFocus(f);
              }}
              onPointerDown={(e) => {
                touch.current = e.pointerType === "touch";
              }}
              onPointerMove={(e) => {
                if (e.pointerType !== "touch" && f) preview(f);
              }}
              onClick={(e) => {
                if ((e.target as HTMLElement).closest("input, select, textarea, button, a")) return;
                if (touch.current && armed !== item.key) {
                  // Hover has no touch equivalent: the first tap previews.
                  setArmed(item.key);
                  setActive(i);
                  if (f) setFocus(f);
                  return;
                }
                setArmed(null);
                press(item);
              }}
            >
              <span className={s.pip} aria-hidden="true" />
              <span className={s.head}>
                <span className={s.label}>{item.label}</span>
                {item.data ? <span className={s.data}>{item.data}</span> : null}
                {item.tags?.map((t) => (
                  <span key={t.text} className={s.tag} data-tone={t.tone}>
                    {t.text}
                  </span>
                ))}
                {item.na ? (
                  <span className={s.tag} data-tone="na">
                    not applicable
                  </span>
                ) : null}
                {recorded ? (
                  <span className={s.tag} data-tone="recorded">
                    recorded
                  </span>
                ) : null}
                {armed === item.key ? (
                  <span className={s.tapAgain}>
                    tap again to {mode === "multi" ? "choose" : "record"}
                  </span>
                ) : null}
              </span>
              <span className={s.line}>
                <Taught text={item.na ?? item.line} placement="above" />
              </span>
              {item.note ? <span className={s.note}>{item.note}</span> : null}
              {item.extra ? <span className={s.extra}>{item.extra}</span> : null}
            </li>,
            answer ? (
              <li key={`${item.key}-answer`} role="none" className={s.answer}>
                {answer}
              </li>
            ) : null,
          ];
        })}
      </ul>
      <p className={cx(s.keys)} aria-hidden="true">
        <kbd>↑</kbd>
        <kbd>↓</kbd> preview{" "}
        {mode === "multi" ? (
          <>
            <kbd>Space</kbd> choose <kbd>Enter</kbd> record
          </>
        ) : (
          <>
            <kbd>Enter</kbd> record
          </>
        )}{" "}
        <kbd>Esc</kbd> your data now
      </p>
    </>
  );
}
