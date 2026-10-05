/**
 * An option list for a slot or a phrase: a name, the consequence in one line, and the two labels
 * every option carries — customary in the field, sound for this purpose (North star 5) — shown
 * for the option on the canvas only, so the list stays a list. Hover, focus or ↑ ↓ put an option
 * on the canvas; a click or Enter records it. Space is the canvas's flip.
 */
import { useRef, type KeyboardEvent, type ReactNode } from "react";
import { Rich } from "../../components/stage/text";
import o from "../../components/record/Options.module.css";
import c from "./screen.module.css";

export interface Opt {
  key: string;
  label: string;
  line?: string | null;
  data?: string | null;
  customary?: string | null;
  sound?: string | null;
  verdict?: string | null;
  tags?: { text: string; tone?: "usual" | "guess" | "na" | "recorded" | "badge" }[];
  /** Why recording it is not possible here (refused by the engine, or not captured). */
  blocked?: string | null;
}

export function Opts({
  options,
  shown,
  recorded,
  onShow,
  onRecord,
  label,
  mode = "single",
  chosen,
  extra,
}: {
  options: Opt[];
  shown: string | null;
  recorded: string | null;
  onShow: (key: string) => void;
  onRecord: (key: string, via: "pointer" | "key") => void;
  label: string;
  mode?: "single" | "multi";
  chosen?: ReadonlySet<string>;
  extra?: (key: string) => ReactNode;
}) {
  const refs = useRef<(HTMLLIElement | null)[]>([]);
  const at = Math.max(0, options.findIndex((x) => x.key === shown));
  const move = (i: number) => {
    const j = Math.max(0, Math.min(options.length - 1, i));
    refs.current[j]?.focus();
  };
  const onKeyDown = (e: KeyboardEvent<HTMLUListElement>) => {
    if (e.key === "ArrowDown") {
      e.preventDefault();
      move(at + 1);
    } else if (e.key === "ArrowUp") {
      e.preventDefault();
      move(at - 1);
    } else if (e.key === "Enter") {
      e.preventDefault();
      onRecord(options[at]!.key, "key");
    }
  };
  return (
    <ul
      className={`${o.options} ${c.opts}`}
      role="listbox"
      aria-label={label}
      aria-multiselectable={mode === "multi" || undefined}
      onKeyDown={onKeyDown}
    >
      {options.map((x, i) => {
        const isShown = shown === x.key;
        const isRec = recorded === x.key;
        return (
          <li
            key={x.key}
            ref={(el) => {
              refs.current[i] = el;
            }}
            role="option"
            aria-selected={mode === "multi" ? !!chosen?.has(x.key) : isRec}
            tabIndex={i === at ? 0 : -1}
            className={o.option}
            data-mode={mode}
            data-shown={isShown || undefined}
            data-recorded={(mode === "single" && isRec) || undefined}
            data-chosen={(mode === "multi" && chosen?.has(x.key)) || undefined}
            data-na={x.blocked ? true : undefined}
            data-testid={`opt-${x.key}`}
            onFocus={() => onShow(x.key)}
            onPointerEnter={() => onShow(x.key)}
            onClick={() => onRecord(x.key, "pointer")}
          >
            <span className={o.pip} aria-hidden="true" />
            <span className={o.head}>
              <span className={o.label}>{x.label}</span>
              {x.data ? <span className={o.data}>{x.data}</span> : null}
              {x.tags?.map((t) => (
                <span key={t.text} className={o.tag} data-tone={t.tone ?? "badge"}>
                  {t.text}
                </span>
              ))}
              {x.verdict ? (
                <span className={c.verdict} data-verdict={x.verdict}>
                  {x.verdict}
                </span>
              ) : null}
              {isRec && mode === "single" ? (
                <span className={o.tag} data-tone="recorded">
                  recorded
                </span>
              ) : null}
            </span>
            {x.line ? (
              <span className={o.line}>
                <Rich text={x.line} />
              </span>
            ) : null}
            {isShown && (x.customary || x.sound) ? (
              <span className={c.labels}>
                {x.customary ? (
                  <span className={c.labelRow}>
                    <span className={c.labelKey}>customary</span>
                    <span>
                      <Rich text={x.customary} />
                    </span>
                  </span>
                ) : null}
                {x.sound ? (
                  <span className={c.labelRow}>
                    <span className={c.labelKey}>sound?</span>
                    <span>
                      <Rich text={x.sound} />
                    </span>
                  </span>
                ) : null}
              </span>
            ) : null}
            {isShown && x.blocked ? (
              <span className={c.blocked}>
                <Rich text={x.blocked} />
              </span>
            ) : null}
            {extra ? extra(x.key) : null}
          </li>
        );
      })}
    </ul>
  );
}
