/**
 * A question's options for the prototype, styled by the production Record's Options.module.css:
 * a name, one consequence line, hover / focus / ↑ ↓ to put it on the stage, Enter to record.
 * Space is the stage's flip (frontend CLAUDE.md), so it is not taken here.
 */
import { useRef, type KeyboardEvent } from "react";
import { Rich } from "../../components/stage/text";
import o from "../../components/record/Options.module.css";

export interface Opt {
  key: string;
  label: string;
  line: string;
  tag?: string;
}

export function OptionList({
  options,
  shown,
  recorded,
  onShow,
  onRecord,
}: {
  options: Opt[];
  shown: string | null;
  recorded: string | null;
  onShow: (key: string) => void;
  onRecord: (key: string) => void;
}) {
  const refs = useRef<(HTMLLIElement | null)[]>([]);
  const at = Math.max(0, options.findIndex((x) => x.key === shown));
  const move = (i: number) => {
    const j = Math.max(0, Math.min(options.length - 1, i));
    refs.current[j]?.focus();
  };
  const onKeyDown = (e: KeyboardEvent<HTMLUListElement>) => {
    if (e.key === "ArrowDown" || e.key === "ArrowRight") {
      e.preventDefault();
      move(at + 1);
    } else if (e.key === "ArrowUp" || e.key === "ArrowLeft") {
      e.preventDefault();
      move(at - 1);
    } else if (e.key === "Enter") {
      e.preventDefault();
      onRecord(options[at]!.key);
    }
  };
  return (
    <>
      <ul className={o.options} role="listbox" aria-label="Options: ↑ ↓ preview, Enter records" onKeyDown={onKeyDown}>
        {options.map((x, i) => (
          <li
            key={x.key}
            ref={(el) => {
              refs.current[i] = el;
            }}
            role="option"
            aria-selected={recorded === x.key}
            tabIndex={i === at ? 0 : -1}
            className={o.option}
            data-shown={shown === x.key || undefined}
            data-recorded={recorded === x.key || undefined}
            data-key={x.key}
            onFocus={() => onShow(x.key)}
            onPointerEnter={() => onShow(x.key)}
            onClick={() => onRecord(x.key)}
          >
            <span className={o.pip} aria-hidden="true" />
            <span className={o.head}>
              <span className={o.label}>{x.label}</span>
              {x.tag ? <span className={o.tag}>{x.tag}</span> : null}
              {recorded === x.key ? (
                <span className={o.tag} data-tone="recorded">
                  recorded
                </span>
              ) : null}
            </span>
            <span className={o.line}>
              <Rich text={x.line} />
            </span>
          </li>
        ))}
      </ul>
      <p className={o.keys} aria-hidden="true">
        <kbd>↑</kbd>
        <kbd>↓</kbd> preview <kbd>Space</kbd> flip <kbd>Enter</kbd> record
      </p>
    </>
  );
}
