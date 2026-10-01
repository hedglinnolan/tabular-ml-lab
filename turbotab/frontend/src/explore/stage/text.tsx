/**
 * The app's voice with data in chips (§03) and terms that define themselves (§11.8).
 * Fixture captions name columns without backticks, so `chipify` marks the names it knows.
 */
import { Fragment, useId, useState, type ReactNode } from "react";
import { A } from "./fixture";
import s from "./StageScreen.module.css";

const KNOWN = new Set([...A.dataset.columns, "kcal"]);
const IDENT = /^[a-z][a-z0-9]*(?:_[a-z0-9]+)+$/;

/** Wrap column names in backticks unless the text already marks its data. */
export function chipify(text: string): string {
  if (text.includes("`")) return text;
  const tokens = text.split(/(\s+)/);
  return tokens
    .map((tok, i) => {
      const m = /^([(]?)([A-Za-z][A-Za-z0-9_]*)([,.;:)]*)$/.exec(tok);
      if (!m) return tok;
      const [, pre, word, post] = m;
      const prev = tokens[i - 2] ?? "";
      const afterNumber = /\d$/.test(prev.replace(/[,.;:)]$/, ""));
      if (IDENT.test(word!) || (KNOWN.has(word!) && !afterNumber)) return `${pre}\`${word}\`${post}`;
      return tok;
    })
    .join("");
}

function Term({ term, definition }: { term: string; definition: string }) {
  const [open, setOpen] = useState(false);
  const id = useId();
  return (
    <span className={s.termWrap}>
      <span
        className={s.term}
        tabIndex={0}
        aria-describedby={id}
        onMouseEnter={() => setOpen(true)}
        onMouseLeave={() => setOpen(false)}
        onFocus={() => setOpen(true)}
        onBlur={() => setOpen(false)}
      >
        {term}
      </span>
      <span id={id} role="tooltip" className={s.termCard} data-open={open || undefined}>
        {definition}
      </span>
    </span>
  );
}

/** Backticks become mono chips; any listed term becomes a self-defining term. */
export function Rich({ text, terms }: { text: string; terms?: Record<string, string> }) {
  const parts = text.split("`");
  const out: ReactNode[] = [];
  parts.forEach((part, i) => {
    if (i % 2 === 1) {
      out.push(
        <code key={i} className="v">
          {part}
        </code>,
      );
      return;
    }
    let rest = part;
    let k = 0;
    const names = terms ? Object.keys(terms) : [];
    while (rest) {
      const hit = names
        .map((t) => ({ t, at: rest.indexOf(t) }))
        .filter((h) => h.at >= 0)
        .sort((a, b) => a.at - b.at)[0];
      if (!hit) {
        out.push(<Fragment key={`${i}-${k}`}>{rest}</Fragment>);
        break;
      }
      if (hit.at > 0) out.push(<Fragment key={`${i}-${k++}`}>{rest.slice(0, hit.at)}</Fragment>);
      out.push(<Term key={`${i}-${k++}`} term={hit.t} definition={terms![hit.t]!} />);
      rest = rest.slice(hit.at + hit.t.length);
    }
  });
  return <>{out}</>;
}
