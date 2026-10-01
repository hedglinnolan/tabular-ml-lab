/**
 * The app's voice with data in chips (DESIGN_LANGUAGE §03). Server captions mark data with
 * backticks; titles sometimes name columns bare, so `chipify` marks the names it knows.
 */
import { createContext, Fragment, useContext, type ReactNode } from "react";

const IDENT = /^[a-z][a-z0-9]*(?:_[a-z0-9]+)+$/;

/** The project's column names, so a bare `fat_total` in a title becomes a chip. */
export const ColumnsContext = createContext<ReadonlySet<string>>(new Set());

/** Wrap column names in backticks unless the text already marks its data. */
export function chipify(text: string, known: ReadonlySet<string>): string {
  if (text.includes("`")) return text;
  const tokens = text.split(/(\s+)/);
  return tokens
    .map((tok, i) => {
      const m = /^([(]?)([A-Za-z][A-Za-z0-9_]*)([,.;:)]*)$/.exec(tok);
      if (!m) return tok;
      const [, pre, word, post] = m;
      const prev = tokens[i - 2] ?? "";
      const afterNumber = /\d$/.test(prev.replace(/[,.;:)]$/, ""));
      if (IDENT.test(word!) || (known.has(word!) && !afterNumber)) return `${pre}\`${word}\`${post}`;
      return tok;
    })
    .join("");
}

/** Backticks become mono chips. */
export function Rich({ text, chips = true }: { text: string; chips?: boolean }) {
  const known = useContext(ColumnsContext);
  // A correlation of −1.9e−17 is zero to the printed precision, never "-0.00".
  const tidy = text.replace(/(^|[^\w.])[-−]0\.00(?!\d)/g, (_, pre: string) => `${pre}0.00`);
  const marked = chips ? chipify(tidy, known) : tidy;
  const parts = marked.split("`");
  const out: ReactNode[] = parts.map((part, i) =>
    i % 2 === 1 ? (
      <code key={i} className="v">
        {part}
      </code>
    ) : (
      <Fragment key={i}>{part}</Fragment>
    ),
  );
  return <>{out}</>;
}
