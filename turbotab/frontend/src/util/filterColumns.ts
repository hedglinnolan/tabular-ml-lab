/**
 * Column-name search for the target picker. Built once per column list; each
 * keystroke is one pass over pre-lowercased names, so 20,000 columns stay fast.
 *
 * Every whitespace-separated token must appear (case-insensitive). Ranking is
 * exact match, then prefix match, then the rest — each group in table order,
 * because table order is the order the researcher recorded the columns in.
 */
export interface ColumnIndex {
  names: readonly string[];
  lower: readonly string[];
}

export function buildColumnIndex(names: readonly string[]): ColumnIndex {
  return { names, lower: names.map((n) => n.toLowerCase()) };
}

/** Indices into `index.names`, ranked. An empty query returns every column in order. */
export function filterColumns(index: ColumnIndex, query: string): number[] {
  const tokens = query.toLowerCase().split(/\s+/).filter(Boolean);
  const n = index.lower.length;
  if (tokens.length === 0) {
    const all = new Array<number>(n);
    for (let i = 0; i < n; i++) all[i] = i;
    return all;
  }
  const whole = tokens.join(" ");
  const first = tokens[0]!;
  const exact: number[] = [];
  const prefix: number[] = [];
  const rest: number[] = [];
  outer: for (let i = 0; i < n; i++) {
    const name = index.lower[i]!;
    for (const t of tokens) if (!name.includes(t)) continue outer;
    if (name === whole) exact.push(i);
    else if (name.startsWith(first)) prefix.push(i);
    else rest.push(i);
  }
  return exact.concat(prefix, rest);
}
