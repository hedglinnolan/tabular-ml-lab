export const fmtInt = (n: number) => Math.round(n).toLocaleString("en-US");

export function fmtBytes(n: number | null | undefined): string {
  if (n === null || n === undefined) return "—";
  if (n < 1024) return `${n} B`;
  const units = ["KB", "MB", "GB", "TB"];
  let v = n / 1024;
  let u = 0;
  while (v >= 1024 && u < units.length - 1) {
    v /= 1024;
    u++;
  }
  return `${v < 10 ? v.toFixed(1) : Math.round(v)} ${units[u]}`;
}

export function fmtValue(v: unknown): string {
  if (v === null || v === undefined) return "";
  if (typeof v === "number") {
    if (Number.isInteger(v)) return v.toLocaleString("en-US");
    const abs = Math.abs(v);
    return abs !== 0 && (abs < 0.001 || abs >= 1e7) ? v.toExponential(2) : String(+v.toFixed(4));
  }
  if (typeof v === "boolean") return v ? "true" : "false";
  return String(v);
}

const CODE_NAME =
  /(^|_)(id|seqn|code|zip|year|yr|cycle|visit|wave)$|^(seqn|year|cycle)|_year_|year_/i;

/**
 * A column whose integers are labels or dates, not quantities: identifiers and years. Their
 * values are written without digit grouping (SEQN 9966, not 9,966; 2001, not 2,001).
 */
export function codeLike(
  name: string,
  summary?: { min: number | null; max: number | null; dtype?: string } | null,
): boolean {
  if (CODE_NAME.test(name)) return true;
  if (!summary || summary.min === null || summary.max === null) return false;
  const integral = Number.isInteger(summary.min) && Number.isInteger(summary.max);
  return integral && summary.min >= 1800 && summary.max <= 2100 && summary.dtype === "integer";
}

/** An identifier or a year as written: no grouping, no rounding. */
export function fmtCode(v: number | null): string {
  if (v === null) return "—";
  return Number.isInteger(v) ? String(v) : String(+v.toFixed(4));
}

export function fmtStat(v: number | null): string {
  if (v === null) return "—";
  const abs = Math.abs(v);
  if (abs >= 1000) return Math.round(v).toLocaleString("en-US");
  if (abs >= 10) return v.toFixed(1);
  return v.toFixed(2);
}

export function fmtWhen(iso: string): string {
  const t = Date.parse(iso);
  if (Number.isNaN(t)) return iso;
  const d = new Date(t);
  const diff = Date.now() - t;
  if (diff < 60_000) return "just now";
  if (diff < 3_600_000) return `${Math.round(diff / 60_000)} min ago`;
  if (diff < 86_400_000) return `${Math.round(diff / 3_600_000)} h ago`;
  return d.toLocaleDateString("en-US", { month: "short", day: "numeric", year: "numeric" });
}

export function fmtClock(iso: string): string {
  const t = Date.parse(iso);
  if (Number.isNaN(t)) return iso;
  return new Date(t).toLocaleTimeString("en-US", { hour: "2-digit", minute: "2-digit" });
}

export function cx(...parts: Array<string | false | null | undefined>): string {
  return parts.filter(Boolean).join(" ");
}

/** Oxford-comma list: "a", "a and b", "a, b, and c". */
export function listJoin(items: string[]): string {
  if (items.length <= 1) return items.join("");
  if (items.length === 2) return `${items[0]} and ${items[1]}`;
  return `${items.slice(0, -1).join(", ")}, and ${items[items.length - 1]}`;
}

/** What to say about a table that has no size yet. Never claims work that has stopped. */
export function readingText(
  ingest: { status: string; cancelled: boolean } | null | undefined,
): string {
  if (ingest?.status === "error") return "could not be read";
  if (ingest?.cancelled) return "reading stopped";
  return "reading…";
}
