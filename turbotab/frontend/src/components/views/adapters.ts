/**
 * From the engine's artifacts to the views' inputs. Every number passes through unchanged; every
 * sentence is the engine's, with its backticks dropped (the calm budget prints column names as
 * plain text) and its first letter set upper-case, since a sentence that opens on a column name
 * opened on a backtick. Table 2 and the forest share one row list, so their rows align by key.
 */
import type { ProfileArtifact } from "../../api/schema";
import type { EffectsArtifact } from "../../api/m3-types";
import { fmtInt } from "../stage/format";
import type { EffectsLock, ExhibitModel, Table1Artifact, Table1Summary, Table1Variable } from "./contracts";
import type { Cell, Footnote, ForestData, ForestRow, Gate, PageData, Placement, TableData, TableRow } from "./types";

const plain = (s: string | null | undefined) => (s ?? "").replaceAll("`", "");
const cap = (s: string) => (s ? s[0]!.toUpperCase() + s.slice(1) : s);
/** An engine sentence as the paper prints it: no backticks, and a capital to open it. */
export const sentence = (s: string | null | undefined) => cap(plain(s));

/** The line a view says while the plan is not locked (FOUNDATION §5 rule 6). */
export const LOCK_GATE = "Estimates open when Fit is pressed: the plan is fixed before any estimate is shown.";

/** Rule 6's gate from the lock the effects artifact should carry (an engine contract item). It
 *  fails closed: no lock on record is a closed gate, never an open one. */
export function gateFromLock(lock: EffectsLock | null | undefined, line = LOCK_GATE): Gate {
  return lock?.locked ? null : line;
}
const MARKS = "abcdefghijklmnopqrstuvwxyz";

/** The locked primary of the declared sequence (MODELING_SEQUENCE: Model 2 is the reported one). */
const PRIMARY = "model_2";

type Family = EffectsArtifact["families"][number];
type Fit = Family["sequence"][number];

function primaryFit(fam: Family): Fit | undefined {
  return fam.sequence.find((s) => s.key === PRIMARY) ?? fam.sequence[0];
}

/** The outcome's name, from the engine's effect sentence ("Difference in mean `glucose` per …"). */
export function outcomeOf(effects: EffectsArtifact, family = 0): string | null {
  const fit = primaryFit(effects.families[family]!);
  return fit?.inference?.effect?.match(/`([^`]+)`/)?.[1] ?? null;
}

/** The measure per unit of the exposure, in the engine's words: "Difference in mean glucose per
 *  unit of sugar". */
export function measureOf(effects: EffectsArtifact, exposure: string, family = 0): string {
  const effect = primaryFit(effects.families[family]!)?.inference?.effect;
  if (effect) return plain(effect).replace(/,?\s*holding the others\.?$/, "").replace("each input", exposure);
  return `${cap(effects.measure_label ?? "estimate")} per unit of ${exposure}`;
}

const SIDE_WORD: Record<string, string> = {
  mean_difference: "mean",
  odds_ratio: "odds of",
  risk_ratio: "risk of",
  rate_ratio: "rate of",
  hazard_ratio: "hazard of",
};

/** Each side of the reference in plain words, when the measure says which way is which. */
export function sidesOf(effects: EffectsArtifact, family = 0): [string, string] | undefined {
  const word = SIDE_WORD[effects.measure ?? ""];
  const outcome = outcomeOf(effects, family);
  if (!word || !outcome) return undefined;
  return [`Lower ${word} ${outcome}`, `Higher ${word} ${outcome}`];
}

interface SeqRow {
  key: string;
  exposure: string;
  fit: Fit;
  est: number | null;
  lo: number | null;
  hi: number | null;
  p: number | null;
}

/** One row per exposure per declared model, on the display scale (the ratio under a ratio). */
function sequenceRows(effects: EffectsArtifact, family: number): SeqRow[] {
  const fam = effects.families[family];
  if (!fam) return [];
  const ratio = effects.scale === "ratio";
  const many = effects.exposures.length > 1;
  return effects.exposures.flatMap((exposure) =>
    fam.sequence.flatMap((fit) => {
      const c = fit.effects?.find((e) => e.feature === exposure);
      if (!c) return [];
      return [
        {
          key: many ? `${exposure}:${fit.key}` : fit.key,
          exposure,
          fit,
          est: ratio ? c.ratio : c.estimate,
          lo: ratio ? c.ratio_low : c.ci_low,
          hi: ratio ? c.ratio_high : c.ci_high,
          p: c.p,
        },
      ];
    }),
  );
}

const subOf = (fit: Fit) => (fit.adjusted_for.length ? `Adjusted for ${fit.adjusted_for.length} column${fit.adjusted_for.length === 1 ? "" : "s"}` : "No other column in the model");

/** Table 2: the exposure's estimate across the declared model sequence of one model family. */
export function table2FromEffects(effects: EffectsArtifact, family = 0, number = "Table 2"): TableData {
  const rows = sequenceRows(effects, family);
  const many = effects.exposures.length > 1;
  // One footnote per adjusted model, marked on its row; the engine's note for the models beside
  // the primary (Model 3's "possible mediators: not a total effect").
  const footnotes: Footnote[] = [];
  const markOf = new Map<string, string>();
  for (const fit of effects.families[family]?.sequence ?? []) {
    if (!fit.adjusted_for.length) continue;
    const mark = MARKS[markOf.size]!;
    markOf.set(fit.key, mark);
    const note = fit.key !== PRIMARY && fit.note ? `; ${plain(fit.note).replace(/\.$/, "")}` : "";
    footnotes.push({ mark, text: `${plain(fit.label)}: adjusted for ${fit.adjusted_for.join(", ")}${note}.` });
  }
  const primary = primaryFit(effects.families[family]!);
  if (primary?.inference?.caption) footnotes.push({ text: sentence(primary.inference.caption) });
  if (effects.multiplicity) footnotes.push({ text: sentence(effects.multiplicity) });
  const body: TableRow[] = [];
  for (const exposure of effects.exposures) {
    if (many) body.push({ kind: "group", key: `g:${exposure}`, label: `Per unit of ${exposure}` });
    for (const r of rows.filter((x) => x.exposure === exposure)) {
      const mark = markOf.get(r.fit.key);
      body.push({
        kind: "row",
        key: r.key,
        label: plain(r.fit.label),
        sub: subOf(r.fit),
        indent: many,
        primary: r.fit.key === PRIMARY,
        marks: mark ? [mark] : undefined,
        cells: {
          est: { kind: "estimate", est: r.est, lo: r.lo, hi: r.hi },
          p: { kind: "p", p: r.p },
          n: { kind: "count", n: r.fit.n_rows },
        },
      });
    }
  }
  const exposure = effects.exposures.length === 1 ? effects.exposure : null;
  return {
    number,
    title: `${exposure ? measureOf(effects, exposure, family) : cap(effects.measure_label ?? "Estimates")}, across the declared model sequence (${plain(effects.rows)}).`,
    stub: "Model",
    columns: [
      { key: "est", label: `${cap(effects.measure_label ?? "estimate")} (95% CI)` },
      { key: "p", label: "P" },
      { key: "n", label: "n" },
    ],
    rows: body,
    footnotes,
    empty: "The declared model sequence has no estimate for what you study yet.",
  };
}

/** The forest beside Table 2: the same rows, keys and labels, on one axis. With several
 *  exposures, each declared model is a series, in the sequence's order, so a model keeps its color
 *  from one exposure to the next. */
export function forestFromEffects(effects: EffectsArtifact, family = 0): ForestData {
  const many = effects.exposures.length > 1;
  const rows: ForestRow[] = sequenceRows(effects, family).map((r) => ({
    key: r.key,
    label: many ? `${r.exposure} · ${plain(r.fit.label)}` : plain(r.fit.label),
    sub: subOf(r.fit),
    est: r.est,
    lo: r.lo,
    hi: r.hi,
    primary: r.fit.key === PRIMARY,
    ...(many ? { series: r.fit.key } : {}),
  }));
  const log = effects.scale === "ratio";
  const sequence = effects.families[family]?.sequence ?? [];
  return {
    stub: many ? "Exposure and model" : "Model",
    ...(many ? { series: sequence.map((f) => ({ key: f.key, label: plain(f.label) })) } : {}),
    measure: effects.exposures.length === 1 ? measureOf(effects, effects.exposure, family) : cap(effects.measure_label ?? "Estimate"),
    axis: log ? "log" : "linear",
    reference: log ? 1 : 0,
    sides: sidesOf(effects, family),
    rows,
    empty: "The declared model sequence has no estimate for what you study yet.",
  };
}

// ── Table 1 ──────────────────────────────────────────────────────────────────

export interface Table1FromProfileOptions {
  /** The characteristics, in Table 1's order. */
  columns: string[];
  /** The lens's unit in the plural: "participants". */
  unit: string;
  /** The outcome's column: left out, whatever `columns` says, because the profile covers the
   *  whole file, not the rows analyzed its gate opens on (FOUNDATION §5 rule 6). */
  outcome: string | null;
  /** The rows the analysis keeps (the cohort's n_final), to say whether the profile's rows are
   *  the analyzed ones. */
  analyzed: number;
  /** Each column's name in the paper with its unit ("Age, years"), from the codebook; a column
   *  without one keeps its name. */
  labels?: Record<string, string>;
}

/** Table 1's overall column from the profile: mean (SD) for a number, n (%) for each level, with
 *  the levels past the profile's top values gathered in one row so the percents sum to 100. The
 *  profile reads every row of the file, so the header's n and the footnote say so, and say when
 *  Who's in keeps fewer. The groups of what is studied, the design weights and the median for a
 *  skewed column wait for the engine's Table 1 (D1b). */
export function table1FromProfile(profile: ProfileArtifact, opts: Table1FromProfileOptions): Table1Artifact {
  const overall = "overall";
  const n = Math.max(0, ...profile.columns.map((c) => c.n));
  const variables = opts.columns.flatMap((name): Table1Variable[] => {
    if (name === opts.outcome) return [];
    const c = profile.columns.find((x) => x.name === name);
    if (!c) return [];
    const label = opts.labels?.[name] ?? name;
    const missing = c.n_missing ? { [overall]: c.n_missing } : undefined;
    if (c.dtype === "categorical" || c.dtype === "boolean") {
      const present = c.n - c.n_missing;
      const pct = (k: number) => (present ? (100 * k) / present : null);
      const top = c.top ?? [];
      const levels = top.map((t) => ({
        level: String(t.value),
        summary: { [overall]: { kind: "count_pct" as const, n: t.count, pct: pct(t.count) } },
      }));
      const rest = present - top.reduce((a, t) => a + t.count, 0);
      const others = Math.max(1, c.n_unique - top.length);
      if (rest > 0)
        levels.push({
          level: `${others} other level${others === 1 ? "" : "s"}`,
          summary: { [overall]: { kind: "count_pct" as const, n: rest, pct: pct(rest) } },
        });
      return [{ column: name, label, levels, missing }];
    }
    return [{ column: name, label, summary: { [overall]: { kind: "mean_sd" as const, mean: c.mean, sd: c.std } }, missing }];
  });
  const rows =
    n === opts.analyzed
      ? `all ${fmtInt(n)} rows in the file, every one of them analyzed`
      : `all ${fmtInt(n)} rows in the file, before Who's in; the analysis keeps ${fmtInt(opts.analyzed)}`;
  return { unit: opts.unit, outcome: null, groups: [{ key: overall, label: "Overall", n }], variables, weighted: false, rows };
}

function cellOf(s: Table1Summary | undefined): Cell | null {
  if (!s) return null;
  if (s.kind === "mean_sd") return { kind: "mean_sd", mean: s.mean ?? null, sd: s.sd ?? null };
  if (s.kind === "median_iqr") return { kind: "median_iqr", median: s.median ?? null, q1: s.q1 ?? null, q3: s.q3 ?? null };
  return { kind: "count_pct", n: s.n ?? 0, pct: s.pct ?? null };
}

/** Table 1 as the paper prints it. Under a closed lock (rule 6), the outcome beside the groups of
 *  what is studied is held out, and a footnote says when it joins; the outcome alone, in the
 *  overall column, is not an estimate and stays. */
export function table1FromArtifact(t: Table1Artifact, lock: Gate, number = "Table 1"): TableData {
  const rows: TableRow[] = [];
  let anyMissing = false;
  const kinds = new Set<string>();
  const held = lock !== null && t.groups.length > 1 ? t.variables.find((v) => v.column === t.outcome) : undefined;
  for (const v of t.variables) {
    if (v === held) continue;
    if (v.levels) {
      rows.push({ kind: "group", key: v.column, label: `${v.label}, n (%)` });
      for (const l of v.levels) {
        kinds.add("count_pct");
        rows.push({
          kind: "row",
          key: `${v.column}=${l.level}`,
          label: l.level,
          indent: true,
          cells: Object.fromEntries(t.groups.map((g) => [g.key, cellOf(l.summary[g.key])])),
        });
      }
    } else if (v.summary) {
      const kind = Object.values(v.summary)[0]?.kind ?? "mean_sd";
      kinds.add(kind);
      rows.push({
        kind: "row",
        key: v.column,
        label: `${v.label}, ${kind === "median_iqr" ? "median (IQR)" : "mean (SD)"}`,
        cells: Object.fromEntries(t.groups.map((g) => [g.key, cellOf(v.summary![g.key])])),
      });
    }
    if (v.missing) {
      anyMissing = true;
      rows.push({
        kind: "row",
        key: `${v.column}:missing`,
        label: "Missing",
        indent: true,
        cells: Object.fromEntries(t.groups.map((g) => [g.key, v.missing![g.key] !== undefined ? { kind: "count" as const, n: v.missing![g.key]! } : null])),
      });
    }
  }
  const footnotes: Footnote[] = [
    { text: `Characteristics of ${plain(t.rows)}${t.weighted ? ", with counts and percents weighted by the survey design" : ""}.` },
  ];
  if (anyMissing) footnotes.push({ text: "Percents are of the rows with a value." });
  if (held) footnotes.push({ text: `${held.label}, the outcome, is shown by group once the plan is locked.` });
  return {
    number,
    title: `Characteristics of the ${t.unit}`,
    stub: "Characteristic",
    columns: t.groups.map((g) => ({ key: g.key, label: g.label, sub: `n = ${fmtInt(g.n)}` })),
    rows,
    footnotes,
    empty: "No characteristic to describe yet.",
  };
}

// ── the page ─────────────────────────────────────────────────────────────────

/** The exhibit numbers in placement order (C7a's rule): the main text's tables and figures counted
 *  from 1 through Results then Discussion, the Supplement's from S1, each in the exhibit model's
 *  order within its section; an exhibit left out of the paper has no number. */
export function numberInPlacementOrder(exhibits: { key: string; kind: "table" | "figure"; placement: Placement }[]): Map<string, string | null> {
  const out = new Map<string, string | null>();
  const count = { main: { table: 0, figure: 0 }, supplement: { table: 0, figure: 0 } };
  const word = { table: "Table", figure: "Figure" };
  for (const place of ["results", "discussion", "supplement"] as const) {
    const part = place === "supplement" ? "supplement" : "main";
    for (const e of exhibits.filter((x) => x.placement === place)) {
      const k = ++count[part][e.kind];
      out.set(e.key, `${word[e.kind]} ${part === "supplement" ? "S" : ""}${k}`);
    }
  }
  for (const e of exhibits) if (e.placement === "left_out") out.set(e.key, null);
  return out;
}

/** The page preview of one exhibit of the exhibit model (C7a), with every exhibit in the model's
 *  order and the placements the floor allows each. */
export function pageFromExhibits(model: ExhibitModel, key: string): PageData | null {
  if (!model.exhibits.some((e) => e.key === key)) return null;
  return {
    focus: key,
    exhibits: model.exhibits.map((e) => ({
      key: e.key,
      number: e.number,
      kind: e.kind,
      caption: e.caption,
      placement: e.placement,
      allowed: e.placement_allowed,
      fixed: e.fixed_reason ?? undefined,
    })),
    text: model.text,
  };
}
