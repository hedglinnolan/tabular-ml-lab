/**
 * From the engine's artifacts to the views' inputs. Every number passes through unchanged; every
 * sentence is the engine's, with its backticks dropped (the calm budget prints column names as
 * plain text). Table 2 and the forest share one row list, so their rows align by key.
 */
import type { ProfileArtifact } from "../../api/schema";
import type { EffectsArtifact } from "../../api/m3-types";
import { fmtInt } from "../stage/format";
import type { ExhibitModel, Table1Artifact, Table1Summary, Table1Variable } from "./contracts";
import type { Cell, Footnote, ForestData, ForestRow, PageData, TableData, TableRow } from "./types";

const plain = (s: string | null | undefined) => (s ?? "").replaceAll("`", "");
const cap = (s: string) => (s ? s[0]!.toUpperCase() + s.slice(1) : s);
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
  if (primary?.inference?.caption) footnotes.push({ text: plain(primary.inference.caption) });
  if (effects.multiplicity) footnotes.push({ text: plain(effects.multiplicity) });
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

/** The forest beside Table 2: the same rows, keys and labels, on one axis. */
export function forestFromEffects(effects: EffectsArtifact, family = 0): ForestData {
  const rows: ForestRow[] = sequenceRows(effects, family).map((r) => ({
    key: r.key,
    label: effects.exposures.length > 1 ? `${r.exposure} · ${plain(r.fit.label)}` : plain(r.fit.label),
    sub: subOf(r.fit),
    est: r.est,
    lo: r.lo,
    hi: r.hi,
    primary: r.fit.key === PRIMARY,
  }));
  const log = effects.scale === "ratio";
  return {
    measure: effects.exposures.length === 1 ? measureOf(effects, effects.exposure, family) : cap(effects.measure_label ?? "Estimate"),
    axis: log ? "log" : "linear",
    reference: log ? 1 : 0,
    sides: sidesOf(effects, family),
    rows,
    empty: "The declared model sequence has no estimate for what you study yet.",
  };
}

// ── Table 1 ──────────────────────────────────────────────────────────────────

/** Table 1's overall column from the profile: mean (SD) for a number, n (%) for each level. The
 *  groups of what is studied, and the design weights, wait for the engine's Table 1 (D1b). */
export function table1FromProfile(profile: ProfileArtifact, opts: { columns: string[]; unit: string; rows: string; n: number }): Table1Artifact {
  const overall = "overall";
  const variables = opts.columns.flatMap((name): Table1Variable[] => {
    const c = profile.columns.find((x) => x.name === name);
    if (!c) return [];
    const missing = c.n_missing ? { [overall]: c.n_missing } : undefined;
    if (c.dtype === "categorical" || c.dtype === "boolean") {
      const n = c.n - c.n_missing;
      return [
        {
          column: name,
          label: name,
          levels: (c.top ?? []).map((t) => ({
            level: String(t.value),
            summary: { [overall]: { kind: "count_pct" as const, n: t.count, pct: n ? (100 * t.count) / n : null } },
          })),
          missing,
        },
      ];
    }
    return [{ column: name, label: name, summary: { [overall]: { kind: "mean_sd" as const, mean: c.mean, sd: c.std } }, missing }];
  });
  return { unit: opts.unit, groups: [{ key: overall, label: "Overall", n: opts.n }], variables, weighted: false, rows: opts.rows };
}

function cellOf(s: Table1Summary | undefined): Cell | null {
  if (!s) return null;
  if (s.kind === "mean_sd") return { kind: "mean_sd", mean: s.mean ?? null, sd: s.sd ?? null };
  if (s.kind === "median_iqr") return { kind: "median_iqr", median: s.median ?? null, q1: s.q1 ?? null, q3: s.q3 ?? null };
  return { kind: "count_pct", n: s.n ?? 0, pct: s.pct ?? null };
}

export function table1FromArtifact(t: Table1Artifact, number = "Table 1"): TableData {
  const rows: TableRow[] = [];
  let anyMissing = false;
  const kinds = new Set<string>();
  for (const v of t.variables) {
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

/** The page preview of one exhibit of the exhibit model (C7a). */
export function pageFromExhibits(model: ExhibitModel, key: string): PageData | null {
  const ex = model.exhibits.find((e) => e.key === key);
  if (!ex) return null;
  const strip = (e: (typeof model.exhibits)[number]) => ({
    key: e.key,
    number: e.number,
    kind: e.kind,
    caption: e.caption,
    placement: e.placement,
    fixed: e.fixed_reason ?? undefined,
  });
  return { exhibit: strip(ex), others: model.exhibits.filter((e) => e.key !== key).map(strip), text: model.text };
}
