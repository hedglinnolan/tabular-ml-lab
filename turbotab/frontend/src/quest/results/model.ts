/**
 * Predict Results' exhibits, from the fit, explain and evaluation artifacts (plans/W1_PART2_PLAN.md
 * §2 W1-G). Pure: nothing is estimated or rounded here. Every number a table prints is a served
 * field, carried as it was served and formatted by the table view (`components/views/format.ts`);
 * every sentence is the engine's, or composed only from served fields.
 */
import type { FitArtifact, FittedModel, SplitArtifact } from "../../api/m1-stage-types";
import type { Benchmark, EvaluationArtifact, FamilyExplanation, ScoreInterval, VersusBaseline } from "../../api/m3-types";
import type { DecisionRecord, ProjectView } from "../../api/schema";
import { fmtNum, plain } from "../../components/stage/format";
import { fmtEstimate } from "../../components/views/format";
import type { Cell, Footnote, TableData, TableRow } from "../../components/views/types";
import { glyphOf, isExploratory, scoresVisible, type SealPhase } from "../../components/stage/seal/phase";

export interface Metric {
  key: string;
  label: string;
}

/** The metrics in the order the fit names them (`metric_labels`), each one some family is scored
 *  on in `scored`. */
export function metricsOf(fit: FitArtifact, scored: (m: FittedModel) => Record<string, unknown> | null | undefined): Metric[] {
  return Object.keys(fit.metric_labels)
    .filter((k) => fit.models.some((m) => (scored(m) ?? {})[k] !== undefined))
    .map((key) => ({ key, label: fit.metric_labels[key] ?? key }));
}

const estimate = (est: number | null | undefined, lo?: number | null, hi?: number | null): Cell => ({
  kind: "estimate",
  est: est ?? null,
  lo: lo ?? null,
  hi: hi ?? null,
});

/** The verdict against the baseline, in the words its rule defines (`tolerance_basis`). */
export const VERDICT: Record<VersusBaseline["verdict"], string> = {
  better: "Better",
  no_better: "Not shown to be better",
  worse: "Worse",
};

/** "Better: a gain in R² of 0.133 (0.119 to 0.147)", in the metric the verdict reads (`metric`,
 *  named by the fit's `metric_labels`); the gain's interval only where one is served. */
export function verdictCell(v: VersusBaseline | null | undefined, labels: Record<string, string>): Cell | null {
  if (!v) return null;
  const gain = v.gain === null ? "" : `: a gain in ${labels[v.metric] ?? v.metric} of ${fmtGain(v)}`;
  return { kind: "text", text: `${VERDICT[v.verdict]}${gain}` };
}

/** The gain as the table prints an estimate with its interval (views/format.ts). */
const fmtGain = (v: VersusBaseline): string => fmtEstimate(v.gain, v.ci_low, v.ci_high);

const famLabel = (fit: FitArtifact, family: string | null | undefined) =>
  fit.models.find((m) => m.family === family)?.label ?? family ?? "";

/** The families in the fit's order, the final model first once one is declared. */
export function familiesOf(fit: FitArtifact): FittedModel[] {
  const final = fit.final_model;
  return final ? [...fit.models.filter((m) => m.family === final), ...fit.models.filter((m) => m.family !== final)] : fit.models;
}

const ROLE: Record<"final" | "secondary", string> = { final: "the final model", secondary: "secondary" };

// ── the cross-validated comparison ───────────────────────────────────────────

/**
 * Every family's cross-validated score on every metric with its interval (`cv[metric].estimate`,
 * `ci_low`, `ci_high`); the no-predictor baseline on the metric it is scored on; each family's
 * verdict against it; and, with several families, the best family's scores corrected for having
 * been chosen (BBC-CV, `selection`: the primary's `corrected`, the others' `extras`).
 */
export function cvComparison(fit: FitArtifact): TableData {
  const metrics = metricsOf(fit, (m) => m.cv);
  const base = fit.models.find((m) => m.baseline.value !== null)?.baseline ?? null;
  const against = base ? `Against ${base.label}` : "Against the baseline";
  const rows: TableRow[] = familiesOf(fit).map((m, i) => ({
    kind: "row",
    key: m.family,
    label: m.label,
    sub: m.role ? ROLE[m.role] : undefined,
    primary: m.role === "final",
    marks: m.versus_baseline ? [mark(i)] : undefined,
    cells: {
      ...Object.fromEntries(metrics.map((x) => [x.key, m.cv[x.key] ? estimate(m.cv[x.key]!.estimate ?? m.cv[x.key]!.mean, m.cv[x.key]!.ci_low, m.cv[x.key]!.ci_high) : null])),
      versus: verdictCell(m.versus_baseline, fit.metric_labels),
    },
  }));
  if (base) {
    rows.push({
      kind: "row",
      key: "baseline",
      label: `${cap(base.label)}, with no predictors`,
      cells: { [base.metric]: estimate(base.value) },
    });
  }
  const sel = fit.selection;
  if (sel) {
    rows.push({
      kind: "row",
      key: "corrected",
      label: `${famLabel(fit, sel.best)}, corrected for choosing it`,
      sub: "bootstrap bias-corrected cross-validation (BBC-CV)",
      cells: Object.fromEntries(
        metrics.map((x) => {
          if (x.key === sel.metric) return [x.key, estimate(sel.corrected, sel.corrected_low, sel.corrected_high)];
          const e = sel.extras[x.key];
          return [x.key, e && e.corrected !== undefined ? estimate(e.corrected, e.corrected_low, e.corrected_high) : null];
        }),
      ),
    });
  }
  const footnotes: Footnote[] = [
    ...familiesOf(fit).flatMap((m, i) => (m.versus_baseline ? [{ mark: mark(i), text: `${m.label}: ${plain(m.versus_baseline.tolerance_basis)}` }] : [])),
    ...(sel ? [{ text: plain(sel.text) }] : []),
    ...(fit.cv_definition ? [{ text: plain(fit.cv_definition) }] : []),
    ...(fit.se_definition ? [{ text: plain(fit.se_definition) }] : []),
  ];
  return {
    number: "Models compared",
    title: `Cross-validated on ${fit.n_train.toLocaleString("en-US")} training rows${intervalClause(rows, metrics, base ? { baseline: base.label } : {})}`,
    stub: "Model",
    columns: [...metrics.map((x) => ({ key: x.key, label: x.label })), ...(rows.some((r) => r.kind === "row" && r.cells.versus) ? [{ key: "versus", label: against }] : [])],
    rows,
    footnotes,
    empty: "No model was fitted.",
  };
}

const mark = (i: number) => String.fromCharCode(97 + i);
const cap = (s: string) => (s ? s[0]!.toUpperCase() + s.slice(1) : s);

/** "a", "a and b", "a, b and c". */
const listing = (xs: string[]) => (xs.length < 2 ? (xs[0] ?? "") : `${xs.slice(0, -1).join(", ")} and ${xs[xs.length - 1]}`);

const uniq = <T,>(xs: T[]) => xs.filter((x, i) => xs.indexOf(x) === i);

/** Footnotes by their text: each text marked once, the same text on several rows sharing its mark. */
function marker() {
  const texts: string[] = [];
  return {
    mark: (text: string) => {
      if (!texts.includes(text)) texts.push(text);
      return mark(texts.indexOf(text));
    },
    footnotes: (): Footnote[] => texts.map((text, i) => ({ mark: mark(i), text })),
  };
}

/**
 * What a table's caption may say of intervals, true of every score it prints (each estimate cell
 * with a served value): "each score with its interval" only where every one carries its interval.
 * Where some do not, the caption names them when they are whole rows (the no-predictor baseline's
 * scores, served with none) or whole columns (a metric served with none, as macro-F1's: core/models/
 * artifacts.py), "except …"; where they are neither, it claims nothing of intervals, and where none
 * has one, nothing either. `names` gives a row's name in running text, by its key (else its label).
 */
export function intervalClause(rows: readonly TableRow[], metrics: readonly Metric[], names: Record<string, string> = {}): string {
  const cells: { row: string; col: string; has: boolean }[] = [];
  const labels = new Map<string, string>();
  for (const r of rows) {
    if (r.kind !== "row") continue;
    labels.set(r.key, names[r.key] ?? r.label);
    for (const m of metrics) {
      const c = r.cells[m.key];
      if (c && c.kind === "estimate" && c.est !== null) cells.push({ row: r.key, col: m.key, has: c.lo !== null && c.hi !== null });
    }
  }
  if (!cells.some((c) => c.has)) return "";
  if (cells.every((c) => c.has)) return ": each score with its interval";
  const rowsWithout = uniq(cells.map((c) => c.row)).filter((row) => cells.every((c) => c.row !== row || !c.has));
  const rest = cells.filter((c) => !rowsWithout.includes(c.row));
  const colsWithout = uniq(rest.map((c) => c.col)).filter((col) => rest.every((c) => c.col !== col || !c.has));
  if (rest.some((c) => !c.has && !colsWithout.includes(c.col))) return "";
  const named = [
    ...rowsWithout.map((row) => labels.get(row) ?? row),
    ...colsWithout.map((col) => metrics.find((m) => m.key === col)?.label ?? col),
  ];
  return `: each score with its interval, except ${listing(named)}`;
}

// ── the held-out rows, once opened ───────────────────────────────────────────

/** The split as the held-out exhibits read it: its basis, its chronology and its own flag. */
export type SplitBasis = Pick<SplitArtifact, "basis" | "chronology" | "exploratory">;

/**
 * Why the held-out scores are exploratory, in the split's own words, or null where its basis is a
 * verified clean one (`seal/phase.ts` `isExploratory`: an abandoned or undetermined basis, one
 * the split marks exploratory, or one not read). The words are the basis's sentence where the
 * basis is the reason, and the chronology's where time was asked for and no column could order
 * the draw (turbotab/core/seal.py `SealDraw.exploratory`).
 */
export function exploratoryReasons(split: SplitBasis | null | undefined): string[] | null {
  if (!isExploratory(split?.basis?.state, split?.exploratory)) return null;
  if (!split) return ["The split that drew these rows is not read yet, so its basis is not shown as a verified clean one."];
  const said: string[] = [];
  const b = split.basis;
  if (b && (b.exploratory || glyphOf(b.state) !== "closed")) said.push(b.sentence);
  const c = split.chronology;
  if (c && !c.drawn && c.time_column === null) said.push(c.sentence);
  return said.length ? said : ["The split that drew these rows marks its held-out scores exploratory."];
}

/**
 * The engine's notes on the held-out result, each once: the final model's (`final_note`) and the
 * opening's (`at_opening.note`). Once the served scores are not the opening's, the server serves
 * the opening's note as the final model's too (server/service.py `_served_fit`), so it is said once.
 */
export function openingNotes(fit: Pick<FitArtifact, "final_note" | "at_opening">): string[] {
  const notes = [fit.final_note, fit.at_opening?.note].filter((t): t is string => !!t);
  return notes.filter((t, i) => notes.indexOf(t) === i);
}

/**
 * The record that opened the current seal, read by the engine's rule (turbotab/core/seal.py
 * `opening`): among the records no revert cancels, the latest opening or re-seal of the current
 * outcome's seal (or one naming no outcome) — an opening, or null where a re-seal closed it again.
 * The fit's final model is this opening's (selection.py `declared_family`); after a re-seal and a
 * second opening it is not the first one (`OpenSeal.openingRecord`).
 */
export function currentOpening(view: Pick<ProjectView, "decisions" | "state">): DecisionRecord | null {
  const ordered = [...view.decisions].sort((a, b) => a.seq - b.seq);
  // Latest first: a revert counts only while nothing later reverts it (core/decisions.py `reverted`).
  const cancelled = new Set<string>();
  for (const r of [...ordered].reverse()) {
    if (!cancelled.has(r.id) && r.decision.kind === "revert") cancelled.add(r.decision.decision_id);
  }
  const target = view.state.target ?? null;
  for (const r of ordered.reverse()) {
    const d = r.decision;
    if (cancelled.has(r.id) || (d.kind !== "open_seal" && d.kind !== "reseal")) continue;
    if (d.target !== null && d.target !== undefined && d.target !== target) continue;
    return d.kind === "open_seal" ? r : null;
  }
  return null;
}

/**
 * What the held-out scores are, said in the exhibit's title. "Scored once, by no choice seen" only
 * while the served scores are the opening's: the phase is `opened` and the fit does not say the
 * opening's result is no longer current (`at_opening.current`, turbotab/core/seal.py
 * `reported_result`). After a change the fit is post-seal; after the rows are drawn again and
 * opened in turn, the scores are those rows'. The engine's note, a footnote, says which result
 * stays reported.
 */
function heldOutTitle(fit: FitArtifact, phase: SealPhase, exploratory: boolean, intervals: string): string {
  const rows = `${fit.n_holdout.toLocaleString("en-US")} rows`;
  const what =
    phase === "post_seal"
      ? `from a fit changed after the opening, on ${rows}, so not an independent test`
      : fit.at_opening?.current === false
        ? `on ${rows} drawn again after the first opening, so not an independent test`
        : `scored once, on ${rows} no choice has seen`;
  return `${cap(what)}${exploratory ? ", exploratory" : ""}${intervals}`;
}

/**
 * The held-out scores once the rows are opened (`holdout` with `holdout_detail.intervals`), beside
 * the no-predictor baseline scored on the same rows (`holdout_detail.baseline`) and the held-out
 * verdict against it (`holdout_detail.versus_baseline`). Null while the phase hides held-out
 * scores (`seal/phase.ts`): no held-out number is read before the opening, whatever the artifact
 * holds. An exploratory split labels the exhibit, with its reason as the first footnote.
 */
export function heldOutTable(fit: FitArtifact, phase: SealPhase, split: SplitBasis | null): TableData | null {
  if (!scoresVisible(phase)) return null;
  const exploratory = exploratoryReasons(split);
  const metrics = metricsOf(fit, (m) => m.holdout);
  const families = familiesOf(fit).filter((m) => m.holdout);
  const detail = families.find((m) => m.holdout_detail?.baseline)?.holdout_detail ?? null;
  const base = detail?.baseline ?? null;
  const rows: TableRow[] = families.map((m, i) => ({
    kind: "row",
    key: m.family,
    label: m.label,
    sub: m.role ? ROLE[m.role] : undefined,
    primary: m.role === "final",
    marks: m.holdout_detail?.versus_baseline ? [mark(i)] : undefined,
    cells: {
      ...Object.fromEntries(
        metrics.map((x) => {
          const iv = m.holdout_detail?.intervals[x.key];
          const v = m.holdout?.[x.key];
          if (iv) return [x.key, estimate(iv.estimate, iv.ci_low, iv.ci_high)];
          return [x.key, v === undefined ? null : estimate(v)];
        }),
      ),
      versus: verdictCell(m.holdout_detail?.versus_baseline, fit.metric_labels),
    },
  }));
  if (base) {
    rows.push({
      kind: "row",
      key: "baseline",
      label: `${cap(base.label)}, with no predictors`,
      cells: Object.fromEntries(metrics.map((x) => [x.key, base.scores[x.key] === undefined ? null : estimate(base.scores[x.key])])),
    });
  }
  const footnotes: Footnote[] = [
    ...(exploratory ? [{ text: plain(`Exploratory. ${exploratory.join(" ")}`) }] : []),
    ...families.flatMap((m, i) => {
      const v = m.holdout_detail?.versus_baseline;
      return v ? [{ mark: mark(i), text: `${m.label}: ${plain(v.tolerance_basis)}` }] : [];
    }),
    ...openingNotes(fit).map((text) => ({ text: plain(text) })),
  ];
  return {
    number: "Held-out rows",
    title: heldOutTitle(fit, phase, !!exploratory, intervalClause(rows, metrics, base ? { baseline: base.label } : {})),
    stub: "Model",
    columns: [
      ...metrics.map((x) => ({ key: x.key, label: x.label })),
      ...(rows.some((r) => r.kind === "row" && r.cells.versus) ? [{ key: "versus", label: base ? `Against ${base.label}` : "Against the baseline" }] : []),
    ],
    rows,
    footnotes,
    empty: "No held-out score was served.",
  };
}

// ── what drives the model ────────────────────────────────────────────────────

/**
 * One explained family's SHAP importance (`importance`: each input's mean |SHAP| on the family's
 * scale, its rank, and how many reseeded refits keep it among the top `stability.top`).
 */
export function importanceTable(f: FamilyExplanation): TableData {
  const reseeds = f.stability?.reseeds ?? null;
  const top = f.stability?.top ?? null;
  const kept = reseeds !== null && top !== null && f.importance.some((r) => r.in_top !== null && r.in_top !== undefined);
  return {
    number: f.label,
    title: `what its predictions lean on: each input's mean absolute SHAP value${f.scale ? `, on ${plain(f.scale)}` : ""}`,
    stub: "Input",
    columns: [
      { key: "rank", label: "Rank" },
      { key: "mean_abs", label: "Mean |SHAP|" },
      ...(kept ? [{ key: "kept", label: `Among the top ${top} in refits`, sub: `of ${reseeds}` }] : []),
    ],
    rows: [...f.importance]
      .sort((a, b) => a.rank - b.rank)
      .map((r) => ({
        kind: "row" as const,
        key: r.input,
        label: r.input,
        cells: {
          rank: { kind: "count", n: r.rank },
          mean_abs: estimate(r.mean_abs),
          ...(kept ? { kept: r.in_top === null || r.in_top === undefined ? null : ({ kind: "count", n: r.in_top } as Cell) } : {}),
        },
      })),
    footnotes: f.method ? [{ text: plain(f.method) }] : [],
    empty: f.reason ? plain(f.reason) : `${f.label} was not explained.`,
  };
}

// ── the held-out calibration's measures ──────────────────────────────────────

/** The gaps between each prediction and the smoothed observed outcome (turbotab/core/models/
 *  performance.py `calibration`: `|p − smooth(p)|`, summarized as `val.prob` does), in plain words
 *  with the technical name as the quiet line. */
const GAPS = [
  { key: "eavg", label: "Average gap", sub: "Eavg", said: "the average gap (Eavg, the integrated calibration index)" },
  { key: "e90", label: "90th percentile gap", sub: "E90", said: "the 90th percentile gap (E90)" },
  { key: "emax", label: "Largest gap", sub: "Emax", said: "the largest gap (Emax)" },
] as const;

/**
 * Every family's held-out calibration once the rows are opened (`holdout_detail.calibration`): the
 * calibration intercept and slope with their intervals, and the gaps (`eavg`, `e90`, `emax`), each
 * gap's column only where some family's is served (an artifact from before them serves none). Null
 * while the phase hides held-out scores, as the held-out table is. A flagged family's concern, in
 * the engine's words, is its footnote.
 */
export function calibrationMeasures(fit: FitArtifact, phase: SealPhase, split: SplitBasis | null): TableData | null {
  if (!scoresVisible(phase)) return null;
  const families = familiesOf(fit).filter((m) => m.holdout_detail?.calibration);
  if (!families.length) return null;
  const cal = (m: FittedModel) => m.holdout_detail!.calibration!;
  const gaps = GAPS.filter((g) => families.some((m) => typeof cal(m)[g.key] === "number"));
  const notes = marker();
  const rows: TableRow[] = families.map((m) => {
    const c = cal(m);
    return {
      kind: "row",
      key: m.family,
      label: m.label,
      sub: m.role ? ROLE[m.role] : undefined,
      primary: m.role === "final",
      marks: c.concern ? [notes.mark(`${m.label}: ${plain(c.concern)}`)] : undefined,
      cells: {
        intercept: estimate(c.intercept.estimate, c.intercept.ci_low, c.intercept.ci_high),
        slope: estimate(c.slope.estimate, c.slope.ci_low, c.slope.ci_high),
        ...Object.fromEntries(gaps.map((g) => [g.key, typeof c[g.key] === "number" ? estimate(c[g.key]) : null])),
      },
    };
  });
  const smoothers = uniq(families.map((m) => cal(m).smoother).filter((x) => !!x));
  const by = smoothers.length ? `, by ${listing(smoothers)}` : "";
  return {
    number: "Calibration measures",
    title: `On the held-out rows${exploratoryReasons(split) ? ", exploratory" : ""}`,
    stub: "Model",
    columns: [
      { key: "intercept", label: "Calibration intercept" },
      { key: "slope", label: "Calibration slope" },
      ...gaps.map((g) => ({ key: g.key, label: g.label, sub: g.sub })),
    ],
    rows,
    footnotes: [
      ...notes.footnotes(),
      ...(gaps.length
        ? [
            {
              text: `A row's gap is the distance between its prediction and the observed outcome smoothed at that prediction${by}. The columns give ${listing(gaps.map((g) => g.said))} over the rows.`,
            },
          ]
        : []),
    ],
    empty: "No held-out calibration was served.",
  };
}

// ── the evaluation stage's exhibits (G09) ────────────────────────────────────
// Each is read from the evaluation artifact as the stage serves it (turbotab/core/stages/
// evaluation.py `_prediction`), out of fold on the training rows: the benchmark, the decision curve,
// the subgroups, shrinkage as model updating and internal–external validation. The artifact types
// all but the benchmark as plain dicts; the shapes below are the engine's.

/** Each fitted family's name as the fit serves it, and the benchmark's as the evaluation does. */
export function familyLabels(fit: FitArtifact, ev: EvaluationArtifact): Record<string, string> {
  return {
    ...Object.fromEntries(fit.models.map((m) => [m.family, m.label])),
    ...(ev.benchmark ? { [ev.benchmark.key]: ev.benchmark.label } : {}),
  };
}

/**
 * The regression-with-splines benchmark (`benchmark`): its cross-validated score on the primary
 * metric with its interval, and its verdict against the no-predictor baseline (named as the fit
 * names it) in the words its rule defines, the rule its footnote; then the engine's sentence, and
 * its paired difference from each family in the engine's words (`differences[].sentence`), whose
 * sign each sentence says.
 */
export function benchmarkTable(b: Benchmark, fit: FitArtifact): TableData {
  const v = (b.versus_baseline ?? null) as VersusBaseline | null;
  const metric: Metric = { key: b.metric, label: fit.metric_labels[b.metric] ?? b.metric };
  const base = fit.models.find((m) => m.baseline.value !== null)?.baseline ?? null;
  const rows: TableRow[] = [
    {
      kind: "row",
      key: b.key,
      label: b.label,
      marks: v ? [mark(0)] : undefined,
      cells: { [metric.key]: estimate(b.estimate, b.ci_low, b.ci_high), versus: verdictCell(v, fit.metric_labels) },
    },
  ];
  return {
    number: "Benchmark",
    title: `Cross-validated${intervalClause(rows, [metric])}`,
    stub: "Model",
    columns: [{ key: metric.key, label: metric.label }, ...(v ? [{ key: "versus", label: base ? `Against ${base.label}` : "Against the baseline" }] : [])],
    rows,
    footnotes: [
      ...(v ? [{ mark: mark(0), text: `${b.label}: ${plain(v.tolerance_basis)}` }] : []),
      ...(b.sentence ? [{ text: plain(b.sentence) }] : []),
      ...b.differences.flatMap((d) => (typeof d["sentence"] === "string" && d["sentence"] ? [{ text: plain(d["sentence"]) }] : [])),
    ],
    empty: "No benchmark was fitted.",
  };
}

/** One named column's groups, scored with the reported family's out-of-fold predictions
 *  (turbotab/core/models/decision_curve.py `subgroup_performance`); `note` and `ask`, with no
 *  groups, while the column's code-or-amount reading is unsettled. */
interface ServedSubgroups {
  column: string;
  grouped_by: string | null;
  groups: { group: string; n: number; events?: number | null; scores: Record<string, ScoreInterval | undefined>; note: string | null }[];
  note?: string | null;
}

const reportedName = (fit: FitArtifact, ev: EvaluationArtifact) =>
  ev.reported_family ? (familyLabels(fit, ev)[ev.reported_family] ?? ev.reported_family) : "The reported model";

/** A row's key in its own space, the served values it is made of JSON-encoded, so no column's name
 *  and no level (one named "pooled", one holding a colon) can make two rows' keys one. */
const rowKey = (space: string, ...parts: string[]) => `${space}:${JSON.stringify(parts)}`;

/**
 * Subgroup performance (`subgroups`): each named column a group of rows, its groups in the order
 * served, each with its rows, its events under a yes/no outcome, and every metric scored with its
 * interval, the metrics in the order the fit names them. A group the engine names but does not
 * score, an interval it refuses and a column whose groups wait on a reading are each marked, the
 * engine's words their footnotes. Null where none is named. The caption names a group scored with
 * no interval by its column and its level ("sex 1"), since two coded columns can share a level,
 * each by its own row's key, which no column's name or level can make another row's.
 */
export function subgroupTable(ev: EvaluationArtifact, fit: FitArtifact): TableData | null {
  const subs = ev.subgroups as unknown as ServedSubgroups[];
  if (!subs.length) return null;
  const served = uniq(subs.flatMap((s) => s.groups.flatMap((g) => Object.keys(g.scores))));
  const metrics: Metric[] = [...Object.keys(fit.metric_labels).filter((k) => served.includes(k)), ...served.filter((k) => !(k in fit.metric_labels))].map((key) => ({
    key,
    label: fit.metric_labels[key] ?? key,
  }));
  const events = subs.some((s) => s.groups.some((g) => typeof g.events === "number"));
  const groupKey = (column: string, group: string) => rowKey("group", column, group);
  const notes = marker();
  const rows: TableRow[] = subs.flatMap((s): TableRow[] => [
    { kind: "group", key: rowKey("column", s.column), label: s.column, marks: s.note ? [notes.mark(plain(s.note))] : undefined },
    ...s.groups.map((g): TableRow => {
      const said = uniq([g.note, ...metrics.map((m) => g.scores[m.key]?.refused)].filter((t): t is string => !!t));
      return {
        kind: "row",
        key: groupKey(s.column, g.group),
        label: g.group,
        indent: true,
        marks: said.length ? said.map((t) => notes.mark(plain(t))) : undefined,
        cells: {
          n: { kind: "count", n: g.n },
          ...(events && typeof g.events === "number" ? { events: { kind: "count", n: g.events } as Cell } : {}),
          ...Object.fromEntries(
            metrics.map((m) => {
              const sc = g.scores[m.key];
              return [m.key, sc ? estimate(sc.estimate, sc.ci_low, sc.ci_high) : null];
            }),
          ),
        },
      };
    }),
  ]);
  const names = Object.fromEntries(subs.flatMap((s) => s.groups.map((g) => [groupKey(s.column, g.group), `${s.column} ${g.group}`])));
  return {
    number: "Subgroups",
    title: `${reportedName(fit, ev)}'s out-of-fold scores in each group${intervalClause(rows, metrics, names)}`,
    stub: "Subgroup",
    columns: [{ key: "n", label: "Rows" }, ...(events ? [{ key: "events", label: "Events" }] : []), ...metrics.map((m) => ({ key: m.key, label: m.label }))],
    rows,
    footnotes: notes.footnotes(),
    // With no group scored (every named column waiting), the table says why in one line.
    empty: subs.map((s) => (s.note ? plain(s.note) : `No group of ${s.column} was scored.`)).join(" "),
  };
}

type InternalExternal = NonNullable<FittedModel["internal_external"]>;
/** One model's internal–external validation as the evaluation serves it (evaluation.py `_iecv`):
 *  the engine's `InternalExternal` beside the model's key and name. */
type ServedInternalExternal = InternalExternal & { family: string; label: string };

/**
 * Internal–external validation by cluster (`internal_external`): for each model (the reported
 * family and the benchmark), each level of the grouping held out in turn and scored on the primary
 * metric with its interval by the model fitted on the rest, its rows and, under a yes/no outcome,
 * its events; then, where the engine pooled two levels or more (`pooled.k`), the random-effects
 * summary over them with its interval. The engine's sentence on the spread is the footnote of the
 * summary, or of the model's row where there is none. Null where no grouping is named. The caption
 * names a row scored with no interval by its model, as each model holds the same levels: a level
 * with its grouping ("Ridge's site 3"), the summary as the model's random-effects summary, each by
 * its own row's key, so a level named "pooled" is named as its level, never as the summary.
 */
export function internalExternalTable(ev: EvaluationArtifact, fit: FitArtifact): TableData | null {
  const runs = ev.internal_external as unknown as ServedInternalExternal[];
  if (!runs.length) return null;
  const metrics: Metric[] = uniq(runs.map((r) => r.metric)).map((key) => ({ key, label: fit.metric_labels[key] ?? key }));
  const events = runs.some((r) => r.clusters.some((c) => typeof c.n_events === "number"));
  const levelKey = (run: ServedInternalExternal, cluster: string) => rowKey("level", run.family, cluster);
  const summaryKey = (run: ServedInternalExternal) => rowKey("summary", run.family);
  const notes = marker();
  const rows: TableRow[] = runs.flatMap((run): TableRow[] => {
    // The engine's own rule (validation.py `_spread_sentence`): with one level pooled, `pooled` is
    // that level's own score, and with none it holds no estimate, so neither is a summary.
    const summary = run.pooled.k >= 2 && typeof run.pooled.estimate === "number";
    const spread = () => (run.spread ? [notes.mark(`${run.label}: ${plain(run.spread)}`)] : undefined);
    const head: TableRow = { kind: "group", key: rowKey("model", run.family), label: run.label, marks: summary ? undefined : spread() };
    const levels = run.clusters.map((c): TableRow => {
      const said = uniq([c.note, c.primary.refused].filter((t): t is string => !!t));
      return {
        kind: "row",
        key: levelKey(run, c.cluster),
        label: c.cluster,
        indent: true,
        marks: said.length ? said.map((t) => notes.mark(plain(t))) : undefined,
        cells: {
          n: { kind: "count", n: c.n },
          ...(events && typeof c.n_events === "number" ? { events: { kind: "count", n: c.n_events } as Cell } : {}),
          [run.metric]: estimate(c.primary.estimate, c.primary.ci_low, c.primary.ci_high),
        },
      };
    });
    if (!summary) return [head, ...levels];
    return [
      head,
      ...levels,
      {
        kind: "row",
        key: summaryKey(run),
        label: "Random-effects summary",
        indent: true,
        marks: spread(),
        cells: { [run.metric]: estimate(run.pooled.estimate, run.pooled.ci_low, run.pooled.ci_high) },
      },
    ];
  });
  const clusters = uniq(runs.map((r) => r.cluster));
  const names = Object.fromEntries(
    runs.flatMap((run) => [
      ...run.clusters.map((c) => [levelKey(run, c.cluster), `${run.label}'s ${run.cluster} ${c.cluster}`]),
      [summaryKey(run), `${run.label}'s random-effects summary`],
    ]),
  );
  return {
    number: "Internal–external validation",
    title: `Each level of ${listing(clusters)} held out in turn, scored by each model fitted on the rest${intervalClause(rows, metrics, names)}`,
    stub: "Held-out level",
    columns: [{ key: "n", label: "Rows" }, ...(events ? [{ key: "events", label: "Events" }] : []), ...metrics.map((m) => ({ key: m.key, label: m.label }))],
    rows,
    footnotes: notes.footnotes(),
    empty: "No level was scored.",
  };
}

/** Shrinkage offered as model updating (evaluation.py `_shrinkage`): the offer, and once answered
 *  the shrunk model (`intercept`, `coefficients`) with its sentence, or, where it could not be
 *  applied, the sentence saying so with no coefficients. */
interface ServedShrinkage {
  factor: number;
  sentence: string | null;
  intercept?: number | null;
  coefficients?: { feature: string; estimate: number; shrunk: number }[];
}

/**
 * Shrinkage as model updating, once applied: each of the regression's coefficients before and
 * after multiplying by the factor (`coefficients[].estimate`, `shrunk`), and its intercept as
 * re-estimated (`intercept`; the intercept before is not served, so its cell is empty), the
 * engine's sentence the footnote. `estimate` is the deployed coefficient (decision_curve.py
 * `deployed_coefficients`): under an imbalance correction the recalibration's slope times the
 * class-weighted fit's, not the coefficient the regression was fitted with, so the words say
 * before shrinking, not as fitted. Null while it is only offered, or where it was declared and
 * not applied (`updatingNotApplied` says why).
 */
export function updatingTable(ev: EvaluationArtifact): TableData | null {
  const sh = ev.shrinkage as unknown as ServedShrinkage | null;
  if (!sh || !Array.isArray(sh.coefficients) || typeof sh.intercept !== "number") return null;
  return {
    number: "Model updating",
    title: `The regression's coefficients before and after multiplying by ${fmtNum(sh.factor)}`,
    stub: "Term",
    columns: [
      { key: "fitted", label: "Before shrinking" },
      { key: "shrunk", label: "Shrunk" },
    ],
    rows: [
      { kind: "row", key: "intercept", label: "Intercept", sub: "re-estimated", cells: { shrunk: estimate(sh.intercept) } },
      ...sh.coefficients.map((c): TableRow => ({ kind: "row", key: c.feature, label: c.feature, cells: { fitted: estimate(c.estimate), shrunk: estimate(c.shrunk) } })),
    ],
    footnotes: sh.sentence ? [{ text: plain(sh.sentence) }] : [],
    empty: "No coefficient was shrunk.",
  };
}

/** The engine's sentence where shrinkage was declared and not applied (a sentence, no shrunk
 *  coefficients); null otherwise, the offer alone included. */
export function updatingNotApplied(ev: EvaluationArtifact): string | null {
  const sh = ev.shrinkage as unknown as ServedShrinkage | null;
  if (!sh || Array.isArray(sh.coefficients)) return null;
  return typeof sh.sentence === "string" && sh.sentence ? sh.sentence : null;
}

// ── what the interpretable model costs ───────────────────────────────────────

/** Evaluation's sentence on what the interpretable model costs or gains (`interpretable.text`),
 *  or null where it serves none. */
export function interpretableCost(ev: EvaluationArtifact | null): string | null {
  const text = ev?.interpretable?.["text"];
  return typeof text === "string" && text ? text : null;
}
