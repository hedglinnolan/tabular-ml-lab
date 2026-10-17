/**
 * Predict Results' exhibits, from the fit, explain and evaluation artifacts (plans/W1_PART2_PLAN.md
 * §2 W1-G). Pure: nothing is estimated or rounded here. Every number a table prints is a served
 * field, carried as it was served and formatted by the table view (`components/views/format.ts`);
 * every sentence is the engine's, or composed only from served fields.
 */
import type { FitArtifact, FittedModel, SplitArtifact } from "../../api/m1-stage-types";
import type { EvaluationArtifact, FamilyExplanation, VersusBaseline } from "../../api/m3-types";
import type { DecisionRecord, ProjectView } from "../../api/schema";
import { plain } from "../../components/stage/format";
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
    title: `cross-validated on ${fit.n_train.toLocaleString("en-US")} training rows: each score with its interval`,
    stub: "Model",
    columns: [...metrics.map((x) => ({ key: x.key, label: x.label })), ...(rows.some((r) => r.kind === "row" && r.cells.versus) ? [{ key: "versus", label: against }] : [])],
    rows,
    footnotes,
    empty: "No model was fitted.",
  };
}

const mark = (i: number) => String.fromCharCode(97 + i);
const cap = (s: string) => (s ? s[0]!.toUpperCase() + s.slice(1) : s);

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
function heldOutTitle(fit: FitArtifact, phase: SealPhase, exploratory: boolean): string {
  const rows = `${fit.n_holdout.toLocaleString("en-US")} rows`;
  const what =
    phase === "post_seal"
      ? `from a fit changed after the opening, on ${rows}, so not an independent test`
      : fit.at_opening?.current === false
        ? `on ${rows} drawn again after the first opening, so not an independent test`
        : `scored once, on ${rows} no choice has seen`;
  return `${what}: each score with its interval${exploratory ? ", exploratory" : ""}`;
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
    title: heldOutTitle(fit, phase, !!exploratory),
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

// ── what the interpretable model costs ───────────────────────────────────────

/** Evaluation's sentence on what the interpretable model costs or gains (`interpretable.text`),
 *  or null where it serves none. */
export function interpretableCost(ev: EvaluationArtifact | null): string | null {
  const text = ev?.interpretable?.["text"];
  return typeof text === "string" && text ? text : null;
}
