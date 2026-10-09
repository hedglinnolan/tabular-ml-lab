/**
 * From the engine's artifacts to the views' inputs. Pure: nothing is estimated here; every number
 * a view draws is one the engine served, and every sentence is composed from served numbers.
 */
import type { ModelCalibration, SubstitutionArtifact } from "../../api/m3-types";
import type { CalibrationBin, CalibrationData } from "./calibration";
import type { CurveData, CurveLine } from "./curve";
import type { DecisionCurveData, DecisionCurveRow } from "./decisionCurve";
import type { Slot } from "./parts";

const pct = (v: number) => `${Math.round(v * 100)}%`;
const SLOTS: Slot[] = [1, 2, 3, 4, 5];

export interface SubstitutionOptions {
  /** what the model predicts, in plain words ("glucose") */
  outcome: string;
  /** one model: its curve as it stands (gray) beside the fixed-population curve (indigo) when
   *  `fixed` is pointed at; omitted: every model compared, in the comparison palette */
  family?: string;
  fixed?: boolean;
}

/**
 * A substitution artifact as a curve. Per model, `delta` averages each k over its own on-support
 * rows; `fixed_delta` averages every k over the same rows (those on support through
 * `support.fixed_through`), which is the choice the indigo line shows.
 */
export function curveFromSubstitution(art: SubstitutionArtifact, opts: SubstitutionOptions): CurveData {
  const unit = art.scale === "percent_energy" ? "% of energy" : "kcal";
  const models = art.models.filter((m) => !m.refused);
  const lineOf = (m: (typeof models)[number], role: CurveLine["role"], slot?: Slot): CurveLine => ({
    key: m.family,
    label: m.label,
    role,
    slot,
    x: art.ks,
    y: m.delta,
    low: m.ci_low,
    high: m.ci_high,
  });
  let lines: CurveLine[];
  if (opts.family) {
    const m = models.find((x) => x.family === opts.family);
    lines = m
      ? [
          { ...lineOf(m, "now"), label: "Each k's own rows" },
          ...(opts.fixed && m.fixed_delta.length
            ? [{ key: `${m.family}:fixed`, label: `The same ${(art.support?.fixed_rows ?? 0).toLocaleString("en-US")} rows at every k`, role: "choice" as const, x: art.ks, y: m.fixed_delta, low: m.fixed_ci_low, high: m.fixed_ci_high }]
            : []),
        ]
      : [];
  } else {
    lines = models.slice(0, 5).map((m, i) => lineOf(m, "series", SLOTS[i]));
  }
  const reported = models.find((m) => m.family === opts.family) ?? models[0];
  const share = reported?.on_support_fraction ?? [];
  const stopAt = reported?.stopped_at ?? null;
  const stopIdx = stopAt === null ? -1 : art.ks.indexOf(stopAt);
  return {
    xLabel: `${unit} moved from ${art.donor} to ${art.recipient}`,
    xName: `k (${unit})`,
    yLabel: `Change in predicted ${opts.outcome}`,
    lines,
    zero: true,
    support: share.length ? { x: art.ks, share } : null,
    stop: stopAt !== null && stopIdx >= 0 ? { x: stopAt, why: `The curve stops at ${stopAt} ${unit}, where ${pct(share[stopIdx] ?? 0)} of rows stay within the range observed.` } : null,
    basis: art.basis,
    band: art.band?.caption ?? null,
  };
}

/** The engine's calibration of one model's predictions, with the binned points when served. */
export function calibrationFromEngine(
  cal: ModelCalibration & { bins?: CalibrationBin[] | null; bins_method?: string | null },
  opts: { kind: "risk" | "value"; outcome: string; where?: string },
): CalibrationData {
  return {
    kind: opts.kind,
    outcome: opts.outcome,
    n: cal.n,
    observed: cal.observed,
    expected: cal.expected,
    intercept: cal.intercept,
    slope: cal.slope,
    curve: cal.curve,
    bins: cal.bins ?? null,
    binsMethod: cal.bins_method ?? null,
    smoother: cal.smoother,
    concern: cal.concern,
    where: opts.where ?? null,
  };
}

/** The evaluation stage's `decision_curve` (turbotab/core/stages/evaluation.py `_decision`). */
export interface EngineDecisionCurve {
  family: string;
  low: number;
  high: number;
  rows: DecisionCurveRow[];
  useful: [number, number] | number[] | null;
  labels?: Record<string, string>;
  prevalence?: number;
  n?: number;
}

export function decisionFromEngine(dc: EngineDecisionCurve, labels: Record<string, string> = dc.labels ?? {}): DecisionCurveData {
  const keys = Object.keys(dc.rows[0]?.models ?? {});
  // The reported model first, then the others in the engine's order: color follows the entity.
  const ordered = [dc.family, ...keys.filter((k) => k !== dc.family)].filter((k) => keys.includes(k)).slice(0, 5);
  return {
    rows: dc.rows,
    models: ordered.map((k, i) => ({ key: k, label: labels[k] ?? k, slot: SLOTS[i]! })),
    low: dc.low,
    high: dc.high,
    useful: dc.useful && dc.useful.length === 2 ? [dc.useful[0]!, dc.useful[1]!] : null,
    prevalence: dc.prevalence ?? null,
    n: dc.n ?? null,
  };
}
