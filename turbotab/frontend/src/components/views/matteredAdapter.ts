/**
 * From the served specification curve to the specification curve view's input (specCurve.ts
 * `SpecCurveData`). The curve is core/mattered.py's `MatteredCurve`, which the `mattered` stage
 * computes after the plan's lock and the server serves in its artifact (GET …/stages/mattered) and
 * at GET …/mattered. The mapping is the engine's own view contract field for field
 * (core/methods/spec_curve.py `SpecCurve.to_view`): every number passes through unchanged, the
 * no-difference line is drawn, and a sealed curve carries no specification and no level. Every
 * label is the engine's with its backticks dropped (the calm budget prints column names as plain
 * text) and never capitalized, since a choice's label may open on a column's name
 * ("`protein` in the adjustment set").
 *
 * The types below are the served fields the adapter reads, as core/mattered.py declares them; the
 * contract's `MatteredCurve` is not aliased in src/api yet.
 */
import type { SpecCurveData } from "./specCurve";

export interface ServedChoice {
  key: string;
  label: string;
  /** the option the plan recorded */
  primary: string;
  options: { key: string; label: string }[];
}

export interface ServedSpec {
  key: string;
  picks: Record<string, string>;
  primary: boolean;
  estimate: number;
  low: number | null;
  high: number | null;
  n: number;
  scale_key: string;
}

export interface ServedNotEstimated {
  key: string;
  picks: Record<string, string>;
  primary: boolean;
  reason: string;
}

export interface ServedCurve {
  estimate_label: string;
  choices: ServedChoice[];
  specs: ServedSpec[];
  not_estimated: ServedNotEstimated[];
  level: number | null;
  sealed: string | null;
}

const plain = (s: string) => s.replaceAll("`", "");

/** The served curve as SpecCurveView draws it. */
export function specCurveFromMattered(curve: ServedCurve): SpecCurveData {
  const sealed = curve.sealed ? plain(curve.sealed) : null;
  return {
    estimateLabel: plain(curve.estimate_label),
    choices: curve.choices.map((c) => ({ key: c.key, label: plain(c.label), options: c.options.map((o) => ({ key: o.key, label: plain(o.label) })) })),
    specs: sealed
      ? []
      : curve.specs.map((s) => ({ key: s.key, estimate: s.estimate, low: s.low, high: s.high, n: s.n, primary: s.primary, picks: s.picks, scaleKey: s.scale_key })),
    zero: true,
    level: sealed ? null : curve.level,
    sealed,
  };
}

/** A specification the curve could not estimate: the choices it changed from the recorded plan
 *  (each choice's label and the option taken, as the view's matrix names them) and the engine's
 *  reason, as served. */
export interface NotEstimatedLine {
  key: string;
  changed: { choice: string; option: string }[];
  reason: string;
}

export function notEstimatedOf(curve: ServedCurve): NotEstimatedLine[] {
  return curve.not_estimated.map((x) => ({
    key: x.key,
    changed: curve.choices
      .filter((c) => x.picks[c.key] !== undefined && x.picks[c.key] !== c.primary)
      .map((c) => {
        const pick = x.picks[c.key]!;
        return { choice: plain(c.label), option: plain(c.options.find((o) => o.key === pick)?.label ?? pick) };
      }),
    reason: x.reason,
  }));
}
