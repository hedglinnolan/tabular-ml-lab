/**
 * What every living-methods prototype shows the same after the lock (SCENARIO.md): Table 2's rows
 * for the exposure and the declared alternatives "Which of my decisions mattered?" compares. Each
 * prototype draws them in its own design from these rows, printed by one rule, and marks them for
 * the cross-prototype check (e2e/methods-protos.spec.ts):
 *
 *   Table 2         an element with data-testid="table2"; each row data-t2-row=<model key>,
 *                   data-t2-estimate and data-t2-ci (t2Attrs), the estimate printed in the row
 *   mattered        an element with data-testid="mattered"; each alternative
 *                   data-mattered-row=<key>, data-estimate and data-ci (matteredAttrs)
 *
 * Pure functions over the server's own stage artifacts (effects, sensitivity); nothing is computed
 * that the engine did not serve.
 */
import { fmtNum } from "../../components/stage/format";

export interface Effect {
  feature: string;
  estimate: number;
  ci_low: number | null;
  ci_high: number | null;
}

export interface EffectsLike {
  exposure: string | null;
  families: {
    family: string;
    label: string;
    sequence: { key: string; label: string; adjusted_for: string[]; n_rows: number; effects: Effect[] }[];
  }[];
}

export interface SensitivityLike {
  analyses: { label: string; primary: boolean; n_rows: number | null; refused?: unknown }[];
  families: { family: string; fits: { label: string; n_rows: number; coefficients: Effect[] }[] }[];
}

export interface T2Row {
  /** crude · model_1 · model_2 · model_3 */
  key: string;
  label: string;
  adjustedFor: string[];
  n: number;
  estimate: number;
  lo: number | null;
  hi: number | null;
  primary: boolean;
}

/** Table 2: the exposure's estimate in each model of the declared sequence, in its order. */
export function table2Rows(effects: EffectsLike): T2Row[] {
  const fam = effects.families[0];
  if (!fam) return [];
  const exposure = effects.exposure;
  return fam.sequence.flatMap((s) => {
    const e = s.effects.find((x) => x.feature === exposure) ?? s.effects[0];
    if (!e) return [];
    return [
      {
        key: s.key,
        label: s.label,
        adjustedFor: s.adjusted_for,
        n: s.n_rows,
        estimate: e.estimate,
        lo: e.ci_low,
        hi: e.ci_high,
        primary: s.key === "model_2",
      },
    ];
  });
}

export interface MatteredRow {
  key: string;
  label: string;
  /** What the alternative varies from the primary: the adjustment (the model sequence) or the
   *  rows (a declared screen); the primary itself varies nothing. */
  varies: "primary" | "adjustment" | "rows";
  n: number;
  estimate: number;
  lo: number | null;
  hi: number | null;
}

/**
 * The declared alternatives (BLUEPRINT §11.4): the model sequence, and the primary model on each
 * screen's rows. Sensitivity, never a way to choose. In the sequence's order, then the screens.
 */
export function matteredRows(effects: EffectsLike, sensitivity: SensitivityLike | null): MatteredRow[] {
  const rows: MatteredRow[] = table2Rows(effects).map((r) => ({
    key: r.key,
    label: r.label,
    varies: r.primary ? "primary" : "adjustment",
    n: r.n,
    estimate: r.estimate,
    lo: r.lo,
    hi: r.hi,
  }));
  const fits = sensitivity?.families[0]?.fits ?? [];
  const exposure = effects.exposure;
  for (const a of sensitivity?.analyses ?? []) {
    if (a.primary || a.refused) continue;
    const fit = fits.find((f) => f.label === a.label);
    const e = fit?.coefficients.find((c) => c.feature === exposure);
    if (!fit || !e) continue;
    rows.push({
      key: `screen:${a.label}`,
      label: a.label,
      varies: "rows",
      n: fit.n_rows,
      estimate: e.estimate,
      lo: e.ci_low,
      hi: e.ci_high,
    });
  }
  return rows;
}

/** An estimate as every prototype prints it: three significant digits, a real minus sign. */
export const fmtEst = (v: number | null | undefined): string => fmtNum(v);

/** A 95% interval: "−0.0327 to −0.00718". */
export const fmtCI = (lo: number | null | undefined, hi: number | null | undefined): string =>
  lo === null || lo === undefined || hi === null || hi === undefined ? "—" : `${fmtNum(lo)} to ${fmtNum(hi)}`;

export function t2Attrs(r: T2Row): Record<string, string> {
  return { "data-t2-row": r.key, "data-t2-estimate": fmtEst(r.estimate), "data-t2-ci": fmtCI(r.lo, r.hi) };
}

export function matteredAttrs(r: MatteredRow): Record<string, string> {
  return { "data-mattered-row": r.key, "data-estimate": fmtEst(r.estimate), "data-ci": fmtCI(r.lo, r.hi) };
}
