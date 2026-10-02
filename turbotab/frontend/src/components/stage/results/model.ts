/**
 * What the Results draw, computed from the fit, shelf, design, split and substitution artifacts
 * (M1_CONTRACT §13). Pure, so every saved figure and every screen agree on it.
 */
import type {
  Coefficient,
  FitArtifact,
  FittedModel,
  ShelfArtifact,
  SplitArtifact,
  SubstitutionArtifact,
  SubstitutionModel,
} from "../../../api/m1-stage-types";
import { fmtInt, fmtNum } from "../format";
import { shownHoldout, type SealPhase } from "../seal/phase";

/** Fitted models in the shelf's order (judgment is order, never absence). */
export function inShelfOrder(fit: FitArtifact, shelf: ShelfArtifact | null): FittedModel[] {
  if (!shelf) return fit.models;
  const rank = new Map(shelf.families.map((f) => [f.key, f.rank]));
  return [...fit.models].sort((a, b) => (rank.get(a.family) ?? 99) - (rank.get(b.family) ?? 99));
}

export interface ComparisonRow {
  family: string;
  label: string;
  mean: number | null;
  sd: number | null;
  holdout: number | null;
  concerns: string[];
}

export interface Comparison {
  metric: string;
  label: string;
  rows: ComparisonRow[];
  baseline: { value: number; label: string } | null;
  domain: [number, number];
  /** Larger is better (R², AUC, accuracy) or smaller (RMSE, Brier, log loss). */
  higherIsBetter: boolean;
}

const LOWER_BETTER = new Set(["rmse", "mae", "brier", "log_loss", "logloss"]);

/**
 * The comparison the Results draw. Held-out scores are read only in a phase that may show them
 * (`seal/phase.ts`): before the seal is opened every row's held-out score is null, whatever the
 * artifact holds, so no picture, number, tooltip or saved figure can imply one.
 */
export function comparisonOf(
  fit: FitArtifact,
  shelf: ShelfArtifact | null,
  metric = fit.primary_metric,
  phase: SealPhase = "sealed",
): Comparison {
  const rows = inShelfOrder(fit, shelf).map((m) => ({
    family: m.family,
    label: m.label,
    // The reported CV score: pooled over out-of-fold predictions for R², RMSE and MAE (the
    // engine's `estimate`); the fold mean in artifacts from before it existed.
    mean: m.cv[metric]?.estimate ?? m.cv[metric]?.mean ?? null,
    sd: m.cv[metric]?.sd ?? null,
    holdout: shownHoldout(phase, m.holdout?.[metric]),
    concerns: m.concerns,
  }));
  const base =
    fit.models.find((m) => m.baseline.metric === metric && m.baseline.value !== null)?.baseline ?? null;
  const values: number[] = [];
  for (const r of rows) {
    if (r.mean !== null) values.push(r.mean - (r.sd ?? 0), r.mean + (r.sd ?? 0));
    if (r.holdout !== null) values.push(r.holdout);
  }
  if (base?.value != null) values.push(base.value);
  if (metric === "r2") values.push(0);
  let lo = Math.min(...values);
  let hi = Math.max(...values);
  if (!Number.isFinite(lo)) [lo, hi] = [0, 1];
  const pad = (hi - lo) * 0.08 || 0.05;
  return {
    metric,
    label: fit.metric_labels[metric] ?? metric,
    rows,
    baseline: base?.value != null ? { value: base.value, label: base.label } : null,
    domain: [lo - pad, hi + pad],
    higherIsBetter: !LOWER_BETTER.has(metric),
  };
}

/** "5-fold cross-validation on 2,294 training rows; 570 held-out rows sealed" (or "scored once"). */
export function metricBasis(fit: FitArtifact, split: SplitArtifact | null, phase: SealPhase = "sealed"): string {
  const folds = split?.folds ?? null;
  const cv = `${folds ? `${folds}-fold ` : ""}cross-validation on ${fmtInt(fit.n_train)} training rows`;
  const grouped = split?.grouped_by ? `, grouped by \`${split.grouped_by}\`` : "";
  const hold = fit.n_holdout
    ? `; ${fmtInt(fit.n_holdout)} held-out rows ${phase === "opened" || phase === "post_seal" ? "scored once" : "sealed"}`
    : "; no rows held out";
  return `Mean ± SD over ${cv}${grouped}${hold}.`;
}

// ── coefficients ─────────────────────────────────────────────────────────────

/** Features that come from an exposure or the energy column (`fat_total_adj` ← `fat_total`). */
export function exposureCoefficients(
  coefs: Coefficient[],
  roles: Record<string, string> | null | undefined,
): Coefficient[] {
  const sources = Object.entries(roles ?? {})
    .filter(([, r]) => r === "exposure" || r === "energy")
    .map(([c]) => c)
    .sort((a, b) => b.length - a.length);
  const pick = coefs.filter((c) => {
    if (c.feature === "(intercept)") return false;
    return sources.some(
      (src) => c.feature === src || c.feature.startsWith(`${src}_`) || c.feature.startsWith(`kcal_from_${src}`),
    );
  });
  return pick.length ? pick : coefs.filter((c) => c.feature !== "(intercept)");
}

export function forestDomain(coefs: Coefficient[]): [number, number] {
  const vals: number[] = [0];
  for (const c of coefs) {
    for (const v of [c.estimate, c.ci_low, c.ci_high]) if (v !== null && Number.isFinite(v)) vals.push(v);
  }
  const lo = Math.min(...vals);
  const hi = Math.max(...vals);
  const pad = (hi - lo) * 0.08 || 0.1;
  return [lo - pad, hi + pad];
}

// ── substitution ─────────────────────────────────────────────────────────────

export const MIN_SUPPORT = 0.5;

export interface CurvePoint {
  k: number;
  delta: number;
}

/** The defined part of a curve: it stops at its support limit. */
export function curvePoints(sub: SubstitutionArtifact, m: SubstitutionModel): CurvePoint[] {
  const out: CurvePoint[] = [];
  sub.ks.forEach((k, i) => {
    const d = m.delta[i];
    if (d !== null && d !== undefined && Number.isFinite(d)) out.push({ k, delta: d });
  });
  return out;
}

export function bandPoints(sub: SubstitutionArtifact, m: SubstitutionModel): { k: number; lo: number; hi: number }[] {
  if (!m.ci_low || !m.ci_high) return [];
  const out: { k: number; lo: number; hi: number }[] = [];
  sub.ks.forEach((k, i) => {
    const lo = m.ci_low![i];
    const hi = m.ci_high![i];
    if (lo !== null && hi !== null && lo !== undefined && hi !== undefined && Number.isFinite(lo) && Number.isFinite(hi))
      out.push({ k, lo, hi });
  });
  return out;
}

export function hasBand(sub: SubstitutionArtifact): boolean {
  return sub.models.some((m) => bandPoints(sub, m).length > 1);
}

/** Where the curves disagree: the span between the lowest and highest curve at each k. */
export function disagreement(sub: SubstitutionArtifact): { k: number; lo: number; hi: number }[] {
  const out: { k: number; lo: number; hi: number }[] = [];
  sub.ks.forEach((k, i) => {
    const vals = sub.models
      .map((m) => m.delta[i])
      .filter((v): v is number => v !== null && v !== undefined && Number.isFinite(v));
    if (vals.length >= 2) out.push({ k, lo: Math.min(...vals), hi: Math.max(...vals) });
  });
  return out;
}

/** Why a curve stops, from the data's own support. */
export function stopReason(sub: SubstitutionArtifact): string | null {
  const m = sub.models.find((x) => x.stopped_at !== null);
  if (!m || m.stopped_at === null) return null;
  const i = sub.ks.findIndex((k) => k === m.stopped_at);
  const share = i >= 0 ? m.on_support_fraction[i] : null;
  const pct = share !== null && share !== undefined ? ` (${Math.round(share * 100)}%)` : "";
  const last = i > 0 ? sub.ks[i - 1] : null;
  const end = last !== null && last !== undefined ? `Stops at ${fmtInt(last)} kcal` : "Stops at the start";
  return `${end}: at ${fmtInt(m.stopped_at)} kcal, fewer than half the rows would stay within observed intakes${pct}.`;
}

export function curveDomain(sub: SubstitutionArtifact): [number, number] {
  const vals: number[] = [0];
  for (const m of sub.models) {
    for (const p of curvePoints(sub, m)) vals.push(p.delta);
    for (const b of bandPoints(sub, m)) vals.push(b.lo, b.hi);
  }
  const lo = Math.min(...vals);
  const hi = Math.max(...vals);
  const pad = (hi - lo) * 0.08 || 0.5;
  return [lo - pad, hi + pad];
}

/** The refits a requested band uses when the server names none (M1_CONTRACT §12.7). */
export const BAND_BOOT = 50;
/** Each refit sees at most this many training rows (§12.7). */
export const BAND_ROWS = 2_000;

/**
 * The band the Results offer: the server's own measurement (it times one refit per family when
 * no band is asked for, and reports the band's refits and seconds). Without one, an estimate from
 * the measured fit times: each family's time per fit scaled to ≤ 2,000 rows, times the refits,
 * doubled for the curve each refit draws.
 */
export function bandOffer(sub: SubstitutionArtifact, fit: FitArtifact, folds: number): { nBoot: number; seconds: number } {
  if (sub.band_estimate) return { nBoot: sub.band_estimate.n_boot, seconds: sub.band_estimate.seconds };
  const share = Math.min(1, BAND_ROWS / Math.max(1, fit.n_train));
  const perFit = fit.models.reduce((t, m) => t + m.fit_seconds / (folds + 2), 0);
  return { nBoot: BAND_BOOT, seconds: 2 * perFit * share * BAND_BOOT };
}

export function aboutSeconds(s: number): string {
  if (s < 5) return "a few seconds";
  if (s < 90) return `${Math.max(5, Math.round(s / 5) * 5)} s`;
  return `${Math.round(s / 60)} min`;
}

/** "−1.71 per 100 kcal at k = 100", as the server labels it (a real minus sign). */
export function effectLabel(m: SubstitutionModel): string {
  return (m.effect_label ?? "").replace(/^-/, "−");
}

export function fmtDelta(v: number): string {
  return fmtNum(v);
}

const ticked = (names: string[]) => {
  const t = names.map((n) => `\`${n}\``);
  return t.length < 2 ? (t[0] ?? "") : `${t.slice(0, -1).join(", ")} and ${t.at(-1)}`;
};

/**
 * The nested nutrients a curve moved along (§12.5): "`fat_sat`, `fat_mon` and `fat_poly` moved with
 * `fat_total`; `sugar` with `carb`" — so the reader sees that a part kept its share of its total.
 */
export function carriedSentence(
  carried: string[],
  nested: { column: string; parent: string }[],
): string | null {
  if (!carried.length) return null;
  const byParent = new Map<string, string[]>();
  const loose: string[] = [];
  for (const c of carried) {
    const parent = nested.find((n) => n.column === c)?.parent;
    if (parent) byParent.set(parent, [...(byParent.get(parent) ?? []), c]);
    else loose.push(c);
  }
  const parts = [...byParent].map(([parent, kids]) => `${ticked(kids)} with \`${parent}\``);
  if (loose.length) parts.push(`${ticked(loose)} with their totals`);
  return `Parts moved with their totals, keeping their shares: ${parts.join("; ")}.`;
}
