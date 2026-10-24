/**
 * Coefficients for the linear families (§13): a forest plot of the exposures, with confidence
 * intervals under inference. Under prediction the estimates are drawn without intervals and a
 * note says they are not interpreted. Under inference a family whose fit gives no coefficient
 * table (a penalized family's shrunk coefficients) is drawn without intervals too, and its note
 * says why they are not read as effects (the ruling of 2026-10-23: the engine reports no effect
 * for it), in the reason the family's own declaration gives.
 */
import { useState } from "react";
import { scaleLinear } from "d3-scale";
import type { Coefficient, FittedModel } from "../../../api/m1-stage-types";
import type { FamilyInfo } from "../../../api/m3-types";
import { fmtNum } from "../format";
import { useSize } from "../views/geometry";
import { forestDomain } from "./model";
import s from "./results.module.css";

/** The table a family declares it gives under inference (``FamilyInfo.inference_table``, as GET
 *  /models serves the engine's ``inference_decl``), or undefined where it is not known. */
export type DeclaredTable = FamilyInfo["inference_table"] | undefined;

interface Props {
  models: { model: FittedModel; coefficients: Coefficient[] }[];
  purpose: string | null;
  target: string;
  /** Each family's declared table, by its key (``declaredTables``). */
  tables: Readonly<Record<string, DeclaredTable>>;
}

const ROW = 24;

/** Under inference, whether ``model``'s fit gives no coefficient table: it says how its intervals
 *  were made (``inference``) only when it has one (``stages.effects.has_coefficient_table``), so
 *  it has neither intervals nor an effect to report (the effects stage refuses it). */
export function withoutTable(model: FittedModel, purpose: string | null): boolean {
  return purpose === "inference" && !model.inference;
}

/** Each family's declared table by its key, from GET /models (none known before it loads). */
export function declaredTables(families: readonly FamilyInfo[] | undefined): Record<string, DeclaredTable> {
  return Object.fromEntries((families ?? []).map((f) => [f.key, f.inference_table]));
}

/** The note beside such a family's coefficients: why they are not read as effects, and neither an
 *  effect nor an interval is claimed. A family that declares a shrunk table with no intervals
 *  (``shrunk_no_intervals``) has its coefficients shrunk toward zero by its penalty; otherwise,
 *  or before its declaration is known, the note says only what the fit serves: no standard errors. */
export function withoutTableNote(label: string, table: DeclaredTable): string {
  const why = table === "shrunk_no_intervals" ? "are shrunk toward zero by its penalty and have no standard errors" : "have no standard errors";
  return `${label}'s coefficients ${why}, so they describe the fitted model; they are not interpreted as effects, and no intervals are drawn.`;
}

/** The saved forest figure's caption (Results' save menu): what the squares are, and the lines only
 *  where the fit drew intervals (its ``inference`` caption says how, or why none could be). */
export function forestCaption(model: FittedModel, purpose: string | null, table: DeclaredTable): string {
  if (purpose !== "inference") return "Squares: estimates of the fitted prediction model, not interpreted as effects.";
  const how = model.inference;
  if (!how) return `Squares: coefficients as fitted. ${withoutTableNote(model.label, table)}`;
  return how.refused ? `Squares: estimates. ${how.caption}` : `Squares: estimates; lines: ${how.caption}`;
}

function fmtP(p: number | null): string {
  if (p === null || !Number.isFinite(p)) return "";
  if (p < 0.001) return "p < 0.001";
  return `p ${p.toFixed(3)}`;
}

export function Coefficients({ models, purpose, target, tables }: Props) {
  const [pick, setPick] = useState(0);
  const [ref, { w }] = useSize<HTMLDivElement>();
  if (!models.length) return null;
  const current = models[Math.min(pick, models.length - 1)]!;
  const coefs = current.coefficients;
  const inference = purpose === "inference";
  // How the intervals were made (turbotab/core/models/inference.py): "95% intervals …" or, when
  // none can be reported, "No intervals: …".
  const how = current.model.inference;
  const domain = forestDomain(coefs);
  const x = scaleLinear().domain(domain).nice().range([10, Math.max(60, w - 10)]);
  const ticks = x.ticks(w < 300 ? 3 : 6);
  const h = coefs.length * ROW + 8;
  return (
    <div className={s.forest} data-testid="coefficients">
      {models.length > 1 ? (
        <div className={s.switcher} role="radiogroup" aria-label="Family">
          {models.map((m, i) => (
            <button
              key={m.model.family}
              type="button"
              role="radio"
              aria-checked={i === pick}
              className={s.switch}
              onClick={() => setPick(i)}
            >
              {m.model.label}
            </button>
          ))}
        </div>
      ) : null}
      <div className={s.forestGrid}>
        <div className={s.forestNames}>
          {coefs.map((c) => (
            <span key={c.feature} className={s.feature} style={{ height: ROW }}>
              {c.feature}
            </span>
          ))}
        </div>
        <div ref={ref} className={s.forestPlot}>
          {w > 0 ? (
            <svg width={w} height={h + 26} className={s.svg} role="img" aria-label={`${current.model.label} coefficients`}>
              {ticks.map((t) => (
                <line key={t} x1={x(t)} x2={x(t)} y1={0} y2={h} className={t === 0 ? s.zeroLine : s.gridLine} />
              ))}
              {coefs.map((c, i) => {
                const y = i * ROW + ROW / 2;
                const hasCi = inference && c.ci_low !== null && c.ci_high !== null;
                return (
                  <g key={c.feature}>
                    {hasCi ? <line x1={x(c.ci_low!)} x2={x(c.ci_high!)} y1={y} y2={y} className={s.interval} /> : null}
                    {c.estimate !== null ? (
                      <rect x={x(c.estimate) - 4} y={y - 4} width={8} height={8} className={s.square} />
                    ) : null}
                  </g>
                );
              })}
              {ticks.map((t) => (
                <text key={`t${t}`} x={x(t)} y={h + 14} textAnchor="middle" className={s.tick}>
                  {fmtNum(t, 2)}
                </text>
              ))}
            </svg>
          ) : null}
        </div>
        <div className={s.forestNums}>
          {coefs.map((c) => (
            <span key={c.feature} className={s.coefNum} style={{ height: ROW }}>
              {fmtNum(c.estimate)}
              {inference && c.ci_low !== null && c.ci_high !== null ? (
                <span className={s.ci}>
                  [{fmtNum(c.ci_low)}, {fmtNum(c.ci_high)}] {fmtP(c.p)}
                </span>
              ) : null}
            </span>
          ))}
        </div>
      </div>
      <p className={s.basis}>
        {withoutTable(current.model, purpose)
          ? withoutTableNote(current.model.label, tables[current.model.family])
          : inference
            ? `Change in predicted ${target} per unit of each input, holding the others${how?.refused ? ". " : "; "}${how?.caption ?? "95% confidence intervals."}`
            : `Under prediction the coefficients describe the fitted model; they are not interpreted as effects, so no intervals are drawn.`}
      </p>
    </div>
  );
}
