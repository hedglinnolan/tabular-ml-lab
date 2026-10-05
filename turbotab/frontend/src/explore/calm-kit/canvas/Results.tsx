/**
 * The results after the lock, in the canvas: Table 2 (the declared model sequence) and "Which of my
 * decisions mattered?" (the estimate across the alternatives declared before any estimate was
 * shown: sensitivity, never a way to choose). Every number is the engine's fit; the comparison
 * palette marks what each alternative varies (sage: the adjustment, plum: the rows).
 */
import { scaleLinear } from "d3-scale";
import { fmtCI, fmtEst, fmtInt, fmtTick, plain } from "../text";
import { matteredAttrs, t2Attrs, type MatteredRow, type Results } from "../walk";
import k from "../kit.module.css";

const GW = 160;

function Interval({ lo, hi, est, min, max, color, bold }: { lo: number | null; hi: number | null; est: number; min: number; max: number; color: string; bold?: boolean }) {
  const x = scaleLinear().domain([min, max]).range([6, GW - 6]);
  return (
    <svg viewBox={`0 0 ${GW} 18`} width={GW} height={18} aria-hidden="true" style={{ display: "block" }}>
      <line x1={x(0)} x2={x(0)} y1={0} y2={18} style={{ stroke: "var(--canvas-line)" }} />
      {lo !== null && hi !== null ? <line x1={x(lo)} x2={x(hi)} y1={9} y2={9} style={{ stroke: color }} strokeWidth={bold ? 2.5 : 1.75} /> : null}
      <circle cx={x(est)} cy={9} r={bold ? 4.5 : 3.5} style={{ fill: color }} />
    </svg>
  );
}

function span(rows: { lo: number | null; hi: number | null; estimate: number }[]) {
  const lo = Math.min(0, ...rows.map((r) => r.lo ?? r.estimate));
  const hi = Math.max(0, ...rows.map((r) => r.hi ?? r.estimate));
  const pad = (hi - lo) * 0.08 || 1e-6;
  return [lo - pad, hi + pad] as const;
}

export function Table2({ results }: { results: Results }) {
  const rows = results.table2;
  const [min, max] = span(rows);
  const exposure = rows[0]?.feature ?? "";
  return (
    <div className={k.panels}>
      <p className={k.lead}>{plain(results.fit.caption).split(". Declared beside it")[0]}.</p>
      <div className={k.tableWrap}>
        <table className={k.t2} data-testid="table2">
          <thead>
            <tr>
              <th scope="col">Model</th>
              <th scope="col">
                {results.fit.measure_label[0]!.toUpperCase() + results.fit.measure_label.slice(1)} per unit of {exposure} (95% CI)
              </th>
              <th scope="col" className={k.ci} aria-label="Interval" />
              <th scope="col">Rows</th>
            </tr>
          </thead>
          <tbody>
            {rows.map((r) => (
              <tr key={r.key} data-primary={r.primary} {...t2Attrs(r)}>
                <th scope="row">
                  {r.label}
                  <small>{r.adjustedFor.length ? `Adjusted for ${r.adjustedFor.join(", ")}` : "No other column in the model"}</small>
                </th>
                <td className="num">
                  {fmtEst(r.estimate)} <span style={{ color: "var(--canvas-muted)" }}>({fmtCI(r.lo, r.hi)})</span>
                </td>
                <td className={k.ci}>
                  <Interval lo={r.lo} hi={r.hi} est={r.estimate} min={min} max={max} color="var(--data-fit)" bold={r.primary} />
                </td>
                <td className="num">{fmtInt(r.n)}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      <p className={k.footnote} data-testid="t2-inference">
        {plain(results.footnote)} The axis spans {fmtTick(min)} to {fmtTick(max)}; the line marks zero.
      </p>
    </div>
  );
}

const HUE: Record<MatteredRow["varies"], string> = {
  primary: "var(--data-fit)",
  adjustment: "var(--cat-1)",
  rows: "var(--cat-2)",
};
const VARIES: Record<MatteredRow["varies"], string> = {
  primary: "The main model",
  adjustment: "Changes what is adjusted for",
  rows: "Changes who is included",
};

export function Mattered({ results }: { results: Results }) {
  const rows = [...results.mattered].sort((a, b) => a.estimate - b.estimate);
  const [min, max] = span(rows);
  const ests = rows.map((r) => r.estimate);
  const width = Math.max(...ests) - Math.min(...ests);
  const below = rows.every((r) => (r.hi ?? r.estimate) < 0);
  const above = rows.every((r) => (r.lo ?? r.estimate) > 0);
  const exposure = results.table2[0]?.feature ?? "";
  return (
    <div className={k.panels}>
      <p className={k.lead}>
        Checks, never a way to choose. Across the {rows.length} versions of the analysis planned before any estimate was shown, the estimate
        for {exposure} spans {fmtEst(width)}; {below ? "every interval lies below zero." : above ? "every interval lies above zero." : "some intervals include zero."}
      </p>
      <div className={k.tableWrap}>
        <table className={k.t2} data-testid="mattered">
          <thead>
            <tr>
              <th scope="col">Version</th>
              <th scope="col">Estimate (95% CI)</th>
              <th scope="col" className={k.ci} aria-label="Interval" />
              <th scope="col">Rows</th>
            </tr>
          </thead>
          <tbody>
            {rows.map((r) => (
              <tr key={r.key} data-primary={r.varies === "primary"} {...matteredAttrs(r)}>
                <th scope="row">
                  {r.label}
                  <small>{VARIES[r.varies]}</small>
                </th>
                <td className="num">
                  {fmtEst(r.estimate)} <span style={{ color: "var(--canvas-muted)" }}>({fmtCI(r.lo, r.hi)})</span>
                </td>
                <td className={k.ci}>
                  <Interval lo={r.lo} hi={r.hi} est={r.estimate} min={min} max={max} color={HUE[r.varies]} bold={r.varies === "primary"} />
                </td>
                <td className="num">{fmtInt(r.n)}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      <div className={k.key}>
        {(["primary", "adjustment", "rows"] as const).map((v) => (
          <span key={v}>
            <i style={{ background: HUE[v], borderRadius: "50%" }} />
            {VARIES[v]}
          </span>
        ))}
      </div>
    </div>
  );
}
