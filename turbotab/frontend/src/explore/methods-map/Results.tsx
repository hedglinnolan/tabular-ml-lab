/**
 * The results once the plan is locked: Table 2 for the exposure only, the adjustment terms one
 * press away (Westreich & Greenland 2013: "adjustment terms, not effect estimates"), the checks,
 * and "Which of my decisions mattered?" — the estimate across the declared alternatives, as
 * sensitivity, never as a way to choose (BLUEPRINT §11.4).
 */
import { useState } from "react";
import { scaleLinear } from "d3-scale";
import { Rich } from "../../components/stage/text";
import { useSize } from "../../components/stage/views/geometry";
import { INF, fmt3, fmtCi, fmtInt, fmtP, type EffectRow, type Fit } from "./fixture";
import type { Answers } from "./model";
import r from "./screen.module.css";

const PRIME = (f: string) => f.replace(/''/g, "″").replace(/'/g, "′");
const cap = (s: string) => s.charAt(0).toUpperCase() + s.slice(1);
/** An axis tick: as short as the value allows. */
const fmtTick = (v: number) => (v === 0 ? "0" : String(+v.toPrecision(3)).replace("-", "−"));

function rowsOf(fit: Fit, a: Answers) {
  return fit.sequence.filter((s) => s.key !== "model_1" || a.model1 === "guess");
}

export function Table2({ fit, a }: { fit: Fit; a: Answers }) {
  const [appendix, setAppendix] = useState(false);
  const curve = fit.tests.length > 0;
  const rows = rowsOf(fit, a);
  return (
    <div className={r.results} data-testid="table2">
      <p className={r.kicker}>Table 2 · the exposure only</p>
      {curve ? (
        <div className={r.tests}>
          {fit.tests.map((t) => (
            <p key={t} className={r.test}>
              <Rich text={t} />
            </p>
          ))}
        </div>
      ) : null}
      <table className={r.t2}>
        <thead>
          <tr>
            <th scope="col">Model</th>
            <th scope="col" className={r.num}>
              n
            </th>
            <th scope="col" className={r.num}>
              {curve ? "Each term of the curve (95% CI)" : `${cap(fit.measure_label)} (95% CI)`}
            </th>
            <th scope="col" className={r.num}>
              p
            </th>
          </tr>
        </thead>
        <tbody>
          {rows.map((s) => (
            <tr key={s.key} data-primary={s.key === "model_2" || undefined}>
              <th scope="row">
                <span className={r.model}>{s.label}</span>
                <span className={r.modelNote}>
                  <Rich text={s.note} />
                </span>
              </th>
              <td className={r.num}>{fmtInt(s.n_rows)}</td>
              <td className={r.num}>
                {s.effects.map((e) => (
                  <span key={e.feature} className={r.est}>
                    {curve ? <span className={r.termName}>{PRIME(e.feature)}</span> : null}
                    {fmtCi(e)}
                  </span>
                ))}
              </td>
              <td className={r.num}>
                {s.effects.map((e) => (
                  <span key={e.feature} className={r.est}>
                    {fmtP(e.p)}
                  </span>
                ))}
              </td>
            </tr>
          ))}
        </tbody>
      </table>
      <p className={r.caption}>
        <Rich text={fit.caption} />
      </p>
      <p className={r.inference}>
        <Rich text={rows.find((s) => s.key === "model_2")!.inference} />
      </p>
      <button
        type="button"
        className={r.disclose}
        aria-expanded={appendix}
        onClick={() => setAppendix((v) => !v)}
        data-testid="appendix-toggle"
      >
        {appendix ? "Hide" : "Show"} the appendix: {fit.appendix_title}
      </button>
      {appendix ? <Appendix fit={fit} a={a} /> : null}
      <div className={r.checks}>
        {fit.diagnostics.map((d) => (
          <p key={d.check} className={r.check}>
            <span className={r.checkName}>{d.check === "influence" ? "Influence" : d.check}</span>
            <Rich text={d.reading} />
          </p>
        ))}
        {fit.unmeasured ? (
          <p className={r.check}>
            <span className={r.checkName}>Unmeasured confounding</span>
            E-value <span className="v">{fmt3(fit.unmeasured.e_point)}</span> (for the limit nearer the null,{" "}
            <span className="v">{fmt3(fit.unmeasured.e_limit)}</span>); robustness value{" "}
            <span className="v">{fmt3(fit.unmeasured.rv)}</span>.
          </p>
        ) : null}
      </div>
    </div>
  );
}

function Appendix({ fit, a }: { fit: Fit; a: Answers }) {
  const shown = fit.appendix.filter((x) => x.key !== "model_1" || a.model1 === "guess");
  return (
    <div className={r.appendix} data-testid="appendix">
      {shown.map((m) => (
        <table key={m.key} className={r.app}>
          <caption>{m.label}</caption>
          <tbody>
            {m.terms.map(([feature, est, lo, hi, p, why]) => (
              <tr key={feature}>
                <th scope="row">
                  <code className="v">{feature}</code>
                </th>
                <td className={r.num}>{fmtCi({ feature, estimate: est, ci_low: lo, ci_high: hi, p })}</td>
                <td className={r.num}>{fmtP(p)}</td>
                <td className={r.whyMark}>{why === 0 ? "†" : "‡"}</td>
              </tr>
            ))}
          </tbody>
        </table>
      ))}
      <p className={r.whys}>
        † {INF.appendix_whys[0]}. ‡ {INF.appendix_whys[1]}.
      </p>
    </div>
  );
}

// ── which of my decisions mattered ──────────────────────────────────────────

interface Spec {
  key: string;
  label: string;
  n: number;
  e: EffectRow;
  primary: boolean;
  adjustment: string;
  rows: string;
}

export function specsOf(fit: Fit, a: Answers): Spec[] {
  const out: Spec[] = [];
  const primaryRows = a.exclusions === "willett_2013_by_sex" ? "Willett 2013, by sex" : "Every row";
  for (const s of rowsOf(fit, a)) {
    const e = s.effects[0];
    if (!e) continue;
    out.push({
      key: s.key,
      label: s.label,
      n: s.n_rows,
      e,
      primary: s.key === "model_2",
      adjustment: s.key === "crude" ? "Unadjusted" : s.key === "model_1" ? "Model 1" : s.key === "model_2" ? "Model 2" : "Model 3",
      rows: primaryRows,
    });
  }
  const declared = new Set(a.sensitivity.map((k) => INF.exclusions.labels.options[k]?.label));
  for (const an of fit.sensitivity.analyses) {
    if (an.primary || !an.effects[0] || an.refused) continue;
    if (an.label === primaryRows) continue; // the primary's own rows, fit again
    if (!an.added && !declared.has(an.label)) continue;
    out.push({ key: `rows:${an.label}`, label: an.label, n: an.n_rows, e: an.effects[0], primary: false, adjustment: "Model 2", rows: an.label });
  }
  return out;
}

export function Mattered({ fit, a }: { fit: Fit; a: Answers }) {
  const [ref, { w }] = useSize<HTMLDivElement>();
  if (fit.tests.length) {
    return (
      <div className={r.results}>
        <p className={r.kicker}>Which of my decisions mattered?</p>
        <p className={r.empty}>
          With <code className="v">sugar</code> as a curve there is no single estimate to line up across the declared
          alternatives; each is a curve of three terms. Change the exposure's form to a straight line to see them side by
          side.
        </p>
      </div>
    );
  }
  const specs = specsOf(fit, a).sort((p, q) => p.e.estimate - q.e.estimate);
  const dims: { name: string; values: string[]; of: (s: Spec) => string }[] = [
    {
      name: "Adjustment set",
      values: ["Unadjusted", "Model 1", "Model 2", "Model 3"].filter((v) => specs.some((s) => s.adjustment === v)),
      of: (s) => s.adjustment,
    },
    { name: "Rows", values: [...new Set(specs.map((s) => s.rows))], of: (s) => s.rows },
  ];
  const plotH = 190;
  const rowH = 17;
  const nDims = dims.reduce((n, d) => n + d.values.length + 1, 0);
  const H = plotH + 18 + nDims * rowH;
  const pad = { l: 150, r: 16, t: 12 };
  const width = Math.max(420, w);
  const lo = Math.min(0, ...specs.map((s) => s.e.ci_low ?? s.e.estimate));
  const hi = Math.max(0, ...specs.map((s) => s.e.ci_high ?? s.e.estimate));
  const y = scaleLinear().domain([lo, hi]).nice().range([plotH, pad.t]);
  const x = (i: number) => pad.l + ((width - pad.l - pad.r) * (i + 0.5)) / specs.length;
  // each dimension: its name on a row of its own, then one row per value it takes
  const lines: { d: (typeof dims)[number]; v: string | null; y: number }[] = [];
  let y0 = plotH + 26;
  for (const d of dims) {
    lines.push({ d, v: null, y: y0 });
    y0 += rowH;
    for (const v of d.values) {
      lines.push({ d, v, y: y0 });
      y0 += rowH;
    }
  }
  return (
    <div className={r.results} data-testid="mattered">
      <p className={r.kicker}>Which of my decisions mattered?</p>
      <p className={r.lede}>
        The estimate of <code className="v">sugar</code> under each declared alternative, with its 95% interval.
        Sensitivity, not a choice: the primary was declared before any of these was seen.
      </p>
      <div ref={ref} className={r.curveWrap}>
        <svg width={width} height={H} className={r.curve} role="img" aria-label="The estimate under each declared alternative">
          {y.ticks(5).map((v) => (
            <g key={v}>
              <line x1={pad.l} x2={width - pad.r} y1={y(v)} y2={y(v)} className={v === 0 ? r.zero : r.grid} />
              <text x={pad.l - 8} y={y(v) + 3.5} textAnchor="end" className={r.tick}>
                {fmtTick(v)}
              </text>
            </g>
          ))}
          {specs.map((s, i) => (
            <g key={s.key} data-primary={s.primary || undefined} className={r.spec}>
              <line x1={x(i)} x2={x(i)} y1={y(s.e.ci_low ?? s.e.estimate)} y2={y(s.e.ci_high ?? s.e.estimate)} />
              <circle cx={x(i)} cy={y(s.e.estimate)} r={s.primary ? 5.5 : 4} />
              <title>{`${s.label}: ${fmtCi(s.e)}, n ${fmtInt(s.n)}`}</title>
            </g>
          ))}
          {lines.map((ln) =>
            ln.v === null ? (
              <text key={`${ln.d.name}-head`} x={0} y={ln.y + 4} className={r.dimName}>
                {ln.d.name}
              </text>
            ) : (
              <g key={`${ln.d.name}-${ln.v}`} transform={`translate(0,${ln.y})`}>
                <text x={pad.l - 10} y={4} textAnchor="end" className={r.dimValue}>
                  {ln.v}
                </text>
                <line x1={pad.l} x2={width - pad.r} y1={0} y2={0} className={r.grid} />
                {specs.map((s, i) =>
                  ln.d.of(s) === ln.v ? (
                    <circle key={s.key} cx={x(i)} cy={0} r={4} className={s.primary ? r.dimOnPrimary : r.dimOn} />
                  ) : null,
                )}
              </g>
            ),
          )}
        </svg>
      </div>
      <ul className={r.specList}>
        {specs.map((s) => (
          <li key={s.key} data-primary={s.primary || undefined}>
            <span className={r.specLabel}>{s.label}</span>
            <span className="num">{fmtCi(s.e)}</span>
            <span className={r.specN}>n {fmtInt(s.n)}</span>
          </li>
        ))}
      </ul>
    </div>
  );
}
