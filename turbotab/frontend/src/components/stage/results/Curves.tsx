/**
 * Substitution curves (§13, PRODUCT_VISION §06c's marks): one line per family told apart by dash
 * pattern, the span between them shaded where the models disagree, each curve stopping at the
 * data's support limit with an end marker and its reason, a strip of the share of rows on support,
 * effect labels "per 100 kcal at k = 100", and the refit band once it is computed.
 */
import { area, line } from "d3-shape";
import { scaleLinear } from "d3-scale";
import type { SubstitutionArtifact } from "../../../api/m1-stage-types";
import { fmtInt, fmtNum } from "../format";
import { useSize } from "../views/geometry";
import { bandPoints, curveDomain, curvePoints, disagreement, effectLabel, stopReason } from "./model";
import s from "./results.module.css";

const DASH = ["", "7 4", "2 3", "9 3 2 3"];
const HUE = ["var(--c1)", "var(--c2)", "var(--c3)", "var(--c4)"];
const PAD = { l: 48, r: 18, t: 26, b: 54 };

interface Props {
  sub: SubstitutionArtifact;
  target: string;
}

export function Curves({ sub, target }: Props) {
  const [ref, { w }] = useSize<HTMLDivElement>();
  const H = 280;
  const box = { x0: PAD.l, x1: Math.max(PAD.l + 60, w - PAD.r), y0: PAD.t, y1: H - PAD.b };
  const kMax = sub.ks[sub.ks.length - 1] ?? 1;
  const x = scaleLinear().domain([0, kMax]).range([box.x0, box.x1]);
  const y = scaleLinear().domain(curveDomain(sub)).nice().range([box.y1, box.y0]);
  const gap = disagreement(sub);
  const shade = area<{ k: number; lo: number; hi: number }>()
    .x((d) => x(d.k))
    .y0((d) => y(d.lo))
    .y1((d) => y(d.hi));
  const path = line<{ k: number; delta: number }>()
    .x((d) => x(d.k))
    .y((d) => y(d.delta));
  const support = sub.models[0]?.on_support_fraction ?? [];
  const cell = sub.ks.length > 1 ? (x(sub.ks[1]!) - x(sub.ks[0]!)) : 20;
  const reason = stopReason(sub);
  return (
    <div className={s.curves} data-testid="substitution-curves">
      <div ref={ref} className={s.curvePlot}>
        {w > 0 ? (
          <svg width={w} height={H} className={s.svg} role="img" aria-label={`Substitution curves, ${sub.donor} to ${sub.recipient}`}>
            {y.ticks(5).map((t) => (
              <g key={t}>
                <line x1={box.x0} x2={box.x1} y1={y(t)} y2={y(t)} className={t === 0 ? s.zeroLine : s.gridLine} />
                <text x={box.x0 - 7} y={y(t)} dy="0.32em" textAnchor="end" className={s.tick}>
                  {fmtNum(t, 2)}
                </text>
              </g>
            ))}
            {gap.length > 1 ? <path d={shade(gap) ?? ""} className={s.disagree} /> : null}
            {sub.models.map((m, i) => {
              const band = bandPoints(sub, m);
              return band.length > 1 ? (
                <path key={`b-${m.family}`} d={shade(band) ?? ""} style={{ fill: HUE[i % HUE.length] }} className={s.band} />
              ) : null;
            })}
            {sub.models.map((m, i) => {
              const pts = curvePoints(sub, m);
              const end = pts[pts.length - 1];
              return (
                <g key={m.family} data-family={m.family}>
                  <path
                    d={path(pts) ?? ""}
                    className={s.curve}
                    style={{ stroke: HUE[i % HUE.length], strokeDasharray: DASH[i % DASH.length] || undefined }}
                  />
                  {end && m.stopped_at !== null ? (
                    <line
                      x1={x(end.k)}
                      x2={x(end.k)}
                      y1={y(end.delta) - 6}
                      y2={y(end.delta) + 6}
                      className={s.stopMark}
                      style={{ stroke: HUE[i % HUE.length] }}
                    />
                  ) : null}
                </g>
              );
            })}
            {sub.ks.map((k, i) => (
              <g key={k}>
                <text x={x(k)} y={box.y1 + 14} textAnchor="middle" className={s.tick}>
                  {fmtInt(k)}
                </text>
                <rect
                  x={Math.max(box.x0, x(k) - cell / 2) + 1}
                  y={box.y1 + 22}
                  width={Math.max(1, Math.min(box.x1, x(k) + cell / 2) - Math.max(box.x0, x(k) - cell / 2) - 2)}
                  height={8}
                  className={s.supportCell}
                  style={{ opacity: 0.12 + 0.88 * (support[i] ?? 0) }}
                >
                  <title>{`${Math.round((support[i] ?? 0) * 100)}% of rows on support at ${fmtInt(k)} kcal`}</title>
                </rect>
              </g>
            ))}
            <line x1={box.x0} x2={box.x1} y1={box.y1} y2={box.y1} className={s.axisLine} />
            <text x={box.x0} y={H - 6} className={s.axisTitle}>
              kcal moved from {sub.donor} to {sub.recipient} · strip: rows on support
            </text>
            <text x={box.x0 - 40} y={box.y0 - 14} className={s.axisTitle}>
              Δ predicted {target}
            </text>
          </svg>
        ) : null}
      </div>
      <ul className={s.curveLegend}>
        {sub.models.map((m, i) => (
          <li key={m.family}>
            <svg width={30} height={10} aria-hidden="true">
              <line
                x1={1}
                x2={29}
                y1={5}
                y2={5}
                className={s.curve}
                style={{ stroke: HUE[i % HUE.length], strokeDasharray: DASH[i % DASH.length] || undefined }}
              />
            </svg>
            <span className={s.family}>{m.label}</span>
            <span className={s.effect}>{effectLabel(m) || "no effect at k = 100"}</span>
          </li>
        ))}
        {gap.length > 1 ? (
          <li>
            <span className={s.keyDisagree} aria-hidden="true" />
            <span className={s.legendText}>where the models disagree</span>
          </li>
        ) : null}
      </ul>
      {reason ? <p className={s.stopNote}>{reason}</p> : null}
    </div>
  );
}
