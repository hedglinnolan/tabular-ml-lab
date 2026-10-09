/**
 * The decision curve view (FOUNDATION §5 rule 9): each model's net benefit across the thresholds
 * at which someone would act, beside the two references every decision curve needs (treat
 * everyone, treat no one), with the declared threshold range shaded. The range is gray as it
 * stands and indigo while a choice that changes it is pointed at.
 */
import { useState } from "react";
import { decisionScales, shadedRange, treatAllLeaves, type DecisionCurveData } from "./decisionCurve";
import { Crosshair, Empty, Legend, Numbers, Tip, TipLines, XAxis, YAxis, fmtNum, fmtTick, leftFor, slotColor, useWidth, viewStyles as s, type KeyItem } from "./parts";
import { nearest, pathOf, runs, ticksIn } from "./scale";

const REF = "var(--canvas-muted)";

export interface DecisionCurveViewProps {
  data: DecisionCurveData | null;
  title?: string;
  /** the pointed choice changes the threshold range: shade it in the choice color */
  rangeTouched?: boolean;
  why?: string;
}

export function DecisionCurveView({ data, title, rangeTouched = false, why }: DecisionCurveViewProps) {
  const [ref, W] = useWidth();
  const [at, setAt] = useState<number | null>(null);
  if (!data || !data.rows.length || !data.models.length) {
    return <Empty kind="decision_curve" title={title} why={why ?? "There is no decision curve: it is drawn for a yes/no outcome's predicted risks."} />;
  }
  const d = data;
  const pre = decisionScales(d, W, 40)!;
  const left = leftFor(ticksIn(pre.y.domain() as [number, number]), fmtTick);
  const direct = d.models.length <= 4 && W >= 480;
  const right = direct ? Math.min(150, 14 + Math.max(...d.models.map((m) => m.label.length), "Treat no one".length) * 6.4) : 12;
  const g = decisionScales(d, W, left, right)!;
  const { box, x, y } = g;
  const xs = d.rows.map((r) => r.threshold);
  const leaves = treatAllLeaves(d, g.floor);
  const y0 = box.top;
  const y1 = box.height - box.bottom;
  const lastIdx = d.rows.length - 1;
  const ends = d.models.map((m) => {
    for (let i = lastIdx; i >= 0; i--) {
      const v = d.rows[i]!.models[m.key];
      if (v !== null && v !== undefined) return { key: m.key, label: m.label, x: x(d.rows[i]!.threshold), y: y(v) };
    }
    return null;
  });
  const endYs = [...ends.filter(Boolean).map((e) => e!.y), y(0)].sort((a, b) => a - b);
  const collide = endYs.some((v, i) => i > 0 && v - endYs[i - 1]! < 14);
  const range = shadedRange(d);
  const key: KeyItem[] = [
    ...d.models.map((m) => ({ label: m.label, color: slotColor(m.slot), mark: "line" as const })),
    { label: "Treat everyone", color: REF, mark: "line" },
    { label: "Treat no one", color: REF, mark: "thin" },
    ...(range ? [{ label: "Your threshold range", color: rangeTouched ? "var(--data-affected)" : "var(--canvas-line)", mark: "band" as const, opacity: rangeTouched ? 0.3 : 1 }] : []),
  ];
  const row = at !== null ? d.rows[at] : undefined;
  const clipId = `dc-clip-${Math.round(box.width)}-${d.rows.length}`;
  return (
    <figure className={s.fig} data-view="decision_curve">
      {title ? <h3>{title}</h3> : null}
      <Legend items={key} />
      <div className={s.plot} ref={ref}>
        <svg viewBox={`0 0 ${box.width} ${box.height}`} role="img" aria-label="Net benefit across thresholds for acting, beside treating everyone and treating no one">
          <defs>
            <clipPath id={clipId}>
              <rect x={box.left} y={y0 - 1} width={box.width - box.left - box.right} height={y1 - y0 + 2} />
            </clipPath>
          </defs>
          {range ? (
            <g data-mark="range">
              <rect x={x(range[0])} y={y0} width={x(range[1]) - x(range[0])} height={y1 - y0} style={{ fill: rangeTouched ? "var(--data-affected)" : "var(--canvas-line)" }} opacity={rangeTouched ? 0.14 : 0.55} />
            </g>
          ) : null}
          <YAxis y={y} ticks={g.yTicks} box={box} fmt={fmtTick} title="Net benefit (per row)" />
          <XAxis x={x} ticks={g.xTicks} box={box} fmt={fmtTick} title="Threshold risk for acting" />
          <line data-ref="treat-none" x1={box.left} x2={box.width - box.right} y1={y(0)} y2={y(0)} style={{ stroke: REF }} strokeWidth={1} />
          <g clipPath={`url(#${clipId})`}>
            <path data-ref="treat-all" d={pathOf(d.rows.map((r) => [x(r.threshold), y(r.treat_all)] as const))} fill="none" style={{ stroke: REF }} strokeWidth={1.5} strokeLinejoin="round" />
            {d.models.map((m) => (
              <g key={m.key} data-line={m.key}>
                {runs(d.rows, (r) => r.models[m.key] !== null && r.models[m.key] !== undefined).map((seg, i) =>
                  seg.length === 1 ? (
                    <circle key={i} cx={x(seg[0]!.threshold)} cy={y(seg[0]!.models[m.key]!)} r={4} className={s.ring} style={{ fill: slotColor(m.slot) }} />
                  ) : (
                    <path key={i} d={pathOf(seg.map((r) => [x(r.threshold), y(r.models[m.key]!)] as const))} fill="none" style={{ stroke: slotColor(m.slot) }} strokeWidth={2} strokeLinejoin="round" strokeLinecap="round" />
                  ),
                )}
              </g>
            ))}
          </g>
          {/* direct labels at the right ends, when they stand apart; the legend carries identity otherwise */}
          {direct && !collide ? (
            <>
              {ends.map((e) => (e ? (
                <text key={e.key} className={s.label} x={e.x + 7} y={e.y + 4}>
                  {e.label}
                </text>
              ) : null))}
              <text className={s.axis} x={box.width - box.right + 7} y={y(0) + 4}>
                Treat no one
              </text>
            </>
          ) : null}
          {row ? (
            <g>
              <line className={s.cross} x1={x(row.threshold)} x2={x(row.threshold)} y1={y0} y2={y1} />
              {d.models.map((m) => {
                const v = row.models[m.key];
                return v === null || v === undefined ? null : <circle key={m.key} cx={x(row.threshold)} cy={y(v)} r={4.5} className={s.ring} style={{ fill: slotColor(m.slot) }} />;
              })}
            </g>
          ) : null}
          <Crosshair box={box} count={xs.length} index={at} setIndex={setAt} indexAt={(px) => nearest(xs, x.invert(px))} label="Read the net benefit at each threshold" />
        </svg>
        <Tip at={row ? { x: x(row.threshold), y: Math.min(...d.models.map((m) => y(row.models[m.key] ?? g.floor))) } : null} width={box.width}>
          {row ? (
            <TipLines
              head={`Threshold ${fmtTick(row.threshold)}`}
              lines={[
                ...d.models.map((m) => ({ label: `${m.label}:`, value: fmtNum(row.models[m.key]), color: slotColor(m.slot) })),
                { label: "Treat everyone:", value: fmtNum(row.treat_all), color: REF },
                { label: "Treat no one:", value: "0" },
              ]}
            />
          ) : null}
        </Tip>
      </div>
      <p className={s.quiet}>
        {[
          d.useful ? `${d.models[0]!.label} does better than treating everyone or no one from ${fmtTick(d.useful[0])} to ${fmtTick(d.useful[1])}.` : `${d.models[0]!.label} does better than treating everyone or no one at no threshold in the range.`,
          leaves !== null ? `Treating everyone falls below the chart past ${fmtTick(leaves)}; the table has every value.` : null,
        ]
          .filter(Boolean)
          .join(" ")}
      </p>
      <Numbers
        table={{
          caption: `Net benefit per row at each threshold${d.prevalence !== null && d.prevalence !== undefined ? `; ${fmtNum(d.prevalence)} of rows had the outcome` : ""}`,
          head: ["Threshold", ...d.models.map((m) => m.label), "Treat everyone", "Treat no one"],
          rows: d.rows.map((r) => ({ key: String(r.threshold), cells: [fmtTick(r.threshold), ...d.models.map((m) => fmtNum(r.models[m.key])), fmtNum(r.treat_all), "0"] })),
        }}
      />
    </figure>
  );
}
