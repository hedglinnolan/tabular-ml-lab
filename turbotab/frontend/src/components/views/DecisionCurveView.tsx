/**
 * The decision curve view (FOUNDATION §5 rule 9): each model's net benefit across the thresholds
 * at which someone would act, beside the two references every decision curve needs (treat
 * everyone, treat no one), with the declared threshold range shaded in gray. While an option that
 * changes the range is pointed at, the range it would declare is drawn in indigo: exactly what the
 * option changes (§4, §5 rule 5).
 */
import { useState } from "react";
import { decisionScales, directRoom, pointedRange, refusal, shadedRange, treatAllLabelAt, treatAllLeaves, usefulLine, type DecisionCurveData } from "./decisionCurve";
import { CHOICE, Crosshair, Empty, Legend, MARK_R, Numbers, Tip, TipLines, XAxis, YAxis, fmtNum, fmtTick, leftFor, slotColor, useWidth, viewStyles as s, type KeyItem } from "./parts";
import { inner, nearest, pathOf, runs } from "./scale";

const REF = "var(--canvas-muted)";

export interface DecisionCurveViewProps {
  data: DecisionCurveData | null;
  title?: string;
  why?: string;
}

export function DecisionCurveView({ data, title, why }: DecisionCurveViewProps) {
  const [ref, W] = useWidth();
  const [at, setAt] = useState<number | null>(null);
  if (!data || !data.rows.length || !data.models.length) {
    return <Empty kind="decision_curve" title={title} why={why ?? "There is no decision curve: it is drawn for a yes/no outcome's predicted risks."} />;
  }
  if (data.sealed) return <Empty kind="decision_curve" title={title} why={data.sealed} />;
  const refused = refusal(data);
  if (refused) return <Empty kind="decision_curve" title={title} why={refused} />;
  const d = data;
  const pre = decisionScales(d, W, 40)!;
  const left = leftFor(pre.yTicks, fmtTick);
  const one = d.rows.length === 1;
  // Direct labels at the line ends draw only when they fit and stand apart; their margin is
  // reserved only then. The y scale does not depend on the right margin, so the check comes first.
  const lastIdx = d.rows.length - 1;
  const endOf = (key: string) => {
    for (let i = lastIdx; i >= 0; i--) {
      const v = d.rows[i]!.models[key];
      if (v !== null && v !== undefined) return { i, v };
    }
    return null;
  };
  const endYs = [...d.models.flatMap((m) => { const e = endOf(m.key); return e ? [pre.y(e.v)] : []; }), pre.y(0), ...(one ? [pre.y(d.rows[0]!.treat_all)] : [])].sort((a, b) => a - b);
  const collide = endYs.some((v, i) => i > 0 && v - endYs[i - 1]! < 14);
  const room = d.models.length <= 4 && W >= 480 && !collide ? directRoom(d) : null;
  const direct = room !== null;
  const g = decisionScales(d, W, left, room ?? 12)!;
  const { box, x, y } = g;
  const { x0, y0, y1 } = inner(box);
  const xs = d.rows.map((r) => r.threshold);
  const leaves = treatAllLeaves(d, g.floor);
  const ends = d.models.map((m) => {
    const e = endOf(m.key);
    return e ? { key: m.key, label: m.label, x: x(d.rows[e.i]!.threshold), y: y(e.v) } : null;
  });
  const range = shadedRange(d);
  const pointed = pointedRange(d);
  const allLabel = treatAllLabelAt(d, x, y, { x0, y0, y1 });
  const key: KeyItem[] = [
    ...d.models.map((m) => ({ label: m.label, color: slotColor(m.slot), mark: one ? ("dot" as const) : ("line" as const) })),
    { label: "Treat everyone", color: REF, mark: one ? "dot" : "line" },
    { label: "Treat no one", color: REF, mark: "thin" },
    ...(range ? [{ label: "Your threshold range", color: "var(--canvas-line)", mark: "band" as const }] : []),
    ...(pointed ? [{ label: "With this choice", color: CHOICE, mark: "band" as const, opacity: 0.3 }] : []),
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
            {/* only treat everyone is clipped: it plunges below the floor, while every model value
                is inside the domain and its markers draw whole */}
            <clipPath id={clipId}>
              <rect x={box.left} y={y0} width={box.width - box.left - box.right} height={y1 - y0} />
            </clipPath>
          </defs>
          {range ? (
            <g data-mark="range">
              <rect x={x(range[0])} y={y0} width={x(range[1]) - x(range[0])} height={y1 - y0} style={{ fill: "var(--canvas-line)" }} opacity={0.55} />
            </g>
          ) : null}
          {pointed ? (
            <g data-mark="pointed-range">
              <rect x={x(pointed[0])} y={y0} width={x(pointed[1]) - x(pointed[0])} height={y1 - y0} style={{ fill: CHOICE }} opacity={0.18} />
            </g>
          ) : null}
          <YAxis y={y} ticks={g.yTicks} box={box} fmt={fmtTick} title="Net benefit (per row)" />
          <XAxis x={x} ticks={g.xTicks} box={box} fmt={fmtTick} title="Threshold risk for acting" />
          <line data-ref="treat-none" x1={box.left} x2={box.width - box.right} y1={y(0)} y2={y(0)} style={{ stroke: REF }} strokeWidth={1} />
          <g clipPath={`url(#${clipId})`}>
            {one ? (
              <circle data-ref="treat-all" cx={x(d.rows[0]!.threshold)} cy={y(d.rows[0]!.treat_all)} r={MARK_R} className={s.ring} style={{ fill: REF }} />
            ) : (
              <path data-ref="treat-all" d={pathOf(d.rows.map((r) => [x(r.threshold), y(r.treat_all)] as const))} fill="none" style={{ stroke: REF }} strokeWidth={1.5} strokeLinejoin="round" />
            )}
          </g>
          {allLabel ? (
            <text data-label="treat-all" className={s.axis} x={allLabel.x} y={allLabel.y} textAnchor={allLabel.anchor}>
              Treat everyone
            </text>
          ) : null}
          {d.models.map((m) => (
            <g key={m.key} data-line={m.key}>
              {runs(d.rows, (r) => r.models[m.key] !== null && r.models[m.key] !== undefined).map((seg, i) =>
                seg.length === 1 ? (
                  <circle key={i} cx={x(seg[0]!.threshold)} cy={y(seg[0]!.models[m.key]!)} r={MARK_R} className={s.ring} style={{ fill: slotColor(m.slot) }} />
                ) : (
                  <path key={i} d={pathOf(seg.map((r) => [x(r.threshold), y(r.models[m.key]!)] as const))} fill="none" style={{ stroke: slotColor(m.slot) }} strokeWidth={2} strokeLinejoin="round" strokeLinecap="round" />
                ),
              )}
            </g>
          ))}
          {/* direct labels at the right ends, when they stand apart; the legend carries identity otherwise */}
          {direct ? (
            <>
              {ends.map((e) => (e ? (
                <text key={e.key} className={s.label} data-label={e.key} x={e.x + 7 + (one ? MARK_R : 0)} y={e.y + 4}>
                  {e.label}
                </text>
              ) : null))}
              <text className={s.axis} data-label="treat-none" x={box.width - box.right + 7} y={y(0) + 4}>
                Treat no one
              </text>
            </>
          ) : null}
          {row ? (
            <g>
              <line className={s.cross} x1={x(row.threshold)} x2={x(row.threshold)} y1={y0} y2={y1} />
              {d.models.map((m) => {
                const v = row.models[m.key];
                return v === null || v === undefined ? null : <circle key={m.key} cx={x(row.threshold)} cy={y(v)} r={MARK_R + 0.5} className={s.ring} style={{ fill: slotColor(m.slot) }} />;
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
        {[usefulLine(d), leaves !== null ? `Treating everyone falls below the chart past ${fmtTick(leaves)}; the table has every value.` : null].filter(Boolean).join(" ")}
      </p>
      <Numbers
        table={{
          caption: `Net benefit per row at each threshold, scored ${d.where === "held_out" ? "on the held-out rows" : "out of fold"}${d.prevalence !== null && d.prevalence !== undefined ? `; ${fmtNum(d.prevalence)} of rows had the outcome` : ""}`,
          head: ["Threshold", ...d.models.map((m) => m.label), "Treat everyone", "Treat no one"],
          rows: d.rows.map((r) => ({ key: String(r.threshold), cells: [fmtTick(r.threshold), ...d.models.map((m) => fmtNum(r.models[m.key])), fmtNum(r.treat_all), "0"] })),
        }}
      />
    </figure>
  );
}
