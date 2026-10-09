/**
 * The curve view (FOUNDATION §5 rule 9): an estimate across one input's values with its band and
 * the input's observed support as a rug. Gray is the curve as it stands; indigo is the same curve
 * with the pointed choice; several entities compared take the comparison palette in fixed order.
 */
import { useState } from "react";
import { bandOf, nearest, pathOf, runs } from "./scale";
import { curveScales, endLabels, hasEstimate, stops, visibleLines, type CurveData, type CurveLine } from "./curve";
import { CHOICE, Crosshair, Empty, Legend, Numbers, NOW, Tip, TipLines, XAxis, YAxis, fmtNum, fmtTick, leftFor, slotColor, useWidth, viewStyles as s, type KeyItem } from "./parts";
import { ticksIn } from "./scale";

export function lineColor(l: CurveLine): string {
  if (l.role === "now") return NOW;
  if (l.role === "choice") return CHOICE;
  return slotColor(l.slot ?? 1);
}

const BAND_OPACITY = { now: 0.4, choice: 0.16, series: 0.13 } as const;

const pct = (v: number) => `${Math.round(v * 100)}%`;

export interface CurveViewProps {
  data: CurveData | null;
  title?: string;
  /** the flip: "Your data now" (false) or "With this choice" (true) */
  showChoice?: boolean;
  /** why there is nothing to draw, when data is null */
  why?: string;
}

export function CurveView({ data, title, showChoice = true, why }: CurveViewProps) {
  const [ref, W] = useWidth();
  const [at, setAt] = useState<number | null>(null);
  if (!data || (!hasEstimate(data) && !data.sealed)) {
    return <Empty kind="curve" title={title} why={why ?? "There is no curve to draw: no estimate was served for any value of the input."} />;
  }
  if (data.sealed && !(data.rug?.length || data.support?.x.length)) {
    return <Empty kind="curve" title={title} why={data.sealed} />;
  }
  const lines = data.sealed ? [] : visibleLines(data, showChoice);
  // Provisional ticks size the left margin; the right margin makes room for direct labels.
  const pre = curveScales(data, W, 40);
  if (!pre) return <Empty kind="curve" title={title} why={why ?? "There is no curve to draw."} />;
  const left = data.sealed ? 12 : leftFor(ticksIn(pre.y.domain() as [number, number]), fmtTick);
  const labelRoom = lines.length >= 2 && lines.length <= 4 && W >= 480 ? Math.min(170, 12 + Math.max(...lines.map((l) => l.label.length)) * 6.4) : 12;
  const g = curveScales(data, W, left, labelRoom)!;
  const { box, x, y } = g;
  const xs = stops(lines);
  const labels = endLabels(lines, x, y);
  const supportAt = (v: number) => {
    const i = data.support?.x.indexOf(v) ?? -1;
    return i >= 0 ? data.support!.share[i] : undefined;
  };
  const key: KeyItem[] = lines.map((l) => ({ label: l.label, color: lineColor(l), mark: "line" }));
  const xName = data.xName ?? data.xLabel;
  const hasBand = lines.some((l) => l.low?.some((v) => v !== null));
  const cur = at !== null ? xs[at] : undefined;
  return (
    <figure className={s.fig} data-view="curve">
      {title ? <h3>{title}</h3> : null}
      <Legend items={key} />
      <div className={s.plot} ref={ref}>
        <svg viewBox={`0 0 ${box.width} ${box.height}`} role="img" aria-label={`${data.yLabel} across ${data.xLabel}`}>
          {data.sealed ? null : <YAxis y={y} ticks={g.yTicks} box={box} fmt={fmtTick} title={data.yLabel} />}
          <XAxis x={x} ticks={g.xTicks} box={box} fmt={fmtTick} title={data.xLabel} />
          {data.zero && !data.sealed ? <line data-ref="zero" x1={box.left} x2={box.width - box.right} y1={y(0)} y2={y(0)} style={{ stroke: "var(--canvas-muted)" }} strokeWidth={1} /> : null}
          {data.stop && !data.sealed ? (
            <g data-ref="stop">
              <line x1={x(data.stop.x)} x2={x(data.stop.x)} y1={box.top} y2={box.height - box.bottom} style={{ stroke: "var(--canvas-muted)" }} strokeWidth={1} />
              <text className={s.axis} x={x(data.stop.x) - 5} y={box.top + 12} textAnchor="end">
                Stops
              </text>
            </g>
          ) : null}
          {/* the rug: where the input was observed (reads no outcome) */}
          <g data-mark="rug" style={{ stroke: "var(--canvas-muted)" }}>
            {data.rug?.length
              ? data.rug.map((v, i) => <line key={i} x1={x(v)} x2={x(v)} y1={box.height - box.bottom - 7} y2={box.height - box.bottom} strokeWidth={1} opacity={0.55} />)
              : data.support?.x.map((v, i) => (v < x.domain()[0]! || v > x.domain()[1]! ? null : (
                  <line key={i} x1={x(v)} x2={x(v)} y1={box.height - box.bottom - 3 - 18 * (data.support!.share[i] ?? 0)} y2={box.height - box.bottom} strokeWidth={2} strokeLinecap="butt" />
                )))}
          </g>
          {lines.map((l) => {
            const pts = l.x.map((v, i) => ({ x: v, y: l.y[i], lo: l.low?.[i], hi: l.high?.[i] }));
            const color = lineColor(l);
            const bands = runs(pts, (p) => p.lo !== null && p.lo !== undefined && p.hi !== null && p.hi !== undefined);
            const segs = runs(pts, (p) => p.y !== null && p.y !== undefined);
            return (
              <g key={l.key} data-line={l.key} data-role={l.role}>
                {bands.map((b, i) => (
                  <path
                    key={`b${i}`}
                    className={s.morph}
                    data-mark="band"
                    d={bandOf(
                      b.map((p) => [x(p.x), y(p.hi!)] as const),
                      b.map((p) => [x(p.x), y(p.lo!)] as const),
                    )}
                    style={{ fill: color }}
                    opacity={BAND_OPACITY[l.role]}
                  />
                ))}
                {segs.map((seg, i) =>
                  seg.length === 1 ? (
                    <circle key={`p${i}`} data-mark="point" cx={x(seg[0]!.x)} cy={y(seg[0]!.y!)} r={4} className={s.ring} style={{ fill: color }} />
                  ) : (
                    <path key={`l${i}`} className={s.morph} data-mark="line" d={pathOf(seg.map((p) => [x(p.x), y(p.y!)] as const))} fill="none" style={{ stroke: color }} strokeWidth={2} strokeLinejoin="round" strokeLinecap="round" />
                  ),
                )}
              </g>
            );
          })}
          {labels?.map((lb) => (
            <text key={lb.key} className={s.label} x={lb.x + 7} y={lb.y + 4}>
              {lb.label}
            </text>
          ))}
          {cur !== undefined ? (
            <g>
              <line className={s.cross} x1={x(cur)} x2={x(cur)} y1={box.top} y2={box.height - box.bottom} />
              {lines.map((l) => {
                const i = l.x.indexOf(cur);
                const v = i >= 0 ? l.y[i] : null;
                return v === null || v === undefined ? null : <circle key={l.key} cx={x(cur)} cy={y(v)} r={4.5} className={s.ring} style={{ fill: lineColor(l) }} />;
              })}
            </g>
          ) : null}
          {xs.length ? (
            <Crosshair box={box} count={xs.length} index={at} setIndex={setAt} indexAt={(px) => nearest(xs, x.invert(px))} label={`Read the curve at each ${xName}`} />
          ) : null}
        </svg>
        <Tip at={cur !== undefined ? { x: x(cur), y: Math.min(...lines.map((l) => { const i = l.x.indexOf(cur); const v = i >= 0 ? l.y[i] : null; return v === null || v === undefined ? box.height : y(v); })) } : null} width={box.width}>
          {cur !== undefined ? (
            <TipLines
              head={`${xName} ${fmtTick(cur)}`}
              lines={[
                ...lines.flatMap((l) => {
                  const i = l.x.indexOf(cur);
                  const v = i >= 0 ? l.y[i] : null;
                  if (v === null || v === undefined) return [];
                  const lo = l.low?.[i];
                  const hi = l.high?.[i];
                  const band = lo !== null && lo !== undefined && hi !== null && hi !== undefined ? ` (${fmtNum(lo)} to ${fmtNum(hi)})` : "";
                  return [{ label: lines.length > 1 ? `${l.label}:` : "", value: `${fmtNum(v)}${band}`, color: lines.length > 1 ? lineColor(l) : undefined }];
                }),
                ...(supportAt(cur) !== undefined ? [{ label: "Rows in range:", value: pct(supportAt(cur)!) }] : []),
              ]}
            />
          ) : null}
        </Tip>
      </div>
      {data.sealed ? (
        <p className={s.empty} role="status">
          {data.sealed}
        </p>
      ) : null}
      {!data.sealed && (data.stop || data.basis || (hasBand && data.band)) ? (
        <p className={s.quiet}>{[data.stop?.why, hasBand ? data.band : null, data.basis].filter(Boolean).join(" ")}</p>
      ) : null}
      {data.sealed ? null : <Numbers table={curveTable(data, lines)} />}
    </figure>
  );
}

export function curveTable(data: CurveData, lines: CurveLine[]) {
  const xs = [...new Set(lines.flatMap((l) => l.x))].sort((a, b) => a - b);
  const banded = lines.map((l) => !!l.low?.some((v) => v !== null));
  return {
    caption: `${data.yLabel}, by ${data.xName ?? data.xLabel}`,
    head: [data.xName ?? data.xLabel, ...lines.map((l, i) => (banded[i] ? `${l.label} (band)` : l.label)), ...(data.support ? ["Rows in range"] : [])],
    rows: xs.map((v) => ({
      key: String(v),
      cells: [
        fmtTick(v),
        ...lines.map((l) => {
          const i = l.x.indexOf(v);
          const e = i >= 0 ? l.y[i] : null;
          if (e === null || e === undefined) return "—";
          const lo = l.low?.[i];
          const hi = l.high?.[i];
          return lo !== null && lo !== undefined && hi !== null && hi !== undefined ? `${fmtNum(e)} (${fmtNum(lo)} to ${fmtNum(hi)})` : fmtNum(e);
        }),
        ...(data.support ? [(() => { const i = data.support.x.indexOf(v); return i >= 0 ? pct(data.support.share[i]!) : "—"; })()] : []),
      ],
    })),
  };
}
