/**
 * The curve view (FOUNDATION §5 rule 9): an estimate across one input's values with its band and
 * the input's observed support, as a rug or as a labelled strip of the share of rows in range.
 * Gray is the curve as it stands; indigo is the same curve with the pointed choice; several
 * entities compared take the comparison palette in fixed order.
 */
import { useState } from "react";
import { bandOf, nearest, pathOf, runs, type Domain } from "./scale";
import { curveFrame, hasEstimate, hasStrip, readAt, visibleLines, type CurveData, type CurveLine } from "./curve";
import { CHOICE, Crosshair, Empty, Legend, MARK_R, Numbers, NOW, Tip, TipLines, XAxis, YAxis, fmtNum, fmtTick, slotColor, useWidth, viewStyles as s, type KeyItem, type TableSpec } from "./parts";

export function lineColor(l: CurveLine): string {
  if (l.role === "now") return NOW;
  if (l.role === "choice") return CHOICE;
  return slotColor(l.slot ?? 1);
}

const BAND_OPACITY = { now: 0.3, choice: 0.16, series: 0.13 } as const;

const pct = (v: number) => `${Math.round(v * 100)}%`;

/** The strip's label: what a bar's height means, read against its 0 and 100% marks. */
export const STRIP_LABEL = "Share of rows within the observed range";

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
  const frame = curveFrame(data, lines, W);
  if (!frame) return <Empty kind="curve" title={title} why={why ?? "There is no curve to draw."} />;
  const { g, labels } = frame;
  const { box, x, y, strip } = g;
  const xs = readAt(data, lines, x.domain() as unknown as Domain);
  const supportAt = (v: number) => {
    const i = data.support?.x.indexOf(v) ?? -1;
    return i >= 0 ? data.support!.share[i] : undefined;
  };
  const key: KeyItem[] = lines.map((l) => ({ label: l.label, color: lineColor(l), mark: "line" }));
  const xName = data.xName ?? data.xLabel;
  const hasBand = lines.some((l) => l.low?.some((v) => v !== null));
  const cur = at !== null ? xs[at] : undefined;
  const [dLo, dHi] = x.domain() as [number, number];
  const barW = strip ? Math.max(2, Math.min(6, (0.5 * (box.width - box.left - box.right)) / Math.max(1, data.support!.x.length))) : 0;
  const tipY = (v: number) => {
    const ys = lines.flatMap((l) => {
      const i = l.x.indexOf(v);
      const e = i >= 0 ? l.y[i] : null;
      return e === null || e === undefined ? [] : [y(e)];
    });
    if (ys.length) return Math.min(...ys);
    const sh = supportAt(v);
    return strip && sh !== undefined ? strip.y(sh) : g.axisY;
  };
  return (
    <figure className={s.fig} data-view="curve" data-sealed={data.sealed ? "true" : undefined}>
      {title ? <h3>{title}</h3> : null}
      <Legend items={key} />
      <div className={s.plot} ref={ref}>
        <svg viewBox={`0 0 ${box.width} ${box.height}`} role="img" aria-label={data.sealed ? `${STRIP_LABEL}, across ${data.xLabel}` : `${data.yLabel} across ${data.xLabel}`}>
          {data.sealed ? null : <YAxis y={y} ticks={g.yTicks} box={{ ...box, bottom: box.height - g.plotBottom }} fmt={fmtTick} title={data.yLabel} />}
          <XAxis x={x} ticks={g.xTicks} box={{ ...box, bottom: box.height - g.axisY }} fmt={fmtTick} title={data.xLabel} />
          {data.zero && !data.sealed ? <line data-ref="zero" x1={box.left} x2={box.width - box.right} y1={y(0)} y2={y(0)} style={{ stroke: "var(--canvas-muted)" }} strokeWidth={1} /> : null}
          {data.stop && !data.sealed ? (
            <g data-ref="stop">
              <line x1={x(data.stop.x)} x2={x(data.stop.x)} y1={g.plotTop} y2={g.plotBottom} style={{ stroke: "var(--canvas-muted)" }} strokeWidth={1} />
              <text className={s.axis} x={x(data.stop.x) - 5} y={g.plotTop + 12} textAnchor="end">
                Stops
              </text>
            </g>
          ) : null}
          {/* the rug: where the input was observed (reads no outcome) */}
          {data.rug?.length ? (
            <g data-mark="rug" style={{ stroke: "var(--canvas-muted)" }}>
              {data.rug.map((v, i) => (
                <line key={i} x1={x(v)} x2={x(v)} y1={g.axisY - 7} y2={g.axisY} strokeWidth={1} opacity={0.55} />
              ))}
            </g>
          ) : null}
          {/* the support strip: the share of rows within the observed range at each x, on its own
              labelled 0–100% scale under the plot (reads no outcome) */}
          {strip ? (
            <g data-mark="strip">
              <text className={s.axis} x={box.left} y={strip.top - 7}>
                {STRIP_LABEL}
              </text>
              <line className={s.grid} x1={box.left} x2={box.width - box.right} y1={strip.top} y2={strip.top} />
              <text className={s.axis} x={box.left - 6} y={strip.top + 4} textAnchor="end">
                100%
              </text>
              <text className={s.axis} x={box.left - 6} y={strip.bottom + 4} textAnchor="end">
                0
              </text>
              {data.support!.x.map((v, i) => {
                const sh = data.support!.share[i] ?? 0;
                if (v < dLo || v > dHi || sh <= 0) return null;
                return <rect key={i} data-share={sh} x={x(v) - barW / 2} y={strip.y(sh)} width={barW} height={strip.bottom - strip.y(sh)} style={{ fill: NOW }} />;
              })}
            </g>
          ) : null}
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
                    <circle key={`p${i}`} data-mark="point" cx={x(seg[0]!.x)} cy={y(seg[0]!.y!)} r={MARK_R} className={s.ring} style={{ fill: color }} />
                  ) : (
                    <path key={`l${i}`} className={s.morph} data-mark="line" d={pathOf(seg.map((p) => [x(p.x), y(p.y!)] as const))} fill="none" style={{ stroke: color }} strokeWidth={2} strokeLinejoin="round" strokeLinecap="round" />
                  ),
                )}
              </g>
            );
          })}
          {labels?.map((lb) => (
            <text key={lb.key} className={s.label} data-label={lb.key} x={lb.x + 7} y={lb.y + 4}>
              {lb.label}
            </text>
          ))}
          {cur !== undefined ? (
            <g>
              <line className={s.cross} x1={x(cur)} x2={x(cur)} y1={g.plotTop} y2={g.axisY} />
              {lines.map((l) => {
                const i = l.x.indexOf(cur);
                const v = i >= 0 ? l.y[i] : null;
                return v === null || v === undefined ? null : <circle key={l.key} cx={x(cur)} cy={y(v)} r={MARK_R + 0.5} className={s.ring} style={{ fill: lineColor(l) }} />;
              })}
            </g>
          ) : null}
          {xs.length ? (
            <Crosshair
              box={{ ...box, top: strip && data.sealed ? strip.top : g.plotTop, bottom: box.height - g.axisY }}
              count={xs.length}
              index={at}
              setIndex={setAt}
              indexAt={(px) => nearest(xs, x.invert(px))}
              label={`Read the ${data.sealed ? "rows in range" : "curve"} at each ${xName}`}
            />
          ) : null}
        </svg>
        <Tip at={cur !== undefined ? { x: x(cur), y: tipY(cur) } : null} width={box.width}>
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
      <Numbers table={curveTable(data, lines)} />
    </figure>
  );
}

/** The table alternative: every x with each line's estimate and the share of rows in range. Before
 *  the gate it holds what is drawn: the share in range at each x (or the rug's values). */
export function curveTable(data: CurveData, lines: CurveLine[]): TableSpec {
  const strip = hasStrip(data);
  const xName = data.xName ?? data.xLabel;
  if (data.sealed && !strip) {
    return { caption: `Where ${xName} was observed`, head: [xName], rows: (data.rug ?? []).map((v, i) => ({ key: `${i}`, cells: [fmtTick(v)] })) };
  }
  const withSupport = !!data.support && (strip || !data.sealed);
  const xs = [...new Set([...lines.flatMap((l) => l.x), ...(data.sealed && data.support ? data.support.x : [])])].sort((a, b) => a - b);
  const banded = lines.map((l) => !!l.low?.some((v) => v !== null));
  return {
    caption: data.sealed ? `Share of rows within the observed range, by ${xName}` : `${data.yLabel}, by ${xName}`,
    head: [xName, ...lines.map((l, i) => (banded[i] ? `${l.label} (band)` : l.label)), ...(withSupport ? ["Rows in range"] : [])],
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
        ...(withSupport ? [(() => { const i = data.support!.x.indexOf(v); return i >= 0 ? pct(data.support!.share[i]!) : "—"; })()] : []),
      ],
    })),
  };
}
