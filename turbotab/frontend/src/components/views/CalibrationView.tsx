/**
 * The calibration view (FOUNDATION §5 rule 9): what the model predicted against what was observed,
 * on one shared scale with the 45° line of perfect agreement, the engine's smoothed curve, the
 * grouped rows with their intervals when served, and one quiet paragraph of what they show.
 * One entity (the model), so one ink; the marks differ by shape, and the key names them.
 */
import { useState } from "react";
import { calibrationRefusal, calibrationScales, hasCalibration, type CalibrationData } from "./calibration";
import { OneLine, FIT, Legend, MARK_R, Numbers, Tip, TipLines, XAxis, YAxis, fmtNum, fmtTick, leftFor, useWidth, type LegendItem } from "./common/parts";
import s from "./curves.module.css";
import { nearest, pathOf } from "./common/scale";
import { plain } from "../../explore/calm-kit/text";

const fmtInt = (n: number) => Math.round(n).toLocaleString("en-US");

const WHERE = { out_of_fold: "scored out of fold", held_out: "scored on the held-out rows" } as const;

type Iv = CalibrationData["slope"];

/** ", 95% interval 0.547 to 0.76": the level as served, never assumed; "" with no interval. */
export function intervalOf(i: Iv): string {
  if (i.ci_low === null || i.ci_low === undefined || i.ci_high === null || i.ci_high === undefined) return "";
  const lvl = i.level !== null && i.level !== undefined ? `${+(i.level * 100).toFixed(1)}% ` : "";
  return `, ${lvl}interval ${fmtNum(i.ci_low)} to ${fmtNum(i.ci_high)}`;
}

const avg = (d: CalibrationData, v: number) => (d.kind === "risk" ? `${+(v * 100).toFixed(1)}%` : fmtNum(v));

/**
 * The one quiet paragraph under the picture. When the engine flags a concern, its verdict is the
 * paragraph (it carries the numbers, in plain words with the term beside them); otherwise the
 * numbers are said plainly, each technical name riding beside its meaning (FOUNDATION §2).
 */
export function calibrationLine(d: CalibrationData): string {
  const rows = `${fmtInt(d.n)} rows, ${WHERE[d.where]}.`;
  if (d.concern) return `${plain(d.concern)} ${rows}`;
  const parts = [
    `Predicted ${avg(d, d.expected)} on average against ${avg(d, d.observed)} observed${d.intercept.estimate !== null ? ` (calibration intercept ${fmtNum(d.intercept.estimate)}${intervalOf(d.intercept)}; 0 when they agree)` : ""}.`,
    d.slope.estimate !== null
      ? `Outcomes moved ${fmtNum(d.slope.estimate)} times as far as the predictions did (calibration slope${intervalOf(d.slope)}; 1 when they agree, below 1 when predictions are too extreme).`
      : null,
    rows,
  ];
  return parts.filter(Boolean).join(" ");
}

export function calibrationLabels(d: CalibrationData): { x: string; y: string } {
  return d.kind === "risk"
    ? { x: `Predicted risk of ${d.outcome}`, y: `Observed share with ${d.outcome}` }
    : { x: `Predicted ${d.outcome}`, y: `Observed ${d.outcome}` };
}

/** The smoothed curve with repeated points dropped (lowess at tied predictions repeats them). */
export function curvePoints(d: CalibrationData): { x: number; y: number }[] {
  return d.curve.filter((p, i) => i === 0 || p.x !== d.curve[i - 1]!.x || p.y !== d.curve[i - 1]!.y);
}

export function CalibrationView({ data, title, why }: { data: CalibrationData | null; title?: string; why?: string }) {
  const [ref, W] = useWidth();
  const [hover, setHover] = useState<{ kind: "bin" | "curve"; i: number } | null>(null);
  if (!hasCalibration(data)) {
    return <OneLine view="calibration" title={title} text={why ?? "Calibration was not assessed: no predictions were scored against observed outcomes."} />;
  }
  if (data.sealed) return <OneLine view="calibration" title={title} text={data.sealed} />;
  const refused = calibrationRefusal(data);
  if (refused) return <OneLine view="calibration" title={title} text={refused} />;
  const d = data;
  const pre = calibrationScales(d, W, 40)!;
  const left = leftFor(pre.ticks, fmtTick);
  const g = calibrationScales(d, W, left)!;
  const { box, x, y } = g;
  const [lo, hi] = x.domain() as [number, number];
  const pts = curvePoints(d);
  const bins = d.bins ?? [];
  const labels = calibrationLabels(d);
  const key: LegendItem[] = [
    ...(pts.length ? [{ label: "Smoothed", color: FIT, mark: "line" as const }] : []),
    ...(bins.length ? [{ label: "Groups of rows, with intervals", color: FIT, mark: "dot" as const }] : []),
    { label: "Perfect agreement", color: "var(--canvas-muted)", mark: "thin" },
  ];
  const tipAt =
    hover?.kind === "bin" ? { x: x(bins[hover.i]!.predicted), y: y(bins[hover.i]!.high ?? bins[hover.i]!.observed) - 4 } : hover?.kind === "curve" ? { x: x(pts[hover.i]!.x), y: y(pts[hover.i]!.y) - 6 } : null;
  const fmtV = (v: number) => fmtNum(v);
  return (
    <figure className={s.fig} data-exhibit-view="calibration">
      {title ? <h3>{title}</h3> : null}
      <Legend items={key} />
      <div className={s.plot} ref={ref}>
        <svg viewBox={`0 0 ${box.width} ${box.height}`} role="img" aria-label={`${labels.y} against ${labels.x}: ${calibrationLine(d)}`}>
          <YAxis y={y} ticks={g.ticks} box={box} fmt={fmtTick} title={labels.y} />
          <XAxis x={x} ticks={g.ticks} box={box} fmt={fmtTick} title={labels.x} />
          <line data-ref="diagonal" x1={x(lo)} y1={y(lo)} x2={x(hi)} y2={y(hi)} style={{ stroke: "var(--canvas-muted)" }} strokeWidth={1} />
          {bins.map((b, i) => (
            <g key={i} data-mark="bin">
              {b.low !== null && b.high !== null ? <line x1={x(b.predicted)} x2={x(b.predicted)} y1={y(b.low)} y2={y(b.high)} style={{ stroke: FIT }} strokeWidth={1.5} opacity={0.7} /> : null}
              <circle cx={x(b.predicted)} cy={y(b.observed)} r={MARK_R} className={s.ring} style={{ fill: FIT }} />
            </g>
          ))}
          {pts.length > 1 ? <path data-mark="curve" d={pathOf(pts.map((p) => [x(p.x), y(p.y)] as const))} fill="none" style={{ stroke: FIT }} strokeWidth={2} strokeLinejoin="round" strokeLinecap="round" /> : null}
          {pts.length === 1 ? <circle data-mark="curve" cx={x(pts[0]!.x)} cy={y(pts[0]!.y)} r={MARK_R} className={s.ring} style={{ fill: FIT }} /> : null}
          {hover?.kind === "curve" ? <circle cx={x(pts[hover.i]!.x)} cy={y(pts[hover.i]!.y)} r={MARK_R + 0.5} className={s.ring} style={{ fill: FIT }} pointerEvents="none" /> : null}
          {/* hit targets, larger than the marks */}
          {pts.length > 1 ? (
            <path
              className={s.hit}
              d={pathOf(pts.map((p) => [x(p.x), y(p.y)] as const))}
              fill="none"
              strokeWidth={16}
              onPointerMove={(e) => {
                const r = e.currentTarget.ownerSVGElement?.getBoundingClientRect();
                const px = (e.clientX - (r?.left ?? 0)) * (r && r.width ? box.width / r.width : 1);
                setHover({ kind: "curve", i: nearest(pts.map((p) => p.x), x.invert(px)) });
              }}
              onPointerLeave={() => setHover(null)}
            />
          ) : null}
          {bins.map((b, i) => (
            <circle
              key={i}
              className={s.hit}
              cx={x(b.predicted)}
              cy={y(b.observed)}
              r={12}
              tabIndex={0}
              aria-label={`Group ${i + 1}: predicted ${fmtV(b.predicted)}, observed ${fmtV(b.observed)}`}
              onPointerEnter={() => setHover({ kind: "bin", i })}
              onPointerLeave={() => setHover(null)}
              onFocus={() => setHover({ kind: "bin", i })}
              onBlur={() => setHover(null)}
            />
          ))}
        </svg>
        <Tip at={tipAt} width={box.width}>
          {hover?.kind === "bin" ? (
            <TipLines
              head={`Group ${hover.i + 1} · ${fmtInt(bins[hover.i]!.n)} rows`}
              lines={[
                { label: "Predicted", value: fmtV(bins[hover.i]!.predicted) },
                {
                  label: "Observed",
                  value: `${fmtV(bins[hover.i]!.observed)}${bins[hover.i]!.low !== null && bins[hover.i]!.high !== null ? ` (${fmtV(bins[hover.i]!.low!)} to ${fmtV(bins[hover.i]!.high!)})` : ""}`,
                },
              ]}
            />
          ) : hover?.kind === "curve" ? (
            <TipLines lines={[{ label: "Predicted", value: fmtV(pts[hover.i]!.x) }, { label: "Smoothed observed", value: fmtV(pts[hover.i]!.y) }]} />
          ) : null}
        </Tip>
      </div>
      <p className={s.quiet} data-testid="calibration-line">
        {calibrationLine(d)}
      </p>
      <Numbers
        table={{
          caption: `${labels.y} against ${labels.x}${d.binsMethod ? `; ${d.binsMethod}` : ""}${d.smoother ? `; smoothed by ${d.smoother}` : ""}`,
          head: ["", "Predicted", "Observed", "Interval", "Rows"],
          rows: [
            ...bins.map((b, i) => ({ key: `b${i}`, cells: [`Group ${i + 1}`, fmtV(b.predicted), fmtV(b.observed), b.low !== null && b.high !== null ? `${fmtV(b.low)} to ${fmtV(b.high)}` : "—", fmtInt(b.n)] })),
            ...pts.map((p, i) => ({ key: `c${i}`, cells: ["Smoothed", fmtV(p.x), fmtV(p.y), "—", "—"] })),
          ],
        }}
      />
    </figure>
  );
}
