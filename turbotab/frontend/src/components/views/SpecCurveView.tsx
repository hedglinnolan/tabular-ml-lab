/**
 * The specification curve view (FOUNDATION §5 rule 9): every declared specification's estimate,
 * sorted, its interval, and below it the choices each one made, on the same x. One series, so the
 * primary is ink and the rest gray (highlight one, gray the rest); no palette is needed.
 */
import { useState } from "react";
import { columnAt, mixedScales, sortSpecs, specLayout, type SpecCurveData } from "./specCurve";
import { Crosshair, OneLine, FIT, MARK_R, NOW, Numbers, Tip, TipLines, YAxis, fmtNum, fmtTick, leftLabelWidth, useWidth } from "./common/parts";
import s from "./curves.module.css";
import v from "./common/views.module.css";
import type { Box } from "./common/scale";

const fmtInt = (n: number) => Math.round(n).toLocaleString("en-US");
const ci = (lo: number | null, hi: number | null) => (lo === null || hi === null ? "" : ` (${fmtNum(lo)} to ${fmtNum(hi)})`);

/** The interval's name in the table head: its level as served, never assumed. */
export const intervalHead = (level: number | null) => (level === null ? "Estimate (interval)" : `Estimate (${+(level * 100).toFixed(1)}% interval)`);

/**
 * One quiet line of computed facts: how many specifications, and how far the others move the
 * estimate from the primary. It never counts intervals that exclude zero: that is vote-counting,
 * and this view is sensitivity, never a way to choose.
 */
export function specLine(d: SpecCurveData): string {
  if (d.specs.length === 1) return "One specification: the primary, which is the reported estimate.";
  const est = d.specs.map((x) => x.estimate);
  return [
    `${d.specs.length} specifications; the primary is the reported estimate, and the others show how far the declared choices move it.`,
    `Estimates run from ${fmtNum(Math.min(...est))} to ${fmtNum(Math.max(...est))}.`,
  ].join(" ");
}

export function SpecCurveView({ data, title, why }: { data: SpecCurveData | null; title?: string; why?: string }) {
  const [ref, W] = useWidth();
  const [at, setAt] = useState<number | null>(null);
  if (!data || !data.specs.length) {
    return <OneLine view="spec_curve" title={title} text={why ?? "There is no specification curve: no alternative specification was declared."} />;
  }
  if (data.sealed) return <OneLine view="spec_curve" title={title} text={data.sealed} />;
  const mixed = mixedScales(data.specs);
  if (mixed) {
    return <OneLine view="spec_curve" title={title} text={`These specifications estimate on different scales (${mixed.join(", ")}), so one axis cannot hold them; each scale needs its own curve.`} />;
  }
  const d = { ...data, specs: sortSpecs(data.specs) };
  const pre = specLayout(d, W, 60)!;
  const optionW = Math.max(...d.choices.flatMap((c) => [c.label.length * 6.6, ...c.options.map((o) => 10 + o.label.length * 6.4)]), 0);
  const gutter = Math.min(Math.round(W * 0.4), Math.ceil(Math.max(leftLabelWidth(pre.yTicks, fmtTick) + 12, optionW + 16)));
  const l = specLayout(d, W, gutter)!;
  const { box, y, cx, colW } = l;
  const n = d.specs.length;
  // at least 9 px across (the ring is painted under the fill), at most MARK_R
  const r = Math.max(4.5, Math.min(MARK_R, colW / 3));
  const maxChars = Math.floor((gutter - 16) / 6.4);
  const cut = (t: string) => (t.length > maxChars ? `${t.slice(0, Math.max(1, maxChars - 1))}…` : t);
  const curveBox: Box = { ...box, bottom: box.height - l.curveBottom };
  const hoverBox: Box = { ...box, bottom: 8 };
  const primaryAt = d.specs.findIndex((x) => x.primary);
  const cur = at !== null ? d.specs[at] : undefined;
  const choiceLabel = (c: string, o: string) => d.choices.find((x) => x.key === c)?.options.find((x) => x.key === o)?.label ?? o;
  return (
    <figure className={s.fig} data-exhibit-view="spec_curve">
      {title ? <h3>{title}</h3> : null}
      <div className={s.plot} ref={ref}>
        <svg viewBox={`0 0 ${box.width} ${box.height}`} role="img" aria-label={`${d.estimateLabel} in each of ${n} specifications, sorted, with the choices each made`}>
          {cur ? <rect x={cx(at!) - colW / 2} y={box.top} width={colW} height={box.height - box.top - 8} style={{ fill: "var(--canvas-line)" }} opacity={0.7} /> : null}
          <YAxis y={y} ticks={l.yTicks} box={curveBox} fmt={fmtTick} title={d.estimateLabel} />
          <line className={v.grid} x1={box.left} x2={box.width - box.right} y1={l.curveBottom} y2={l.curveBottom} />
          {d.zero ? <line data-ref="zero" x1={box.left} x2={box.width - box.right} y1={y(0)} y2={y(0)} style={{ stroke: "var(--canvas-muted)" }} strokeWidth={1} /> : null}
          {d.specs.map((sp, i) => {
            const color = sp.primary ? FIT : NOW;
            return (
              <g key={sp.key} data-spec={sp.key} data-primary={sp.primary || undefined}>
                {sp.low !== null && sp.high !== null ? <line x1={cx(i)} x2={cx(i)} y1={y(sp.low)} y2={y(sp.high)} style={{ stroke: color }} strokeWidth={sp.primary ? 2 : 1.5} /> : null}
                <circle cx={cx(i)} cy={y(sp.estimate)} r={sp.primary ? r + 1 : r} className={s.ring} style={{ fill: color }} />
              </g>
            );
          })}
          {primaryAt >= 0 ? (
            <text className={s.label} x={cx(primaryAt)} y={Math.max(box.top + 10, y(d.specs[primaryAt]!.high ?? d.specs[primaryAt]!.estimate) - 8)} textAnchor={cx(primaryAt) > box.width - 60 ? "end" : cx(primaryAt) < box.left + 30 ? "start" : "middle"} style={{ fontWeight: 600 }}>
              Primary
            </text>
          ) : null}
          {/* the choices each specification made, on the same x */}
          {l.rows.map((row) =>
            row.kind === "choice" ? (
              <text key={`c:${row.choice}`} className={s.label} x={4} y={row.y} style={{ fontWeight: 600 }}>
                {cut(row.label)}
              </text>
            ) : (
              <g key={`o:${row.choice}:${row.option}`} data-option={`${row.choice}:${row.option}`}>
                <text className={v.axis} x={14} y={row.y + 4} style={{ fontSize: 12 }}>
                  <title>{row.label}</title>
                  {cut(row.label)}
                </text>
                {d.specs.map((sp, i) =>
                  sp.picks[row.choice] === row.option ? <circle key={sp.key} cx={cx(i)} cy={row.y} r={4.5} style={{ fill: sp.primary ? FIT : NOW }} /> : null,
                )}
              </g>
            ),
          )}
          <Crosshair box={hoverBox} count={n} index={at} setIndex={setAt} indexAt={(px) => columnAt(l, px, n)} label="Read each specification" />
        </svg>
        <Tip at={cur ? { x: cx(at!), y: y(cur.high ?? cur.estimate) } : null} width={box.width}>
          {cur ? (
            <TipLines
              head={`${cur.primary ? "Primary · " : ""}${fmtNum(cur.estimate)}${ci(cur.low, cur.high)}`}
              lines={[...d.choices.map((c) => ({ label: `${c.label}:`, value: choiceLabel(c.key, cur.picks[c.key] ?? "") })), { label: "Rows:", value: fmtInt(cur.n) }]}
            />
          ) : null}
        </Tip>
      </div>
      <p className={s.quiet}>{specLine(d)}</p>
      <Numbers
        table={{
          caption: `${d.estimateLabel}, in each specification, sorted by estimate`,
          head: ["Rank", intervalHead(d.level), "Rows", ...d.choices.map((c) => c.label)],
          rows: d.specs.map((sp, i) => ({
            key: sp.key,
            primary: sp.primary,
            cells: [`${i + 1}${sp.primary ? " (primary)" : ""}`, `${fmtNum(sp.estimate)}${ci(sp.low, sp.high)}`, fmtInt(sp.n), ...d.choices.map((c) => choiceLabel(c.key, sp.picks[c.key] ?? "—"))],
          })),
        }}
      />
    </figure>
  );
}
