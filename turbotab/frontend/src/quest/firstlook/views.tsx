/**
 * The gallery's drawings: the closed vocabulary's distribution and relationship (FOUNDATION §5
 * rule 9), drawn small and still. A look changes nothing, so each is "your data now" in gray, with
 * no flip (FIRST_LOOK_BRIEF §1.4). Each composes the exhibit views' shared parts
 * (components/views/common/parts: the gray of the data now, the container's width, the one line in
 * place of an empty drawing, the table alternative) and restyles none of them; the scale math is
 * model.ts's (`barsOf`, `curveOf`).
 */
import type { Histogram } from "../../api/schema";
import { NOW, Numbers, OneLine, fmtNum, fmtPct, fmtTick, useWidth } from "../../components/views/common/parts";
import { fmtInt } from "../../util/format";
import { type BinPoint, type Box, type GroupQuality, barsOf, curveOf } from "./model";
import s from "./firstlook.module.css";

const box = (width: number, height: number, left = 6): Box => ({ width, height, top: 6, right: 6, bottom: 18, left });

/** A column's histogram: rows in each bin, between its first and last edges. */
export function Bars({ histogram, label }: { histogram: Pick<Histogram, "edges" | "counts">; label: string }) {
  const [ref, width] = useWidth<HTMLDivElement>(240, 160);
  const b = box(width, 76);
  const bars = barsOf(histogram, b);
  if (!bars.length) return <OneLine text="No value to draw: every row is blank or not a finite number." view="distribution" />;
  const lo = bars[0]!.from;
  const hi = bars[bars.length - 1]!.to;
  const total = bars.reduce((n, x) => n + x.count, 0);
  return (
    <div ref={ref} className={s.drawing} data-view="distribution">
      <svg
        className={s.svg}
        viewBox={`0 0 ${b.width} ${b.height}`}
        role="img"
        aria-label={`${label}: ${fmtInt(total)} values from ${fmtTick(lo)} to ${fmtTick(hi)}`}
      >
        {bars.map((x, i) => (
          <rect key={i} x={x.x} y={x.y} width={x.w} height={x.h} rx={1.5} fill={NOW} data-count={x.count} />
        ))}
        <line className={s.baseline} x1={b.left} x2={b.width - b.right} y1={b.height - b.bottom} y2={b.height - b.bottom} />
        <text className={s.tick} x={b.left} y={b.height - 4}>
          {fmtTick(lo)}
        </text>
        <text className={s.tick} x={b.width - b.right} y={b.height - 4} textAnchor="end">
          {fmtTick(hi)}
        </text>
      </svg>
      <Numbers
        table={{
          caption: `${label}: rows in each bin`,
          head: ["From", "To", "Rows"],
          rows: bars.map((x, i) => ({ key: String(i), cells: [fmtTick(x.from), fmtTick(x.to), fmtInt(x.count)] })),
        }}
      />
    </div>
  );
}

/** A category's most frequent levels, each bar in proportion to its rows. */
export function Levels({ levels, n, more, label }: { levels: { label: string; count: number }[]; n: number; more: number; label: string }) {
  const top = Math.max(1, ...levels.map((l) => l.count));
  return (
    <div className={s.drawing} data-view="distribution">
      <ul className={s.levels} aria-label={`${label}: rows with each value`}>
        {levels.map((l) => (
          <li key={l.label} className={s.level} data-count={l.count}>
            <span className={s.levelName}>{l.label}</span>
            <span className={s.levelCount}>{fmtInt(l.count)}</span>
            <span className={s.levelTrack} aria-hidden="true">
              <span className={s.levelBar} style={{ width: `${(l.count / top) * 100}%`, background: NOW }} />
            </span>
          </li>
        ))}
      </ul>
      {more ? <p className={s.quiet}>and {more === 1 ? "1 more value" : `${fmtInt(more)} more values`}</p> : null}
      <Numbers
        table={{
          caption: `${label}: rows with each value`,
          head: ["Value", "Rows", "Share of the rows with a value"],
          rows: levels.map((l) => ({ key: l.label, cells: [l.label, fmtInt(l.count), n ? fmtPct(l.count / n) : "—"] })),
        }}
      />
    </div>
  );
}

/** A relationship: the outcome in each tenth of a predictor, at the tenth's middle value. `y` says
 *  what is drawn up the side (the outcome's mean, or the share of rows with its event), `share`
 *  whether it is a share. */
export function Curve({ points, column, y, share }: { points: readonly BinPoint[]; column: string; y: string; share: boolean }) {
  const [ref, width] = useWidth<HTMLDivElement>(240, 160);
  const b = box(width, 96, 40);
  const pts = curveOf(points, b);
  if (!pts.length) return <OneLine text="No tenth of it has a value to draw." view="relationship" />;
  const fmtY = share ? fmtPct : (v: number) => fmtNum(v);
  const ys = pts.map((p) => p.y);
  const [ylo, yhi] = [Math.min(...ys), Math.max(...ys)];
  const path = pts.map((p, i) => `${i ? "L" : "M"}${p.cx},${p.cy}`).join(" ");
  return (
    <div ref={ref} className={s.drawing} data-view="relationship">
      <svg className={s.svg} viewBox={`0 0 ${b.width} ${b.height}`} role="img" aria-label={`${y} in each tenth of ${column}`}>
        <text className={s.tick} x={b.left - 4} y={b.top + 8} textAnchor="end">
          {fmtY(yhi)}
        </text>
        <text className={s.tick} x={b.left - 4} y={b.height - b.bottom} textAnchor="end">
          {fmtY(ylo)}
        </text>
        <path className={s.line} d={path} stroke={NOW} />
        {pts.map((p, i) => (
          <circle key={i} cx={p.cx} cy={p.cy} r={3} fill={NOW} data-n={p.n} />
        ))}
        <text className={s.tick} x={b.left} y={b.height - 4}>
          {fmtTick(pts[0]!.x)}
        </text>
        <text className={s.tick} x={b.width - b.right} y={b.height - 4} textAnchor="end">
          {fmtTick(pts[pts.length - 1]!.x)}
        </text>
      </svg>
      <Numbers
        table={{
          caption: `${y} in each tenth of ${column}`,
          head: [`${column} (the tenth's middle value)`, y, "Rows"],
          rows: pts.map((p, i) => ({ key: String(i), cells: [fmtTick(p.x), fmtY(p.y), fmtInt(p.n)] })),
        }}
      />
    </div>
  );
}

/** Data quality across a column's groups: the share of rows with any candidate predictor blank. */
export function GroupShares({ groups, column }: { groups: readonly GroupQuality[]; column: string }) {
  const top = Math.max(...groups.map((g) => g.missing_share));
  return (
    <div className={s.drawing} data-view="distribution">
      <ul className={s.levels} aria-label={`Rows with a predictor blank in each of ${column}'s groups`}>
        {groups.map((g) => (
          <li key={g.group} className={s.level} data-share={g.missing_share}>
            <span className={s.levelName}>{g.group}</span>
            <span className={s.levelCount}>{fmtPct(g.missing_share)} blank</span>
            <span className={s.levelTrack} aria-hidden="true">
              <span className={s.levelBar} style={{ width: top > 0 ? `${(g.missing_share / top) * 100}%` : "0%", background: NOW }} />
            </span>
          </li>
        ))}
      </ul>
      <Numbers
        table={{
          caption: `Rows with a predictor blank in each of ${column}'s groups`,
          head: ["Group", "Rows", "Share with a predictor blank"],
          rows: groups.map((g) => ({ key: g.group, cells: [g.group, fmtInt(g.n), fmtPct(g.missing_share)] })),
        }}
      />
    </div>
  );
}
