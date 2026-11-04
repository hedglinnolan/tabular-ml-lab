/**
 * The gallery's drawings: the closed vocabulary's distribution and relationship (FOUNDATION §5
 * rule 9), drawn small and still. A look changes nothing, so each is "your data now" in gray, with
 * no flip (FIRST_LOOK_BRIEF §1.4). Each is drawn in the exhibit views' shared frame
 * (components/views/common/parts `ViewFrame`: the one line in place of an empty drawing, and the
 * table alternative, drawn only once it is opened, since a gallery holds many), in the gray of the
 * data now, at the container's width; it restyles none of them. The scale math is model.ts's
 * (`barsOf`, `curveOf`). `Unread` is the line said in place of a drawing, or of the guide's beats,
 * whose read the server did not answer.
 */
import type { Histogram } from "../../api/schema";
import { NOW, ViewFrame, fmtNum, fmtPct, fmtTick, sharedStyles as v, useWidth, type TableSpec } from "../../components/views/common/parts";
import { Plain } from "../../kit";
import { fmtInt } from "../../util/format";
import { type BinPoint, type Box, type GroupQuality, barsOf, curveOf } from "./model";
import s from "./firstlook.module.css";

const box = (width: number, height: number, left = 6): Box => ({ width, height, top: 6, right: 6, bottom: 18, left });

/** A table alternative from its spec, in the shared table's markup (its first cell heads its row),
 *  for ViewFrame to draw once its disclosure is opened. */
const tableOf = (t: TableSpec) => () => (
  <table className={v.table}>
    <caption>{t.caption}</caption>
    <thead>
      <tr>
        {t.head.map((h) => (
          <th key={h} scope="col">
            {h}
          </th>
        ))}
      </tr>
    </thead>
    <tbody>
      {t.rows.map((r) => (
        <tr key={r.key}>
          {r.cells.map((c, i) =>
            i === 0 ? (
              <th key={i} scope="row">
                {c}
              </th>
            ) : (
              <td key={i}>{c}</td>
            ),
          )}
        </tr>
      ))}
    </tbody>
  </table>
);

/** A read the server did not answer, said where what it reads would be, in the client's words. */
export function Unread({ message }: { message: string }) {
  return (
    <p className={s.refused} data-testid="look-unread">
      Not read: <Plain text={message} />
    </p>
  );
}

/** A column's histogram: rows in each bin, between its first and last edges. */
export function Bars({ histogram, label }: { histogram: Pick<Histogram, "edges" | "counts">; label: string }) {
  const [ref, width] = useWidth<HTMLElement>(240, 160);
  const b = box(width, 76);
  const bars = barsOf(histogram, b);
  const lo = bars[0]?.from ?? 0;
  const hi = bars[bars.length - 1]?.to ?? 0;
  const total = bars.reduce((n, x) => n + x.count, 0);
  return (
    <ViewFrame
      kind="distribution"
      frameRef={ref}
      empty={bars.length ? null : "No value to draw: every row is blank or not a finite number."}
      table={tableOf({
        caption: `${label}: rows in each bin`,
        head: ["From", "To", "Rows"],
        rows: bars.map((x, i) => ({ key: String(i), cells: [fmtTick(x.from), fmtTick(x.to), fmtInt(x.count)] })),
      })}
    >
      <svg className={s.svg} viewBox={`0 0 ${b.width} ${b.height}`} role="img" aria-label={`${label}: ${fmtInt(total)} values from ${fmtTick(lo)} to ${fmtTick(hi)}`}>
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
    </ViewFrame>
  );
}

/** A category's most frequent levels, each bar in proportion to its rows. */
export function Levels({ levels, n, more, label }: { levels: { label: string; count: number }[]; n: number; more: number; label: string }) {
  const top = Math.max(1, ...levels.map((l) => l.count));
  return (
    <ViewFrame
      kind="distribution"
      table={tableOf({
        caption: `${label}: rows with each value`,
        head: ["Value", "Rows", "Share of the rows with a value"],
        rows: levels.map((l) => ({ key: l.label, cells: [l.label, fmtInt(l.count), n ? fmtPct(l.count / n) : "—"] })),
      })}
    >
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
    </ViewFrame>
  );
}

/** A relationship: the outcome in each tenth of a predictor, at the tenth's middle value. `y` says
 *  what is drawn up the side (the outcome's mean, or the share of rows with its event), `share`
 *  whether it is a share. */
export function Curve({ points, column, y, share }: { points: readonly BinPoint[]; column: string; y: string; share: boolean }) {
  const [ref, width] = useWidth<HTMLElement>(240, 160);
  const b = box(width, 96, 40);
  const pts = curveOf(points, b);
  const fmtY = share ? fmtPct : (n: number) => fmtNum(n);
  const ys = pts.map((p) => p.y);
  const path = pts.map((p, i) => `${i ? "L" : "M"}${p.cx},${p.cy}`).join(" ");
  return (
    <ViewFrame
      kind="relationship"
      frameRef={ref}
      empty={pts.length ? null : "No tenth of it has a value to draw."}
      table={tableOf({
        caption: `${y} in each tenth of ${column}`,
        head: [`${column} (the tenth's middle value)`, y, "Rows"],
        rows: pts.map((p, i) => ({ key: String(i), cells: [fmtTick(p.x), fmtY(p.y), fmtInt(p.n)] })),
      })}
    >
      <svg className={s.svg} viewBox={`0 0 ${b.width} ${b.height}`} role="img" aria-label={`${y} in each tenth of ${column}`}>
        <text className={s.tick} x={b.left - 4} y={b.top + 8} textAnchor="end">
          {pts.length ? fmtY(Math.max(...ys)) : ""}
        </text>
        <text className={s.tick} x={b.left - 4} y={b.height - b.bottom} textAnchor="end">
          {pts.length ? fmtY(Math.min(...ys)) : ""}
        </text>
        <path className={s.line} d={path} stroke={NOW} />
        {pts.map((p, i) => (
          <circle key={i} cx={p.cx} cy={p.cy} r={3} fill={NOW} data-n={p.n} />
        ))}
        <text className={s.tick} x={b.left} y={b.height - 4}>
          {pts.length ? fmtTick(pts[0]!.x) : ""}
        </text>
        <text className={s.tick} x={b.width - b.right} y={b.height - 4} textAnchor="end">
          {pts.length ? fmtTick(pts[pts.length - 1]!.x) : ""}
        </text>
      </svg>
    </ViewFrame>
  );
}

/** Data quality across a column's groups: the share of rows with any candidate predictor blank. */
export function GroupShares({ groups, column }: { groups: readonly GroupQuality[]; column: string }) {
  const top = Math.max(...groups.map((g) => g.missing_share));
  return (
    <ViewFrame
      kind="distribution"
      table={tableOf({
        caption: `Rows with a predictor blank in each of ${column}'s groups`,
        head: ["Group", "Rows", "Share with a predictor blank"],
        rows: groups.map((g) => ({ key: g.group, cells: [g.group, fmtInt(g.n), fmtPct(g.missing_share)] })),
      })}
    >
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
    </ViewFrame>
  );
}
