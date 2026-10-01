/**
 * Which columns the window shows, out of 20,000 (BLUEPRINT §11.3: wide data shows what the choice
 * touched). Every column's change under the combination — a sample's two assays' mean absolute
 * difference over the column's spread — as one distribution; the shown columns are the ticks at
 * its top. The picture is the same at 17 columns and at 20,000; only the counts differ.
 */
import { scaleLinear } from "d3-scale";
import { barPath, useSize } from "../../../components/stage/views/geometry";
import { fmtInt, fmtNum } from "../data";
import s from "./views.module.css";

const H = 96;

export function ColumnRank({
  rank,
  total,
}: {
  rank: { edges: number[]; counts: number[]; shown: number[]; median: number };
  total: number;
}) {
  const [ref, { w }] = useSize<HTMLDivElement>();
  const width = Math.max(200, w);
  const x = scaleLinear().domain([rank.edges[0]!, rank.edges[rank.edges.length - 1]!]).range([4, width - 4]);
  const ymax = Math.max(...rank.counts);
  const y = scaleLinear().domain([0, ymax]).range([H, 14]);
  const bars = new Float64Array(rank.counts.length * 4);
  rank.counts.forEach((c, i) => {
    bars[i * 4] = x(rank.edges[i]!) + 0.5;
    bars[i * 4 + 1] = x(rank.edges[i + 1]!) - 0.5;
    bars[i * 4 + 2] = c > 0 ? Math.min(y(c), H - 1.5) : H;
    bars[i * 4 + 3] = H;
  });
  const ticks = x.ticks(5);
  return (
    <div className={s.spread} ref={ref} data-view="column_rank">
      {w > 0 ? (
        <svg width={width} height={H + 20} className={s.svg} role="img" aria-label="How much combining changes each column">
          <path d={barPath(bars, rank.counts.length)} className={s.barWith} />
          <line x1={4} x2={width - 4} y1={H + 0.5} y2={H + 0.5} className={s.axisLine} />
          {rank.shown.map((v, i) => (
            <line key={i} x1={x(v)} x2={x(v)} y1={4} y2={H} className={s.rankTick} />
          ))}
          {ticks.map((t) => (
            <text key={t} x={x(t)} y={H + 14} textAnchor="middle" className={s.tick}>
              {fmtNum(t, 2)}
            </text>
          ))}
        </svg>
      ) : null}
      <div className={s.spreadLegend}>
        <span className={s.keyWith} /> {fmtInt(total)} columns · median {fmtNum(rank.median, 2)}
        <span className={s.keyTick} /> the {rank.shown.length} the table shows: the most changed
        <span className={s.keyAxis}>mean |assay 1 − assay 2| ÷ SD</span>
      </div>
    </div>
  );
}
