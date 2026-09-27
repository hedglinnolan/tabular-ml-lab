/**
 * lineage — raw columns -> adjusted -> the model matrix, three lanes.
 *
 * A node is keyed by its lane and the raw column it descends from, so `adj:fat_total` and
 * `adj:fat_total_adj` are one node whose label changes, and `kcal` leaving the adjusted lane is a
 * node fading out of its row. A collapsed group (495 count columns) is one node with its count.
 */
import { useCallback, useMemo } from "react";
import { sourceOf, type Lineage, type LineageNode } from "../data";
import { SceneBuilder, cached, unionKeys, type Scene } from "../engine/morph";
import { attrs, css, place, useMorph, useRegistry } from "../engine/scrub";
import { useWidth } from "../engine/useSize";
import s from "./views.module.css";

export interface LineageState {
  lineage: Lineage;
  emphasis: string[];
  after: boolean;
}

interface Props {
  states: Record<string, LineageState>;
  /** Raw column names, longest match wins, to give derived nodes their source's identity. */
  sources: string[];
  rowH?: number;
  span?: readonly [number, number];
}

const LANES = ["raw", "adjusted", "matrix"] as const;
const LANE_NAME = { raw: "raw", adjusted: "adjusted", matrix: "model matrix" } as const;

/**
 * The raw column a node descends from: follow its first incoming link back to the raw lane (the
 * engine lists a derived column's own nutrient first, then kcal), else read it from the name.
 */
function slotResolver(lineage: Lineage, sources: string[]) {
  const byId = new Map(lineage.nodes.map((n) => [n.id, n]));
  const firstIn = new Map<string, string>();
  for (const l of lineage.links) if (!firstIn.has(l.target)) firstIn.set(l.target, l.source);
  return (node: LineageNode): string => {
    if (node.group) return `group-${node.group}`;
    let cur: LineageNode | undefined = node;
    for (let i = 0; cur && cur.lane !== "raw" && i < 4; i++) cur = byId.get(firstIn.get(cur.id) ?? "");
    if (cur?.lane === "raw") return cur.column ?? cur.label;
    const name = node.column ?? node.label;
    return sourceOf(name, sources) ?? name;
  };
}

export function ScrubLineage({ states, sources, rowH = 21, span }: Props) {
  const [wrap, width] = useWidth<HTMLDivElement>(600);
  const { map, reg } = useRegistry<HTMLElement | SVGElement>();
  const laneW = (width - 8) / 3;
  const nodeW = Math.min(180, laneW - 22);
  const laneX = (i: number) => 2 + i * laneW;
  const maxRows = Math.max(
    ...Object.values(states).flatMap((st) => LANES.map((l) => st.lineage.nodes.filter((n) => n.lane === l).length)),
  );
  const groupRows = Math.max(
    0,
    ...Object.values(states).map((st) => st.lineage.nodes.filter((n) => n.count > 1).length > 0 ? 1 : 0),
  );
  const height = (maxRows + groupRows) * rowH + 26;

  const sceneOf = useMemo(() => {
    return cached((state: string): Scene => {
      const st = states[state] ?? states.now!;
      const sb = new SceneBuilder();
      const keyOf = new Map<string, string>();
      const at = new Map<string, { x: number; y: number }>();
      // Rows follow the raw lane's order, so a derived node sits level with its source.
      const slotOf = slotResolver(st.lineage, sources);
      const order = new Map<string, number>();
      st.lineage.nodes.filter((n) => n.lane === "raw").forEach((n, i) => order.set(slotOf(n), i));
      for (const lane of LANES) {
        const nodes = st.lineage.nodes.filter((n) => n.lane === lane);
        const used = new Map<string, number>();
        let next = 0;
        for (const n of nodes) {
          const slot = slotOf(n);
          const k = used.get(slot) ?? 0;
          used.set(slot, k + 1);
          const key = `${lane}:${slot}#${k}`;
          keyOf.set(n.id, key);
          const row = Math.max(next, (order.get(slot) ?? next) + k);
          next = row + (n.count > 1 ? 2 : 1);
          const x = laneX(LANES.indexOf(lane));
          const y = 22 + row * rowH;
          at.set(n.id, { x, y });
          const em = st.emphasis.some((e) => n.id === e || n.id.endsWith(`:${e}`) || n.column === e || n.label === e)
            ? 1
            : 0;
          sb.set(`nd:${key}`, { x, y, o: 1, em, c: st.after ? 1 : 0, g: n.count > 1 ? 1 : 0 });
          sb.set(`nl:${key}:${n.label}`, { o: 1, swap: 1 });
        }
      }
      for (const l of st.lineage.links) {
        const a = at.get(l.source);
        const b = at.get(l.target);
        const ka = keyOf.get(l.source);
        const kb = keyOf.get(l.target);
        if (!a || !b || !ka || !kb) continue;
        const changed = l.operation !== "kept" && l.operation !== "scaled" && l.operation !== "one-hot" ? 1 : 0;
        const ga = st.lineage.nodes.find((n) => n.id === l.source)?.count ?? 1;
        const gb = st.lineage.nodes.find((n) => n.id === l.target)?.count ?? 1;
        sb.set(`lk:${ka}>${kb}`, {
          x1: a.x + nodeW,
          y1: a.y + (ga > 1 ? rowH : rowH / 2) - 1,
          x2: b.x,
          y2: b.y + (gb > 1 ? rowH : rowH / 2) - 1,
          o: 1,
          em: changed,
        });
      }
      return sb.scene;
    });
    // laneX depends only on laneW
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [states, sources, rowH, laneW, nodeW]);

  const keys = useMemo(() => unionKeys(Object.keys(states).map((k) => sceneOf(k))), [states, sceneOf]);

  const apply = useCallback(
    (sc: Scene) => {
      for (const [k, el] of map.current) {
        const it = sc.items.get(k);
        const kind = k.slice(0, k.indexOf(":"));
        if (kind === "lk") {
          const o = it ? Math.min(1, it.o ?? 1) : 0;
          css(el, { opacity: String(o) });
          if (it) {
            const mx = (it.x1! + it.x2!) / 2;
            attrs(el, { d: `M${it.x1},${it.y1} C${mx},${it.y1} ${mx},${it.y2} ${it.x2},${it.y2}` });
            css(el, { stroke: `color-mix(in oklab, var(--c2) ${Math.round((it.em ?? 0) * 100)}%, var(--line))` });
          }
        } else if (kind === "nd") {
          place(el, it);
          if (it) {
            css(el, { borderColor: `color-mix(in oklab, var(--c2) ${Math.round((it.em ?? 0) * 100)}%, var(--line))` });
          }
        } else place(el, it, { x: false, y: false });
      }
    },
    [map],
  );

  useMorph({ sceneOf, apply, span });

  const nodeKeys = keys.filter((k) => k.startsWith("nd:")).map((k) => k.slice(3));
  return (
    <div ref={wrap} className={s.lineage} style={{ height }}>
      {LANES.map((l, i) => (
        <span key={l} className={s.laneHead} style={{ left: laneX(i) }}>
          {LANE_NAME[l]}
        </span>
      ))}
      <svg className={s.lineageSvg} width={width} height={height} aria-hidden="true">
        {keys
          .filter((k) => k.startsWith("lk:"))
          .map((k) => (
            <path key={k} ref={reg(k)} className={s.link} style={{ opacity: 0 }} />
          ))}
      </svg>
      {nodeKeys.map((nk) => (
        <div
          key={nk}
          ref={reg(`nd:${nk}`)}
          className={s.node}
          data-group={nk.includes(":group-") || undefined}
          style={{ width: nodeW, height: nk.includes(":group-") ? 2 * rowH - 4 : rowH - 4, opacity: 0 }}
        >
          {keys
            .filter((k) => k.startsWith(`nl:${nk}:`))
            .map((k) => (
              <span key={k} ref={reg(k)} className={s.nodeLabel} style={{ opacity: 0 }}>
                {k.slice(`nl:${nk}:`.length)}
              </span>
            ))}
        </div>
      ))}
    </div>
  );
}
