/**
 * table_focus — the working table narrowed to the columns this choice touches.
 *
 * A column is keyed by its source ("slot": the raw column it comes from), so `fat_total` becoming
 * `fat_total_adj` is the same column growing a suffix, not a new one. A cell's value is keyed by
 * its text: a changed value crossfades (the old slides up and out, the new rises in) and nothing
 * ever shows a number between the two — an interpolated value would be a claim the data never made.
 * A column that leaves the model keeps its place, struck and dimmed, so the eye can find where it
 * went. Headnotes (≤ 8 words, or a formula) sit over exactly the columns they describe.
 */
import { useCallback, useMemo, type ReactNode } from "react";
import { Prose } from "../../../components/Prose";
import { SceneBuilder, cached, unionKeys, type Scene } from "../engine/morph";
import { css, useMorph, useRegistry } from "../engine/scrub";
import { useCh } from "../engine/useSize";
import s from "./views.module.css";

export type ColStatus = "same" | "changed" | "dropped";

export interface ColState {
  /** The name the column carries in this state. */
  name: string;
  values: string[];
  status: ColStatus;
}

export interface HeadNote {
  id: string;
  text: string;
  from: string;
  to: string;
  tone?: "formula" | "note";
}

export interface TableState {
  cols: Record<string, ColState>;
  notes: HeadNote[];
}

interface Props {
  slots: string[];
  rowIds: number[];
  /** Must include "now". */
  states: Record<string, TableState>;
  fontPx?: number;
  rowH?: number;
  span?: readonly [number, number];
  /** Values shown in "now", to tint a cell only when it differs from the user's data. */
  label?: string;
  onRowHover?: (row: number | null) => void;
  hotRow?: number | null;
  /** Gutter labels in place of row ids (e.g. a level's row count), and the gutter's head. */
  rowLabels?: string[];
  gutterHead?: string;
  /** Keep every column's width across states (for wide tables with a minimap beneath). */
  stableWidths?: boolean;
  /** Drawn under the table with each column's position (stable widths only). */
  footer?: (pos: Record<string, { x: number; w: number }>, width: number) => ReactNode;
}

const PAD_CH = 1.8;
const GUTTER_CH = 6;

function splitName(name: string, slot: string): { pre: string; core: string; suf: string } {
  const i = name.indexOf(slot);
  if (i < 0) return { pre: "", core: name, suf: "" };
  return { pre: name.slice(0, i), core: slot, suf: name.slice(i + slot.length) };
}

export function MorphTable({
  slots,
  rowIds,
  states,
  fontPx = 12,
  rowH = 25,
  span,
  label,
  onRowHover,
  hotRow = null,
  stableWidths = false,
  footer,
  rowLabels,
  gutterHead = "row",
}: Props) {
  const ch = useCh(fontPx);
  const { map, reg } = useRegistry<HTMLElement>();
  const now = states.now!;

  const stable = useMemo(() => {
    if (!stableWidths) return null;
    const out: Record<string, number> = {};
    for (const st of Object.values(states))
      for (const slot of slots) {
        const c = st.cols[slot];
        if (!c) continue;
        out[slot] = Math.max(out[slot] ?? 0, c.name.length, ...c.values.map((v) => v.length));
      }
    return out;
  }, [stableWidths, states, slots]);

  const gutter = Math.max(GUTTER_CH, ...(rowLabels ?? []).map((l) => l.length + 2)) * ch;

  const layout = useMemo(() => {
    if (!stable) return null;
    let x = gutter;
    const pos: Record<string, { x: number; w: number }> = {};
    for (const slot of slots) {
      const w = ((stable[slot] ?? 4) + PAD_CH) * ch;
      pos[slot] = { x, w };
      x += w;
    }
    return { pos, width: x };
  }, [stable, slots, ch, gutter]);

  const sceneOf = useMemo(() => {
    return cached((state: string): Scene => {
      const st = states[state] ?? now;
      const sb = new SceneBuilder();
      let x = gutter;
      const pos: Record<string, { x: number; w: number }> = {};
      for (const slot of slots) {
        const c = st.cols[slot] ?? now.cols[slot]!;
        const { pre, core, suf } = splitName(c.name, slot);
        const longest = stable
          ? (stable[slot] ?? 4)
          : Math.max(c.name.length, ...(c.status === "dropped" ? [] : c.values.map((v) => v.length)));
        const w = (longest + PAD_CH) * ch;
        pos[slot] = { x, w };
        sb.set(`col:${slot}`, { x, w });
        if (pre) sb.set(`pre:${slot}:${pre}`, { w: pre.length * ch, o: 1, grow: 1 });
        sb.set(`core:${slot}:${core}`, { o: 1 });
        if (suf) sb.set(`suf:${slot}:${suf}`, { w: suf.length * ch, o: 1, grow: 1 });
        if (c.status === "dropped") sb.set(`strike:${slot}`, { o: 1 });
        c.values.forEach((v, i) => {
          const row = rowIds[i]!;
          sb.set(`cell:${row}:${slot}:${v}`, { o: c.status === "dropped" ? 0.2 : 1, slide: Math.round(rowH * 0.8) });
          if (state !== "now" && c.status === "changed" && v !== now.cols[slot]?.values[i])
            sb.set(`tint:${row}:${slot}`, { o: 1 });
        });
        x += w;
      }
      sb.set("width", { w: x });
      for (const n of st.notes) {
        const a = pos[n.from];
        const b = pos[n.to];
        if (!a || !b) continue;
        const tone = n.tone === "formula" ? "formula" : n.from === n.to ? "side" : "note";
        sb.set(`hn:${n.id}:${tone}:${n.text}`, { x: a.x + 4, w: b.x + b.w - a.x - 8, o: 1, slide: 10, swap: 1 });
      }
      return sb.scene;
    });
  }, [states, now, slots, rowIds, ch, stable, gutter, rowH]);

  const keys = useMemo(
    () => unionKeys(Object.keys(states).map((k) => sceneOf(k))),
    [states, sceneOf],
  );

  const apply = useCallback(
    (sc: Scene) => {
      for (const [k, el] of map.current) {
        const it = sc.items.get(k);
        const o = it ? Math.min(1, it.o ?? 1) : 0;
        const kind = k.slice(0, k.indexOf(":"));
        switch (kind) {
          case "col":
            if (it) {
              css(el, { transform: `translateX(${it.x}px)` });
              css(el, { width: `${it.w}px` });
            }
            break;
          case "pre":
          case "suf":
            css(el, { width: `${it?.w ?? 0}px` });
            css(el, { opacity: String(o) });
            break;
          case "hn":
            css(el, { opacity: String(o) });
            css(el, { visibility: o > 0.01 ? "visible" : "hidden" });
            if (it) {
              css(el, { transform: `translate(${it.x}px, ${it.dy ?? 0}px)` });
              css(el, { width: `${it.w}px` });
            }
            break;
          case "cell":
            css(el, { opacity: String(o) });
            css(el, { visibility: o > 0.01 ? "visible" : "hidden" });
            css(el, { transform: `translateY(${it?.dy ?? 0}px)` });
            break;
          case "width":
            break;
          default:
            css(el, { opacity: String(o) });
            css(el, { visibility: o > 0.01 ? "visible" : "hidden" });
        }
      }
      const w = sc.items.get("width")?.w;
      const box = map.current.get("width:box");
      if (box && w) css(box, { width: `${w}px` });
    },
    [map],
  );

  useMorph({ sceneOf, apply, span });

  const byPrefix = (slot: string, prefix: string) =>
    keys.filter((k) => k.startsWith(`${prefix}:${slot}:`)).map((k) => k.slice(prefix.length + slot.length + 2));

  return (
    <div className={s.table} style={{ fontSize: fontPx }} aria-label={label} role="group">
      <div className={s.headnotes}>
        {keys
          .filter((k) => k.startsWith("hn:"))
          .map((k) => {
            const [, , tone, ...rest] = k.split(":");
            const text = rest.join(":");
            return (
              <div key={k} ref={reg(k)} className={s.headnote} data-tone={tone} style={{ opacity: 0 }}>
                <span className={s.headnoteText}>
                  {tone === "formula" ? <span className={s.formula}>{text}</span> : <Prose text={text} />}
                </span>
              </div>
            );
          })}
      </div>
      <div className={s.grid} ref={reg("width:box")} style={{ height: rowH * (rowIds.length + 1) }}>
        <div className={s.gutter} style={{ width: gutter }}>
          <div className={s.hdrCell} style={{ height: rowH }}>
            <span className={s.rowHead}>{gutterHead}</span>
          </div>
          {rowIds.map((r, i) => (
            <div
              key={r}
              className={s.rowId}
              style={{ height: rowH }}
              data-hot={hotRow === r || undefined}
              onPointerEnter={() => onRowHover?.(r)}
              onPointerLeave={() => onRowHover?.(null)}
            >
              {rowLabels?.[i] ?? r}
            </div>
          ))}
        </div>
        {slots.map((slot) => (
          <div key={slot} ref={reg(`col:${slot}`)} className={s.col}>
            <div className={s.hdrCell} style={{ height: rowH }}>
              <span className={s.hdrName}>
                {byPrefix(slot, "pre").map((t) => (
                  <span key={t} ref={reg(`pre:${slot}:${t}`)} className={s.affix} style={{ width: 0 }}>
                    {t}
                  </span>
                ))}
                <span className={s.coreStack}>
                  {byPrefix(slot, "core").map((t) => (
                    <span key={t} ref={reg(`core:${slot}:${t}`)} className={s.core} style={{ opacity: 0 }}>
                      {t}
                    </span>
                  ))}
                </span>
                {byPrefix(slot, "suf").map((t) => (
                  <span key={t} ref={reg(`suf:${slot}:${t}`)} className={s.affix} style={{ width: 0 }}>
                    {t}
                  </span>
                ))}
              </span>
              <span ref={reg(`strike:${slot}`)} className={s.strike} style={{ opacity: 0 }} />
            </div>
            {rowIds.map((r) => (
              <div
                key={r}
                className={s.cell}
                style={{ height: rowH }}
                data-hot={hotRow === r || undefined}
                onPointerEnter={() => onRowHover?.(r)}
                onPointerLeave={() => onRowHover?.(null)}
              >
                <span ref={reg(`tint:${r}:${slot}`)} className={s.tint} style={{ opacity: 0 }} />
                {keys
                  .filter((k) => k.startsWith(`cell:${r}:${slot}:`))
                  .map((k) => (
                    <span key={k} ref={reg(k)} className={s.value} style={{ opacity: 0 }}>
                      {k.slice(`cell:${r}:${slot}:`.length)}
                    </span>
                  ))}
              </div>
            ))}
          </div>
        ))}
      </div>
      {footer && layout ? footer(layout.pos, layout.width) : null}
    </div>
  );
}
