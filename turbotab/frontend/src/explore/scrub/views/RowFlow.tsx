/**
 * row_flow — the participant flow, one bar per step, keyed by step.
 *
 * Steps that exist before and after keep their key, so a new step arriving pushes the ones below
 * it down rather than redrawing the list; the rows a step removes spill off its bar as a hatched
 * segment with the count, and the sidenote sits on that step's line.
 */
import { useCallback, useMemo } from "react";
import { Prose } from "../../../components/Prose";
import { fmtInt, type RowStep } from "../data";
import { SceneBuilder, cached, unionKeys, type Scene } from "../engine/morph";
import { css, place, useMorph, useRegistry } from "../engine/scrub";
import { useWidth } from "../engine/useSize";
import s from "./views.module.css";

export interface FlowState {
  steps: RowStep[];
  after: boolean;
  /** A sidenote on one step's line. */
  note?: { step: string; text: string };
}

interface Props {
  states: Record<string, FlowState>;
  rowH?: number;
  labelW?: number;
  noteW?: number;
  span?: readonly [number, number];
}

export function ScrubRowFlow({ states, rowH = 34, labelW = 262, noteW = 210, span }: Props) {
  const [wrap, width] = useWidth<HTMLDivElement>(700);
  const barW = Math.max(200, width - labelW - noteW - 90);
  const { map, reg } = useRegistry<HTMLElement>();
  const nMax = useMemo(
    () => Math.max(...Object.values(states).flatMap((st) => st.steps.map((x) => x.n + x.dropped))),
    [states],
  );
  const maxSteps = Math.max(...Object.values(states).map((st) => st.steps.length));

  const sceneOf = useMemo(() => {
    return cached((state: string): Scene => {
      const st = states[state] ?? states.now!;
      const sb = new SceneBuilder();
      st.steps.forEach((step, i) => {
        const w = (step.n / nMax) * barW;
        const dw = (step.dropped / nMax) * barW;
        sb.set(`step:${step.key}`, { y: i * rowH, o: 1 });
        sb.set(`bar:${step.key}`, { w, c: st.after ? 1 : 0 });
        sb.set(`lab:${step.key}:${step.label}`, { o: 1 });
        sb.set(`n:${step.key}:${step.n}`, { o: 1, slide: 6 });
        if (step.dropped > 0) {
          sb.set(`drop:${step.key}`, { x: w, w: Math.max(2, dw), o: 1 });
          sb.set(`dropn:${step.key}:${step.dropped}`, { x: w + Math.max(2, dw) + 6, o: 1 });
        }
      });
      if (st.note) {
        const i = st.steps.findIndex((x) => x.key === st.note!.step);
        if (i >= 0) sb.set(`sn:${st.note.text}`, { y: i * rowH, o: 1 });
      }
      return sb.scene;
    });
  }, [states, nMax, barW, rowH]);

  const keys = useMemo(() => unionKeys(Object.keys(states).map((k) => sceneOf(k))), [states, sceneOf]);

  const apply = useCallback(
    (sc: Scene) => {
      for (const [k, el] of map.current) {
        const it = sc.items.get(k);
        const kind = k.slice(0, k.indexOf(":"));
        if (kind === "bar") {
          css(el, { width: `${it?.w ?? 0}px` });
          css(el, { background: `color-mix(in oklab, var(--c2) ${Math.round((it?.c ?? 0) * 100)}%, var(--c4))` });
        } else if (kind === "drop") {
          place(el, it, { y: false });
          if (it) css(el, { width: `${it.w}px` });
        } else if (kind === "dropn") place(el, it, { y: false });
        else if (kind === "step" || kind === "sn") place(el, it, { x: false });
        else place(el, it, { x: false, y: false });
      }
    },
    [map],
  );

  useMorph({ sceneOf, apply, span });

  const stepKeys = keys.filter((k) => k.startsWith("step:")).map((k) => k.slice(5));
  const sub = (prefix: string, key: string) =>
    keys.filter((k) => k.startsWith(`${prefix}:${key}:`)).map((k) => k.slice(prefix.length + key.length + 2));

  return (
    <div ref={wrap} className={s.flow} style={{ height: maxSteps * rowH }}>
      {stepKeys.map((key) => (
        <div key={key} ref={reg(`step:${key}`)} className={s.step} style={{ height: rowH, opacity: 0 }}>
          <span className={s.stepLabel} style={{ width: labelW }}>
            {sub("lab", key).map((t) => (
              <span key={t} ref={reg(`lab:${key}:${t}`)} className={s.stack} style={{ opacity: 0 }}>
                <Prose text={t} />
              </span>
            ))}
          </span>
          <span className={s.barTrack} style={{ width: barW }}>
            <span ref={reg(`bar:${key}`)} className={s.flowBar} />
            <span ref={reg(`drop:${key}`)} className={s.flowDrop} style={{ opacity: 0 }} />
            {sub("dropn", key).map((t) => (
              <span key={t} ref={reg(`dropn:${key}:${t}`)} className={s.flowDropN} style={{ opacity: 0 }}>
                −{fmtInt(Number(t))}
              </span>
            ))}
          </span>
          <span className={s.stepN}>
            {sub("n", key).map((t) => (
              <span key={t} ref={reg(`n:${key}:${t}`)} className={s.stack} style={{ opacity: 0 }}>
                {fmtInt(Number(t))}
              </span>
            ))}
          </span>
        </div>
      ))}
      {keys
        .filter((k) => k.startsWith("sn:"))
        .map((k) => (
          <div
            key={k}
            ref={reg(k)}
            className={s.flowNote}
            style={{ height: rowH, left: labelW + barW + 96, opacity: 0 }}
          >
            <Prose text={k.slice(3)} />
          </div>
        ))}
    </div>
  );
}
