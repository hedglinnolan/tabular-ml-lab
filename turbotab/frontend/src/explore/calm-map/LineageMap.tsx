/**
 * The map's band: the analysis as a compact lineage, in one band of at most 120 px. Each stage of
 * the stage bar is a region (its name over its stretch of the line); each decision of the walk is a
 * node on it, solid once stated in the record, open while it is asked, faint while it waits for an
 * earlier answer; the decision on the card is the ink one. Clicking a node opens its card (the
 * walk's own `open`, so only what the walk can reach). Pointing at a node, or focusing it, names
 * its question and the answer it holds, in the kit's tooltip. Arrow keys move along the line.
 *
 * Every word on it is the kit's: the stage names, the questions and the option names. Where the
 * band is too narrow for every decision (720 px and narrower), the stages other than the open one
 * fold to one mark each, as the stage bar folds its names on a phone.
 */
import { useLayoutEffect, useRef, useState, type CSSProperties, type KeyboardEvent } from "react";
import { STEP_BY_ID, isResult, kit as k, optionOf, plain, reachable, stepsOfStage, type WalkApi, type WalkState } from "../calm-kit";
import m from "./map.module.css";

type Status = "stated" | "asked" | "waiting";

interface MapNode {
  id: string;
  status: Status;
  /** Its question and the answer it holds (or that it is not answered yet). */
  label: string;
}

/** The result moments' names, as their cards title them. */
const RESULT_NAME: Record<string, string> = {
  table2: "Table 2: the estimate in each declared model",
  mattered: "Which of my decisions mattered?",
};

function statusOf(s: WalkState, id: string): Status {
  if (isResult(id)) return s.locked ? "stated" : "waiting";
  if (id in s.answers) return "stated";
  return reachable(s, id) ? "asked" : "waiting";
}

function labelOf(s: WalkState, id: string, status: Status): string {
  if (isResult(id)) return RESULT_NAME[id] ?? id;
  const step = STEP_BY_ID[id]!;
  const question = plain(step.question);
  if (status === "stated") return `${question} · ${plain(optionOf(step, s.answers[id])?.name)}`;
  if (status === "waiting") return `${question} · Waits for an earlier answer`;
  return `${question} · Not answered yet`;
}

/** A stage folded to one mark: stated when every decision in it is, waiting when none can open. */
function stageStatus(nodes: MapNode[]): Status {
  if (nodes.every((n) => n.status === "stated")) return "stated";
  return nodes.some((n) => n.status !== "waiting") ? "asked" : "waiting";
}

/** Where the tooltip points: a node (its words follow the walk) and its place on screen. */
interface Tip {
  id: string;
  x: number;
  y: number;
}

/** The kit's tooltip (k.tip), under the node, kept inside the window. */
function NodeTip({ text, x, y }: { text: string; x: number; y: number }) {
  const ref = useRef<HTMLDivElement>(null);
  useLayoutEffect(() => {
    const el = ref.current;
    if (!el) return;
    el.style.left = `${x}px`;
    const r = el.getBoundingClientRect();
    const room = document.documentElement.clientWidth - 8;
    if (r.left < 8) el.style.left = `${x + 8 - r.left}px`;
    else if (r.right > room) el.style.left = `${x - (r.right - room)}px`;
  }, [text, x, y]);
  return (
    <div ref={ref} className={k.tip} style={{ left: x, top: y }} aria-hidden="true" data-testid="map-tip">
      {text}
    </div>
  );
}

export function LineageMap({ walk }: { walk: WalkApi }) {
  const s = walk.state;
  const band = useRef<HTMLOListElement>(null);
  const [tip, setTip] = useState<Tip | null>(null);
  // The one tab stop on the line: the node last focused on this card, else the decision on it.
  const [stop, setStop] = useState<{ at: string; id: string } | null>(null);

  const regions = walk.chain.map((c) => {
    const nodes = stepsOfStage(c.id).map((id) => {
      const status = statusOf(s, id);
      return { id, status, label: labelOf(s, id, status) };
    });
    return { ...c, nodes, current: c.status === "current", reach: nodes.some((n) => n.status !== "waiting") };
  });
  const tabStop = stop?.at === s.open ? stop.id : s.open;
  const tipText = tip && regions.flatMap((r) => r.nodes).find((n) => n.id === tip.id)?.label;

  const show = (el: HTMLElement, id: string) => {
    const r = el.getBoundingClientRect();
    setTip({ id, x: r.left + r.width / 2, y: r.bottom + 40 });
  };
  const hide = () => setTip(null);

  /** Arrow keys walk the line: every node (and folded stage) that can open, as drawn. */
  const onKey = (e: KeyboardEvent<HTMLOListElement>) => {
    const keys = ["ArrowRight", "ArrowLeft", "Home", "End"];
    if (!keys.includes(e.key) || !band.current) return;
    const all = Array.from(band.current.querySelectorAll<HTMLButtonElement>("button[data-nav]")).filter(
      (b) => !b.disabled && b.offsetParent !== null,
    );
    const at = all.indexOf(document.activeElement as HTMLButtonElement);
    if (at < 0) return;
    e.preventDefault();
    const to = e.key === "Home" ? 0 : e.key === "End" ? all.length - 1 : at + (e.key === "ArrowRight" ? 1 : -1);
    all[Math.max(0, Math.min(all.length - 1, to))]?.focus();
  };

  return (
    <nav aria-label="The analysis, decision by decision" data-testid="map">
      <ol className={m.map} ref={band} onKeyDown={onKey}>
        {regions.map((r) => {
          const folded = stageStatus(r.nodes);
          return (
            <li
              key={r.id}
              className={m.region}
              style={{ "--n": Math.max(2, r.nodes.length) } as CSSProperties}
              data-current={r.current || undefined}
              data-reach={r.reach}
              data-testid={`map-region-${r.id}`}
            >
              <span className={m.name}>{r.label}</span>
              <button
                type="button"
                className={m.stage}
                data-nav
                data-status={folded}
                disabled={folded === "waiting"}
                tabIndex={-1}
                aria-label={r.label}
                onClick={() => walk.open(r.nodes.find((n) => n.status === "asked")?.id ?? r.first)}
                data-testid={`map-stage-${r.id}`}
              >
                <i />
              </button>
              <ol className={m.nodes} aria-label={r.label}>
                {r.nodes.map((n) => {
                  const open = n.id === s.open;
                  return (
                    // The tooltip listens on the item, so a waiting (disabled) node still names itself.
                    <li key={n.id} onPointerEnter={(e) => show(e.currentTarget, n.id)} onPointerLeave={hide}>
                      <button
                        type="button"
                        className={m.node}
                        data-nav
                        data-status={n.status}
                        data-testid={`map-node-${n.id}`}
                        aria-current={open ? "step" : undefined}
                        aria-label={n.label}
                        disabled={n.status === "waiting"}
                        tabIndex={n.id === tabStop ? 0 : -1}
                        onClick={() => walk.open(n.id)}
                        onFocus={(e) => {
                          setStop({ at: s.open, id: n.id });
                          show(e.currentTarget, n.id);
                        }}
                        onBlur={hide}
                      >
                        <i />
                      </button>
                    </li>
                  );
                })}
              </ol>
            </li>
          );
        })}
      </ol>
      {tip && tipText ? <NodeTip text={tipText} x={tip.x} y={tip.y} /> : null}
    </nav>
  );
}
