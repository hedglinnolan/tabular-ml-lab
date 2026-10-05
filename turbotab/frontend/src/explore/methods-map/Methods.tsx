/**
 * The methods text, read alongside the map: the same decisions as sentences, by the same regions
 * (the guideline's sections). It is derived, never typed: each line is the engine's sentence for
 * an answer in force; an asked slot is a gap that names itself; a section still waiting says on
 * what. Clicking a line opens its node on the map. A change after the lock is kept and marked.
 *
 * Opening a node (a press whose purpose is to go there) brings its first line into view, once;
 * nothing else moves the text.
 */
import { useEffect, useRef, type KeyboardEvent } from "react";
import { Rich } from "../../components/stage/text";
import { NODE_TITLE, REGIONS, record, type Answers, type NodeId, type Purpose } from "./model";
import c from "./screen.module.css";

const SOURCE: NodeId[] = ["lens", "outcome", "purpose", "grain"];

export function Methods({
  a,
  purpose,
  focus,
  walking,
  onFocus,
  fresh,
}: {
  a: Answers;
  purpose: Purpose;
  focus: NodeId | null;
  walking: NodeId | null;
  onFocus: (n: NodeId) => void;
  /** The sentence just recorded, marked for a moment. */
  fresh: string | null;
}) {
  const sections = record(a, purpose);
  const regions = REGIONS[purpose];
  const root = useRef<HTMLElement>(null);
  useEffect(() => {
    if (!focus) return;
    const target = focus === "source" ? "lens" : focus;
    root.current?.querySelector(`[data-node="${target}"]`)?.scrollIntoView({ block: "nearest" });
  }, [focus]);
  const key = (n: NodeId) => (e: KeyboardEvent) => {
    if (e.key === "Enter") {
      e.preventDefault();
      onFocus(n);
    }
  };
  return (
    <aside ref={root} className={c.methods} aria-label="The methods text" data-testid="methods">
      <p className={c.methodsKicker}>Methods · the record</p>
      {sections.map((s) => {
        const r = regions.find((x) => x.id === s.region)!;
        return (
          <section key={s.region} className={c.mSection}>
            <h3 className={c.mHead}>
              {r.title} <span className={c.items}>{r.items}</span>
            </h3>
            {s.lines.length === 0 ? (
              <p className={c.mWait}>Waits on the lock.</p>
            ) : (
              s.lines.map((l, i) => {
                const lit = !walking || walking === l.node;
                const focused = focus === l.node || (focus === "source" && SOURCE.includes(l.node));
                if (l.text === null) {
                  return (
                    <button
                      key={i}
                      type="button"
                      className={c.mSlot}
                      data-tier={l.tier}
                      data-node={l.node}
                      data-focus={focused || undefined}
                      data-dim={!lit || undefined}
                      onClick={() => onFocus(l.node)}
                      data-testid={`slot-${l.node}`}
                    >
                      {l.tier === "waiting" ? `${NODE_TITLE[l.node]}: waiting` : `${NODE_TITLE[l.node]}: asked of you`}
                    </button>
                  );
                }
                return (
                  <p
                    key={i}
                    className={c.mLine}
                    role="button"
                    tabIndex={0}
                    data-tier={l.tier}
                    data-node={l.node}
                    data-after={l.after || undefined}
                    data-focus={focused || undefined}
                    data-dim={!lit || undefined}
                    data-fresh={fresh === l.text || undefined}
                    onClick={() => onFocus(l.node)}
                    onKeyDown={key(l.node)}
                  >
                    {l.after ? <span className={c.afterTag}>after the estimates</span> : null}
                    {l.tier === "silent" ? <span className={c.silentTag}>export only</span> : null}
                    <Rich text={l.text} />
                  </p>
                );
              })
            )}
          </section>
        );
      })}
    </aside>
  );
}
