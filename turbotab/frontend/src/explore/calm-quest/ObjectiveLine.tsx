/**
 * The objective line: the quest log's interface, where the kit puts its chain. The methods
 * section's guideline sections in order, each with how many of its objectives are recorded; the
 * one the card is on is marked as the chain marks its stage. Choosing a section discloses its
 * objectives beneath it (recorded ones marked, the open one ready, later ones waiting); choosing
 * one opens it on the card. "Next objective" returns the card to the open slot. The look is the
 * chain's own (kit.module.css); quest.module.css only places it.
 */
import { useEffect, useId, useMemo, useRef, useState } from "react";
import { kit as k, type SectionId, type WalkApi } from "../calm-kit";
import { nextObjective, questSections, type QuestSection } from "./objectives";
import q from "./quest.module.css";

export function ObjectiveLine({ walk }: { walk: WalkApi }) {
  const sections = useMemo(
    () => questSections(walk.manuscript, walk.state.open),
    [walk.manuscript, walk.state.open],
  );
  const [shown, setShown] = useState<{ id: SectionId; left: number } | null>(null);
  const wrap = useRef<HTMLDivElement>(null);
  const buttons = useRef<Partial<Record<SectionId, HTMLButtonElement | null>>>({});
  const menuId = useId();
  const next = nextObjective(walk.state);

  // A pointer outside the line or Escape closes the disclosed objectives.
  useEffect(() => {
    if (!shown) return;
    const away = (e: PointerEvent) => {
      if (wrap.current && !wrap.current.contains(e.target as Node)) setShown(null);
    };
    const esc = (e: KeyboardEvent) => {
      if (e.key !== "Escape") return;
      buttons.current[shown.id]?.focus();
      setShown(null);
    };
    document.addEventListener("pointerdown", away);
    window.addEventListener("keydown", esc);
    return () => {
      document.removeEventListener("pointerdown", away);
      window.removeEventListener("keydown", esc);
    };
  }, [shown]);

  const toggle = (sec: QuestSection) => {
    if (sec.id === "results") {
      setShown(null);
      walk.open("table2");
      return;
    }
    if (shown?.id === sec.id) {
      setShown(null);
      return;
    }
    const b = buttons.current[sec.id];
    const box = wrap.current?.getBoundingClientRect();
    const at = b && box ? b.getBoundingClientRect().left - box.left : 0;
    // Keep the disclosure inside the line: it is at most 380 px wide.
    const left = box ? Math.max(0, Math.min(at, box.width - Math.min(380, box.width))) : 0;
    setShown({ id: sec.id, left });
  };

  const open = sections.find((s) => s.id === shown?.id) ?? null;

  return (
    <div className={q.line} ref={wrap}>
      <nav aria-label="The methods section, objective by objective">
        <ol className={k.chain} data-testid="objectives">
          {sections.map((sec) => {
            const total = sec.objectives.length;
            const results = sec.id === "results";
            const status = sec.current
              ? "current"
              : !results && sec.done === total
                ? "done"
                : "waiting";
            return (
              <li key={sec.id} data-status={status}>
                <button
                  type="button"
                  ref={(el) => {
                    buttons.current[sec.id] = el;
                  }}
                  className={k.chainBtn}
                  data-status={status}
                  data-testid={`section-${sec.id}`}
                  aria-current={sec.current ? "step" : undefined}
                  aria-expanded={results ? undefined : shown?.id === sec.id}
                  aria-controls={!results && shown?.id === sec.id ? menuId : undefined}
                  disabled={results && !walk.state.locked}
                  onClick={() => toggle(sec)}
                >
                  <span className={k.chainName}>{sec.title}</span>
                  {results ? null : (
                    <span className={q.count} data-testid={`count-${sec.id}`}>
                      {sec.done} of {total}
                    </span>
                  )}
                </button>
              </li>
            );
          })}
        </ol>
      </nav>
      {next ? (
        <button
          type="button"
          className={k.linkish}
          onClick={() => walk.open(next)}
          data-testid="next-objective"
        >
          Next objective
        </button>
      ) : null}
      {open ? (
        <div
          className={q.menu}
          id={menuId}
          style={{ left: shown!.left }}
          data-testid={`objectives-${open.id}`}
        >
          <p className={q.menuHead}>
            {open.title} <small>{open.item}</small>
          </p>
          <ul className={q.items} aria-label={`${open.title}: its objectives`}>
            {open.objectives.map((o) => (
              <li key={o.id}>
                <button
                  type="button"
                  className={k.chainBtn}
                  data-status={o.current ? "current" : o.status === "done" ? "done" : "waiting"}
                  aria-current={o.current ? "step" : undefined}
                  disabled={!o.step}
                  data-testid={`objective-${o.id}`}
                  autoFocus={
                    o ===
                    (open.objectives.find((x) => x.current) ?? open.objectives.find((x) => x.step))
                  }
                  onClick={() => {
                    walk.open(o.step!);
                    setShown(null);
                  }}
                >
                  {o.name}
                </button>
              </li>
            ))}
          </ul>
        </div>
      ) : null}
    </div>
  );
}
