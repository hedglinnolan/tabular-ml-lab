/**
 * The objective list: the methods section's sections in the guideline's order, each with what it
 * still holds open. A finished section collapses to its sentence; the current one opens to its
 * objectives; a section that waits says on what. The tally and the segmented bar are the whole
 * quest at a glance: settled, open now, waiting, and the items only the author can supply.
 */
import { Rich } from "../../components/stage/text";
import { progressOf, type Guideline, type Item, type Section } from "./sections";
import s from "./questlog.module.css";

interface Props {
  guideline: Guideline;
  sections: Section[];
  current: { section: string; item: string } | null;
  /** The item whose card moves the walk on. */
  objective: string | null;
  locked: boolean;
  docOpen: boolean;
  onPick: (section: string, item: string) => void;
  onNext: () => void;
  onDoc: () => void;
}

const MARK: Record<Item["tier"], string> = {
  stated: "✓",
  engine: "✓",
  asked: "●",
  waiting: "○",
  silent: "–",
  author: "◇",
};

function Ring({ section, current }: { section: Section; current: boolean }) {
  const counted = section.items.filter((i) => i.tier !== "silent" && i.tier !== "author");
  const done = counted.filter((i) => i.tier === "stated" || i.tier === "engine").length;
  const frac = counted.length ? done / counted.length : 1;
  if (!counted.length)
    // Only the author can supply it: a hollow diamond, never a check (nothing was recorded).
    return (
      <svg className={s.glyph} viewBox="0 0 16 16" aria-hidden="true">
        <path d="M8 2.2l5.8 5.8L8 13.8 2.2 8z" fill="none" stroke="var(--faint)" strokeWidth="1.4" strokeDasharray="2.2 1.6" />
      </svg>
    );
  if (section.done) {
    // Green only where a person recorded something; a section the engine settled alone is neutral.
    const human = section.items.some((i) => i.tier === "stated");
    return (
      <svg className={s.glyph} viewBox="0 0 16 16" aria-hidden="true">
        <circle cx="8" cy="8" r="7" fill={human ? "var(--ok)" : "none"} stroke={human ? "none" : "var(--faint)"} strokeWidth="1.4" />
        <path
          d="M4.6 8.3l2.2 2.2 4.6-4.8"
          fill="none"
          stroke={human ? "var(--surface)" : "var(--faint)"}
          strokeWidth="1.8"
          strokeLinecap="round"
          strokeLinejoin="round"
        />
      </svg>
    );
  }
  const c = 2 * Math.PI * 6;
  return (
    <svg className={s.glyph} viewBox="0 0 16 16" aria-hidden="true">
      <circle cx="8" cy="8" r="6" fill="none" stroke="var(--line)" strokeWidth="2.2" />
      <circle
        cx="8"
        cy="8"
        r="6"
        fill="none"
        stroke={current || section.open ? "var(--accent)" : "var(--faint)"}
        strokeWidth="2.2"
        strokeDasharray={`${c * frac} ${c}`}
        transform="rotate(-90 8 8)"
        strokeLinecap="round"
      />
      {section.open ? <circle cx="8" cy="8" r="2.4" fill="var(--accent)" /> : null}
    </svg>
  );
}

function badge(section: Section) {
  if (section.open) return <span className={s.badgeOpen}>{section.open} open</span>;
  if (section.waiting) return <span className={s.badgeWait}>waiting</span>;
  if (section.done && section.author === section.items.length) return <span className={s.badgeYours}>yours</span>;
  if (section.done && !section.items.some((i) => i.tier === "stated")) return <span className={s.badgeWait}>read</span>;
  if (section.done) return <span className={s.badgeDone}>done</span>;
  return null;
}

/** What a finished section reads as, collapsed: its first stated sentence. */
function summaryOf(section: Section): string | null {
  for (const i of section.items) if ((i.tier === "stated" || i.tier === "engine") && i.sentences[0]) return i.sentences[0];
  const yours = section.items.filter((i) => i.tier === "author").length;
  return yours ? `Only you can supply ${yours === 1 ? "this item" : `these ${yours} items`}.` : null;
}

function waitingOf(section: Section): string | null {
  const names = [...new Set(section.items.flatMap((i) => (i.tier === "waiting" ? i.waitingOn : [])))];
  return names.length ? `after ${names.slice(0, 2).join(" and ")}` : null;
}

function objectiveNote(i: Item) {
  if (i.tier === "asked" && i.optional) return <span className={s.objNote}>optional</span>;
  if (i.tier === "asked") return <span className={s.objNoteOpen}>{i.count > 1 ? `${i.count} open` : "open"}</span>;
  if (i.tier === "waiting") return <span className={s.objNote}>waits</span>;
  if (i.tier === "author") return <span className={s.objNote}>yours</span>;
  if (i.tier === "engine") return <span className={s.objNote}>read</span>;
  if (i.tier === "silent") return <span className={s.objNote}>silent</span>;
  return null;
}

export function Rail({ guideline, sections, current, objective, locked, docOpen, onPick, onNext, onDoc }: Props) {
  const p = progressOf(sections);
  const ticks = sections.flatMap((sec) =>
    sec.items
      .filter((i) => i.tier !== "silent")
      .map((i) => ({ key: `${sec.key}:${i.key}`, tier: i.tier })),
  );
  return (
    <nav className={s.rail} aria-label="The methods section, as objectives" data-testid="rail">
      <div className={s.railHead}>
        <span className={s.kicker}>Methods · {guideline}</span>
        <h2 className={s.railTitle}>{locked ? "Plan locked" : p.open ? "What is left" : "Nothing open now"}</h2>
        <div className={s.bar} aria-hidden="true">
          {ticks.map((t) => (
            <span key={t.key} className={s.tick} data-tier={t.tier} />
          ))}
        </div>
        <div className={s.tally} data-testid="tally">
          <span>
            <b>{p.settled}</b> settled
          </span>
          {p.open ? (
            <span className={s.tallyOpen}>
              <b>{p.open}</b> open now
            </span>
          ) : null}
          {p.waiting ? (
            <span>
              <b>{p.waiting}</b> waiting
            </span>
          ) : null}
          {p.author ? (
            <span>
              <b>{p.author}</b> only you supply
            </span>
          ) : null}
        </div>
      </div>
      <ol className={s.sections}>
        {sections.map((sec) => {
          const isCurrent = current?.section === sec.key;
          const state = sec.done ? "done" : sec.open ? "open" : "waiting";
          const expand = isCurrent && !docOpen;
          const summary = sec.done ? summaryOf(sec) : null;
          const wait = !sec.done && !sec.open && !isCurrent ? waitingOf(sec) : null;
          const firstObjective = sec.items.find((i) => i.key === objective) ?? sec.items.find((i) => i.tier === "asked") ?? sec.items[0]!;
          return (
            <li key={sec.key} className={s.section} data-current={isCurrent || undefined} data-state={state}>
              <button
                type="button"
                className={s.sectionButton}
                data-testid={`section-${sec.key}`}
                onClick={() => onPick(sec.key, firstObjective.key)}
                aria-current={isCurrent ? "step" : undefined}
                aria-label={`${sec.title}: ${sec.open ? `${sec.open} open` : sec.done ? "done" : "waiting"}. Go there.`}
              >
                <Ring section={sec} current={isCurrent} />
                <span className={s.sectionName}>{sec.title}</span>
                {badge(sec)}
              </button>
              {expand ? (
                <ul className={s.objectives}>
                  {sec.items.map((i) => (
                    <li key={i.key}>
                      <button
                        type="button"
                        className={s.objective}
                        data-tier={i.tier}
                        data-now={current?.item === i.key || undefined}
                        data-objective={i.key === objective || undefined}
                        data-testid={`item-${i.key}`}
                        onClick={() => onPick(sec.key, i.key)}
                      >
                        <span className={s.objMark} data-tier={i.tier} aria-hidden="true">
                          {MARK[i.tier]}
                        </span>
                        <span>{i.title}</span>
                        {objectiveNote(i)}
                      </button>
                    </li>
                  ))}
                </ul>
              ) : summary ? (
                <p className={s.sentenceLine}>
                  <Rich text={summary} />
                </p>
              ) : wait ? (
                <p className={s.waitLine}>{wait}</p>
              ) : null}
            </li>
          );
        })}
      </ol>
      <div className={s.railFoot}>
        {!locked && p.open ? (
          <button type="button" className={s.railButtonPrimary} onClick={onNext} data-testid="next-objective">
            Next objective <span className={s.key}>N</span>
          </button>
        ) : null}
        <button type="button" className={s.railButton} onClick={onDoc} aria-pressed={docOpen} data-testid="open-doc">
          {docOpen ? (locked ? "Back to the results" : "Back to the objective") : "Read the whole methods section"} <span className={s.key}>M</span>
        </button>
      </div>
    </nav>
  );
}
