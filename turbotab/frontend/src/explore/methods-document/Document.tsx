/**
 * The Record as a typeset methods section (angle A, document-first). Sections by the purpose's
 * guideline; each paragraph a run-in head and the server's sentence; changeable phrases marked in
 * place; open slots set where their sentence will stand. A gutter on the left carries each
 * paragraph's tier (recorded by you, stated by TurboTab, asked, waiting, yours to write) and the
 * right margin its checklist item, so the document is also the STROBE-nut / TRIPOD+AI checklist.
 *
 * "Next slot" walks a newcomer through the open slots one at a time: the focused slot opens in
 * place and the rest of the document dims to a map. An expert ignores it, reads, and edits any
 * marked phrase directly.
 */
import { Fragment, useState, type ReactNode } from "react";
import { Rich } from "../../components/stage/text";
import { cx } from "../../util/format";
import { segments, sentences, type Doc, type Mark, type Para, type Section } from "./model";
import s from "./doc.module.css";

const TIER_WORDS: Record<Para["tier"], string> = {
  recorded: "Recorded by you",
  stated: "Stated by TurboTab from your data",
  asked: "Asked: a slot to fill",
  waiting: "Waiting on an earlier answer",
  author: "Only you can write this",
  silent: "Changes no number: in the export only",
};

export interface DocHandlers {
  focus: string | null;
  edit: string | null;
  concept: string | null;
  onNext: () => void;
  onExit: () => void;
  onFocus: (id: string) => void;
  onPhrase: (id: string) => void;
  onConcept: (key: string) => void;
  /** Go to a section or paragraph (a navigation press: the one scroll it makes). */
  onGo: (id: string) => void;
  renderSlot: (p: Para) => ReactNode;
  renderEdit: (p: Para) => ReactNode;
  renderResults: (p: Para) => ReactNode;
  renderConcept: (p: Para, key: string) => ReactNode;
  /** What an asked slot says inline, where its sentence will stand. */
  slotLabel: (p: Para) => string;
}

export function DocumentPane({ doc, h, title }: { doc: Doc; h: DocHandlers; title: string }) {
  const focusMode = h.focus !== null || h.edit !== null;
  const idx = h.focus ? doc.objectives.indexOf(h.focus) : -1;
  const next = doc.objectives.find((id) => id !== h.focus) ?? null;
  const nextPara = next ? doc.sections.flatMap((x) => x.paras).find((p) => p.id === next) : null;
  return (
    <div className={s.docPane} data-focus-mode={focusMode || undefined}>
      <div className={s.objectives} role="toolbar" aria-label="The draft and its open slots">
        {h.focus ? (
          <>
            <span className={s.objKicker}>
              Slot {idx + 1} of {doc.objectives.length}
            </span>
            <button type="button" className={s.objQuiet} onClick={h.onExit}>
              Read the whole draft <kbd>Esc</kbd>
            </button>
            {nextPara ? (
              <button type="button" className={s.objNext} onClick={h.onNext}>
                Next: {nextPara.head.toLowerCase()}
              </button>
            ) : null}
          </>
        ) : (
          <>
            <span className={s.objKicker}>Methods · {doc.guideline}</span>
            <span className={s.objStatus}>
              <b>{doc.counts.written}</b> written · <b>{doc.counts.open}</b> open
              {doc.counts.waiting ? ` · ${doc.counts.waiting} waiting` : ""} · {doc.counts.author} yours
            </span>
            {doc.objectives.length ? (
              <button type="button" className={s.objNext} onClick={h.onNext} data-testid="next-slot">
                Next: {nextPara?.head.toLowerCase() ?? ""}
              </button>
            ) : (
              <span className={s.objDone}>No slot is open</span>
            )}
          </>
        )}
      </div>
      <article className={s.paper} aria-label={title}>
        <header className={s.paperHead}>
          <h1 className={s.paperTitle}>{title}</h1>
          <p className={s.paperMeta}>
            Drafted from your decisions, organized by {doc.guideline}. Underlined phrases can be changed; boxed ones
            are asked.
          </p>
          <DocMap doc={doc} onGo={h.onGo} />
          <Legend />
        </header>
        {doc.sections.map((sec) => (
          <SectionView key={sec.id} sec={sec} h={h} />
        ))}
      </article>
    </div>
  );
}

/** The map (§11.4 rule 5): the whole section's shape at a glance, one mark per paragraph by tier,
 *  grouped by the guideline's sections; a section's name goes there (a navigation press). */
function DocMap({ doc, onGo }: { doc: Doc; onGo: (id: string) => void }) {
  return (
    <nav className={s.map} aria-label="The draft at a glance">
      {doc.sections.map((sec) => (
        <button key={sec.id} type="button" className={s.mapSection} onClick={() => onGo(`sec-${sec.id}`)}>
          <span className={s.mapName}>{sec.title}</span>
          <span className={s.mapMarks} aria-hidden="true">
            {sec.paras.map((p) => (
              <span key={p.id} className={s.mapMark} data-tier={p.tier} />
            ))}
          </span>
        </button>
      ))}
    </nav>
  );
}

function Legend() {
  return (
    <ul className={s.legend} aria-label="What the margin marks mean">
      {(["recorded", "stated", "asked", "waiting", "author"] as const).map((t) => (
        <li key={t}>
          <span className={s.mark} data-tier={t} aria-hidden="true" />
          {TIER_WORDS[t].split(":")[0]}
        </li>
      ))}
    </ul>
  );
}

function SectionView({ sec, h }: { sec: Section; h: DocHandlers }) {
  const lit = sec.paras.some((p) => p.id === h.focus || p.id === h.edit);
  return (
    <section className={s.section} id={`sec-${sec.id}`} data-lit={lit || undefined}>
      <h2 className={s.sectionTitle}>
        <span>{sec.title}</span>
        <span className={s.sectionItem}>{sec.item}</span>
      </h2>
      {sec.paras.map((p) => (
        <ParaView key={p.id} p={p} h={h} />
      ))}
      {sec.silent.length ? (
        <details className={s.silent}>
          <summary>
            {sec.silent.length} {sec.silent.length === 1 ? "question" : "questions"} did not apply · in the export only
          </summary>
          <ul>
            {sec.silent.map((x) => (
              <li key={x.head}>
                <span className={s.silentHead}>{x.head}.</span> <Rich text={x.reason} />
              </li>
            ))}
          </ul>
        </details>
      ) : null}
    </section>
  );
}

function ParaView({ p, h }: { p: Para; h: DocHandlers }) {
  const focused = h.focus === p.id;
  const editing = h.edit === p.id;
  const lit = focused || editing;
  return (
    <div className={s.para} id={`para-${p.id}`} data-tier={p.tier} data-lit={lit || undefined} data-after={p.afterEstimates || undefined}>
      <span className={s.mark} data-tier={p.tier} title={TIER_WORDS[p.tier]} aria-label={TIER_WORDS[p.tier]} role="img" />
      <div className={s.paraBody}>
        <p className={s.paraText}>
          <span className={s.runIn}>{p.head}.</span> <Body p={p} h={h} focused={focused} />
        </p>
        {editing ? h.renderEdit(p) : null}
        {p.marks
          .filter((m) => m.kind === "concept" && m.concept === h.concept)
          .map((m) => (
            <Fragment key={m.text}>{h.renderConcept(p, m.concept!)}</Fragment>
          ))}
        {p.subs.length ? (
          <ul className={s.subs}>
            {p.subs.map((t, i) => (
              <li key={i}>
                <Rich text={t} />
              </li>
            ))}
          </ul>
        ) : null}
        {p.fold ? (
          <details className={s.fold}>
            <summary>{p.fold.label}, no question asked</summary>
            <p>
              <Rich text={p.fold.text} />
            </p>
          </details>
        ) : null}
        {focused ? h.renderSlot(p) : null}
        {p.id.startsWith("results-") && p.tier === "recorded" ? h.renderResults(p) : null}
      </div>
      {p.item ? <span className={s.item}>{p.item}</span> : null}
    </div>
  );
}

function Body({ p, h, focused }: { p: Para; h: DocHandlers; focused: boolean }) {
  const [whole, setWhole] = useState(false);
  switch (p.tier) {
    case "recorded":
    case "stated": {
      if (!p.text) return null;
      const all = sentences(p.text);
      const long = all.length > 2 && p.text.split(/\s+/).length > 70;
      const shown = long && !whole ? all.slice(0, 2).join(" ") : p.text;
      return (
        <>
          <Marked text={shown} marks={p.marks} h={h} id={p.id} />
          {long ? (
            <>
              {" "}
              <button type="button" className={s.more} onClick={() => setWhole((v) => !v)} aria-expanded={whole}>
                {whole ? "fewer" : `${all.length - 2} more ${all.length - 2 === 1 ? "sentence" : "sentences"}`}
              </button>
            </>
          ) : null}
        </>
      );
    }
    case "asked":
      return (
        <button
          type="button"
          className={cx(s.slotInline, focused && s.slotInlineOn)}
          onClick={() => h.onFocus(p.id)}
          aria-expanded={focused}
        >
          {h.slotLabel(p)}
        </button>
      );
    case "waiting":
      return (
        <span className={s.waiting}>
          {p.text ? <Rich text={p.text} /> : <>Waits on {lowerFirst(p.waitingOn ?? "")}.</>}
        </span>
      );
    case "author":
      return (
        <span className={s.author}>
          <span className={s.authorKicker}>Yours to write</span> {p.text}
        </span>
      );
    default:
      return null;
  }
}

function lowerFirst(t: string): string {
  return t.charAt(0).toLowerCase() + t.slice(1);
}

function Marked({ text, marks, h, id }: { text: string; marks: Mark[]; h: DocHandlers; id: string }) {
  return (
    <>
      {segments(text, marks).map((seg, i) =>
        seg.mark?.kind === "phrase" ? (
          <button
            key={i}
            type="button"
            className={s.phrase}
            data-open={h.edit === id || undefined}
            onClick={() => h.onPhrase(id)}
            aria-label={`Change: ${seg.text.replace(/`/g, "")}`}
          >
            <Rich text={seg.text} />
          </button>
        ) : seg.mark?.kind === "concept" ? (
          <button
            key={i}
            type="button"
            className={s.term}
            data-open={h.concept === seg.mark.concept || undefined}
            onClick={() => h.onConcept(seg.mark!.concept!)}
          >
            <Rich text={seg.text} />
          </button>
        ) : (
          <Rich key={i} text={seg.text} />
        ),
      )}
    </>
  );
}
