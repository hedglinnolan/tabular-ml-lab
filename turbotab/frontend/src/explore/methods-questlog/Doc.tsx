/**
 * The whole methods section, one press away: the sections in the guideline's order, each a
 * paragraph of the engine's sentences (verbatim) with the changeable phrase marked; what the
 * engine stated without asking in the muted voice; open and waiting slots as dashed blanks; the
 * items only the author can supply as their own blanks; silent decisions counted, in the export.
 */
import { useState } from "react";
import { Rich } from "../../components/stage/text";
import type { MethodsLine } from "./data";
import { phraseOf, type Guideline, type Section } from "./sections";
import s from "./questlog.module.css";

const READ_FROM = "Read from the values, no question asked:";

const sentenceCase = (t: string) => (/^[a-z]/.test(t) ? t[0]!.toUpperCase() + t.slice(1) : t);

function Sentence({ text, kind }: { text: string; kind: string | null }) {
  const [open, setOpen] = useState(false);
  let body = text;
  let folded: string | null = null;
  const at = text.indexOf(READ_FROM);
  if (at > 0) {
    body = text.slice(0, at).trim();
    folded = text.slice(at + READ_FROM.length).trim();
  }
  const parts = kind ? phraseOf(kind, body) : null;
  return (
    <>
      {parts ? (
        <>
          <Rich text={parts[0]} />
          <span className={s.phrase} title="A stated phrase: press to see the alternatives on the canvas">
            <Rich text={parts[1]} />
          </span>
          <Rich text={parts[2]} />
        </>
      ) : (
        <Rich text={body} />
      )}{" "}
      {folded ? (
        open ? (
          <span className={s.docEngine}>
            <Rich text={`${READ_FROM} ${folded}`} />
          </span>
        ) : (
          <button type="button" className={s.fold} onClick={() => setOpen(true)}>
            {folded.split("); `").length} readings were read from the values, no question asked
          </button>
        )
      ) : null}{" "}
    </>
  );
}

export function MethodsDoc({
  guideline,
  sections,
  lines,
  title,
  locked,
}: {
  guideline: Guideline;
  sections: Section[];
  lines: MethodsLine[];
  title: string;
  locked: string | null;
}) {
  const kindOf = (sentence: string) => lines.find((l) => l.sentence === sentence)?.kind ?? null;
  const silent = sections.reduce((n, sec) => n + sec.silent, 0);
  return (
    <article className={s.doc} data-testid="methods-doc">
      <header className={s.docHead}>
        <span className={s.kicker}>Methods · organized by {guideline}</span>
        <h2 className={s.docTitle}>
          <Rich text={title} />
        </h2>
      </header>
      {locked ? (
        <div className={s.locked}>
          <span className={s.lockedTag}>Plan locked</span>
          <span>
            <Rich text={locked} />
          </span>
        </div>
      ) : null}
      {sections.map((sec) => (
        <section key={sec.key} className={s.docSection}>
          <div className={s.docSectionHead}>
            <h3 className={s.docSectionTitle}>{sec.title}</h3>
            <span className={s.ref}>{sec.ref}</span>
          </div>
          <p className={s.docPara}>
            {sec.items.map((i) => {
              if (i.tier === "stated")
                return i.sentences
                  .filter((t) => !/^The analysis plan recorded above/.test(t))
                  .map((t) => <Sentence key={t} text={t} kind={kindOf(t)} />);
              if (i.tier === "engine")
                // Stated by the engine without asking: its reason, as a sentence of the paragraph.
                return i.sentences.map((t) => (
                  <span key={t} className={s.docEngine}>
                    <Sentence text={sentenceCase(/[.)]$/.test(t) ? t : `${t}.`)} kind={null} />
                  </span>
                ));
              if (i.tier === "asked")
                return (
                  <span key={i.key}>
                    <span className={i.optional ? s.docSlot : s.docSlotOpen}>
                      {i.title}
                      {i.optional ? ": optional" : i.count > 1 ? `: ${i.count} open` : ": open"}
                    </span>{" "}
                  </span>
                );
              if (i.tier === "waiting")
                return (
                  <span key={i.key}>
                    <span className={s.docSlot}>
                      {i.title}: waits on {i.waitingOn.join(" and ") || "an earlier answer"}
                    </span>{" "}
                  </span>
                );
              if (i.tier === "author")
                return (
                  <span key={i.key} className={s.docYours}>
                    <b>Only you can supply · {i.title}.</b> {i.ask}
                  </span>
                );
              return null;
            })}
          </p>
        </section>
      ))}
      <p className={s.hint}>
        {silent} silent decisions change no number; they appear only in the export.
      </p>
    </article>
  );
}
