/**
 * First look's guide, on the card (FOUNDATION §3 stage 3, "a guide plus a gallery"; FIRST_LOOK_BRIEF
 * §1.2, §4.3–§4.4). A few beats, one per section of the gallery beside the card and in its order
 * (each column, what was noticed, what Explore found, the outcome last), each a line on what it
 * shows and a link that brings that section into view.
 *
 * "Decide now" in the gallery hands a look to the card (the stage focus, `{ kind: "finding" }`):
 * the beats give way to that look's question, asked from First look, its lever's options as any
 * question's (Options: each previewed on the canvas while pointed at, Enter or a click records),
 * recorded with `decide_now` (POST /decisions?decide_now=true), which the server refuses, saying
 * what it waits for, where the question cannot be answered ahead of the Router (FOUNDATION §3, the
 * one exception to "Waiting for"). An option the lever names but no decision records is said, not
 * offered. The options are offered only while what Explore found is current: a lever's decision
 * carries the answer it changes as Explore computed it, so while Explore reads the rows again the
 * card says so and offers nothing to record. A record, "Back to First look" or Escape returns to
 * the beats.
 */
import { useEffect, useId, useState, type ReactNode } from "react";
import { isRefusalError } from "../../api/client";
import { useDecideNow, useProjectView } from "../../api/queries";
import type { Decision, ProjectView, QuestLog, Refusal } from "../../api/schema";
import { Options, type OptionItem } from "../../components/record/Options";
import { Plain, kit as k, plain } from "../../kit";
import { useStageFocus } from "../../state/focus";
import { useFirstLook } from "./data";
import { type ExploreBeat, type Look, beats, decideNow, exploreLine, sectionId } from "./model";
import { Unread } from "./views";
import s from "./firstlook.module.css";

const isLook = (id: string) => id.startsWith("explore::");

export function FirstLookGuide({ pid, log }: { pid: string; log: QuestLog }) {
  const view = useProjectView(pid).data;
  return view ? <Guide pid={pid} view={view} log={log} /> : null;
}

function Guide({ pid, view, log }: { pid: string; view: ProjectView; log: QuestLog }) {
  const fl = useFirstLook(pid, view, log);
  const { focus, release, reset } = useStageFocus();
  const held = focus.kind === "finding" && isLook(focus.findingId) ? focus.findingId : null;
  // The look the card asks about: the one last handed over, kept while its options are previewed
  // (an option pointed at holds the focus in its place).
  const [asked, setAsked] = useState<string | null>(held);
  const [seen, setSeen] = useState<string | null>(held);
  if (held !== seen) {
    setSeen(held);
    if (held) setAsked(held);
  }
  // A look held from First look does not follow the person to another stage's canvas.
  useEffect(() => {
    if (!held) return;
    return () => release({ kind: "finding", findingId: held });
  }, [held, release]);

  const looks = fl.explore.looks;
  const d = looks?.distribution ?? null;
  const all: Look[] = looks ? [...looks.free, ...looks.relationships, ...(d ? [{ finding: d, state: { kind: "free" as const }, options: decideNow(d) }] : [])] : [];
  const look = all.find((l) => l.finding.id === asked && l.options.length);
  const back = () => {
    setAsked(null);
    reset();
  };
  if (look) return <Asked pid={pid} look={look} beat={fl.explore.beat} onBack={back} />;
  if (!fl.read) {
    return (
      <section data-kit data-testid="first-look-guide" aria-label="First look's guide">
        {fl.unread.columns ? <Unread message={fl.unread.columns} /> : <p className={s.beatLine}>Reading the columns…</p>}
      </section>
    );
  }

  const said = beats({
    columns: fl.columns.length,
    drawn: fl.drawn.length,
    noticed: fl.noticed,
    target: fl.target,
    door: fl.door,
    rows: fl.rows,
    relationships: looks?.relationships ?? [],
    explore: fl.explore.beat,
  });
  return (
    <section data-kit data-testid="first-look-guide" aria-label="First look's guide">
      <ol className={s.beats}>
        {said.map((b) => (
          <li key={b.key} className={s.beat} data-testid={`beat-${b.key}`}>
            <span className={s.beatTitle}>
              <Plain text={b.title} />
            </span>
            <p className={s.beatLine}>
              <Plain text={b.line} />
            </p>
            {b.also.map((line, i) => (
              <p key={i} className={s.beatLine} data-testid={`beat-${b.key}-also`}>
                <Plain text={line} />
              </p>
            ))}
            {b.key !== "noticed" ? (
              <button type="button" className={`${k.linkish} ${s.act}`} onClick={() => show(b.key)} data-testid={`beat-${b.key}-show`}>
                Show it beside the card
              </button>
            ) : null}
          </li>
        ))}
      </ol>
    </section>
  );
}

/** Bring a gallery section into view, and give it focus. */
function show(key: string) {
  const el = document.getElementById(sectionId(key));
  if (!el) return;
  if (typeof el.scrollIntoView === "function") el.scrollIntoView({ block: "start" });
  el.focus({ preventScroll: true });
}

function Asked({ pid, look, beat, onBack }: { pid: string; look: Look; beat: ExploreBeat; onBack: () => void }) {
  const decideNow = useDecideNow(pid);
  // Its options are Explore's: offered only while what it found is current (model.ts `exploreBeat`).
  const current = beat.kind === "fresh";
  const [answer, setAnswer] = useState<{ key: string; node: ReactNode } | null>(null);
  const id = useId();
  const f = look.finding;
  const lever = f.lever!;
  const items: OptionItem[] = look.options.map((o) => ({
    key: o.key,
    label: <Plain text={o.label} />,
    line: o.sound,
    decision: o.decision as unknown as Decision,
    previewLabel: plain(o.label),
    tags: o.rung === "recommended" ? [{ text: "recommended", tone: "usual" }] : undefined,
  }));
  const told = lever.options.filter((o) => !o.decision);
  const send = (key: string, decision: Decision) => {
    setAnswer(null);
    decideNow.mutate(decision, {
      onSuccess: onBack,
      onError: (err) =>
        setAnswer({
          key,
          node: isRefusalError(err) ? (
            <Refused refusal={err.refusal} onExit={(d) => send(key, d)} onDismiss={() => setAnswer(null)} />
          ) : (
            <p className={s.refused} role="alert" data-testid="decide-now-failed">
              Not recorded: the server did not record this: <Plain text={err.message} />
            </p>
          ),
        }),
    });
  };
  return (
    <section
      className={s.asked}
      data-kit
      data-testid="decide-now"
      aria-labelledby={id}
      onKeyDown={(e) => {
        if (e.key === "Escape" && !e.defaultPrevented) onBack();
      }}
    >
      <p className={s.askedFrom}>Asked from First look</p>
      <h3 id={id} className={s.askedQuestion}>
        <Plain text={lever.question} />
      </h3>
      {!f.outcome_view ? (
        <p className={s.beatLine}>
          <Plain text={f.summary} />
        </p>
      ) : null}
      {current ? (
        <Options
          items={items}
          mode="single"
          onRecord={(item) => item.decision && send(item.key, item.decision)}
          pending={decideNow.isPending}
          answerAt={answer}
          label={plain(lever.question)}
          testId="decide-now-options"
        />
      ) : (
        <p className={s.beatLine} role="status" data-testid="decide-now-waits">
          <Plain text={exploreLine(beat)} />
        </p>
      )}
      {current && told.length ? (
        <ul className={s.told}>
          {told.map((o) => (
            <li key={o.key}>
              <Plain text={o.label} />. <Plain text={o.sound} />
            </li>
          ))}
        </ul>
      ) : null}
      <button type="button" className={`${k.linkish} ${s.act}`} onClick={onBack} data-testid="decide-now-back">
        Back to First look
      </button>
    </section>
  );
}

/** A refusal at the option: why, and each way forward the server gives. One that records is a
 *  button; one that records nothing is said, not offered as a button that does nothing. */
function Refused({ refusal, onExit, onDismiss }: { refusal: Refusal; onExit: (d: Decision) => void; onDismiss: () => void }) {
  const exits = refusal.error.exits;
  return (
    <div className={s.asked} role="alert" data-testid="decide-now-refusal">
      <p className={s.refused}>
        Not recorded: <Plain text={refusal.error.message} />
      </p>
      {exits.map((exit) =>
        exit.decision ? (
          <button key={exit.label} type="button" className={`${k.linkish} ${s.act}`} onClick={() => onExit(exit.decision as Decision)} data-testid="decide-now-exit">
            <Plain text={exit.label} /> instead
          </button>
        ) : (
          <p key={exit.label} className={s.beatLine} data-testid="decide-now-way">
            <Plain text={exit.label} />
          </p>
        ),
      )}
      <button type="button" className={`${k.linkish} ${s.act}`} onClick={onDismiss}>
        Understood
      </button>
    </div>
  );
}
