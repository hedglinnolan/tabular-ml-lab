/**
 * Predict Results' card column (plans/W1_PART2_PLAN.md §2 W1-G; rulings 2 and 4), in Predict's
 * order (FOUNDATION §3, §10): the Results declarations before the opening (the shrinkage
 * question), the triage of the open noticings, the opening of the held-out rows with the chain of
 * refusals that names the final model, and the explanations. Every declaration is drawn through
 * the shell's dispatcher, so W1-F's cards are their only renderers, here as everywhere.
 *
 * The opening is the card's one primary action once the Results questions before it and the
 * triage are done (FOUNDATION §1): "Open the held-out rows" sits at the foot of the card column
 * (the shell's FootSlot), and until then the foot says, in the button's own words, what it waits
 * for. The opening's card above it says what the press does, and holds the chain of refusals.
 *
 * Today's card idiom inside the shell (no Direction A styling yet): the seal's CONSEQUENCE card,
 * the stage's quiet captions. Every sentence is the engine's, or says only what the code does; no
 * number is computed here.
 */
import { Fragment, useContext, useId, useState } from "react";
import { createPortal } from "react-dom";
import { useQueryClient } from "@tanstack/react-query";
import { isRefusalError } from "../../api/client";
import type { FitArtifact, SplitArtifact } from "../../api/m1-stage-types";
import { keys, useConfirmSweep, useDecide, useQuest, useTriage } from "../../api/queries";
import type { Decision, ProjectView, QuestLine, Refusal, Triage, TriageItem } from "../../api/schema";
import { useM1Stage } from "../../components/stage/data";
import { fmtInt } from "../../components/stage/format";
import r from "../../components/stage/results/results.module.css";
import { useRefetchOnOpening } from "../../components/stage/seal/OpenSeal";
import { glyphOf, sealPhase } from "../../components/stage/seal/phase";
import { SealGlyph } from "../../components/stage/seal/SealGlyph";
import seal from "../../components/stage/seal/seal.module.css";
import { Rich } from "../../components/stage/text";
import { Foot, Plain, kit as k } from "../../kit";
import { CardFor } from "../cards/CardFor";
import { FootSlot } from "../contracts";
import { runningName } from "../log";
import q from "../quest.module.css";
import { said } from "../words";
import { currentOpening, exploratoryReasons, openingNotes } from "./model";
import s from "./results.module.css";

/** The opening's decision with no family named: the engine asks for one where several were fit. */
export const OPEN: Decision = { kind: "open_seal" } as Decision;

/** The Results stage's lines, as the quest log serves them. */
export function resultsLines(quest: { stages: { key: string; lines: QuestLine[] }[] } | undefined): QuestLine[] {
  return quest?.stages.find((x) => x.key === "results")?.lines ?? [];
}

export function Cards({ pid, view }: { pid: string; view: ProjectView }) {
  const quest = useQuest(pid).data;
  const triageQ = useTriage(pid);
  const served = triageQ.data;
  const fit = useM1Stage(pid, view, "fit");
  const split = useM1Stage(pid, view, "split");
  const triageId = useId();
  const opened = !!view.state.seal_opened;
  useRefetchOnOpening(pid, opened, fit.artifact, view.stages.fit?.key ?? undefined);
  const lines = resultsLines(quest);
  // The explanations come after the opening; every other declaration before it.
  const declarations = lines.filter((l) => l.source === "declaration" && l.key !== "set_explain");
  const explain = lines.find((l) => l.source === "declaration" && l.key === "set_explain") ?? null;
  const triage = served && served.stage === "results" ? served : null;
  // No rows held out (the fit's `n_holdout` is 0): there is nothing to open, and the triage says so.
  const heldOut = !(fit.artifact && sealPhase(fit.artifact, opened) === "cv_only");
  // Until the quest log and the triage are read, the press waits for them (a triage the server
  // cannot serve leaves the engine's own refusals to say what is open).
  const waits = quest === undefined || triageQ.isPending ? READING : openingWaits(lines, triage);
  return (
    <div className={s.cards} data-testid="results-cards">
      {declarations.map((line) => (
        <Declaration key={line.id} pid={pid} line={line} view={view} />
      ))}
      {triage ? <TriageCard pid={pid} triage={triage} id={triageId} heldOut={heldOut} /> : null}
      {fit.artifact ? (
        <SealCard pid={pid} view={view} fit={fit.artifact} fresh={fit.veil === "fresh"} split={split.artifact} waits={waits} />
      ) : null}
      {explain ? <Declaration pid={pid} line={explain} view={view} /> : null}
    </div>
  );
}

/**
 * A Results declaration through W1-F's card while it waits for its answer; once recorded it
 * collapses to its answer, as the other stages' cards do (finding 24; the re-walk's N8): its name,
 * the record's sentence and Change, which opens the card again until a new record collapses it.
 */
function Declaration({ pid, line, view }: { pid: string; line: QuestLine; view: ProjectView }) {
  const [changing, setChanging] = useState(false);
  const [seen, setSeen] = useState(line.decision_id);
  if (seen !== line.decision_id) {
    setSeen(line.decision_id);
    setChanging(false);
  }
  if (line.status !== "answered" || changing) return <CardFor pid={pid} line={line} />;
  const sentence = view.decisions.find((d) => d.id === line.decision_id)?.sentence ?? null;
  return (
    <div className={q.line} data-kit data-testid={`results-recorded-${line.key}`} data-line={line.id} data-status={line.status}>
      <div className={q.lineRow}>
        <span className={q.lineName}>{said(line.name)}</span>
        <button type="button" className={k.linkish} onClick={() => setChanging(true)}>
          Change
        </button>
      </div>
      {sentence ? (
        <p className={q.lineAnswer}>
          <Plain text={sentence} />
        </p>
      ) : null}
    </div>
  );
}

/** What the press waits for while the quest log and the triage are being read. */
const READING = "Reading what the opening waits for…";

/** "a", "a and b", "a, b and c". */
const listing = (xs: string[]) => (xs.length < 2 ? (xs[0] ?? "") : `${xs.slice(0, -1).join(", ")} and ${xs[xs.length - 1]}`);

/**
 * What the opening waits for, in the button's own words, or null when it can be pressed: the
 * Results questions before it, whose answers the held-out score reads (FOUNDATION §3 and §10: what
 * the held-out rows read is fixed before they are scored), then the triage's confirmation (§7: the
 * triage stands before the held-out rows open). The engine's own refusals stand behind it: a
 * noticing still to act on, or the final model to name, is refused at the press, with its exits.
 */
export function openingWaits(lines: readonly QuestLine[], triage: Triage | null): string | null {
  const before = lines.filter((l) => l.label === "Decide" && l.counted && l.key !== "open_seal" && l.status !== "answered");
  if (before.length) return `To open the held-out rows, first answer ${listing(before.map((l) => runningName(l.name)))}.`;
  if (triage && triage.items.length && !triage.answered && !triage.passed) {
    const n = triage.items.length;
    return `To open the held-out rows, first confirm the ${n === 1 ? "noticing" : `${n} noticings`} above.`;
  }
  return null;
}

// ── the triage of the open noticings ─────────────────────────────────────────

/** A recorded disposition, in the triage's own words for each (core/sweep.py, its summary). It
 *  can differ from the recommendation it stands beside (`recorded` is kept while that holds). */
const RECORDED: Record<NonNullable<TriageItem["recorded"]>, string> = {
  no_change: "as changing no number here",
  could_bias: "as a limitation",
  act_on_it: "to act on",
};

/**
 * The noticings the held-out rows wait for (`GET /triage`, ruling 4): each with the engine's
 * recommended disposition and its reason; a blocker is decided on its own first. "Confirm all"
 * records each as listed (`confirm_sweep`, stage "results", sweep "noticings"). Once the rows are
 * opened (`passed`) the card is read, not confirmed, and its heading says so in the past. With no
 * rows held out (`heldOut` false: cross-validation only) there are none to open, so the gate never
 * passes (core/seal.py refuses the opening, `nothing_sealed`) and the heading names no held-out rows:
 * it says what confirming records.
 */
export function TriageCard({ pid, triage, id, heldOut = true }: { pid: string; triage: Triage; id?: string; heldOut?: boolean }) {
  const confirm = useConfirmSweep(pid);
  const [refused, setRefused] = useState<Refusal | null>(null);
  const [failed, setFailed] = useState<string | null>(null);
  const items = triage.items;
  if (!items.length) return null;
  const blockers = items.filter((i) => i.blocker).length;
  // Once the gate has passed (under Predict, the held-out rows are open) a confirmation is refused
  // (core/sweep.py `gate_passed`), so a disposition that stands was recorded before the opening; a
  // noticing with none is open now, after it.
  const open = items.filter((i) => i.recorded === null || i.recorded === undefined).length;
  const settled = triage.passed && open === 0;
  const counted = items.length === 1 ? "1 noticing" : `${items.length} noticings`;
  const press = () => {
    setRefused(null);
    setFailed(null);
    confirm.mutate(
      { stage: "results", sweep: "noticings", lines: [] },
      { onError: (e) => (isRefusalError(e) ? setRefused(e.refusal) : setFailed(e.message)) },
    );
  };
  return (
    <section className={r.section} id={id} tabIndex={-1} aria-labelledby={`${id}-h`} data-testid="results-triage" data-purpose="triage">
      <header className={r.sectionHead}>
        <h3 id={`${id}-h`} className={r.kicker}>
          {!heldOut
            ? open === 0
              ? `${counted}, each with its disposition recorded`
              : `${counted}, each recorded as listed once confirmed`
            : !triage.passed
              ? `${counted} before the held-out rows open`
              : settled
                ? `${counted}, settled before the held-out rows were opened`
                : `${counted}, ${open} open after the held-out rows were opened`}
        </h3>
      </header>
      <ul className={s.items}>
        {items.map((i) => (
          <li key={i.id} className={s.item} data-item={i.id} data-blocker={i.blocker || undefined}>
            <span className={s.itemSummary}>
              <Rich text={i.summary} />
            </span>
            <span className={s.itemLabel}>
              {i.blocker ? "Resolve first · " : ""}
              {i.recorded && i.recorded !== i.recommended ? `Recommended: ${i.label} · recorded ${RECORDED[i.recorded]}` : i.label}
              {i.recorded && i.recorded === i.recommended ? " · recorded" : ""}
            </span>
            <span className={s.itemReason}>
              <Rich text={i.reason} />
            </span>
          </li>
        ))}
      </ul>
      {settled ? (
        <p className={r.said} role="status" data-testid="triage-settled">
          Settled: the held-out rows no longer wait for these noticings.
        </p>
      ) : !triage.passed && !triage.answered ? (
        <div className={r.bandRow}>
          <button type="button" className={r.bandButton} onClick={press} disabled={confirm.isPending || !triage.confirmable} data-testid="triage-confirm">
            {confirm.isPending ? "Recording…" : `Confirm all ${items.length}`}
          </button>
          {blockers > 0 ? (
            <span className={r.bandNote}>
              {blockers === 1 ? "The noticing marked to resolve first is" : `The ${blockers} noticings marked to resolve first are`} decided on
              {blockers === 1 ? " its" : " their"} own before the rest are confirmed.
            </span>
          ) : null}
        </div>
      ) : null}
      {refused ? (
        <p className={r.bandNote} role="alert">
          <Rich text={`Not recorded: ${refused.error.message}`} />
        </p>
      ) : failed ? (
        <p className={r.bandNote} role="alert">
          Not recorded: {failed}
        </p>
      ) : null}
    </section>
  );
}

// ── the opening, and the chain of refusals that names the final model ────────

/**
 * The opening of the held-out rows (`open_seal`). The engine refuses it while a noticing is open
 * (`triage_open`) and, with several families, until the final model is named on cross-validation
 * (`final_model_needed`). Each refusal's exits are buttons: an exit that is itself the opening (one
 * naming a final model) is the next attempt; any other exit is recorded, and then the last attempt
 * is posted again, so the chain is walked press by press. An exit with no decision says where to
 * act and records nothing. Sealed again by a re-seal after an opening, the card says the engine's
 * note on the first opening's result in place of that opening's claims.
 *
 * The press itself is the card column's one primary action, "Open the held-out rows", drawn at the
 * foot (the shell's FootSlot; in place, where no shell provides one). Before the opening the card
 * says what will happen ("Opens once"), and wears no recorded glyph: nothing is opened yet. Until
 * `waits` is null the press waits, and the foot says for what.
 */
export function SealCard({
  pid,
  view,
  fit,
  fresh,
  split,
  waits = null,
}: {
  pid: string;
  view: ProjectView;
  fit: FitArtifact;
  fresh: boolean;
  split: SplitArtifact | null;
  /** What the opening waits for, in the button's words (`openingWaits`), or null. */
  waits?: string | null;
}) {
  const decide = useDecide(pid);
  const qc = useQueryClient();
  const slot = useContext(FootSlot);
  const [attempt, setAttempt] = useState<Decision>(OPEN);
  const [refused, setRefused] = useState<Refusal | null>(null);
  const [failed, setFailed] = useState<string | null>(null);
  const [busy, setBusy] = useState<string | null>(null);
  const opened = !!view.state.seal_opened;
  const phase = sealPhase(fit, opened);
  if (fit.withheld_for) return null; // nothing is fitted to open yet: Fit is pressed in Models
  if (phase === "cv_only") {
    return (
      <p className={s.quietLine} data-testid="seal-none">
        No rows were held out, so there is nothing to open: the models are judged on cross-validation alone.
      </p>
    );
  }
  if (phase === "opened" || phase === "post_seal") return <OpenedLine view={view} fit={fit} split={split} />;

  const metric = fit.metric_labels[fit.primary_metric] ?? fit.primary_metric;
  const families = fit.models.length;
  const models = families === 1 ? "model" : `${families} models`;
  const glyph = glyphOf(split?.basis?.state);
  const exploratory = exploratoryReasons(split);
  // Sealed again after an opening (a re-seal): the first opening's scores stay the reported result,
  // and the rows drawn since are not an independent test. The engine's note says so
  // (`at_opening`, turbotab/core/seal.py `reported_result`); the first opening's claims would not.
  const before = fit.at_opening && !fit.at_opening.current ? fit.at_opening : null;

  /** Post one decision; a refusal is kept with its exits, any other failure said as it is. */
  const post = async (d: Decision): Promise<boolean> => {
    try {
      await decide.mutateAsync(d);
      return true;
    } catch (e) {
      if (isRefusalError(e)) setRefused(e.refusal);
      else setFailed(e instanceof Error ? e.message : String(e));
      return false;
    }
  };
  const open = async (d: Decision) => {
    setAttempt(d);
    setRefused(null);
    setFailed(null);
    setBusy("Opening…");
    if (await post(d)) void qc.invalidateQueries({ queryKey: keys.stage(pid, "fit"), exact: true });
    setBusy(null);
  };
  const take = async (d: Decision) => {
    if (d.kind === "open_seal") return open(d);
    setRefused(null);
    setFailed(null);
    setBusy("Recording…");
    const taken = await post(d);
    setBusy(null);
    if (taken) await open(attempt);
  };

  // The card column's one primary action, at its foot (FOUNDATION §1).
  const foot = (
    <Foot
      label={busy ?? "Open the held-out rows"}
      onClick={() => void open(OPEN)}
      disabled={!!busy || !fresh || waits !== null}
      left={
        <span data-testid="results-open-hint">
          {!fresh ? "The fit shown is not current; the rows can be opened once it is." : (waits ?? "Nothing is opened until you press it.")}
        </span>
      }
      testId="results-open-press"
    />
  );

  return (
    <section className={seal.consequence} aria-labelledby="results-open-title" data-testid="results-open" data-purpose="open_seal">
      <div className={seal.conRule} aria-hidden="true" />
      <div className={seal.conHead}>
        {/* Nothing is opened yet, so no recorded glyph: only a basis that is not a verified clean
            one is drawn, in the coach's amber, as its exploratory label says. */}
        {glyph !== "closed" ? <SealGlyph state={glyph} recorded={false} size={26} /> : null}
        <span className={seal.conSignal}>{before ? "Re-sealed" : "Opens once"}</span>
      </div>
      <h3 id="results-open-title" className={seal.conTitle}>
        {before
          ? `Opening the held-out rows scores the ${models} on ${fmtInt(fit.n_holdout)} rows drawn again after the first opening.`
          : `Opening the held-out rows scores the ${models} on ${fmtInt(fit.n_holdout)} rows no choice has seen.`}
      </h3>
      <p className={seal.conBody}>
        {before ? (
          <>
            <Rich text={before.note} /> Any change after this opening still refits, and is marked as made after it in the Results and the Record.
          </>
        ) : (
          `It happens once. The held-out ${metric} is then fixed in the record; any later change still refits, and is marked as made after the opening in the Results and the Record.`
        )}
        {exploratory ? (
          <>
            {" "}
            <Rich text={`${exploratory.join(" ")} In the Results the held-out scores are then marked exploratory.`} />
          </>
        ) : null}
      </p>
      {refused ? (
        <div role="alert" data-testid="results-open-refused" data-code={refused.error.code}>
          <p className={seal.conRefused}>
            <Rich text={`Not opened: ${refused.error.message}`} />
          </p>
          <div className={seal.conExits}>
            {refused.error.exits.map((x) =>
              x.decision ? (
                <button
                  key={x.label}
                  type="button"
                  className={seal.conAttest}
                  disabled={!!busy}
                  onClick={() => void take(x.decision as Decision)}
                  data-testid="results-open-exit"
                >
                  <Rich text={x.label} />
                </button>
              ) : (
                <span key={x.label} className={s.exitNote} data-testid="results-open-where">
                  <Rich text={x.label} />
                </span>
              ),
            )}
          </div>
        </div>
      ) : failed ? (
        <p className={seal.conRefused} role="alert">
          Not opened: the server did not record it ({failed}).
        </p>
      ) : null}
      {slot ? createPortal(foot, slot) : foot}
    </section>
  );
}

/** Once opened: the current opening's record (the fit's final model is its), and the engine's notes;
 *  an exploratory split's label with its reason, in the split's words. "Once", and the scores fixed
 *  in the record, only while the served scores are the first opening's (`at_opening.current`);
 *  otherwise the engine's note says which result stays reported, and why these are not it. */
function OpenedLine({ view, fit, split }: { view: ProjectView; fit: FitArtifact; split: SplitArtifact | null }) {
  const opening = currentOpening(view);
  const exploratory = exploratoryReasons(split);
  const label = fit.models.find((m) => m.family === fit.final_model)?.label ?? null;
  const first = fit.at_opening?.current !== false;
  const notes = first ? (fit.final_note ? [fit.final_note] : []) : openingNotes(fit);
  const named = `${opening ? ` (#${opening.seq})` : ""}${label ? `, with ${label} as the final model` : ""}`;
  return (
    <p className={seal.sealRecorded} data-testid="results-opened" data-post={fit.changed_after_seal || undefined}>
      <span>
        {first ? `The held-out rows were opened once${named}: their scores are fixed in the record.` : `The held-out rows were opened${named}.`}
        {notes.map((t) => (
          <Fragment key={t}>
            {" "}
            <Rich text={t} />
          </Fragment>
        ))}
        {exploratory ? (
          <>
            {" "}
            <Rich text={`These scores are exploratory: ${exploratory.join(" ")}`} />
          </>
        ) : null}
      </span>
    </p>
  );
}
