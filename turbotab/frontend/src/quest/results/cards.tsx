/**
 * Predict Results' card column (plans/W1_PART2_PLAN.md §2 W1-G; rulings 2 and 4), in Predict's
 * order (FOUNDATION §3, §10): the Results declarations before the opening (the shrinkage
 * question's own card), the triage of the open noticings, the opening of the held-out rows with
 * the chain of refusals that names the final model, and the explain offer.
 *
 * Today's card idiom inside the shell (no Direction A styling yet): the seal's CONSEQUENCE card,
 * the stage's quiet captions. Every sentence is the engine's, or says only what the code does; no
 * number is computed here.
 */
import { Fragment, useId, useState } from "react";
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
import { DECLARATION_CARDS } from "../cards/predict";
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
  const triage = useTriage(pid).data;
  const fit = useM1Stage(pid, view, "fit");
  const split = useM1Stage(pid, view, "split");
  const triageId = useId();
  useRefetchOnOpening(pid, !!view.state.seal_opened, fit.artifact, view.stages.fit?.key ?? undefined);
  const lines = resultsLines(quest);
  const declarations = lines.filter((l) => l.source === "declaration" && l.key !== "set_explain");
  const explain = lines.find((l) => l.key === "set_explain") ?? null;
  return (
    <div className={s.cards} data-testid="results-cards">
      {declarations.map((line) => (
        <Declaration key={line.id} pid={pid} line={line} />
      ))}
      {triage && triage.stage === "results" ? <TriageCard pid={pid} triage={triage} id={triageId} /> : null}
      {fit.artifact ? <SealCard pid={pid} view={view} fit={fit.artifact} fresh={fit.veil === "fresh"} split={split.artifact} /> : null}
      {explain ? <ExplainOffer pid={pid} view={view} line={explain} /> : null}
    </div>
  );
}

/** A Results declaration through its Predict card (W1-F's), else its line as the quest log names it. */
function Declaration({ pid, line }: { pid: string; line: QuestLine }) {
  const Card = DECLARATION_CARDS[line.key];
  if (Card) return <Card pid={pid} line={line} />;
  return (
    <p className={s.quietLine} data-testid="results-line" data-key={line.key}>
      {line.label}: {line.name}
      {line.status === "answered" ? ", answered" : line.status === "waiting" ? ", waiting" : ""}
    </p>
  );
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
 * opened (`passed`) the card is read, not confirmed, and its heading says so in the past.
 */
export function TriageCard({ pid, triage, id }: { pid: string; triage: Triage; id?: string }) {
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
          {!triage.passed
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
 */
export function SealCard({ pid, view, fit, fresh, split }: { pid: string; view: ProjectView; fit: FitArtifact; fresh: boolean; split: SplitArtifact | null }) {
  const decide = useDecide(pid);
  const qc = useQueryClient();
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

  return (
    <section className={seal.consequence} aria-labelledby="results-open-title" data-testid="results-open" data-purpose="open_seal">
      <div className={seal.conRule} aria-hidden="true" />
      <div className={seal.conHead}>
        <SealGlyph state={glyph} recorded size={26} />
        <span className={seal.conSignal}>{before ? "Re-sealed" : "Opened once"}</span>
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
      <div className={seal.conExits}>
        <button
          type="button"
          className={seal.conAttest}
          onClick={() => void open(OPEN)}
          disabled={!!busy || !fresh}
          data-testid="results-open-press"
        >
          {busy ?? "Open the held-out rows"}
        </button>
        <span className={seal.conOr}>{fresh ? "Nothing is opened until you press it." : "The fit shown is not current; the rows can be opened once it is."}</span>
      </div>
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

// ── the explain offer ────────────────────────────────────────────────────────

/**
 * The explanations (`set_explain`): W1-F's Predict card where it serves one; else the offer as
 * the engine's defaults, recorded with no field set, so every choice in it is the engine's.
 */
function ExplainOffer({ pid, view, line }: { pid: string; view: ProjectView; line: QuestLine }) {
  const Card = DECLARATION_CARDS[line.key];
  const decide = useDecide(pid);
  const [failed, setFailed] = useState<string | null>(null);
  if (Card) return <Card pid={pid} line={line} />;
  const record = line.decision_id ? view.decisions.find((d) => d.id === line.decision_id) : undefined;
  const status = view.stages.explain;
  const press = () => {
    setFailed(null);
    decide.mutate({ kind: "set_explain" } as Decision, {
      onError: (e) => setFailed(isRefusalError(e) ? e.refusal.error.message : e.message),
    });
  };
  return (
    <section className={r.section} data-testid="results-explain-offer" data-purpose="set_explain">
      <header className={r.sectionHead}>
        <h3 className={r.kicker}>What drives each model</h3>
      </header>
      {line.status === "answered" ? (
        <p className={r.caption} data-testid="explain-recorded">
          {record?.sentence ? <Rich text={record.sentence} /> : "Recorded."}
          {status && (status.status === "running" || status.status === "queued") ? " Explaining each fitted model…" : ""}
        </p>
      ) : (
        <>
          {line.would_change ? (
            <p className={r.basis}>
              <Rich text={line.would_change} />
            </p>
          ) : null}
          <div className={r.bandRow}>
            <button type="button" className={r.bandButton} onClick={press} disabled={decide.isPending || line.status === "waiting"} data-testid="explain-offer">
              {decide.isPending ? "Recording…" : "Explain each model"}
            </button>
            <span className={r.bandNote}>Pressing it records the explanations with the engine&apos;s own settings.</span>
            {line.status === "waiting" && line.waiting_for.length ? (
              <span className={r.bandNote}>Waiting for {line.waiting_for.map((w) => w.name).join(", ")}.</span>
            ) : null}
          </div>
          {failed ? (
            <p className={r.bandNote} role="alert">
              <Rich text={`Not recorded: ${failed}`} />
            </p>
          ) : null}
        </>
      )}
    </section>
  );
}
