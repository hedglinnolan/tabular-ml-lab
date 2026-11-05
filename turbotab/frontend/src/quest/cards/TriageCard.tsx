/**
 * The triage of the open noticings at the gate (`GET /triage`; FOUNDATION §7), wherever the shell
 * draws it (log.ts `triageStage`): under Predict in Results, before the held-out rows open (W1-G's
 * card column, results/cards.tsx); under Estimate at Models' end, before the plan's lock and the
 * analysis flowchart (StageCard). The gate is the served triage's own `stage` (core/sweep.py
 * `gate_stage`), so "Confirm all" records that stage's sweep and the card speaks of its gate.
 *
 * Today's card idiom inside the shell (no Direction A styling yet): the stage's results sections.
 * Every sentence is the engine's, or says only what the code does; no number is computed here.
 */
import { useState } from "react";
import { isRefusalError } from "../../api/client";
import { useConfirmSweep } from "../../api/queries";
import type { ConfirmSweepInput, Refusal, SweepStage, Triage, TriageItem } from "../../api/schema";
import r from "../../components/stage/results/results.module.css";
import { Rich } from "../../components/stage/text";
import s from "../results/results.module.css";

/** A recorded disposition, in the triage's own words for each (core/sweep.py, its summary). It
 *  can differ from the recommendation it stands beside (`recorded` is kept while that holds). */
const RECORDED: Record<NonNullable<TriageItem["recorded"]>, string> = {
  no_change: "as changing no number here",
  could_bias: "as a limitation",
  act_on_it: "to act on",
};

/** What the triage stands before, by the gate's stage (core/sweep.py `gate_passed`): the held-out
 *  rows' opening under Predict (Results), else the plan's lock (Models). */
const GATE = {
  results: { before: "before the held-out rows open", settled: "settled before the held-out rows were opened", after: "after the held-out rows were opened" },
  models: { before: "before the plan is fixed", settled: "settled before the plan was fixed", after: "after the plan was fixed" },
} as const;

/**
 * Whether a limitation sentence is owed for `item` as it stands (FOUNDATION §7: "could bias the
 * estimate", a limitation sentence): the engine says so (`limitation`: it could bias and nothing
 * was done about it, core/sweep.py `triage`), and no other disposition is recorded over the
 * recommendation that owes it (the ledger owes one only for a noticing left as could bias,
 * core/materiality.py `ledger`).
 */
function limitationOwed(item: TriageItem): boolean {
  return item.limitation && (item.recorded === null || item.recorded === undefined || item.recorded === item.recommended);
}

/**
 * Whether the card says that a limitation sentence is owed for `item`: where one is, and its
 * reason does not say so already. A finding's reason names none (core/sweep.py `recommend`,
 * `_left_as_it_is`); a measured noticing that owes one ends its reason saying the sentence is
 * drafted (core/materiality.py `recommend`), so the card does not say it twice.
 */
function saysLimitation(item: TriageItem): boolean {
  return item.family === "finding" && limitationOwed(item);
}

/**
 * The decision that records `item`, a noticing recorded "act on it", as a limitation instead: that
 * one line of the triage's own confirmation set to "could bias the estimate", at the gate's stage,
 * as the gate's refusal offers it while the noticing is not acted on ("Record as a limitation: …":
 * core/fit_press.py `triage_refusal` under Estimate, core/seal.py under Predict). The engine
 * confirms each other noticing as recorded while that stands (core/sweep.py `_confirm_the_triage`).
 */
function asLimitation(triage: Triage, item: TriageItem): Omit<ConfirmSweepInput, "kind"> {
  return { stage: triage.stage as SweepStage, sweep: "noticings", lines: [{ id: item.line, key: item.id, value: "could_bias" }] };
}

/**
 * The noticings the gate waits for (`GET /triage`, ruling 4): each with the engine's recommended
 * disposition and its reason, and, where one is owed and the reason does not say so, that a
 * limitation sentence is; a blocker is decided on its own first. "Confirm all" records each as listed (`confirm_sweep`, the gate's
 * stage, sweep "noticings"). A noticing recorded "act on it" holds the gate until it is acted on
 * where it is decided, or its disposition is changed (the owner's Q8): before the gate passes its
 * line offers "Record as a limitation", which records that change (`asLimitation`), the decision
 * the gate's refusal names, whose way forward opens this card (FOUNDATION §1: the card is the one
 * place a choice is recorded); its line then reads as recorded so, and offers it no more.
 * Once the gate has passed (`passed`: under Predict the held-out rows
 * are open, under Estimate the plan is fixed) the card is read, not confirmed, and its heading says
 * so in the past: settled only where no noticing is open and none recorded "act on it" is still
 * listed, since one acted on drops out (core/sweep.py `triage`); under Predict the opening refuses
 * while one is (core/seal.py), but under Estimate the plan's lock reads no triage, so the plan can
 * be fixed with one, and the heading counts it still to act on. Under Predict with no rows held
 * out (`heldOut` false: cross-validation only) there are none to open, so the gate never passes
 * (core/seal.py refuses the opening, `nothing_sealed`) and the heading names no held-out rows: it
 * says what confirming records.
 */
export function TriageCard({ pid, triage, id, heldOut = true }: { pid: string; triage: Triage; id?: string; heldOut?: boolean }) {
  const confirm = useConfirmSweep(pid);
  const [refused, setRefused] = useState<Refusal | null>(null);
  const [failed, setFailed] = useState<string | null>(null);
  const items = triage.items;
  if (!items.length) return null;
  const predict = triage.stage === "results";
  const gate = predict ? GATE.results : GATE.models;
  const blockers = items.filter((i) => i.blocker).length;
  // Once the gate has passed a confirmation is refused (core/sweep.py `gate_passed`), so a
  // disposition that stands was recorded before it; a noticing with none is open now, after it.
  const open = items.filter((i) => i.recorded === null || i.recorded === undefined).length;
  // A noticing recorded "act on it" is listed only until it is acted on (core/sweep.py `triage`;
  // core/materiality.py `in_triage`): one still listed is still to act on.
  const toAct = items.filter((i) => i.recorded === "act_on_it").length;
  const settled = triage.passed && open === 0 && toAct === 0;
  const unsettled = [open ? `${open} open` : "", toAct ? `${toAct} still to act on` : ""].filter(Boolean).join(" and ");
  const counted = items.length === 1 ? "1 noticing" : `${items.length} noticings`;
  /** Record the triage: every noticing as listed (Confirm all, no lines), or one line set. */
  const press = (sweep: Omit<ConfirmSweepInput, "kind"> = { stage: triage.stage as SweepStage, sweep: "noticings", lines: [] }) => {
    setRefused(null);
    setFailed(null);
    confirm.mutate(sweep, { onError: (e) => (isRefusalError(e) ? setRefused(e.refusal) : setFailed(e.message)) });
  };
  return (
    <section className={r.section} id={id} tabIndex={-1} aria-labelledby={`${id}-h`} data-testid="results-triage" data-purpose="triage">
      <header className={r.sectionHead}>
        <h3 id={`${id}-h`} className={r.kicker}>
          {predict && !heldOut
            ? open === 0
              ? `${counted}, each with its disposition recorded`
              : `${counted}, each recorded as listed once confirmed`
            : !triage.passed
              ? `${counted} ${gate.before}`
              : settled
                ? `${counted}, ${gate.settled}`
                : `${counted}, ${unsettled} ${gate.after}`}
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
              {i.recorded === "act_on_it" && !triage.passed ? (
                <button
                  type="button"
                  className={r.linkButton}
                  onClick={() => press(asLimitation(triage, i))}
                  disabled={confirm.isPending || !triage.confirmable}
                  data-testid="triage-as-limitation"
                >
                  Record as a limitation
                </button>
              ) : null}
            </span>
            <span className={s.itemReason}>
              <Rich text={i.reason} />
            </span>
            {saysLimitation(i) ? (
              <span className={s.itemReason} data-testid="triage-limitation">
                A limitation sentence is owed for it.
              </span>
            ) : null}
          </li>
        ))}
      </ul>
      {settled && predict ? (
        <p className={r.said} role="status" data-testid="triage-settled">
          Settled: the held-out rows no longer wait for these noticings.
        </p>
      ) : !triage.passed && !triage.answered ? (
        <div className={r.bandRow}>
          <button type="button" className={r.bandButton} onClick={() => press()} disabled={confirm.isPending || !triage.confirmable} data-testid="triage-confirm">
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
