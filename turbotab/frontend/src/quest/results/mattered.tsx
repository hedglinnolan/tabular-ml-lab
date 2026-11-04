/**
 * "Which of my decisions mattered?" in Estimate Results v0 (the front-end audit's G10; the owner's
 * Q10 of 2026-11-04: in Results, right after the estimate, before Write-up, as its own exhibit).
 * EstimateTapestry mounts it under Table 2 and the diagnostics, past the plan-lock gate the
 * estimate is drawn behind (FOUNDATION §5 rule 6), so before the lock nothing here is read or
 * drawn. Nothing is recorded here and nothing is computed: every number is a served field,
 * formatted by the view, and every sentence is the engine's, save the headings and the lines that
 * say a part is not served yet, which choice a specification changed, and in which quest stage a
 * way forward is taken.
 *
 *   1. the specification curve the `mattered` stage computes after the lock (core/mattered.py),
 *      read from its served artifact (GET …/stages/mattered) and drawn by SpecCurveView through
 *      matteredAdapter; withheld (as the estimates are, while a question they rest on is open, or
 *      where the goal, the measure or the outcome does not offer it) or refused (a curve that cannot
 *      be drawn) in the engine's words, a refusal's ways forward named, never pressed; each
 *      specification not estimated named by the choice it changed, with the engine's reason; the
 *      stage's methods sentence beneath it. Sensitivity, never a way to choose: the primary stays
 *      the reported estimate (the view's own line says so). Under a yes/no outcome the curve is of
 *      log odds while Table 2 prints the odds ratio, so beneath the view the engine's own account
 *      of the curve says the primary is the log of the reported odds ratio, quoting both
 *      (matteredAdapter `onLogOdds`); under a difference in means, which Table 2 prints as the
 *      curve does, it is not said;
 *   2. the materiality ledger's rows (GET …/materiality, core/materiality.py `ledger`), read-only:
 *      each noticing's alternative, the ledger's own line for it (the supplement line, the
 *      limitation draft or the card it was decided on), the open relabel of a row that moved more
 *      than predicted, what was done about it, and its instruments' technical names as quiet labels.
 *
 * A failed or stopped curve is said through StageFailure with `recordHere` false and no change
 * offered, as the estimate's is; one under way in the engine's label for the work; one of earlier
 * answers (the curve, its withheld line or its refusal with the ways forward) veiled and inert
 * while it is computed again; one whose stage is blocked on an answer, with nothing served, not at
 * all (and with no ledger row either, no part of this is drawn).
 */
import { useQueryClient } from "@tanstack/react-query";
import { useEffect, useRef } from "react";
import type { AnyStageName } from "../../api/m1-types";
import { keys, useMateriality, useQuest } from "../../api/queries";
import type { Ledger, ProjectView } from "../../api/schema";
import { StageFailure, rootFailure } from "../../components/Failure";
import { stageLabel } from "../../components/JobChips";
import { useM1Stage } from "../../components/stage/data";
import { Rich } from "../../components/stage/text";
import { OneLine } from "../../components/views/common/parts";
import { notEstimatedOf, onLogOdds, specCurveFromMattered, type ServedCurve } from "../../components/views/matteredAdapter";
import { SpecCurveView } from "../../components/views/SpecCurveView";
import { StaleVeil, type VeilState } from "../../motion/StaleVeil";
import s from "./results.module.css";

const STAGE = "mattered";
/** The stage as the stage hook reads it. The client types the stages it knows (src/api's stage
 *  maps), and `mattered` is not among them yet, so its artifact is typed here (`Served`). */
const MATTERED = STAGE as unknown as AnyStageName;

/** The stage's artifact as served (core/mattered.py `MatteredArtifact`), with the gate's words while
 *  it is withheld (`withheld`, core/mattered.py `withheld`, outside its schema). */
interface Served {
  curve: ServedCurve | null;
  says: string | null;
  methods: string | null;
  refused: { message: string; exits: { label?: string; stage?: string }[] } | null;
  withheld?: string | null;
}

type Row = Ledger["rows"][number];

const TITLE = "Which of my decisions mattered?";

/** The stages whose refits the ledger's realized rows quote once fresh (server/service.py
 *  `materiality`). The client reads the ledger again on a decision (api/events.ts), not when one
 *  of these changes, so it is read again here then. */
const VERIFIED_BY = ["sensitivity", "calibration", "secondary"] as const;

const busy = (status: string | undefined) => status === "running" || status === "queued";
const plain = (text: string) => text.replaceAll("`", "");
/** A phrase opened as a heading: its first letter upper-case, unless it opens on a column's name. */
const opening = (text: string) => (text.startsWith("`") || !text ? text : text[0]!.toUpperCase() + text.slice(1));

export function Mattered({ pid, view }: { pid: string; view: ProjectView }) {
  const stage = useM1Stage(pid, view, MATTERED);
  const rows = useLedgerRows(pid, view);
  const status = stage.status;
  // A stage blocked on an answer, with nothing served, says nothing: under Describe it waits for an
  // estimand that goal never records (core/mattered.py `MATTERED_REQUIRES`), so a line that the
  // curve is not served yet would promise one. With no ledger row either, nothing is drawn.
  const waiting = !stage.artifact && status?.status === "blocked";
  if (waiting && !rows.length) return null;
  // Drawn once what it reads is served, as the estimate waits for the effects.
  const reading = !stage.result && (status?.status === "fresh" || status?.status === "stale");
  const failed = !!rootFailure(view.stages, STAGE) || !!status?.cancelled;
  return (
    <div className={s.tapestry} data-purpose="mattered" data-testid="results-mattered">
      <section className={s.exhibit} data-purpose="spec_curve">
        <h3 className={s.exhibitTitle}>{TITLE}</h3>
        {failed ? <StageFailure pid={pid} view={view} stage={STAGE} recordHere={false} changeable={false} /> : null}
        {waiting || reading ? null : <CurveBody pid={pid} art={stage.artifact as Served | null} veil={stage.veil} status={status?.status} failed={failed} />}
      </section>
      {rows.length ? <LedgerRows rows={rows} /> : null}
    </div>
  );
}

function CurveBody({ pid, art, veil, status, failed }: { pid: string; art: Served | null; veil: VeilState; status: string | undefined; failed: boolean }) {
  if (!art) {
    if (failed) return null;
    return <OneLine view="spec_curve" text={busy(status) ? `${stageLabel(STAGE)}…` : "No specification curve is served yet."} />;
  }
  // Whatever the artifact says (a curve, its gate's words, a refusal and its ways forward) is of
  // the answers it was computed for, so all of it is veiled and inert while it is computed again.
  return (
    <StaleVeil state={veil} label={TITLE}>
      <ServedBody pid={pid} art={art} />
    </StaleVeil>
  );
}

function ServedBody({ pid, art }: { pid: string; art: Served }) {
  if (art.withheld) return <OneLine view="spec_curve" text={<Rich text={art.withheld} />} />;
  if (art.refused) return <Refused pid={pid} refused={art.refused} />;
  if (!art.curve) return <OneLine view="spec_curve" text="No specification curve is served yet." />;
  const data = specCurveFromMattered(art.curve);
  const missing = notEstimatedOf(art.curve);
  // With no specification estimated the view says one line: the gate's while sealed, else the
  // engine's account of the curve.
  const why = data.sealed ?? (art.says ? plain(art.says) : "No specification of this curve is estimated.");
  // On the log-odds scale the view's own line calls the primary the reported estimate, while Table 2
  // prints the odds ratio: the engine's account beneath it says the primary is the log of that
  // ratio, quoting both.
  const logOdds = onLogOdds(art.curve) && !!art.says;
  return (
    <>
      <SpecCurveView data={data} why={why} />
      {logOdds ? (
        <p className={s.quietLine} data-testid="results-mattered-says">
          <Rich text={art.says!} />
        </p>
      ) : null}
      {missing.length ? (
        <ul className={s.items}>
          {missing.map((x) => (
            <li key={x.key} className={s.item} data-testid="results-not-estimated">
              <span className={s.itemLabel}>{x.changed.length ? `Not estimated: ${x.changed.map((c) => `${c.choice}: ${c.option}`).join("; ")}` : "Not estimated"}</span>
              <span className={s.itemReason}>
                <Rich text={x.reason} />
              </span>
            </li>
          ))}
        </ul>
      ) : null}
      {art.methods ? (
        <p className={s.quietLine} data-testid="results-mattered-methods">
          <Rich text={art.methods} />
        </p>
      ) : null}
    </>
  );
}

/** A curve that cannot be drawn, in the engine's words, and its ways forward named, never pressed
 *  (the card is where a choice is recorded): each with the quest stage it opens, where it names one
 *  the quest log serves. */
function Refused({ pid, refused }: { pid: string; refused: NonNullable<Served["refused"]> }) {
  const stages = useQuest(pid).data?.stages ?? [];
  const exits = refused.exits.flatMap((x) => {
    if (!x.label) return [];
    const at = stages.find((st) => st.key === x.stage)?.name;
    return [{ label: x.label, at }];
  });
  return (
    <>
      <OneLine view="spec_curve" text={<Rich text={refused.message} />} />
      {exits.length ? (
        <ul className={s.items}>
          {exits.map((x) => (
            <li key={x.label} className={s.exitNote} data-testid="results-mattered-exit">
              <Rich text={x.at ? `${x.label}, in ${x.at}.` : x.label} />
            </li>
          ))}
        </ul>
      ) : null}
    </>
  );
}

/** The ledger's rows, read again whenever a stage its realized rows quote changes state or key. */
function useLedgerRows(pid: string, view: ProjectView): Row[] {
  const qc = useQueryClient();
  const ledger = useMateriality(pid);
  const reads = VERIFIED_BY.map((name) => `${name}:${view.stages[name]?.status ?? ""}:${view.stages[name]?.key ?? ""}`).join("|");
  const seen = useRef(reads);
  useEffect(() => {
    if (seen.current === reads) return;
    seen.current = reads;
    void qc.invalidateQueries({ queryKey: keys.materiality(pid), exact: true });
  }, [reads, qc, pid]);
  return ledger.data?.rows ?? [];
}

function LedgerRows({ rows }: { rows: Row[] }) {
  return (
    <section className={s.exhibit} data-purpose="ledger" data-testid="results-ledger">
      <h4 className={s.exhibitTitle}>The noticings</h4>
      <ul className={s.items}>
        {rows.map((row) => (
          <LedgerRow key={row.thread} row={row} />
        ))}
      </ul>
    </section>
  );
}

/** The relabel of a row that moved more than predicted (`label`: "Moved more than predicted.", the
 *  realized movement, then what was expected before the plan was fixed; core/materiality.py
 *  `verify`), without the realized movement the row's line already says. */
function relabelOf(row: Row): string | null {
  if (!row.label) return null;
  const said = row.realized?.words;
  const text = said && row.sentence.includes(said) ? row.label.replace(said, " ") : row.label;
  return text.replace(/\s+/g, " ").trim();
}

function LedgerRow({ row }: { row: Row }) {
  const relabel = relabelOf(row);
  const names = [row.predicted.label, row.realized?.label].filter((x): x is string => !!x).map(plain);
  return (
    <li className={s.item} data-testid="results-ledger-row" data-thread={row.thread}>
      <span className={s.itemLabel}>
        <Rich text={opening(row.alternative)} />
      </span>
      <span className={s.itemSummary}>
        <Rich text={row.sentence} />
      </span>
      {relabel ? (
        <span className={s.itemReason} data-testid="results-ledger-relabel">
          <Rich text={relabel} />
        </span>
      ) : null}
      {row.done ? (
        <span className={s.itemReason}>
          <Rich text={row.done} />
        </span>
      ) : null}
      <span className={s.quietLine}>{names.join(" · ")}</span>
    </li>
  );
}
