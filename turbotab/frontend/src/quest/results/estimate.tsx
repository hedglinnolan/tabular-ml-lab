/**
 * Estimate Results v0 (the front-end audit's G02 v0, G11 and G18; the owner's ruling of 2026-11-03:
 * a read-only first version from the views that exist, which W4's designed Results replaces or
 * wraps): the shell's Results tapestry under any goal but prediction. Nothing is recorded here
 * (FOUNDATION §1, the tapestry's rule) and nothing is computed: every number is the effects
 * stage's served field, formatted by the view, and every sentence is the engine's.
 *
 *   1. the plan-lock line, "Plan fixed at …" with the locked plan's SHA-256 as its quiet label
 *      (FOUNDATION §7), from the quest log's lock (core/fit_press.py `FitLock`);
 *   2. per model family, Table 2 with its forest as the last column (table2FromEffects and
 *      forestFromEffects over GET /stages/effects; on a ratio scale, which prints each ratio and
 *      no estimate, the caption's sentence on the coefficients' scale is left out, `asTable2`),
 *      and beside it the primary model's diagnostics in the engine's words: a failed check's ways
 *      forward are named, never pressed, and a recorded response is said as its record words it.
 *
 * Gated on the plan's lock as the engine gates it (FOUNDATION §5 rule 6): the quest log's lock
 * through gateFromLock, which fails closed, and the served artifact's own withholding. A fit refused
 * or failed is said through StageFailure (StageRefusal for a refusal) with `recordHere` false; one
 * under way in the engine's label for the work.
 */
import { useState } from "react";
import type { EffectsArtifact } from "../../api/m3-types";
import { useQuest } from "../../api/queries";
import type { DecisionRecord, FitLock, ProjectView } from "../../api/schema";
import { StageFailure, rootFailure } from "../../components/Failure";
import { stageLabel } from "../../components/JobChips";
import { useM1Stage } from "../../components/stage/data";
import { Rich } from "../../components/stage/text";
import { forestFromEffects, gateFromLock, LOCK_GATE, table2FromEffects } from "../../components/views/adapters";
import { OneLine } from "../../components/views/common/parts";
import type { EffectsLock } from "../../components/views/contracts";
import { TableWithForest } from "../../components/views/Forest";
import { StaleVeil, type VeilState } from "../../motion/StaleVeil";
import s from "./results.module.css";

type Family = EffectsArtifact["families"][number];
type Diagnostic = Family["diagnostics"][number];
/** The served effects: an estimate stage's artifact carries the gate's words while it is withheld
 *  (core/estimand.py `withhold`), outside its schema. */
type ServedEffects = EffectsArtifact & { withheld?: string | null };

/** The plan's lock as the views' gate reads it, from the lock the quest log serves: locked only
 *  where the quest log says so, and its record the lock_plan record whose digest is the one served
 *  (null when none is). No lock served reads as not locked. */
export function lockOf(fit: FitLock | null | undefined, decisions: readonly DecisionRecord[]): EffectsLock {
  if (!fit?.locked) return { locked: false, plan_lock: null };
  const record = [...decisions]
    .reverse()
    .find((r) => r.decision.kind === "lock_plan" && (r.decision as { digest?: unknown }).digest === fit.sha256);
  return { locked: true, plan_lock: record?.id ?? null };
}

const two = (n: number) => String(n).padStart(2, "0");

/** "Plan fixed at 20:28 UTC on 2026-11-03": FOUNDATION §7's words, with the day and the zone the
 *  engine words the same lock in (core/fit_press.py `_when`). */
function fixedAt(at: string): string | null {
  const t = new Date(at);
  if (Number.isNaN(t.getTime())) return null;
  const day = `${t.getUTCFullYear()}-${two(t.getUTCMonth() + 1)}-${two(t.getUTCDate())}`;
  return `Plan fixed at ${two(t.getUTCHours())}:${two(t.getUTCMinutes())} UTC on ${day}`;
}

const busy = (status: string | undefined) => status === "running" || status === "queued";

export function EstimateTapestry({ pid, view }: { pid: string; view: ProjectView }) {
  const quest = useQuest(pid);
  const effects = useM1Stage(pid, view, "effects");
  const status = effects.status;
  // Drawn once what it reads is served: the quest log, and the effects where the stage holds them.
  const fetching = !effects.result && (status?.status === "fresh" || status?.status === "stale");
  if (quest.isPending || fetching) return null;
  const fit = quest.data?.fit ?? null;
  const art = effects.artifact as ServedEffects | null;
  const gate = gateFromLock(lockOf(fit, view.decisions), art?.withheld ?? fit?.reason ?? LOCK_GATE);
  if (gate !== null) {
    return (
      <div className={s.tapestry} data-testid="results-estimate">
        <OneLine view="table" text={<Rich text={gate} />} />
      </div>
    );
  }
  const sha256 = fit?.sha256 ?? null;
  const line = fit?.at && sha256 ? fixedAt(fit.at) : null;
  const failed = !!rootFailure(view.stages, "effects") || !!status?.cancelled;
  return (
    <div className={s.tapestry} data-testid="results-estimate">
      {line ? (
        // The digest is one unbroken word: it scrolls in its own container on a narrow screen.
        <section className={s.exhibit} data-purpose="plan_lock">
          <p className={s.quietLine} data-testid="results-plan-lock">
            {line} · <span data-testid="results-plan-sha256">SHA-256 {sha256}</span>
          </p>
        </section>
      ) : null}
      {failed ? <StageFailure pid={pid} view={view} stage="effects" recordHere={false} /> : null}
      <Estimates art={art} view={view} veil={effects.veil} status={status?.status} failed={failed} />
    </div>
  );
}

function Estimates({
  art,
  view,
  veil,
  status,
  failed,
}: {
  art: ServedEffects | null;
  view: ProjectView;
  veil: VeilState;
  status: string | undefined;
  failed: boolean;
}) {
  if (!art) {
    if (failed) return null;
    return <OneLine view="table" text={busy(status) ? `${stageLabel("effects")}…` : "No estimate is served yet."} />;
  }
  // Withheld after the lock too, while a question an estimate rests on is open (WP17).
  if (art.withheld) return <OneLine view="table" text={<Rich text={art.withheld} />} />;
  // No family: the stage reports no effect here, and its methods say why (`_not_applicable`).
  if (!art.families.length) return <OneLine view="table" text={<Rich text={art.methods} />} />;
  const many = art.families.length > 1;
  return (
    <>
      {art.families.map((fam, i) => (
        <StaleVeil key={fam.family} state={veil} order={i} label={many ? `Table 2, ${fam.label}` : "Table 2"}>
          <FamilyEstimates art={art} index={i} many={many} decisions={view.decisions} />
        </StaleVeil>
      ))}
    </>
  );
}

/** A caption's sentence on the coefficients' scale names the exponential that turns a coefficient
 *  into its ratio: "… each hazard ratio is exp(estimate)." (core/models/inference.py `_on_scale`,
 *  for the hazard, odds and relative-risk ratios) or "… exp(β) multiplies the odds …"
 *  (core/models/ordinal.py, the cumulative odds ratio). */
const ON_COEFFICIENTS = /exp\((?:estimate|β)\)/;

/** The sentences of an engine text: each ends at a full stop before a space, outside a backticked
 *  name (a level such as `1.0`, a column such as `a. b`). Joined by one space they are the text. */
function sentencesOf(text: string): string[] {
  const out: string[] = [];
  let [from, ticked] = [0, false];
  for (let i = 0; i < text.length; i++) {
    if (text[i] === "`") ticked = !ticked;
    else if (!ticked && text[i] === "." && /\s/.test(text[i + 1] ?? "")) {
      out.push(text.slice(from, i + 1).trim());
      from = i + 1;
    }
  }
  const tail = text.slice(from).trim();
  return tail ? [...out, tail] : out;
}

/** The effects as Table 2 prints them. On a ratio scale its column is each ratio, exp(estimate),
 *  and it has no estimate column, so the caption's sentence on the coefficients' scale (the
 *  coefficient table's: "Estimates are log hazard ratios; …") would describe a column it does not
 *  print, and a reader who trusted it would exponentiate a ratio again: that sentence is left out,
 *  and the caption's others (its intervals, its rows) stay in the engine's words. A difference
 *  scale prints the estimate itself, and its caption whole. */
function asTable2(art: ServedEffects): ServedEffects {
  if (art.scale !== "ratio") return art;
  const caption = (text: string) => sentencesOf(text).filter((x) => !ON_COEFFICIENTS.test(x)).join(" ");
  return {
    ...art,
    families: art.families.map((fam) => ({
      ...fam,
      sequence: fam.sequence.map((fit) => (fit.inference ? { ...fit, inference: { ...fit.inference, caption: caption(fit.inference.caption) } } : fit)),
    })),
  };
}

function FamilyEstimates({ art, index, many, decisions }: { art: ServedEffects; index: number; many: boolean; decisions: readonly DecisionRecord[] }) {
  const [lit, setLit] = useState<string | null>(null);
  const fam = art.families[index]!;
  const table = table2FromEffects(asTable2(art), index, many ? `Table 2 (${fam.label})` : "Table 2");
  const primary = fam.sequence.find((q) => q.key === "model_2") ?? fam.sequence[0];
  // A family blocked and recorded has no estimate: its refusal says why, in place of an empty table.
  const refused = !table.rows.some((r) => r.kind === "row") ? (primary?.inference?.refused ?? null) : null;
  return (
    <div className={s.tapestry}>
      <section className={s.exhibit} data-purpose="table2" data-testid="results-table2" data-family={fam.family}>
        {refused ? (
          <OneLine view="table" title={table.number} text={<Rich text={refused} />} />
        ) : (
          <TableWithForest table={table} forest={forestFromEffects(art, index)} gate={null} lit={lit} onLit={setLit} />
        )}
      </section>
      <Diagnostics fam={fam} exposure={art.exposure} decisions={decisions} />
    </div>
  );
}

const CHECK: Record<string, string> = { proportional_hazards: "Proportional hazards", influence: "Influence" };
const STATUS: Record<string, string> = { passed: "passed", failed: "failed", not_assessed: "not assessed" };

/** The sentence of the newest record of the response the check serves, for this exposure. */
function responseSentence(d: Diagnostic, exposure: string, decisions: readonly DecisionRecord[]): string | null {
  const record = [...decisions].reverse().find((r) => {
    const x = r.decision as { kind: string; check?: unknown; exposure?: unknown; action?: unknown };
    return x.kind === "respond_diagnostic" && x.check === d.check && x.exposure === exposure && x.action === d.response;
  });
  return record?.sentence ?? null;
}

/** The primary model's diagnostics (G18), as words: each check's state and the engine's reading;
 *  a failed check's ways forward named, never pressed (the card is where a choice is recorded);
 *  a recorded response as its record says it. */
function Diagnostics({ fam, exposure, decisions }: { fam: Family; exposure: string; decisions: readonly DecisionRecord[] }) {
  if (!fam.diagnostics.length) return null;
  return (
    <section className={s.exhibit} data-purpose="diagnostics" data-testid="results-diagnostics" data-family={fam.family}>
      <h3 className={s.exhibitTitle}>Diagnostics of the primary model</h3>
      <ul className={s.items}>
        {fam.diagnostics.map((d) => (
          <li key={d.check} className={s.item} data-testid="results-diagnostic" data-check={d.check} data-status={d.status}>
            <span className={s.itemLabel}>
              {CHECK[d.check] ?? d.check.replace(/_/g, " ")}: {STATUS[d.status] ?? d.status.replace(/_/g, " ")}
            </span>
            <span className={s.itemSummary}>
              <Rich text={d.reading} />
            </span>
            {d.response ? (
              <span className={s.itemReason}>
                <Rich text={responseSentence(d, exposure, decisions) ?? "A response is recorded for this check."} />
              </span>
            ) : d.status === "failed" ? (
              <span className={s.itemReason}>No response is recorded yet.{d.exits.length ? " Its ways forward:" : ""}</span>
            ) : null}
            {!d.response && d.exits.length ? (
              <ul className={s.items}>
                {d.exits.map((x) => (
                  <li key={x.label} className={s.exitNote}>
                    <Rich text={x.label} />
                  </li>
                ))}
              </ul>
            ) : null}
          </li>
        ))}
      </ul>
    </section>
  );
}
