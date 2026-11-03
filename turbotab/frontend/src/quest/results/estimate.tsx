/**
 * Estimate Results v0 (the front-end audit's G02 v0, G11 and G18; the owner's ruling of 2026-11-03:
 * a read-only first version from the views that exist, which W4's designed Results replaces or
 * wraps): the shell's Results tapestry under any goal but prediction. Nothing is recorded here
 * (FOUNDATION §1, the tapestry's rule) and nothing is computed: every number is the effects
 * stage's served field, formatted by the view, and every sentence is the engine's, save the lines
 * that name what this Table 2 does not draw.
 *
 *   1. the plan-lock line, "Plan fixed at …" with the locked plan's SHA-256 as its quiet label
 *      (FOUNDATION §7), from the quest log's lock (core/fit_press.py `FitLock`);
 *   2. per model family, Table 2 with its forest as the last column (table2FromEffects and
 *      forestFromEffects over GET /stages/effects; on a ratio scale, which prints each ratio and
 *      no estimate, the caption's sentence on the coefficients' scale is left out, `asTable2`):
 *      each declared model's one served row for each exposure, under the name of the column it is
 *      per unit of: the exposure's own column under the exposure's name, and a column a method
 *      made of it (the residual method's `<exposure>_adj`, a density's `<exposure>_per_<E>`, a
 *      cut's `<exposure>_C2`, a level's indicator), whose unit is not the exposure's, under its
 *      served name; a model that reports no estimate is said in its refusal's words beside it,
 *      and an exposure served as several rows (a category's levels) is named, not drawn, nor
 *      headed (`asTable2`, `unreportedOf`); where an exposure family's models serve more than one
 *      member, neither Table 2 nor its forest is drawn, and one line says why and that the
 *      estimates are in the export (`membersOf`, `FAMILY_NOT_DRAWN`);
 *   3. beside it the primary model's diagnostics in the engine's words: a failed check's ways
 *      forward are named, never pressed, and a recorded response is said as its record words it,
 *      with the estimates the engine serves for it.
 *
 * Gated on the plan's lock as the engine gates it (FOUNDATION §5 rule 6): the quest log's lock
 * through gateFromLock, which fails closed, and the served artifact's own withholding. A fit refused
 * or failed is said through StageFailure (StageRefusal for a refusal) with `recordHere` false and
 * no change offered; one under way in the engine's label for the work.
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
import { fmtEstimate } from "../../components/views/format";
import { StaleVeil, type VeilState } from "../../motion/StaleVeil";
import s from "./results.module.css";

type Family = EffectsArtifact["families"][number];
type Fit = Family["sequence"][number];
type Row = NonNullable<Fit["effects"]>[number];
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
      {/* No change is offered: the Record that reopens a question is not in the shell. */}
      {failed ? <StageFailure pid={pid} view={view} stage="effects" recordHere={false} changeable={false} /> : null}
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

/** Whose a served row is, where several exposures share a model (core/estimand.py
 *  `primary_features`): the exposure's own column, the columns an energy model made of it
 *  (`<exposure>_adj`, `<exposure>_per_<E>`, `kcal_from_<exposure>`) and a form's terms (`x'`,
 *  `x_Q2`, a level's `x_b`); the longest exposure's where two names begin alike. */
function ownerOf(feature: string, exposures: readonly string[]): string | null {
  const base = feature.startsWith("kcal_from_") ? feature.slice("kcal_from_".length) : feature;
  const mine = exposures.filter((e) => base.replace(/'+$/, "") === e || base.startsWith(`${e}_`) || base.startsWith(`${e}[`));
  return mine.sort((a, b) => b.length - a.length)[0] ?? null;
}

/** The exposure a served row is of: a model's rows are the exposures' only (core/stages/effects.py
 *  `SequenceFit`), so with one exposure every row is its own. */
const whose = (feature: string, exposures: readonly string[]) => (exposures.length === 1 ? exposures[0]! : ownerOf(feature, exposures));

/** Whether a declared model reports an estimate: a served row with a number (core/stages/
 *  effects.py `_estimates`, the rule its methods text says "reports none" by). */
const reports = (fit: Fit) => (fit.effects ?? []).some((r) => r.estimate !== null);

/** A declared model as Table 2 draws it: each exposure's one served row, as served, so under the
 *  name of the column its number is per unit of (the exposure's own, or one a method made of it on
 *  another unit: `fiber_per_kcal` is per unit of `fiber` / `kcal`, never of `fiber`), and the rows
 *  of an exposure it serves as several, which one row per model cannot hold (a category's levels,
 *  a curve's terms), by exposure. */
function drawnOf(fit: Fit, exposures: readonly string[]): { rows: Row[]; several: [string, Row[]][] } {
  const by = new Map<string, Row[]>();
  for (const r of fit.effects ?? []) {
    const e = whose(r.feature, exposures);
    if (e !== null) by.set(e, [...(by.get(e) ?? []), r]);
  }
  const rows = [...by.values()].flatMap((own) => (own.length === 1 ? own : []));
  return { rows, several: [...by].filter(([, own]) => own.length > 1) };
}

/** What Table 2 says beside its rows for a declared model it does not draw, in the engine's words
 *  for why where it gives them (core/stages/effects.py `_family_sentence`: the table's refusal,
 *  else its first concern); the primary first, as the methods text says it. */
function unreportedOf(fam: Family, exposures: readonly string[]): { key: string; text: string }[] {
  const plain = (x: string) => x.replaceAll("`", "");
  const out = fam.sequence.flatMap((fit) => {
    const label = plain(fit.label);
    if (!reports(fit)) {
      const why = fit.inference?.refused ?? fit.concerns[0] ?? null;
      return [{ key: fit.key, text: why ? `${label}: no estimate. ${why}` : `${label}: no estimate.` }];
    }
    return drawnOf(fit, exposures).several.map(([e, own]) => ({
      key: fit.key,
      text: `${label}: \`${e}\` is reported as ${own.length} rows (${own.map((r) => `\`${r.feature}\``).join(", ")}); this Table 2 draws one row of it per model, so they are not drawn here.`,
    }));
  });
  return [...out.filter((x) => x.key === "model_2"), ...out.filter((x) => x.key !== "model_2")];
}

/** The effects of family `index` as Table 2 and its forest draw them. Each declared model keeps the
 *  rows it draws (`drawnOf`), and a model that draws none is left out, so no footnote is marked on
 *  a row that is not there and the caption is the first drawn model's (the primary's where it
 *  reports); what is left out is said beside the table (`unreportedOf`). The adapter heads and
 *  titles its rows by the artifact's exposures and finds each model's row by that name, so they are
 *  the drawn rows' columns, each exposure's in turn: a column on another unit than its exposure's
 *  is said per unit of its own name, a member with no drawn row gets no heading, and with one
 *  column drawn the title is per unit of it, never of a family's name for itself (`*`, core/
 *  decisions.py `EXPOSURE_FAMILY`). On a ratio scale its column is each ratio,
 *  exp(estimate), and it has no estimate column, so the caption's sentence on the coefficients'
 *  scale (the coefficient table's: "Estimates are log hazard ratios; …") would describe a column
 *  it does not print, and a reader who trusted it would exponentiate a ratio again: that sentence
 *  is left out, and the caption's others (its intervals, its rows) stay in the engine's words. A
 *  difference scale prints the estimate itself, and its caption whole. */
function asTable2(art: ServedEffects, index: number): ServedEffects {
  const caption = (text: string) => (art.scale === "ratio" ? sentencesOf(text).filter((x) => !ON_COEFFICIENTS.test(x)).join(" ") : text);
  const sequence = art.families[index]!.sequence.flatMap((fit) => {
    const rows = reports(fit) ? drawnOf(fit, art.exposures).rows : [];
    if (!rows.length) return [];
    return [{ ...fit, effects: rows, inference: fit.inference ? { ...fit.inference, caption: caption(fit.inference.caption) } : fit.inference }];
  });
  const drawn = sequence.flatMap((fit) => fit.effects);
  const columns = [...new Set(art.exposures.flatMap((e) => drawn.filter((r) => whose(r.feature, art.exposures) === e).map((r) => r.feature)))];
  return {
    ...art,
    exposure: columns.length === 1 ? columns[0]! : art.exposure,
    exposures: columns,
    families: art.families.map((fam, i) => (i === index ? { ...fam, sequence } : fam)),
  };
}

/** The members of what is studied a model family's declared models serve rows for, any model's
 *  (`whose`; a row no member owns counts as its own): a single exposure's rows are all its own. */
function membersOf(fam: Family, exposures: readonly string[]): Set<string> {
  return new Set(fam.sequence.flatMap((fit) => (fit.effects ?? []).map((r) => whose(r.feature, exposures) ?? r.feature)));
}

/** Said in place of Table 2 and its forest where an exposure family's models serve more than one
 *  member (`membersOf`). The engine serves a model's adjustment as its covariates alone
 *  (`adjusted_for`, the unadjusted model's "no other column in the model"), which is wrong where
 *  the members share one fit (the linear model: each is adjusted for the others there) and right
 *  where each is fitted on its own (feature-wise regression), and both serve each member's rows
 *  per model alike; until it says which, no row of a family is drawn with a label beside it. A
 *  family whose models serve one member (feature-wise regression's, beside a category it adjusts
 *  for and names) is drawn as a single exposure's. Every served row is written to the export's
 *  Table 2 (core/export/tables.py `table2`). */
const FAMILY_NOT_DRAWN =
  "This first version of Results does not draw Table 2 for a family of exposures, because what the engine says each estimate is adjusted for is not yet right for every family; the estimates are in the export.";

function FamilyEstimates({ art, index, many, decisions }: { art: ServedEffects; index: number; many: boolean; decisions: readonly DecisionRecord[] }) {
  const [lit, setLit] = useState<string | null>(null);
  const fam = art.families[index]!;
  const number = many ? `Table 2 (${fam.label})` : "Table 2";
  const diagnostics = <Diagnostics fam={fam} exposure={art.exposure} scale={art.scale} decisions={decisions} />;
  if (membersOf(fam, art.exposures).size > 1) {
    return (
      <div className={s.tapestry}>
        <section className={s.exhibit} data-purpose="table2" data-testid="results-table2" data-family={fam.family}>
          <OneLine view="table" title={number} text={<span data-testid="results-family-not-drawn">{FAMILY_NOT_DRAWN}</span>} />
        </section>
        {diagnostics}
      </div>
    );
  }
  const shown = asTable2(art, index);
  const table = table2FromEffects(shown, index, number);
  const unreported = unreportedOf(fam, art.exposures);
  const said = unreported.map((x, i) => (
    <span key={`${x.key}:${i}`} className={s.itemReason} data-testid="results-unreported" data-fit={x.key}>
      <Rich text={x.text} />
    </span>
  ));
  return (
    <div className={s.tapestry}>
      <section className={s.exhibit} data-purpose="table2" data-testid="results-table2" data-family={fam.family}>
        {shown.families[index]!.sequence.length ? (
          <>
            <TableWithForest table={table} forest={forestFromEffects(shown, index)} gate={null} lit={lit} onLit={setLit} />
            {said.length ? <div className={s.items}>{said}</div> : null}
          </>
        ) : (
          // No declared model draws a row: what each serves is said in place of an empty table.
          <OneLine view="table" title={table.number} text={<span className={s.items}>{said}</span>} />
        )}
      </section>
      {diagnostics}
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
function Diagnostics({ fam, exposure, scale, decisions }: { fam: Family; exposure: string; scale: string | null; decisions: readonly DecisionRecord[] }) {
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
            {d.response && d.change?.length ? <Change d={d} scale={scale} /> : null}
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

/** What a recorded response shows beside the estimate (core/stages/effects.py `Diagnostic.change`:
 *  the hazard ratio before and after the median event time, the primary refit without the
 *  influential rows), under the engine's label for it: each served row on the display scale, the
 *  ratio and its limits under a ratio, as Table 2 prints them. */
function Change({ d, scale }: { d: Diagnostic; scale: string | null }) {
  const ratio = scale === "ratio";
  return (
    <div className={s.item} data-testid="results-diagnostic-change">
      {d.change_label ? (
        <span className={s.itemReason}>
          <Rich text={d.change_label} />
        </span>
      ) : null}
      <ul className={s.items}>
        {d.change!.map((e) => (
          <li key={e.feature} className={s.exitNote} data-testid="results-change-row">
            <span data-testid="results-change-label">{e.feature.replaceAll("`", "")}</span>:{" "}
            <span data-testid="results-change-value">{ratio ? fmtEstimate(e.ratio ?? null, e.ratio_low ?? null, e.ratio_high ?? null) : fmtEstimate(e.estimate, e.ci_low, e.ci_high)}</span>
          </li>
        ))}
      </ul>
    </div>
  );
}
