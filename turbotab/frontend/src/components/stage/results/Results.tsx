/**
 * The Results (M1_CONTRACT §13): what the recorded pipeline fitted, every number traced to the
 * decision that produced it. A changed earlier answer veils each section until its stage
 * recomputes; nothing is deleted.
 */
import { useState } from "react";
import { isRefusalError } from "../../../api/client";
import {
  substitutionDecision,
  type DesignArtifact,
  type FitArtifact,
  type ShelfArtifact,
  type SplitArtifact,
  type SubstitutionArtifact,
} from "../../../api/m1-stage-types";
import { useCancelJob, useDecide } from "../../../api/queries";
import type { ProjectView, Refusal } from "../../../api/schema";
import { RefusalNote } from "../../record/Refusal";
import { StaleVeil, type VeilState } from "../../../motion/StaleVeil";
import { StageRetry } from "../../StageRetry";
import { DesignWarnings, warningsAbout } from "../DesignWarnings";
import { plain } from "../format";
import { comparisonFigure, curvesFigure, forestFigure } from "../save/resultsJournal";
import {
  HeldOutAside,
  OpenSealCard,
  openingRecord,
  PostSealBand,
  SealOpenedLine,
  useRefetchOnOpening,
} from "../seal/OpenSeal";
import { glyphOf, sealPhase } from "../seal/phase";
import { SaveMenu } from "../save/SaveMenu";
import { Rich } from "../text";
import { Coefficients } from "./Coefficients";
import { Comparison } from "./Comparison";
import { Curves } from "./Curves";
import {
  aboutSeconds,
  BAND_ROWS,
  bandOffer,
  carriedSentence,
  comparisonOf,
  exposureCoefficients,
  hasBand,
  metricBasis,
} from "./model";
import { PairNavigator } from "./PairNavigator";
import s from "./results.module.css";

export interface ResultsData {
  fit: { artifact: FitArtifact | null; veil: VeilState };
  shelf: ShelfArtifact | null;
  design: DesignArtifact | null;
  split: SplitArtifact | null;
  substitution: { artifact: SubstitutionArtifact | null; veil: VeilState };
}

interface Props {
  pid: string;
  view: ProjectView;
  data: ResultsData;
}

/** The sentence of the newest recorded decision of a kind (the one in effect). */
export function sentenceOf(view: ProjectView, kind: string): string | null {
  const rec = [...view.decisions].reverse().find((d) => d.decision.kind === kind);
  return rec?.sentence ?? null;
}

function energyProvenance(view: ProjectView, design: DesignArtifact | null): string {
  const sentence = sentenceOf(view, "set_energy_adjustment");
  const method = view.state.energy_adjustment?.method;
  const said = sentence ?? (method ? `Energy was adjusted by the ${method} method.` : "No energy adjustment was recorded.");
  return design?.estimand ? `${said} ${design.estimand}` : said;
}

export function Results({ pid, view, data }: Props) {
  const fit = data.fit.artifact;
  const decide = useDecide(pid);
  const cancel = useCancelJob(pid);
  const [pending, setPending] = useState<{ donor: string; recipient: string } | null>(null);
  const [said, setSaid] = useState<string | null>(null);
  // A refused pair answers with the server's ways forward (a nutrient's unit, a part of a total),
  // never a message alone: pressing one records it, and the pair is chosen again.
  const [refused, setRefused] = useState<Refusal | null>(null);

  const opened = !!view.state.seal_opened;
  useRefetchOnOpening(pid, opened, fit, view.stages.fit?.key ?? undefined);
  if (!fit) return null;
  const target = view.state.target ?? "the outcome";
  // The seal (M2_CONTRACT §3): held-out scores reach the comparison only once it is opened.
  const phase = sealPhase(fit, opened);
  const opening = openingRecord(view);
  const openedSeq = opening?.seq ?? null;
  const comparison = comparisonOf(fit, data.shelf, fit.primary_metric, phase);
  const basis = metricBasis(fit, data.split, phase);
  const recorded = `Fitted as recorded. ${basis}`;
  // The family with intervals first (the plain linear model under inference), then the shelf's order.
  const hasCi = (m: (typeof fit.models)[number]) => !!m.coefficients?.some((c) => c.ci_low !== null);
  const linear = fit.models
    .filter((m) => m.coefficients && m.coefficients.length)
    .sort((a, b) => Number(hasCi(b)) - Number(hasCi(a)))
    .map((m) => ({ model: m, coefficients: exposureCoefficients(m.coefficients!, view.state.roles) }));
  const sub = data.substitution.artifact;
  const subStatus = view.stages.substitution;
  const subBusy = subStatus?.status === "running" || subStatus?.status === "queued";
  const folds = data.split?.folds ?? 5;
  const step = view.state.substitution?.step_kcal ?? 100;
  const nBoot = sub?.band?.n_boot ?? view.state.substitution?.n_boot ?? 0;
  const offer = sub && fit ? bandOffer(sub, fit, folds) : null;

  const record = (donor: string, recipient: string, boot = 0) => {
    setPending({ donor, recipient });
    setSaid(null);
    setRefused(null);
    decide.mutate(substitutionDecision(donor, recipient, step, boot), {
      onSuccess: () => setSaid(boot ? `Recorded: a band from ${boot} refits per family.` : `Recorded: ${donor} → ${recipient}.`),
      onError: (e) => (isRefusalError(e) ? setRefused(e.refusal) : setSaid(e.message)),
      onSettled: () => setPending(null),
    });
  };
  const refusal = refused ? (
    <RefusalNote
      refusal={refused}
      onDismiss={() => setRefused(null)}
      onExit={(exit) => {
        setRefused(null);
        if (!exit.decision) return;
        decide.mutate(exit.decision, {
          onSuccess: () => setSaid("Recorded. Choose the pair again to draw its curve."),
          onError: (e) => (isRefusalError(e) ? setRefused(e.refusal) : setSaid(e.message)),
        });
      }}
    />
  ) : null;

  return (
    <div className={s.results} data-testid="results" data-seal-phase={phase}>
      {phase === "post_seal" ? <PostSealBand view={view} fit={fit} openedSeq={openedSeq} /> : null}
      <StaleVeil
        state={data.fit.veil}
        order={0}
        label="Model comparison"
        action={<StageRetry pid={pid} status={view.stages.fit} />}
      >
        <section className={s.section} data-purpose="model_comparison">
          <header className={s.sectionHead}>
            <h3 className={s.kicker}>Models compared</h3>
            <span className={s.sectionAside}>
              <HeldOutAside phase={phase} split={data.split} openedSeq={openedSeq} />
            </span>
            <SaveMenu
              title="Model comparison"
              choices={null}
              build={() =>
                comparisonFigure(comparison, {
                  title: `Model comparison: ${comparison.label} by family`,
                  caption: `Filled dots: cross-validated mean ± SD${phase === "opened" || phase === "post_seal" ? "; hollow dots: held-out rows, scored once" : ""}${comparison.baseline ? `; dashed line: ${comparison.baseline.label}` : ""}.`,
                  provenance: plain(`${energyProvenance(view, data.design)} ${basis}`),
                })
              }
            />
          </header>
          <Comparison data={comparison} fit={fit} basis={basis} phase={phase} glyph={glyphOf(data.split?.basis?.state)} />
        </section>
      </StaleVeil>

      {linear.length ? (
        <StaleVeil
          state={data.fit.veil}
          order={1}
          label="Coefficients"
          action={<StageRetry pid={pid} status={view.stages.fit} />}
        >
          <section className={s.section} data-purpose="coefficients">
            <header className={s.sectionHead}>
              <h3 className={s.kicker}>Coefficients of the exposures</h3>
              <SaveMenu
                title="Coefficients"
                choices={null}
                build={() =>
                  forestFigure(linear[0]!.coefficients, view.state.purpose === "inference", {
                    title: `${linear[0]!.model.label}: coefficients of the exposures`,
                    caption:
                      view.state.purpose === "inference"
                        ? linear[0]!.model.inference?.refused
                          ? `Squares: estimates. ${linear[0]!.model.inference.caption}`
                          : `Squares: estimates; lines: ${linear[0]!.model.inference?.caption ?? "95% confidence intervals."}`
                        : "Squares: estimates of the fitted prediction model, not interpreted as effects.",
                    provenance: plain(recorded),
                  })
                }
              />
            </header>
            <Coefficients models={linear} purpose={view.state.purpose} target={target} />
          </section>
        </StaleVeil>
      ) : null}

      {sub ? (
        <section className={s.section} aria-label="Substitution curves" data-purpose="substitution">
          <StaleVeil
            state={data.substitution.veil}
            order={2}
            label="Substitution curves"
            action={<StageRetry pid={pid} status={subStatus} />}
          >
            <header className={s.sectionHead}>
              <h3 className={s.kicker}>
                Substitution: <code className="v">{sub.donor}</code> → <code className="v">{sub.recipient}</code>
              </h3>
              <SaveMenu
                title={`Substitution ${sub.donor} to ${sub.recipient}`}
                choices={null}
                build={() =>
                  curvesFigure(sub, target, {
                    title: `Moving energy from ${sub.donor} to ${sub.recipient}`,
                    caption: `${sub.estimand ?? ""} ${hasBand(sub) ? (sub.band?.caption ?? `Shaded bands: 95% intervals from ${nBoot || "bootstrap"} refits per family.`) : ""} Hatched: where the models disagree.`,
                    provenance: plain(`${energyProvenance(view, data.design)} ${sub.basis}`),
                  })
                }
              />
            </header>
            <Curves sub={sub} target={target} />
            {carriedSentence(sub.carried, data.design?.nested ?? []) ? (
              <p className={s.caption} data-testid="carried">
                <Rich text={carriedSentence(sub.carried, data.design?.nested ?? [])!} />
              </p>
            ) : null}
            <p className={s.caption}>
              <Rich text={energyProvenance(view, data.design)} />
            </p>
            {sub.note ? (
              <details className={s.details}>
                <summary>How this curve was drawn</summary>
                <p className={s.basis}>
                  <Rich text={sub.note} />
                </p>
              </details>
            ) : null}
            <DesignWarnings warnings={warningsAbout(data.design?.warnings ?? [], "substitution")} />
          </StaleVeil>
          {/* The levers stay live under a veil: a stopped band can be asked for again, and
              another pair is a new answer, whatever the state of this curve. */}
          <div className={s.bandRow}>
              {hasBand(sub) ? (
                <p className={s.bandNote}>
                  {sub.band?.caption ??
                    `Bands: 95% intervals from ${nBoot || "bootstrap"} refits of each family on ${(sub.band?.n_rows ?? BAND_ROWS).toLocaleString("en-US")} training rows.`}
                </p>
              ) : subBusy && nBoot > 0 ? (
                <p className={s.bandNote} role="status">
                  Refitting each family {nBoot} times for the band
                  {subStatus?.progress != null ? ` · ${Math.round(subStatus.progress * 100)}%` : "…"}
                  {subStatus?.job_id ? (
                    <button type="button" className={s.linkButton} onClick={() => cancel.mutate(subStatus.job_id!)}>
                      Stop
                    </button>
                  ) : null}
                </p>
              ) : (
                <button
                  type="button"
                  className={s.bandButton}
                  disabled={decide.isPending || subBusy}
                  onClick={() => record(sub.donor, sub.recipient, offer?.nBoot ?? 0)}
                  data-testid="add-band"
                >
                  Add an uncertainty band (about {aboutSeconds(offer?.seconds ?? 0)})
                </button>
              )}
              {said ? (
                <span className={s.said} role="status">
                  {said}
                </span>
              ) : null}
              {refusal}
            </div>
            {data.design ? (
              <PairNavigator
                pairs={data.design.substitution_pairs}
                current={{ donor: sub.donor, recipient: sub.recipient }}
                pending={pending}
                disabled={decide.isPending}
                onPick={(d, r) => record(d, r)}
              />
            ) : null}
        </section>
      ) : data.design && data.design.substitution_pairs.length ? (
        <section className={s.section} data-purpose="substitution">
          <header className={s.sectionHead}>
            <h3 className={s.kicker}>Substitution</h3>
          </header>
          <p className={s.caption}>Choose a pair to draw its curve: energy moves from the row's nutrient to the column's.</p>
          <PairNavigator
            pairs={data.design.substitution_pairs}
            current={null}
            pending={pending}
            disabled={decide.isPending}
            onPick={(d, r) => record(d, r)}
          />
          {said ? (
            <span className={s.said} role="status">
              {said}
            </span>
          ) : null}
          {refusal}
        </section>
      ) : null}

      {phase === "sealed" && data.fit.veil === "fresh" ? (
        <OpenSealCard pid={pid} fit={fit} split={data.split} metric={comparison.label} />
      ) : phase === "opened" || phase === "post_seal" ? (
        <SealOpenedLine seq={openedSeq} metric={comparison.label} nHoldout={fit.n_holdout} post={phase === "post_seal"} />
      ) : null}
    </div>
  );
}
