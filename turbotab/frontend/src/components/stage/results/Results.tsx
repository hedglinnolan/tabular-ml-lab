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
import type { ProjectView } from "../../../api/schema";
import { StaleVeil, type VeilState } from "../../../motion/StaleVeil";
import { plain } from "../format";
import { comparisonFigure, curvesFigure, forestFigure } from "../save/resultsJournal";
import { SaveMenu } from "../save/SaveMenu";
import { Rich } from "../text";
import { Coefficients } from "./Coefficients";
import { Comparison } from "./Comparison";
import { Curves } from "./Curves";
import {
  aboutSeconds,
  BAND_BOOT,
  BAND_ROWS,
  bandSeconds,
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

  if (!fit) return null;
  const target = view.state.target ?? "the outcome";
  const comparison = comparisonOf(fit, data.shelf);
  const basis = metricBasis(fit, data.split);
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
  const nBoot = (view.state.substitution as { n_boot?: number } | null)?.n_boot ?? sub?.n_boot ?? 0;

  const record = (donor: string, recipient: string, boot = 0) => {
    setPending({ donor, recipient });
    setSaid(null);
    decide.mutate(substitutionDecision(donor, recipient, step, boot), {
      onSuccess: () => setSaid(boot ? `Recorded: a band from ${boot} refits per family.` : `Recorded: ${donor} → ${recipient}.`),
      onError: (e) => setSaid(isRefusalError(e) ? e.refusal.error.message : e.message),
      onSettled: () => setPending(null),
    });
  };

  return (
    <div className={s.results} data-testid="results">
      <StaleVeil state={data.fit.veil} order={0} label="Model comparison">
        <section className={s.section}>
          <header className={s.sectionHead}>
            <h3 className={s.kicker}>Models compared</h3>
            <SaveMenu
              title="Model comparison"
              choices={null}
              build={() =>
                comparisonFigure(comparison, {
                  title: `Model comparison: ${comparison.label} by family`,
                  caption: `Filled dots: cross-validated mean ± SD; hollow dots: held-out rows${comparison.baseline ? `; dashed line: ${comparison.baseline.label}` : ""}.`,
                  provenance: plain(`${energyProvenance(view, data.design)} ${basis}`),
                })
              }
            />
          </header>
          <Comparison data={comparison} fit={fit} basis={basis} />
        </section>
      </StaleVeil>

      {linear.length ? (
        <StaleVeil state={data.fit.veil} order={1} label="Coefficients">
          <section className={s.section}>
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
                        ? "Squares: estimates; lines: 95% confidence intervals."
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
        <StaleVeil state={data.substitution.veil} order={2} label="Substitution curves">
          <section className={s.section}>
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
                    caption: `${sub.estimand ?? ""} ${hasBand(sub) ? `Shaded bands: 95% intervals from ${nBoot || "bootstrap"} refits per family.` : ""} Hatched: where the models disagree.`,
                    provenance: plain(`${energyProvenance(view, data.design)} ${sub.basis}`),
                  })
                }
              />
            </header>
            <Curves sub={sub} target={target} />
            <div className={s.bandRow}>
              {hasBand(sub) ? (
                <p className={s.bandNote}>
                  Bands: 95% intervals from {nBoot || "bootstrap"} refits of each family on up to {BAND_ROWS.toLocaleString("en-US")} training
                  rows.
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
                  onClick={() => record(sub.donor, sub.recipient, BAND_BOOT)}
                  data-testid="add-band"
                >
                  Add an uncertainty band (about {aboutSeconds(bandSeconds(sub, fit, folds))})
                </button>
              )}
              {said ? (
                <span className={s.said} role="status">
                  {said}
                </span>
              ) : null}
            </div>
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
        </StaleVeil>
      ) : data.design && data.design.substitution_pairs.length ? (
        <section className={s.section}>
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
        </section>
      ) : null}
    </div>
  );
}
