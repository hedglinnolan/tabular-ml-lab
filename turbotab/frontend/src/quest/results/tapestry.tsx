/**
 * Predict Results' tapestry (plans/W1_PART2_PLAN.md §2 W1-G; rulings 2 and 5): the exhibits, each
 * through a view kind (FOUNDATION §5 rule 9).
 *
 *   1. the cross-validated comparison of every metric, with intervals, the baseline and BBC-CV;
 *   2. once the held-out rows are opened, their scores beside the no-predictor baseline's and the
 *      held-out verdict (sealed until then: no held-out number is read before the opening), titled
 *      by whether they are still the opening's, and labeled exploratory, with the split's reason,
 *      wherever its basis is not a verified clean one (the calibration too);
 *   3. the final model's held-out calibration (the served curve, intercept and slope; binned
 *      points wait for W2, ruling 5);
 *   4. what drives each model: SHAP importance as a table, ALE curves with their support mask;
 *   5. evaluation's word on what the interpretable model costs.
 *
 * Every number is a served field, formatted by the view; nothing is computed here.
 */
import type { FitArtifact } from "../../api/m1-stage-types";
import type { EvaluationArtifact, ExplainArtifact } from "../../api/m3-types";
import type { ProjectView, StageStatus } from "../../api/schema";
import { StaleVeil, type VeilState } from "../../motion/StaleVeil";
import { useM1Stage } from "../../components/stage/data";
import { fmtInt } from "../../components/stage/format";
import r from "../../components/stage/results/results.module.css";
import { openingRecord, PostSealBand, useRefetchOnOpening } from "../../components/stage/seal/OpenSeal";
import { sealPhase, type SealPhase } from "../../components/stage/seal/phase";
import { Rich } from "../../components/stage/text";
import { CalibrationView } from "../../components/views/CalibrationView";
import { calibrationFromEngine, curveFromExplain, pastPalette } from "../../components/views/curveAdapters";
import { CurveView } from "../../components/views/CurveView";
import { ExhibitTable } from "../../components/views/ExhibitTable";
import { OneLine } from "../../components/views/common/parts";
import { cvComparison, exploratoryReasons, heldOutTable, importanceTable, interpretableCost, type SplitBasis } from "./model";
import s from "./results.module.css";

const busy = (status: StageStatus | undefined) => status?.status === "running" || status?.status === "queued";

export function Tapestry({ pid, view }: { pid: string; view: ProjectView }) {
  const fit = useM1Stage(pid, view, "fit");
  const explain = useM1Stage(pid, view, "explain");
  const evaluation = useM1Stage(pid, view, "evaluation");
  const split = useM1Stage(pid, view, "split").artifact;
  const opened = !!view.state.seal_opened;
  useRefetchOnOpening(pid, opened, fit.artifact, view.stages.fit?.key ?? undefined);
  const art = fit.artifact;
  if (!art) {
    return (
      <div className={s.tapestry} data-testid="results-tapestry">
        <OneLine view="table" text={busy(view.stages.fit) ? "The models are being fitted." : "No model is fitted yet."} />
      </div>
    );
  }
  // Nothing scored before Fit is pressed, or while a question an estimate rests on is open
  // (FOUNDATION §5 rule 6): the engine's own words for why.
  if (art.withheld_for) {
    return (
      <div className={s.tapestry} data-testid="results-tapestry">
        <OneLine view="table" text={<Rich text={art.withheld ?? "No score is shown yet."} />} />
      </div>
    );
  }
  const phase = sealPhase(art, opened);
  const target = view.state.target ?? "the outcome";
  return (
    <div className={s.tapestry} data-testid="results-tapestry" data-seal-phase={phase}>
      {phase === "post_seal" ? <PostSealBand view={view} fit={art} openedSeq={openingRecord(view)?.seq ?? null} /> : null}
      <StaleVeil state={fit.veil} order={0} label="Models compared">
        <section className={s.exhibit} data-purpose="cv_comparison" data-testid="results-cv">
          <ExhibitTable data={cvComparison(art)} gate={null} />
        </section>
      </StaleVeil>
      <StaleVeil state={fit.veil} order={1} label="Held-out rows">
        <section className={s.exhibit} data-purpose="held_out" data-testid="results-held-out">
          <HeldOut fit={art} phase={phase} split={split} />
        </section>
      </StaleVeil>
      <StaleVeil state={fit.veil} order={2} label="Calibration">
        <section className={s.exhibit} data-purpose="calibration" data-testid="results-calibration">
          <Calibration fit={art} phase={phase} target={target} split={split} />
        </section>
      </StaleVeil>
      <Explanations artifact={explain.artifact} veil={explain.veil} status={view.stages.explain} task={art.task} target={target} />
      <Interpretable artifact={evaluation.artifact} veil={evaluation.veil} />
    </div>
  );
}

function HeldOut({ fit, phase, split }: { fit: FitArtifact; phase: SealPhase; split: SplitBasis | null }) {
  if (phase === "cv_only") return <OneLine view="table" title="Held-out rows" text="No rows were held out." />;
  const table = heldOutTable(fit, phase, split);
  if (!table) {
    return (
      <OneLine
        view="table"
        title="Held-out rows"
        text={`The ${fmtInt(fit.n_holdout)} held-out rows are sealed: no score on them is shown until they are opened.`}
      />
    );
  }
  return <ExhibitTable data={table} gate={null} />;
}

/** The final model's held-out calibration, once opened; the engine's note where it served none. An
 *  exploratory split labels it as it labels the held-out table, whose footnote says why. */
function Calibration({ fit, phase, target, split }: { fit: FitArtifact; phase: SealPhase; target: string; split: SplitBasis | null }) {
  const title = "Calibration on the held-out rows";
  if (phase === "cv_only") return null;
  if (phase === "sealed" || phase === "none") {
    return <OneLine view="calibration" title={title} text="Calibration on the held-out rows is drawn once they are opened." />;
  }
  const final = fit.models.find((m) => m.family === fit.final_model) ?? null;
  const detail = final?.holdout_detail ?? null;
  if (!final || !detail) return <OneLine view="calibration" title={title} text="No final model's held-out calibration was served." />;
  if (!detail.calibration) {
    return <OneLine view="calibration" title={title} text={<Rich text={detail.calibration_note ?? "Calibration was not assessed on the held-out rows."} />} />;
  }
  const kind = fit.task === "binary" ? "risk" : "value";
  return (
    <CalibrationView
      title={`${final.label}: ${title.toLowerCase()}${exploratoryReasons(split) ? ", exploratory" : ""}`}
      data={calibrationFromEngine(detail.calibration, { kind, outcome: target, where: "held_out", sealed: null })}
    />
  );
}

function Explanations({
  artifact,
  veil,
  status,
  task,
  target,
}: {
  artifact: ExplainArtifact | null;
  veil: VeilState;
  status: StageStatus | undefined;
  task: string;
  target: string;
}) {
  const title = "What drives each model";
  if (!artifact) {
    const text = busy(status)
      ? "Explaining each fitted model…"
      : status?.status === "blocked" && status.missing.includes("explain")
        ? "No explanation is asked for yet."
        : status?.status === "blocked"
          ? "The explanations wait for an earlier answer."
        : status?.status === "error"
          ? `The explanations did not finish: ${status.error ?? "no reason was given"}.`
          : "No explanation is served yet.";
    return (
      <section className={s.exhibit} data-purpose="explain" data-testid="results-explain">
        <OneLine view="table" title={title} text={text} />
      </section>
    );
  }
  if (artifact.withheld) {
    return (
      <section className={s.exhibit} data-purpose="explain" data-testid="results-explain">
        <OneLine view="table" title={title} text={<Rich text={artifact.withheld} />} />
      </section>
    );
  }
  return (
    <StaleVeil state={veil} order={3} label={title}>
      <section className={s.exhibit} data-purpose="explain" data-testid="results-explain">
        <h3 className={s.exhibitTitle}>{title}</h3>
        <p className={r.basis}>
          <Rich text={`${artifact.describes} ${artifact.rows_basis}`} />
        </p>
        {artifact.families.map((f) =>
          f.explained ? (
            <div key={f.family} data-testid="results-importance" data-family={f.family}>
              <ExhibitTable data={importanceTable(f)} gate={null} />
            </div>
          ) : (
            <p key={f.family} className={r.basis} data-testid="results-not-explained" data-family={f.family}>
              <Rich text={f.reason ?? `${f.label} was not explained.`} />
            </p>
          ),
        )}
        {artifact.curves.map((ic) => {
          const past = pastPalette(ic);
          return (
            <div key={ic.input} data-testid="results-curve" data-input={ic.input}>
              <CurveView title={`How the predictions move with ${ic.input}`} data={curveFromExplain(ic, { outcome: target, task })} />
              {past ? (
                <p className={r.basis} data-testid="results-curve-past-palette">
                  {past}
                </p>
              ) : null}
              {ic.curves
                .filter((c) => !c.drawn && c.reason)
                .map((c) => (
                  <p key={c.family} className={r.basis} data-testid="results-curve-not-drawn">
                    <Rich text={c.reason!} />
                  </p>
                ))}
            </div>
          );
        })}
        {artifact.notes.map((n, i) => (
          <p key={i} className={r.basis}>
            <Rich text={n} />
          </p>
        ))}
        <details className={r.details}>
          <summary>How these were computed</summary>
          <p className={r.basis}>
            <Rich text={artifact.methods} />
          </p>
        </details>
      </section>
    </StaleVeil>
  );
}

function Interpretable({ artifact, veil }: { artifact: EvaluationArtifact | null; veil: VeilState }) {
  if (!artifact) return null;
  const text = artifact.scores_shown ? interpretableCost(artifact) : (artifact.note ?? null);
  if (!text) return null;
  return (
    <StaleVeil state={veil} order={4} label="What the interpretable model costs">
      <section className={s.exhibit} data-purpose="interpretable_cost" data-testid="results-interpretable">
        <h3 className={s.exhibitTitle}>What the interpretable model costs</h3>
        <p className={r.caption}>
          <Rich text={text} />
        </p>
      </section>
    </StaleVeil>
  );
}
