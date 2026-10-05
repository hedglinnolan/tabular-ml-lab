/**
 * The Results section of the document: the participant flow (STROBE 13 / TRIPOD+AI 20), Table 2
 * for the exposure only with the adjustment terms one press away (STROBE 16; the Table 2 fallacy,
 * Westreich & Greenland 2013), the declared secondary and sensitivity analyses with the
 * specification-curve teaser (STROBE 17), and, under prediction, the cross-validated performance
 * (TRIPOD+AI 23). Every number is a server estimate; every caption a server sentence.
 */
import { useState } from "react";
import type { CohortArtifact } from "../../api/m1-types";
import { Rich } from "../../components/stage/text";
import {
  artifactOf,
  type EffectsArtifact,
  type FitLite,
  type Moment,
  type SecondaryArtifact,
  type SensitivityArtifact,
} from "./fixture";
import type { SpecRow } from "./views/SpecCurve";
import s from "./doc.module.css";

const n = (x: number) => x.toLocaleString("en-US");
const est = (x: number | null | undefined, d = 3) =>
  x === null || x === undefined ? "—" : x < 0 ? `−${Math.abs(x).toFixed(d)}` : x.toFixed(d);

export function Flow({ m }: { m: Moment }) {
  const cohort = artifactOf<CohortArtifact>(m, "cohort");
  const fit = artifactOf<FitLite>(m, "fit");
  if (!cohort) return null;
  return (
    <table className={s.flow}>
      <tbody>
        {cohort.steps.map((st) => (
          <tr key={st.key}>
            <td>
              <Rich text={st.label} />
              {st.dropped ? (
                <span className={s.flowWhy}>
                  {" "}
                  − {n(st.dropped)} <Rich text={st.reason ?? ""} />
                </span>
              ) : null}
            </td>
            <td className={s.num}>{n(st.n)}</td>
          </tr>
        ))}
        {fit && fit.n_holdout ? (
          <>
            <tr>
              <td>Training rows (cross-validated)</td>
              <td className={s.num}>{n(fit.n_train)}</td>
            </tr>
            <tr>
              <td>Held-out rows, sealed</td>
              <td className={s.num}>{n(fit.n_holdout)}</td>
            </tr>
          </>
        ) : null}
      </tbody>
    </table>
  );
}

export function TableTwo({ m }: { m: Moment }) {
  const effects = artifactOf<EffectsArtifact>(m, "effects");
  const fit = artifactOf<FitLite>(m, "fit");
  const [appendix, setAppendix] = useState(false);
  const family = effects?.families[0];
  if (!effects || !family) return null;
  const terms = fit?.models[0]?.adjustment_terms ?? [];
  const seq = family.sequence;
  return (
    <figure className={s.t2}>
      <figcaption className={s.t2Title}>
        <span className={s.t2No}>Table 2.</span> <Rich text={fit?.estimand?.caption ?? ""} />
      </figcaption>
      <div className={s.t2Wrap}>
        <table className={s.t2Table}>
          <thead>
            <tr>
              <th scope="col" />
              {seq.map((x) => (
                <th key={x.key} scope="col" data-primary={x.key === "model_2" || undefined}>
                  {x.label}
                </th>
              ))}
            </tr>
          </thead>
          <tbody>
            <tr>
              <th scope="row">
                <code className="v">{effects.exposure}</code>
              </th>
              {seq.map((x) => {
                const e = x.effects[0];
                return (
                  <td key={x.key} data-primary={x.key === "model_2" || undefined}>
                    <span className={s.t2Est}>{est(e?.estimate)}</span>
                    <span className={s.t2Ci}>
                      ({est(e?.ci_low)}, {est(e?.ci_high)})
                    </span>
                  </td>
                );
              })}
            </tr>
            <tr className={s.t2N}>
              <th scope="row">n</th>
              {seq.map((x) => (
                <td key={x.key}>{n(x.n_rows)}</td>
              ))}
            </tr>
          </tbody>
        </table>
      </div>
      <ol className={s.t2Notes}>
        {seq.map((x) => (
          <li key={x.key}>
            <b>{x.label}:</b> <Rich text={x.note} />
          </li>
        ))}
        {effects.model_1 && !effects.model_1.declared ? (
          <li className={s.t2Missing}>
            <b>Model 1</b> was not declared before the estimates were shown: <Rich text={effects.model_1.reason} />
          </li>
        ) : null}
      </ol>
      <p className={s.t2Inference}>
        <Rich text={seq[1]?.inference?.caption ?? ""} />
      </p>
      <button type="button" className={s.appendixToggle} aria-expanded={appendix} onClick={() => setAppendix((v) => !v)}>
        Appendix: {effects.appendix_title} · {terms.length} rows {appendix ? "▾" : "▸"}
      </button>
      {appendix ? (
        <table className={s.appendix}>
          <tbody>
            {terms.map((t) => (
              <tr key={t.feature}>
                <td>
                  <code className="v">{t.feature}</code>
                </td>
                <td className={s.num}>{est(t.estimate, 2)}</td>
                <td className={s.why}>{t.why}</td>
              </tr>
            ))}
          </tbody>
        </table>
      ) : null}
    </figure>
  );
}

export function specRows(m: Moment): { rows: SpecRow[]; unit: string; exposure: string } | null {
  const effects = artifactOf<EffectsArtifact>(m, "effects");
  const sens = artifactOf<SensitivityArtifact>(m, "sensitivity");
  const family = effects?.families[0];
  if (!effects || !family) return null;
  const rows: SpecRow[] = [];
  const primary = family.sequence.find((x) => x.key === "model_2");
  for (const x of family.sequence) {
    const e = x.effects[0];
    if (!e || e.ci_low === null || e.ci_high === null) continue;
    rows.push({
      key: x.key,
      label: x.label,
      changes:
        x.key === "crude"
          ? "adjustment set: none"
          : x.key === "model_2"
            ? "the reported estimate"
            : `adjustment set: + ${x.adjusted_for.filter((c) => !primary?.adjusted_for.includes(c)).join(", ")}`,
      estimate: e.estimate,
      low: e.ci_low,
      high: e.ci_high,
      n: x.n_rows,
      primary: x.key === "model_2",
      source: x.note,
    });
  }
  for (const fit of sens?.families[0]?.fits ?? []) {
    if (fit.label === "Primary") continue;
    const e = fit.coefficients?.find((c) => c.feature === effects.exposure);
    if (!e || e.ci_low === null || e.ci_high === null) continue;
    rows.push({
      key: `sens:${fit.label}`,
      label: fit.label,
      changes: "eligibility: no exclusion",
      estimate: e.estimate,
      low: e.ci_low,
      high: e.ci_high,
      n: fit.n_rows,
      primary: false,
      source: sens?.methods ?? "",
    });
  }
  const unit = family.sequence[0]?.inference?.effect ?? "";
  return { rows, unit, exposure: effects.exposure };
}

export function OtherAnalyses({ m, onSpec, specOn }: { m: Moment; onSpec: () => void; specOn: boolean }) {
  const sec = artifactOf<SecondaryArtifact>(m, "secondary");
  const sens = artifactOf<SensitivityArtifact>(m, "sensitivity");
  const spec = specRows(m);
  return (
    <div className={s.other}>
      {spec ? (
        <button type="button" className={s.teaser} onClick={onSpec} aria-pressed={specOn} data-testid="spec-teaser">
          <span className={s.teaserQ}>Which of my decisions mattered?</span>
          <span className={s.teaserA}>
            {spec.rows.length} declared specifications, from {est(Math.min(...spec.rows.map((r) => r.estimate)))} to{" "}
            {est(Math.max(...spec.rows.map((r) => r.estimate)))}
          </span>
        </button>
      ) : null}
      {spec ? (
        <p className={s.otherText}>
          <Rich
            text={`Declared beside the primary: ${[
              ...(sec?.families[0]?.fits.filter((f) => f.label !== "Primary").map((f) => f.label.toLowerCase()) ?? []),
              ...(sens?.analyses.filter((a) => !a.primary).map((a) => `“${a.label}”`) ?? []),
            ].join("; ")}. Each estimate is in the specification curve.`}
          />
        </p>
      ) : null}
    </div>
  );
}

export function Performance({ m }: { m: Moment }) {
  const fit = artifactOf<FitLite>(m, "fit");
  const model = fit?.models[0];
  if (!fit || !model) return null;
  const metrics = ["r2", "rmse", "mae"].filter((k) => model.cv[k]);
  return (
    <figure className={s.t2}>
      <figcaption className={s.t2Title}>
        <span className={s.t2No}>Table 3.</span> {model.label}: cross-validated on the {n(fit.n_train)} training rows
        {fit.holdout_sealed && fit.n_holdout ? `; the ${n(fit.n_holdout)} held-out rows are still sealed` : ""}.
      </figcaption>
      <table className={s.t2Table}>
        <thead>
          <tr>
            <th scope="col">Metric</th>
            <th scope="col">Estimate</th>
            <th scope="col">95% CI</th>
          </tr>
        </thead>
        <tbody>
          {metrics.map((k) => {
            const c = model.cv[k]!;
            return (
              <tr key={k}>
                <th scope="row">{fit.metric_labels[k] ?? k}</th>
                <td>{est(c.estimate)}</td>
                <td className={s.t2Ci}>
                  {est(c.ci_low)} to {est(c.ci_high)}
                </td>
              </tr>
            );
          })}
        </tbody>
      </table>
    </figure>
  );
}
