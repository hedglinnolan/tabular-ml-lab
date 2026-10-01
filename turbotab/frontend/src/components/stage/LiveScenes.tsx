/**
 * What the stage shows when no option is focused: the recorded pipeline. Before a fit, "your data
 * now" — the row flow and the column lineage of the current state; once fitted, the Results. The
 * banner's segments open each of these full size. Every figure here is recorded fact, captioned
 * with the decisions that produced it.
 */
import type {
  CohortArtifact,
  DesignArtifact,
  FitArtifact,
  Lineage as LineageData,
  RowStep,
  ShelfArtifact,
  SplitArtifact,
} from "../../api/m1-stage-types";
import type { DecisionKind, ProjectView } from "../../api/schema";
import { StaleVeil, type VeilState } from "../../motion/StaleVeil";
import { fmtInt, plain } from "./format";
import { Results, type ResultsData } from "./results/Results";
import { Shelf } from "./results/Shelf";
import { figureSvg, lineagePanel, rowFlowPanel } from "./save/journal";
import { SaveMenu } from "./save/SaveMenu";
import { Rich } from "./text";
import { Lineage } from "./views/Lineage";
import { RowFlow } from "./views/RowFlow";
import s from "./Stage.module.css";

export interface LiveData extends ResultsData {
  cohort: { artifact: CohortArtifact | null; veil: VeilState };
  splitVeil: VeilState;
  designVeil: VeilState;
}

/** The participant flow as recorded: the cohort's steps, then the split's fork. */
export function liveSteps(cohort: CohortArtifact | null, split: SplitArtifact | null): RowStep[] {
  if (!cohort) return [];
  const steps = [...cohort.steps];
  if (split) {
    steps.push({
      key: "train",
      label: split.n_holdout ? "Training rows" : "Training rows (cross-validation only)",
      n: split.n_train,
      dropped: split.n_holdout,
      reason: split.n_holdout ? "held out for the final check, sealed" : null,
      decision_id: null,
    });
    if (split.n_holdout)
      steps.push({ key: "holdout", label: "Held-out rows, sealed", n: split.n_holdout, dropped: 0, reason: null, decision_id: null });
  }
  return steps;
}

/** Before the design exists: the predictors as recorded, entering as they are. */
export function predictorsLineage(predictors: string[], roles: Record<string, string> | null): LineageData {
  const nodes: LineageData["nodes"] = [];
  const links: LineageData["links"] = [];
  for (const c of predictors) {
    const role = (roles?.[c] ?? null) as LineageData["nodes"][number]["role"];
    nodes.push({ id: `raw:${c}`, column: c, lane: "raw", role, label: c, formula: null, group: null, count: 1 });
    nodes.push({ id: `adj:${c}`, column: c, lane: "adjusted", role, label: c, formula: null, group: null, count: 1 });
    nodes.push({ id: `mx:${c}`, column: c, lane: "matrix", role, label: c, formula: null, group: null, count: 1 });
    links.push({ source: `raw:${c}`, target: `adj:${c}`, operation: "kept" });
    links.push({ source: `adj:${c}`, target: `mx:${c}`, operation: "kept" });
  }
  return { nodes, links, collapsed: false };
}

/** The recorded sentences behind a figure, newest of each kind, in asking order. */
export function recordedSentences(view: ProjectView, kinds: DecisionKind[]): string {
  const out: string[] = [];
  for (const kind of kinds) {
    const rec = [...view.decisions].reverse().find((d) => d.decision.kind === kind);
    if (rec?.sentence) out.push(rec.sentence);
  }
  return out.length ? `As recorded: ${out.join(" ")}` : "As recorded.";
}

const ROW_KINDS: DecisionKind[] = ["set_target", "set_exclusions", "set_missing", "set_split"];
const COLUMN_KINDS: DecisionKind[] = ["set_roles", "set_missing", "set_energy_adjustment", "select_models"];

function RowsCard({ view, data, full }: { view: ProjectView; data: LiveData; full: boolean }) {
  const steps = liveSteps(data.cohort.artifact, data.split);
  if (!steps.length) return null;
  const last = steps.filter((x) => x.key !== "holdout").at(-1)!;
  const open = view.interview.find((i) => i.status === "open");
  const openAfter =
    !full && open && ["exclusions", "missing", "split"].includes(open.key)
      ? (open.key === "exclusions" ? "outcome_measured" : steps.filter((x) => x.key !== "train" && x.key !== "holdout").at(-1)?.key) ?? null
      : null;
  const provenance = recordedSentences(view, ROW_KINDS);
  const veil: VeilState = data.cohort.veil !== "fresh" ? data.cohort.veil : data.splitVeil;
  return (
    <StaleVeil state={veil} order={0} label="Rows">
      <section className={full ? s.liveFull : s.liveCard} data-card="rows">
        <header className={s.cardHead}>
          <h3 className={s.kicker}>Rows</h3>
          <span className={s.cardChip}>{fmtInt(last.n)} rows</span>
          <SaveMenu
            title="Participant flow"
            choices={null}
            build={() =>
              figureSvg({
                title: "Participant flow: rows at each step",
                caption: "Bars: rows remaining after each step; hatched: rows the step removes.",
                provenance: plain(provenance),
                panels: [{ label: "", body: rowFlowPanel(steps) }],
                panelHeight: Math.max(160, steps.length * 30 + 30),
              })
            }
          />
        </header>
        <RowFlow
          steps={steps}
          compact={!full}
          reasons={full}
          openAfter={openAfter}
          openLabel={open ? `${open.key.replace(/_/g, " ")}: the open question acts here` : undefined}
        />
        {full ? <Provenance text={provenance} /> : null}
      </section>
    </StaleVeil>
  );
}

/** The recorded decisions behind a figure: one press away, not a wall under it. */
function Provenance({ text }: { text: string }) {
  return (
    <details className={s.provenance}>
      <summary>The recorded decisions behind this figure</summary>
      <p className={s.captionText}>
        <Rich text={text} />
      </p>
    </details>
  );
}

function ColumnsCard({ view, data, full }: { view: ProjectView; data: LiveData; full: boolean }) {
  const lineage =
    data.design?.lineage ??
    (data.cohort.artifact ? predictorsLineage(data.cohort.artifact.predictors, view.state.roles) : null);
  if (!lineage) return null;
  const provenance = recordedSentences(view, COLUMN_KINDS);
  const rawRows = lineage.nodes.filter((n) => n.lane === "raw").length;
  const open = view.interview.find((i) => i.status === "open");
  const openLabel = !full && open?.key === "energy_adjustment" ? "energy adjustment acts here" : null;
  const matrix = data.design?.matrix;
  return (
    <StaleVeil state={data.design ? data.designVeil : "fresh"} order={1} label="Columns">
      <section className={full ? s.liveFull : s.liveCard} data-card="lineage">
        <header className={s.cardHead}>
          <h3 className={s.kicker}>Columns</h3>
          <span className={s.cardChip}>
            {matrix ? `model matrix ${fmtInt(matrix.n_cols)} columns` : "predictors as recorded, before encoding"}
          </span>
          <SaveMenu
            title="Column lineage"
            choices={null}
            build={() =>
              figureSvg({
                title: "Column lineage: raw columns to the model matrix",
                caption: "Solid links: columns a recorded choice rewrites; gray links: columns passed through.",
                provenance: plain(provenance),
                panels: [{ label: "", body: lineagePanel(lineage) }],
                panelHeight: Math.max(240, lineage.nodes.filter((n) => n.lane === "raw").length * 22 + 40),
              })
            }
          />
        </header>
        <div
          className={s.lineageBox}
          style={{ height: full ? Math.max(360, rawRows * 26 + 60) : Math.min(380, Math.max(220, rawRows * 15 + 60)) }}
        >
          <Lineage lineage={lineage} openLabel={openLabel} />
        </div>
        {full ? <Provenance text={provenance} /> : null}
      </section>
    </StaleVeil>
  );
}

export function NowScene({ view, data }: { view: ProjectView; data: LiveData }) {
  const waiting = view.interview.filter((i) => i.status === "open" || i.status === "waiting").map((i) => i.key.replace(/_/g, " "));
  if (!data.cohort.artifact) {
    return (
      <p className={s.emptyLine}>
        Nothing is recorded downstream yet. Hover or focus an option to see what it would do to your rows and columns.
      </p>
    );
  }
  return (
    <div className={s.liveStack}>
      <RowsCard view={view} data={data} full={false} />
      <ColumnsCard view={view} data={data} full={false} />
      <section className={s.liveCard} data-card="results">
        <header className={s.cardHead}>
          <h3 className={s.kicker}>Results</h3>
        </header>
        <p className={s.captionText}>
          Nothing has been fit yet.{waiting.length ? ` It waits on ${waiting.slice(0, 3).join(", then ")}.` : ""}
        </p>
      </section>
    </div>
  );
}

export function RowsScene({ view, data }: { view: ProjectView; data: LiveData }) {
  return <RowsCard view={view} data={data} full />;
}

export function ColumnsScene({ view, data }: { view: ProjectView; data: LiveData }) {
  return <ColumnsCard view={view} data={data} full />;
}

export function ModelsScene({
  view,
  shelf,
  fit,
}: {
  view: ProjectView;
  shelf: ShelfArtifact | null;
  fit: FitArtifact | null;
}) {
  if (!shelf) return <p className={s.emptyLine}>The shelf is drawn once the outcome and the roles are recorded.</p>;
  return <Shelf shelf={shelf} chosen={view.state.models} fit={fit} />;
}

export function ResultsScene({ pid, view, data }: { pid: string; view: ProjectView; data: LiveData }) {
  if (!data.fit.artifact) {
    const waiting = view.interview.filter((i) => i.status === "open" || i.status === "waiting").map((i) => i.key.replace(/_/g, " "));
    return (
      <p className={s.emptyLine}>
        Nothing has been fit yet.{waiting.length ? ` It waits on ${waiting.slice(0, 3).join(", then ")}.` : ""}
      </p>
    );
  }
  return <Results pid={pid} view={view} data={data} />;
}

export type { DesignArtifact };
