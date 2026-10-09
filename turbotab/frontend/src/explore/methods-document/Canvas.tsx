/**
 * The canvas: the right half of the working window shows the focused phrase or slot.
 *
 *   live, option   the production <Stage> (its views, the transform player, the Results), fed by
 *                  the captured server answers through the query cache
 *   evidence       a reading's evidence on the user's columns (the server draws no preview for it)
 *   exposure       the estimand: the sentence the record will hold, and where the exposure sits
 *   adjustment     the adjustment set as a lineage of derived roles
 *   spec           "Which of my decisions mattered?" after the plan is locked
 *
 * The custom canvases wear the stage's own bar, so a preview, evidence and the record read alike.
 */
import { useMemo, type ReactNode } from "react";
import type { CohortArtifact, Lineage as LineageData, RoleProposal } from "../../api/m1-types";
import type { ProjectView } from "../../api/schema";
import { Stage } from "../../components/stage/Stage";
import { StageBar } from "../../components/stage/StageBar";
import { Rich } from "../../components/stage/text";
import { createPlayerStore, PlayerContext } from "../../components/stage/usePlayer";
import { Lineage } from "../../components/stage/views/Lineage";
import type { StageFocus } from "../../state/focus";
import { artifactOf, isPreview, moment, type AdjustmentCard, type Moment } from "./fixture";
import { AdjustmentLanes, type Lane } from "./views/AdjustmentLanes";
import { ReadingEvidence, type EvidenceItem } from "./views/ReadingEvidence";
import { SpecCurve } from "./views/SpecCurve";
import { specRows } from "./Results";
import ss from "../../components/stage/Stage.module.css";
import s from "./doc.module.css";

export type CanvasMode = "live" | "option" | "evidence" | "exposure" | "adjustment" | "spec";

function Frame({ pill, label, aside, children }: { pill: string | null; label: string; aside: ReactNode; children: ReactNode }) {
  return (
    <section className={ss.stage} aria-label="Canvas" data-testid="canvas">
      <StageBar pill={pill} label={label} aside={aside} loading={false} />
      <div className={s.canvasBody}>{children}</div>
    </section>
  );
}

export function Canvas({
  mode,
  pid,
  m,
  focus,
  onFocus,
  evidence,
  exposure,
  adjustmentGroup,
  placed,
  confirmed,
}: {
  mode: CanvasMode;
  pid: string;
  m: Moment;
  focus: StageFocus;
  onFocus: (f: StageFocus) => void;
  evidence: { title: string; items: EvidenceItem[]; label: string } | null;
  exposure: { column: string; contrast: string; effect: string } | null;
  adjustmentGroup: string | null;
  /** The adjustment slot's answers so far (nothing recorded). */
  placed: Record<string, Lane>;
  confirmed: string[];
}) {
  const view = m.view;
  if (mode === "evidence" && evidence) {
    const roles = artifactOf<{ columns: RoleProposal[] }>(m, "roles");
    return (
      <Frame pill="Evidence" label={evidence.label} aside="your data as loaded">
        <ReadingEvidence items={evidence.items} proposals={roles?.columns ?? []} title={evidence.title} />
      </Frame>
    );
  }
  if (mode === "exposure" && exposure) return <ExposureCanvas m={m} choice={exposure} />;
  if (mode === "adjustment") {
    const card = artifactOf<{ adjustment: AdjustmentCard | null }>(m, "proposals")?.adjustment;
    const cohort = artifactOf<CohortArtifact>(m, "cohort");
    const last = cohort?.steps.at(-1);
    if (card)
      return (
        <Frame pill="Preview" label="Where the answers send each covariate" aside="nothing is recorded">
          <AdjustmentLanes
            card={card}
            outcome={view.state.target ?? ""}
            focusGroup={adjustmentGroup}
            placed={placed}
            confirmed={confirmed}
          />
          {last ? (
            <p className={s.readout}>
              <span className={s.readoutKicker}>Rows now</span>
              <Rich text={last.label} /> <span className="num">{last.n.toLocaleString("en-US")}</span> of{" "}
              <span className="num">{cohort!.steps[0]!.n.toLocaleString("en-US")}</span>: every covariate counts
              toward complete cases until the set says which enter the model.
            </p>
          ) : null}
        </Frame>
      );
  }
  if (mode === "spec") {
    const spec = specRows(m);
    if (spec)
      return (
        <Frame pill={null} label="Which of my decisions mattered?" aside="declared before the estimates were shown">
          <SpecCurve rows={spec.rows} unit={spec.unit} exposure={spec.exposure} />
        </Frame>
      );
  }
  return <Stage pid={pid} view={view} focus={focus} onFocus={onFocus} />;
}

/** The estimand before it is recorded: the server draws nothing on the data for it ("Nothing about
 *  this choice can be shown on your data yet"), so the canvas shows the sentence the record will
 *  hold — the server's own, captured when this choice was recorded — and the exposure among the
 *  energy sources it is weighed against. */
function ExposureCanvas({ m, choice }: { m: Moment; choice: { column: string; contrast: string; effect: string } }) {
  const view: ProjectView = m.view;
  const preview = Object.values(m.previews).find((p) => {
    const d = p.decision as { exposure?: string; contrast?: string };
    return d.exposure === choice.column && d.contrast === choice.contrast;
  });
  const note = preview && isPreview(preview.body) ? preview.body.note : null;
  const store = useMemo(() => createPlayerStore(), []);
  const cohort = artifactOf<CohortArtifact>(m, "cohort");
  const roles = useMemo(() => view.state.roles ?? {}, [view.state.roles]);
  const lineage = useMemo<LineageData | null>(() => {
    const cols = cohort?.predictors ?? [];
    if (!cols.length) return null;
    const nodes: LineageData["nodes"] = [];
    const links: LineageData["links"] = [];
    for (const c of cols) {
      const role = (roles[c] ?? null) as LineageData["nodes"][number]["role"];
      nodes.push({ id: `raw:${c}`, column: c, lane: "raw", role, label: c, formula: null, group: null, count: 1 });
      nodes.push({ id: `adj:${c}`, column: c, lane: "adjusted", role, label: c, formula: null, group: null, count: 1 });
      nodes.push({ id: `mx:${c}`, column: c, lane: "matrix", role, label: c, formula: null, group: null, count: 1 });
      links.push({ source: `raw:${c}`, target: `adj:${c}`, operation: "kept" });
      links.push({ source: `adj:${c}`, target: `mx:${c}`, operation: "kept" });
    }
    return { nodes, links, collapsed: false };
  }, [cohort, roles]);
  const energy = useMemo(
    () => new Set(Object.entries(roles).filter(([, r]) => r === "exposure" || r === "energy").map(([c]) => c)),
    [roles],
  );
  const sentence = recordedEstimand(choice);
  return (
    <Frame
      pill="Preview"
      label={[`\`${choice.column}\``, choice.effect && `${choice.effect} effect`, choice.contrast].filter(Boolean).join(" · ")}
      aside="nothing is recorded"
    >
      {note ? (
        <p className={s.canvasNote}>
          <Rich text={note} />
        </p>
      ) : null}
      {sentence ? (
        <div className={s.willRead}>
          <span className={s.readoutKicker}>The record will read</span>
          <p className={s.willReadText}>
            <Rich text={sentence} />
          </p>
        </div>
      ) : null}
      {lineage ? (
        <PlayerContext.Provider value={store}>
          <div className={s.canvasLineage}>
            <span className={s.readoutKicker}>The exposure among the energy sources it is weighed against</span>
            <div className={s.lineageBox}>
              <Lineage lineage={lineage} emphasis={[choice.column]} narrow={energy} />
            </div>
          </div>
        </PlayerContext.Provider>
      ) : null}
    </Frame>
  );
}

/** The server's sentence for an estimand, captured when the journey recorded it (the moment after
 *  it, "adjustment"). Only the choice that was recorded has one; any other choice shows none
 *  rather than a composed one. */
function recordedEstimand(choice: { column: string; contrast: string; effect: string }): string | null {
  const rec = moment("adjustment").view.decisions.find((d) => {
    const x = d.decision as { kind: string; exposure?: string; contrast?: string; effect?: string };
    return x.kind === "set_estimand" && x.exposure === choice.column && x.contrast === choice.contrast && x.effect === choice.effect;
  });
  return rec?.sentence ?? null;
}
