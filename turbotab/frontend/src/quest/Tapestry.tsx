/**
 * The tapestry (FOUNDATION §3, §5): the dynamic window beside the card. Before the fit it is today's
 * stage (record/stage's previews: what the pointed option does to the data, the live scenes) under
 * the kit's canvas head, and under Predict it shows no fit until Fit is pressed (§7: what waits is
 * what is shown; log.ts `fitWithheld`); at Models' end it is the analysis flowchart (today's
 * lineage view) with Fit at its end (§7); in Results it is W1-G's ResultsTapestry inside the kit's
 * canvas; in Write-up the manuscript at full width, in the export bundle's sections (P8).
 */
import type { MethodsSections } from "../api/m3-types";
import type { ProjectView, QuestLog } from "../api/schema";
import { Stage } from "../components/stage/Stage";
import { ManuscriptBody, kit as k } from "../kit";
import { useStageFocus, type StageFocus } from "../state/focus";
import { FitControl } from "./Fit";
import { fitWithheld, type FitPosition } from "./log";
import { ResultsTapestry } from "./results";
import q from "./quest.module.css";

/** The analysis flowchart: the banner's columns view, the user's own pipeline (today's lineage). */
const FLOWCHART: StageFocus = { kind: "banner", segment: "columns" };
const NOOP = () => {};

export function Tapestry({
  pid,
  view,
  log,
  shown,
  fit,
  methods,
  onOpenKind,
}: {
  pid: string;
  view: ProjectView;
  log: QuestLog;
  shown: string;
  fit: FitPosition;
  methods: MethodsSections | undefined;
  onOpenKind: (kind: string) => void;
}) {
  const { focus, setFocus, keepPreview, endPreview } = useStageFocus();

  if (shown === "results") {
    return (
      <section className={`${k.canvas} ${q.tapestry}`} data-kit aria-label="Results" data-testid="tapestry-results">
        <ResultsTapestry pid={pid} />
      </section>
    );
  }
  if (shown === "writeup") {
    return (
      <section className={`${k.msColumn} ${q.tapestry}`} data-kit aria-label="The manuscript" data-testid="tapestry-writeup">
        {methods ? <ManuscriptBody methods={methods} onOpen={onOpenKind} /> : <p className={k.note}>Reading the methods…</p>}
      </section>
    );
  }
  const flowchart = shown === "models" && (fit.kind === "press" || fit.kind === "running");
  return (
    <section
      className={`${k.canvas} ${q.tapestry}`}
      data-kit
      aria-label={flowchart ? "Your analysis" : "Your data"}
      data-testid="tapestry"
    >
      {flowchart ? (
        <div className={k.chead}>
          <h2>Your analysis</h2>
        </div>
      ) : null}
      <div className={q.stageHost} onPointerEnter={keepPreview} onPointerLeave={endPreview}>
        <Stage
          pid={pid}
          view={view}
          focus={flowchart ? FLOWCHART : focus}
          onFocus={flowchart ? NOOP : setFocus}
          fitWithheld={fitWithheld(log)}
        />
      </div>
      {flowchart ? <FitControl pid={pid} log={log} fit={view.stages.fit} /> : null}
    </section>
  );
}
