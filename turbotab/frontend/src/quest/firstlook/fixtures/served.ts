/**
 * What the server served First look's gallery at each of the outcome's gates (served.json, written
 * by capture.py from the real server on the clinical sample; never edited by hand), typed for the
 * tests. Loaded through import.meta.glob, as the other captured fixtures are, so the compiler
 * infers no type from the file itself.
 */
import type { ColumnSummary, Histogram, ProjectView, QuestLog, Refusal, StageResult, TableWindow } from "../../../api/schema";
import type { ExploreArtifact } from "../model";

export interface Served<T> {
  status: number;
  body: T;
}

/** Every read the gallery makes, as served at one moment. */
export interface Moment {
  view: ProjectView;
  quest: QuestLog;
  columns: Served<ColumnSummary[]>;
  explore: Served<StageResult<ExploreArtifact | null>>;
  target_info: Served<StageResult<{ column: string; detected_task: string | null } | null>>;
  table: Served<TableWindow>;
  /** each number column's; a refusal's body where it was refused */
  histograms: Record<string, Served<Histogram | Refusal>>;
}

interface Recorded {
  recorded: ProjectView["decisions"][number];
}

export interface Capture {
  meta: { source: string; outcome: string; rows: number; beside: number; bins: number };
  predict: {
    first_visit: Moment;
    drawn: Moment;
    recorded: Moment;
    /** each record a route handed back, posted, and what the server answered */
    posts: Record<"distribution" | "relationship" | "table", Served<Recorded>>;
    /** one lever option posted with decide_now after the draw */
    decide_now: { finding: string; option: string; answer: Served<Recorded> };
    /** the draw posted with decide_now while Who's in is open: refused */
    decide_now_refused: { decision: Record<string, unknown>; answer: Served<Refusal> };
  };
  estimate: { whos_in: Moment };
}

const FILES = import.meta.glob<Capture>("./served.json", { eager: true, import: "default" });
export const SERVED: Capture = Object.values(FILES)[0]!;
export const OUTCOME = SERVED.meta.outcome;
export const { first_visit: FIRST, drawn: DRAWN, recorded: RECORDED } = SERVED.predict;
export const ESTIMATE = SERVED.estimate.whos_in;

export const summaryOf = (m: Moment): ColumnSummary => m.columns.body.find((c) => c.name === OUTCOME)!;
export const artifactOf = (m: Moment): ExploreArtifact => m.explore.body.artifact!;
export const stageName = (m: Moment, key: string): string => m.quest.stages.find((s) => s.key === key)!.name;
export const histogramOf = (m: Moment, column: string): Histogram => m.histograms[column]!.body as Histogram;
