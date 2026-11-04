/**
 * First look's reads, in TanStack Query (the network only through src/api/client.ts), each keyed
 * by what it depends on (model.ts `gateMark`, `tableMark`) so that a gate that opens, a look
 * recorded or a new table is read again rather than shown as it stood: a decision event leaves
 * GET /columns, the histograms and the row windows cached (events.ts reads them again only when the
 * table changes, and the histograms never), and every project key starts with the pid.
 *
 *   useFirstLook   everything the card's guide and the gallery read, shared by both
 *   useColumnHistogram, useOutcomeHistogram   a column's histogram, the outcome's once open
 */
import { useQuery } from "@tanstack/react-query";
import { api } from "../../api/client";
import type { AnyStageName } from "../../api/m1-types";
import { keys, useColumnSummaries, useStageResult } from "../../api/queries";
import type { ColumnSummary, ProjectView, QuestLog, StageResult, TableWindow } from "../../api/schema";
import { veilFor, type VeilState } from "../../motion/StaleVeil";
import {
  type ColumnLook,
  type Door,
  type ExploreArtifact,
  type ExploreBeat,
  type Looks,
  type RowsDoor,
  COLUMNS_BESIDE,
  GALLERY_MAX,
  ROWS_SHOWN,
  columnLooks,
  exploreBeat,
  exploreLooks,
  gateMark,
  noticedCount,
  outcomeDoor,
  rowsDoor,
  started,
  tableMark,
} from "./model";

/** The most bins a gallery histogram draws (the route aligns them to the values' step). */
export const BINS = 20;

// The explore stage is served at GET /stages/explore, but src/api's stage map does not list it yet.
const EXPLORE = "explore" as string as AnyStageName;

export interface FirstLook {
  /** GET /columns has answered: until then nothing is counted */
  read: boolean;
  /** why a read failed, where it did, in the client's words: GET /columns; the outcome's summary
   *  (GET /columns for the gates now); the row window, which waits on the columns and so fails
   *  with them. Null while a read is on its way or answered. */
  unread: { columns: string | null; outcome: string | null; window: string | null };
  /** every column but the outcome, and those drawn */
  columns: ColumnLook[];
  drawn: ColumnLook[];
  target: string | null;
  /** the outcome's summary as served for the gates now; undefined while it is read */
  outcome: ColumnSummary | null | undefined;
  door: Door | null;
  /** the window of the outcome beside the other columns, and its door */
  window: TableWindow | undefined;
  rows: RowsDoor | null;
  windowColumns: string[];
  explore: { looks: Looks | null; artifact: ExploreArtifact | null; veil: VeilState; beat: ExploreBeat };
  task: ProjectView["state"]["task"];
  noticed: number;
  gate: string;
  table: string;
}

export function useFirstLook(pid: string, view: ProjectView, log: QuestLog): FirstLook {
  const state = view.state;
  const target = state.target ?? null;
  const gate = gateMark(view, log);
  const table = tableMark(view);
  const columnsRead = useColumnSummaries(pid, view.stages.ingest?.status === "fresh");
  const summaries = columnsRead.data;
  const columns = summaries ? columnLooks(summaries, state) : [];
  const outcomeRead = useQuery({
    queryKey: [...keys.columns(pid), "outcome", gate],
    queryFn: ({ signal }) => api.columns(pid, signal),
    enabled: target !== null,
    select: (all: ColumnSummary[]) => all.find((c) => c.name === target) ?? null,
  });
  const outcome = outcomeRead.data;
  const windowColumns = target && summaries ? [target, ...summaries.filter((c) => c.name !== target).slice(0, COLUMNS_BESIDE).map((c) => c.name)] : [];
  const windowRead = useQuery({
    queryKey: [...keys.table(pid, 0, ROWS_SHOWN, windowColumns.join(",")), gate],
    queryFn: ({ signal }) => api.table(pid, { offset: 0, limit: ROWS_SHOWN, columns: windowColumns }, signal),
    enabled: windowColumns.length > 0,
  });
  const window = windowRead.data;
  const columnsFailed = summaries === undefined ? (columnsRead.error?.message ?? null) : null;

  const status = view.stages.explore;
  const read = useQuery<StageResult<ExploreArtifact>>({
    queryKey: keys.stage(pid, "explore"),
    queryFn: ({ signal }) => api.stage(pid, EXPLORE, signal) as unknown as Promise<StageResult<ExploreArtifact>>,
    enabled: started(status?.status),
  });
  const result = read.data;
  const artifact = started(status?.status) ? (result?.artifact ?? null) : null;
  const looks = artifact ? exploreLooks(artifact, state) : null;
  const whosIn = log.stages.find((s) => s.key === "whos_in")?.name ?? "Who's in";
  const beat: ExploreBeat = exploreBeat({ target, status, result, failed: read.error?.message ?? null, artifact, looks, whosIn });

  const targetInfo = useStageResult(pid, "target_info", started(view.stages.target_info?.status)).data?.artifact;
  const task = state.task ?? (targetInfo && targetInfo.column === target ? targetInfo.detected_task : null);

  return {
    read: summaries !== undefined,
    unread: {
      columns: columnsFailed,
      outcome: outcome === undefined ? (outcomeRead.error?.message ?? null) : null,
      window: window === undefined ? (windowRead.error?.message ?? (windowColumns.length ? null : columnsFailed)) : null,
    },
    columns,
    drawn: columns.slice(0, GALLERY_MAX),
    target,
    outcome: target ? outcome : null,
    door: target && outcome ? outcomeDoor(outcome) : null,
    window,
    rows: target && window ? rowsDoor(window, target) : null,
    windowColumns,
    explore: { looks, artifact, veil: veilFor(status, result), beat },
    task,
    noticed: noticedCount(log),
    gate,
    table,
  };
}

/** A column's histogram, on the table the routes read now. */
export function useColumnHistogram(pid: string, column: string, table: string) {
  return useQuery({
    queryKey: [...keys.histogram(pid, column, BINS), table],
    queryFn: ({ signal }) => api.histogram(pid, column, BINS, signal),
  });
}

/** The outcome's histogram, asked for only once its door is open, for the gates as they stand. */
export function useOutcomeHistogram(pid: string, target: string, gate: string, enabled: boolean) {
  return useQuery({
    queryKey: [...keys.histogram(pid, target, BINS), "outcome", gate],
    queryFn: ({ signal }) => api.histogram(pid, target, BINS, signal),
    enabled,
  });
}
