/**
 * Write-up's reads of the export (the owner's Q5, 2026-11-04: the export's tables and figures shown
 * read-only now). The server serves them only inside the bundle (GET …/export; turbotab/server/
 * routes/export.py), so Write-up reads the bundle itself, and only when it can draw what it reads:
 * the checklist lists nothing the export waits for, and the browser can open a zip. Reading the
 * bundle is not a neutral read: the server counts its scores as shown (under Predict every fitted
 * family's cross-validated score) and, under a locked plan, its estimates as shown, which is true
 * because the card draws them (server/service.py `export`). So it is read where its tables are
 * drawn and nowhere else, and the checklist, whose quotes follow what was shown, is read again
 * after it.
 *
 * The bundle and the checklist are built from the decision log and the stages as they stand, and
 * no event names them: they are read again when a decision is recorded, or a stage changes its
 * state, its result or its hold, never on a progress tick (`basisOf`). A decision event already
 * reads the checklist again (src/api/events.ts); a stage event does not, so a stage's change reads
 * it again here, where what it says beside the tables matters.
 *
 * What was drawn stays, veiled and inert, until the export is read for the record as it stands:
 * while it is read again, while it waits, and where it refuses or cannot be read (`useExportContents`
 * keeps the newest read; `veilOf` says how it is veiled, by what moved since).
 */
import { useEffect, useRef } from "react";
import { skipToken, useQuery, useQueryClient } from "@tanstack/react-query";
import { api } from "../../api/client";
import { keys } from "../../api/queries";
import type { ProjectView } from "../../api/schema";
import type { VeilState } from "../../motion/StaleVeil";
import { readContents, type ExportContents } from "./bundle";
import { readZip } from "./zip";

/** What the export is built from, as the view holds it: the newest decision and each stage's
 *  state, result and hold. */
export interface Basis {
  /** The decisions recorded: their count and the newest one's id. */
  decisions: string;
  /** Each stage's state, result and hold, by its name. */
  stages: Record<string, string>;
  /** All of it as one key. */
  key: string;
}

/** The view's basis; null until the view is read. */
export function basisOf(view: ProjectView | undefined): Basis | null {
  if (!view) return null;
  const last = view.decisions[view.decisions.length - 1];
  const decisions = `${view.decisions.length}:${last?.id ?? ""}`;
  const stages: Record<string, string> = {};
  for (const [name, s] of Object.entries(view.stages)) if (s) stages[name] = `${s.status}:${s.key ?? ""}:${s.held ?? ""}`;
  const key = [decisions, ...Object.keys(stages).sort().map((name) => `${name}:${stages[name]}`)].join("|");
  return { decisions, stages, key };
}

/** The checklist read again whenever the basis moves after the first read. */
export function useChecklistFollows(pid: string, basis: string | null): void {
  const qc = useQueryClient();
  const last = useRef(basis);
  useEffect(() => {
    if (last.current !== null && basis !== null && basis !== last.current) {
      void qc.invalidateQueries({ queryKey: keys.checklist(pid), exact: true });
    }
    last.current = basis;
  }, [qc, pid, basis]);
}

/** One read of the export: its tables and figures, and the basis it was read for. */
export interface ExportRead {
  basis: Basis;
  contents: ExportContents;
}

/** Where the newest read is kept, whatever basis it was read for. Only a read sets it. */
const drawnKey = (pid: string) => [pid, "export", "drawn"] as const;

/** The export read for this basis while `enabled`: the newest read of it for any basis (`drawn`,
 *  kept until a newer read arrives, through a wait, a refusal or a read that failed), whether it
 *  is being read now (`again`), and the error of the read for this basis, if it refused or failed. */
export function useExportContents(pid: string, basis: Basis | null, enabled: boolean) {
  const qc = useQueryClient();
  const read = useQuery({
    queryKey: [pid, "export", "contents", basis?.key ?? null] as const,
    queryFn: async ({ signal }): Promise<ExportRead> => {
      const { blob } = await api.exportBundle(pid, signal);
      const contents = readContents(await readZip(new Uint8Array(await blob.arrayBuffer())));
      const out: ExportRead = { basis: basis!, contents };
      // A read for a basis already left behind is not the newest.
      if (!signal.aborted) qc.setQueryData(drawnKey(pid), out);
      // The bundle's scores now count as shown, and the checklist quotes them only once they have.
      void qc.invalidateQueries({ queryKey: keys.checklist(pid), exact: true });
      return out;
    },
    enabled: enabled && basis !== null,
    retry: false,
  });
  const drawn = useQuery<ExportRead>({ queryKey: drawnKey(pid), queryFn: skipToken }).data ?? null;
  return { drawn, again: read.isFetching, error: read.error };
}

/** How the tables and figures of the read for `drawn` stand beside the view now, as StaleVeil says
 *  it. "fresh" only where they are the export's for the record as it stands: nothing moved since
 *  that read, the checklist waits for nothing, and the export is neither being read again nor
 *  refused. Else by what moved in the view since: "recomputing" while the export is read again or
 *  a stage that moved since computes; "failed" where such a stage failed, "stopped" where its
 *  computing was stopped; "stale" where a decision was recorded since. Where none of that moved,
 *  an export that refused or failed when read again is "failed" (its own recompute did not
 *  finish); else "stale". */
export function veilOf(
  drawn: Basis,
  view: ProjectView | undefined,
  read: { ready: boolean; again: boolean; failed: boolean },
): VeilState {
  const now = basisOf(view) ?? drawn;
  if (now.key === drawn.key && read.ready && !read.again && !read.failed) return "fresh";
  const moved = [...new Set([...Object.keys(drawn.stages), ...Object.keys(now.stages)])]
    .filter((name) => drawn.stages[name] !== now.stages[name])
    .flatMap((name) => {
      const s = view?.stages[name];
      return s ? [s] : [];
    });
  if (read.again || moved.some((s) => s.status === "queued" || s.status === "running")) return "recomputing";
  if (moved.some((s) => s.status === "error")) return "failed";
  if (moved.some((s) => s.cancelled)) return "stopped";
  if (now.decisions !== drawn.decisions) return "stale";
  return read.failed ? "failed" : "stale";
}
