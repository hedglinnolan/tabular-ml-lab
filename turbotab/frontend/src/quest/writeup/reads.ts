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
 * state, its result or its hold, never on a progress tick (`basisOf`); what was drawn stays,
 * veiled, while it is read again. A decision event already
 * reads the checklist again (src/api/events.ts); a stage event does not, so a stage's change reads
 * it again here, where what it says beside the tables matters.
 */
import { useEffect, useRef } from "react";
import { keepPreviousData, useQuery, useQueryClient } from "@tanstack/react-query";
import { api } from "../../api/client";
import { keys } from "../../api/queries";
import type { ProjectView } from "../../api/schema";
import { readContents } from "./bundle";
import { readZip } from "./zip";

/** What the export is built from, as the view holds it: the newest decision and each stage's
 *  state, result and hold. Null until the view is read. */
export function basisOf(view: ProjectView | undefined): string | null {
  if (!view) return null;
  const last = view.decisions[view.decisions.length - 1];
  const stages = Object.entries(view.stages)
    .flatMap(([name, s]) => (s ? [`${name}:${s.status}:${s.key ?? ""}:${s.held ?? ""}`] : []))
    .sort();
  return [`${view.decisions.length}:${last?.id ?? ""}`, ...stages].join("|");
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

/** The bundle's tables and figures, read for this basis while `enabled`; until a new basis's read
 *  arrives, the last one read stands in for it (`isPlaceholderData`), for the card to veil. */
export function useExportContents(pid: string, basis: string | null, enabled: boolean) {
  const qc = useQueryClient();
  return useQuery({
    queryKey: [pid, "export", "contents", basis] as const,
    placeholderData: keepPreviousData,
    queryFn: async ({ signal }) => {
      const { blob } = await api.exportBundle(pid, signal);
      const contents = readContents(await readZip(new Uint8Array(await blob.arrayBuffer())));
      // The bundle's scores now count as shown, and the checklist quotes them only once they have.
      void qc.invalidateQueries({ queryKey: keys.checklist(pid), exact: true });
      return contents;
    },
    enabled: enabled && basis !== null,
    retry: false,
  });
}
