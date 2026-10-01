/**
 * The stage's server state: previews, finding evidence and the M1 stage artifacts, all in
 * TanStack Query under the project's id ([pid, …]), so a resync refetches them with the rest.
 *
 * Previews are debounced (rifling through options with the arrow keys sends one request for the
 * option you land on), abort when superseded (the query's signal), and keep the last result on
 * screen while the next one loads, so the stage never flashes empty.
 */
import { useEffect, useMemo, useState } from "react";
import { keepPreviousData, useQuery, useQueryClient } from "@tanstack/react-query";
import { api, isRefusalError } from "../../api/client";
import type { PreviewResult } from "../../api/m1-stage-types";
import type { AnyStageArtifacts, AnyStageName } from "../../api/m1-types";
import { keys } from "../../api/queries";
import type { Decision, ProjectView, Refusal, StageResult } from "../../api/schema";
import { veilFor, type VeilState } from "../../motion/StaleVeil";

export const PREVIEW_DEBOUNCE_MS = 120;

/** `value`, once it has held still for `ms`. A change of identity with the same key is ignored. */
export function useDebounced<T>(value: T, ms: number, keyOf: (v: T) => string): T {
  const [settled, setSettled] = useState(value);
  const key = keyOf(value);
  const settledKey = keyOf(settled);
  useEffect(() => {
    if (key === settledKey) return;
    const id = window.setTimeout(() => setSettled(value), ms);
    return () => window.clearTimeout(id);
    // `value` is read through its key on purpose: a new object for the same decision is no change.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [key, settledKey, ms]);
  return key === settledKey ? value : settled;
}

/** The newest recorded decision's sequence number: previews are true of this state only. */
export function stateSeq(view: ProjectView): number {
  return view.decisions.reduce((m, d) => Math.max(m, d.seq), 0);
}

/** A preview, or the refusal recording it would meet, tagged with the option it answers and the
 *  recorded state (`seq`) it is true of. */
export type Answer =
  | { key: string; seq: number; label: string; decision: Decision; result: PreviewResult; refusal: null }
  | { key: string; seq: number; label: string; decision: Decision; result: null; refusal: Refusal };

export interface PreviewQuery {
  /** The answer on screen: the previous option's while the next one loads — but only another
   *  option of the same question, on the same recorded state. A placeholder from another
   *  question (or from before the last answer) is never shown as this one's preview. */
  answer: Answer | undefined;
  error: Error | null;
  /** The focused option's answer is not on screen yet. */
  loading: boolean;
}

export const decisionKey = (d: Decision | null) => (d ? JSON.stringify(d) : "");

export interface OptionFocus {
  decision: Decision;
  label: string;
}

const optionKey = (o: OptionFocus | null) => decisionKey(o?.decision ?? null);

export function usePreview(pid: string, option: OptionFocus | null, seq: number): PreviewQuery {
  const settled = useDebounced(option, PREVIEW_DEBOUNCE_MS, optionKey);
  const key = optionKey(settled);
  const q = useQuery({
    queryKey: [pid, "preview", seq, key] as const,
    queryFn: async ({ signal }): Promise<Answer> => {
      const { decision, label } = settled!;
      try {
        return { key, seq, label, decision, result: await api.preview(pid, decision, signal), refusal: null };
      } catch (e) {
        // A refused option is still an answer: the stage shows why, and the ways out.
        if (isRefusalError(e)) return { key, seq, label, decision, result: null, refusal: e.refusal };
        throw e;
      }
    },
    enabled: settled !== null,
    placeholderData: keepPreviousData,
    gcTime: 120_000,
    retry: 1,
  });
  const data = q.data;
  const sameQuestion =
    !!data && !!option && data.seq === seq && data.decision.kind === option.decision.kind;
  return {
    answer: sameQuestion ? data : undefined,
    error: q.error,
    loading: data?.key !== optionKey(option),
  };
}

export type EvidenceAnswer = { key: string; result: PreviewResult };

export function useEvidence(pid: string, findingId: string | null, seq: number) {
  const q = useQuery({
    queryKey: [pid, "evidence", findingId ?? "", seq] as const,
    queryFn: async ({ signal }): Promise<EvidenceAnswer> => ({
      key: findingId ?? "",
      result: await api.findingEvidence(pid, findingId!, signal),
    }),
    enabled: findingId !== null,
    placeholderData: keepPreviousData,
    gcTime: 120_000,
  });
  return { answer: q.data, error: q.error, loading: q.data?.key !== findingId };
}

export interface StageData<K extends AnyStageName> {
  result: StageResult<AnyStageArtifacts[K]> | undefined;
  artifact: AnyStageArtifacts[K] | null;
  veil: VeilState;
  status: ProjectView["stages"][string] | undefined;
}

/**
 * An M1 stage's artifact. Kept current the way the SSE handler keeps M0 stages current: when the
 * server reports the stage fresh at a key the cached result does not have, it is fetched again.
 * An older artifact stays on screen, veiled, while the new one computes.
 */
export function useM1Stage<K extends AnyStageName>(
  pid: string,
  view: ProjectView,
  name: K,
  enabled = true,
): StageData<K> {
  const qc = useQueryClient();
  const status = view.stages[name];
  const exists = !!status && status.status !== "idle";
  const q = useQuery({
    queryKey: keys.stage(pid, name),
    queryFn: ({ signal }) => api.stage(pid, name, signal),
    enabled: enabled && exists,
  });
  const behind =
    status?.status === "fresh" && !!q.data && (q.data.key !== status.key || !q.data.fresh);
  useEffect(() => {
    if (behind && !q.isFetching) void qc.invalidateQueries({ queryKey: keys.stage(pid, name), exact: true });
  }, [behind, q.isFetching, qc, pid, name]);
  const veil = veilFor(status, q.data);
  return useMemo(
    () => ({ result: q.data, artifact: q.data?.artifact ?? null, veil, status }),
    [q.data, veil, status],
  );
}
