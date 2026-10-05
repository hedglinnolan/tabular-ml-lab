/**
 * The seal in the Results (M2_CONTRACT §3, lifted from /lab/m2's SealedResults): sealed, opened
 * once, and changed after the opening.
 *
 *   sealed     every family's held-out column says "sealed", with the glyph; at the end of the
 *              Results, the open action as a CONSEQUENCE (DESIGN_LANGUAGE §09: a full-width
 *              interruption, declarative then first person, resolve or attest)
 *   opened     the card settles into a recorded line: the scores are fixed in the record
 *   post_seal  an amber band says which later changes refitted the models after the opening
 *
 * Opening is the one irreversible act in the analysis: it takes CONSEQUENCE's silhouette, not
 * --stop (opening makes no number untrustworthy).
 */
import { useEffect, useRef, useState } from "react";
import { useQueryClient } from "@tanstack/react-query";
import { motion } from "motion/react";
import { isRefusalError } from "../../../api/client";
import type { FitArtifact, SplitArtifact } from "../../../api/m1-stage-types";
import { keys, useDecide } from "../../../api/queries";
import type { Decision, ProjectView, Refusal } from "../../../api/schema";
import { useTransitions } from "../../../motion/prefs";
import { fmtInt } from "../format";
import { Rich } from "../text";
import { glyphOf, isExploratory, type SealPhase } from "./phase";
import { SealGlyph } from "./SealGlyph";
import s from "./seal.module.css";

/** The record that opened the seal, if any (it is never reverted). */
export function openingRecord(view: ProjectView) {
  return [...view.decisions].sort((a, b) => a.seq - b.seq).find((d) => d.decision.kind === "open_seal") ?? null;
}

const SUBJECT: Record<string, string> = {
  set_energy_adjustment: "the energy adjustment",
  select_models: "the model families",
  set_missing: "the missing values",
  set_exclusions: "the eligibility rules",
  set_roles: "the roles",
  set_target: "the outcome",
  set_purpose: "the purpose",
  set_substitution: "the substitution",
  apply_repair: "a repair",
  revert: "an earlier answer",
};

/** "the energy adjustment (#12)" for each post-seal decision the served fit names. */
export function postSealChanges(view: ProjectView, ids: string[]): string[] {
  return ids
    .map((id) => view.decisions.find((d) => d.id === id))
    .filter((d): d is NonNullable<typeof d> => !!d)
    .map((d) => `${SUBJECT[d.decision.kind] ?? d.decision.kind.replace(/^set_/, "the ").replace(/_/g, " ")} (#${d.seq})`);
}

function joined(items: string[]): string {
  return items.length < 2 ? (items[0] ?? "") : `${items.slice(0, -1).join(", ")} and ${items.at(-1)}`;
}

/** The comparison header's aside: what the held-out column holds. */
export function HeldOutAside({ phase, split, openedSeq }: { phase: SealPhase; split: SplitArtifact | null; openedSeq: number | null }) {
  const glyph = glyphOf(split?.basis?.state);
  if (phase === "sealed")
    return (
      <span className={s.aside} data-testid="held-out-sealed">
        <SealGlyph state={glyph} recorded size={14} className={s.inlineGlyph} />
        held-out rows: sealed
      </span>
    );
  if (phase === "opened" || phase === "post_seal")
    return (
      <span className={s.aside} data-testid="held-out-opened">
        held out · opened once{openedSeq !== null ? `, #${openedSeq}` : ""}
      </span>
    );
  return null;
}

/**
 * Once the Record says the seal is open, the fit is fetched again: opening changes no stage's key,
 * so no stage event announces it, and the served fit is what carries the scores.
 */
export function useRefetchOnOpening(pid: string, opened: boolean, fit: FitArtifact | null, fitKey: string | undefined) {
  const qc = useQueryClient();
  const asked = useRef<string | null>(null);
  useEffect(() => {
    if (!opened || !fit?.holdout_sealed) return;
    const key = fitKey ?? "fit";
    if (asked.current === key) return;
    asked.current = key;
    void qc.invalidateQueries({ queryKey: keys.stage(pid, "fit"), exact: true });
  }, [opened, fit, fitKey, pid, qc]);
}

export function OpenSealCard({
  pid,
  fit,
  split,
  metric,
}: {
  pid: string;
  fit: FitArtifact;
  split: SplitArtifact | null;
  metric: string;
}) {
  const decide = useDecide(pid);
  const qc = useQueryClient();
  const [refused, setRefused] = useState<Refusal | null>(null);
  const families = fit.models.length;
  const glyph = glyphOf(split?.basis?.state);
  const exploratory = isExploratory(split?.basis?.state, split?.exploratory);
  // With several families the opening names the final one, declared on cross-validation before
  // any held-out score is seen (WP8): the server's refusal offers each, and pressing one opens.
  const open = (decision: Decision = { kind: "open_seal" } as Decision) => {
    setRefused(null);
    decide.mutate(decision, {
      onSuccess: () => void qc.invalidateQueries({ queryKey: keys.stage(pid, "fit"), exact: true }),
      onError: (e) =>
        setRefused(
          isRefusalError(e) ? e.refusal : { error: { code: "failed", message: e.message, exits: [] } },
        ),
    });
  };
  return (
    <section className={s.consequence} aria-labelledby="open-seal-title" data-testid="open-seal-card" data-purpose="open_seal">
      <div className={s.conRule} aria-hidden="true" />
      <div className={s.conHead}>
        <SealGlyph state={glyph} recorded size={26} />
        <span className={s.conSignal}>Opened once</span>
      </div>
      <h3 id="open-seal-title" className={s.conTitle}>
        Opening the seal scores the {families === 1 ? "model" : `${families} models`} on {fmtInt(fit.n_holdout)} held-out
        rows no choice has seen.
      </h3>
      <p className={s.conBody}>
        It happens once. The held-out {metric} is then fixed in the record; any later change still refits, and is marked
        post-seal in the Results and the Record.
        {exploratory ? (
          <>
            {" "}
            <Rich text={`The seal's basis is ${split?.basis?.label ?? "undetermined"}, so these scores carry an exploratory label.`} />
          </>
        ) : null}
      </p>
      <div className={s.conExits}>
        <button type="button" className={s.conAttest} onClick={() => open()} disabled={decide.isPending} data-testid="open-seal">
          {decide.isPending ? "Opening…" : "I'm done choosing: open the seal"}
        </button>
        <span className={s.conOr}>or keep choosing; nothing is opened until you press it.</span>
      </div>
      {refused ? (
        <div role="alert" data-testid="open-seal-refused">
          <p className={s.conRefused}>
            <Rich text={`Not opened: ${refused.error.message}`} />
          </p>
          {refused.error.exits.some((x) => x.decision) ? (
            <div className={s.conExits}>
              {refused.error.exits.map((x) =>
                x.decision ? (
                  <button
                    key={x.label}
                    type="button"
                    className={s.conAttest}
                    disabled={decide.isPending}
                    onClick={() => open(x.decision as Decision)}
                    data-testid="open-seal-exit"
                  >
                    <Rich text={x.label} />
                  </button>
                ) : null,
              )}
            </div>
          ) : null}
        </div>
      ) : null}
    </section>
  );
}

export function SealOpenedLine({ seq, metric, nHoldout, post }: { seq: number | null; metric: string; nHoldout: number; post: boolean }) {
  const t = useTransitions();
  return (
    <motion.p
      layout
      className={s.sealRecorded}
      initial={{ opacity: 0, y: 6 }}
      animate={{ opacity: 1, y: 0 }}
      transition={t.settle}
      data-testid="seal-opened"
      data-purpose="seal_opened"
      data-post={post || undefined}
    >
      The seal was opened once{seq !== null ? ` (#${seq})` : ""}: held-out {metric} on {fmtInt(nHoldout)} rows is fixed in the
      record.
    </motion.p>
  );
}

export function PostSealBand({ view, fit, openedSeq }: { view: ProjectView; fit: FitArtifact; openedSeq: number | null }) {
  const changes = postSealChanges(view, fit.post_seal_decisions ?? []);
  const said = changes.length ? `${joined(changes)} changed` : "An earlier answer changed";
  const what = said.charAt(0).toUpperCase() + said.slice(1); // a sentence, even when it opens on "the …"
  return (
    <div className={s.postBand} role="status" data-testid="post-seal" data-purpose="post_seal">
      <span className={s.postKicker}>Changed after the seal was opened</span>
      <span className={s.postText}>
        {what} after the held-out scores were seen once{openedSeq !== null ? ` (#${openedSeq})` : ""}. These numbers are post-seal,
        and the Record marks each change.
      </span>
    </div>
  );
}
