/**
 * A preview's (or a finding's evidence's) views: one large primary and up to two secondaries,
 * all driven by the one player. Cards are keyed by what they show, so switching options keeps
 * each chart alive and it morphs; a secondary can be promoted to primary. Every card can be saved.
 */
import { AnimatePresence, motion } from "motion/react";
import type { ConsequenceView, ViewKind } from "../../api/m1-stage-types";
import { useTransitions } from "../../motion/prefs";
import { heading } from "./player";
import { figureForTrack } from "./save/journal";
import { saveIndices, type SaveWhich } from "./save/export";
import { SaveMenu, type SaveChoice } from "./save/SaveMenu";
import { Rich } from "./text";
import { noRepeats, type Storyboard, type Track } from "./tracks";
import { usePlayerStore, usePlayerUi } from "./usePlayer";
import { Distribution } from "./views/Distribution";
import { LineageTrack } from "./views/Lineage";
import { Relationship } from "./views/Relationship";
import { RowFlowTrack } from "./views/RowFlow";
import { TableTrack } from "./views/TableFocus";
import s from "./Stage.module.css";

interface Props {
  tracks: Track[];
  story: Storyboard;
  promoted: ViewKind | null;
  onPromote: (kind: ViewKind) => void;
  basis: string;
  /** The save caption's provenance: which choice, and whether it is recorded. */
  provenance: string;
}

export function renderTrack(track: Track, globalLast: number, compact: boolean) {
  const v = track.view as ConsequenceView;
  switch (v.kind) {
    case "relationship":
      return <Relationship track={track as Track<typeof v>} globalLast={globalLast} compact={compact} />;
    case "distribution":
      return <Distribution track={track as Track<typeof v>} globalLast={globalLast} compact={compact} />;
    case "lineage":
      return <LineageTrack track={track as Track<typeof v>} globalLast={globalLast} compact={compact} />;
    case "row_flow":
      return <RowFlowTrack track={track as Track<typeof v>} globalLast={globalLast} compact={compact} />;
    case "table_focus":
      return <TableTrack track={track as Track<typeof v>} globalLast={globalLast} compact={compact} />;
  }
}

function ordered(tracks: Track[], promoted: ViewKind | null): Track[] {
  const i = promoted ? tracks.findIndex((t) => t.view.kind === promoted) : -1;
  if (i <= 0) return tracks;
  return [tracks[i]!, ...tracks.slice(0, i), ...tracks.slice(i + 1)];
}

function TrackSave({ track, story, provenance }: { track: Track; story: Storyboard; provenance: string }) {
  const store = usePlayerStore();
  const ui = usePlayerUi(store);
  const localLast = track.states.length - 1;
  const title = noRepeats(track.view.title);
  if (track.still) {
    return (
      <SaveMenu
        title={title}
        choices={null}
        build={() => figureForTrack(track, [0], { title, caption: track.view.caption, provenance })}
      />
    );
  }
  const choices: SaveChoice[] = [{ which: "now", label: "Your data now" }];
  const step = saveIndices("step", ui.heading, story.last, localLast)[0]!;
  if (localLast > 1) {
    choices.push({ which: "step", label: `This step: ${track.states[step]?.label ?? ""}` });
  }
  choices.push({ which: "with", label: "With this choice" }, { which: "pair", label: "Before and after" });
  const current: SaveWhich =
    ui.heading === 0 ? "now" : ui.heading === story.last ? "with" : localLast > 1 ? "step" : "with";
  return (
    <SaveMenu
      title={title}
      choices={choices}
      current={current}
      build={(which) => {
        // The state is read when the file is made, from whole steps only (never the position).
        const h = heading(store.get());
        const indices = saveIndices(which === "as-shown" ? "with" : which, h, story.last, localLast);
        return figureForTrack(track, indices, { title, caption: track.view.caption, provenance });
      }}
    />
  );
}

export function PreviewGrid({ tracks, story, promoted, onPromote, basis, provenance }: Props) {
  const t = useTransitions();
  const views = ordered(tracks, promoted);
  const first = views[0]?.view;
  // A view with an intrinsic height (a flow, a table) takes what it needs.
  const intrinsic = !!first && (first.kind === "row_flow" || first.kind === "table_focus");
  const p = intrinsic ? "auto" : "minmax(0, 1fr)";
  const thumbRow = intrinsic ? "minmax(220px, 1fr)" : "236px";
  const template =
    views.length >= 3
      ? `"p p" ${p} "t1 t2" ${thumbRow} / 1fr 1fr`
      : views.length === 2
        ? `"p" ${p} "t1" ${thumbRow} / 1fr`
        : `"p" ${intrinsic ? "auto" : "minmax(0, 1fr)"} / 1fr`;
  const counts = new Map<string, number>();
  return (
    <div className={s.grid} style={{ gridTemplate: template }}>
      <AnimatePresence initial={false} mode="popLayout">
        {views.map((track, i) => {
          const v = track.view;
          const n = counts.get(v.kind) ?? 0;
          counts.set(v.kind, n + 1);
          const primary = i === 0;
          const area = primary ? "p" : `t${i}`;
          return (
            <motion.section
              key={`${v.kind}-${n}`}
              layout="position"
              className={primary ? s.primary : s.thumb}
              style={{ gridArea: area }}
              initial={{ opacity: 0, y: 6 }}
              animate={{ opacity: 1, y: 0 }}
              exit={{ opacity: 0, transition: { duration: 0 } }}
              transition={{ ...t.arrive, delay: t.reduced ? 0 : 0.06 * i }}
              data-card={v.kind}
              data-primary={primary || undefined}
            >
              <header className={s.cardHead}>
                {primary ? (
                  <h3 className={s.cardTitle}>
                    <Rich text={noRepeats(v.title)} />
                  </h3>
                ) : (
                  <button
                    type="button"
                    className={s.promote}
                    onClick={() => onPromote(v.kind)}
                    aria-label={`Show “${v.title.replace(/`/g, "")}” large`}
                  >
                    <span className={s.thumbTitle}>
                      <Rich text={noRepeats(v.title)} />
                    </span>
                    <span className={s.promoteIcon} aria-hidden="true" />
                  </button>
                )}
                <TrackSave track={track} story={story} provenance={provenance} />
              </header>
              <div className={s.cardBody}>{renderTrack(track, story.last, !primary)}</div>
              {primary ? (
                <footer className={s.caption}>
                  <p className={s.captionText}>
                    <Rich text={v.caption} />
                  </p>
                  <p className={s.basis}>{basis}</p>
                </footer>
              ) : null}
            </motion.section>
          );
        })}
      </AnimatePresence>
    </div>
  );
}
