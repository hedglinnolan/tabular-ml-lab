/**
 * The consequence stage — the right-hand pipeline panel.
 *
 * Idle, it is the live pipeline: Rows, Columns, Results, true of the working data now.
 * While an option is hovered or focused it becomes that option's consequence: one large
 * primary view and up to two thumbnails, captioned with a fact about the user's data.
 *
 * The stage is laid out once per question and then only its data changes, so flipping
 * between options is a morph, never a swap. Cards are keyed by what they show (rows,
 * lineage, relationship…): the live Columns lane and the preview's lineage thumbnail are
 * the same card, which is how "the pipeline updates with each selection".
 */
import { AnimatePresence, LayoutGroup, motion } from "motion/react";
import { fmtInt } from "./format";
import type { LiveModel, Preview, StageView } from "./scenarios";
import { chipify, Rich } from "./text";
import type { ViewKind } from "./types";
import { Distribution } from "./views/Distribution";
import { Lineage } from "./views/Lineage";
import { Relationship } from "./views/Relationship";
import { RowFlow } from "./views/RowFlow";
import { TableFocus } from "./views/TableFocus";
import s from "./StageScreen.module.css";
import { useStageTransitions } from "./motion";

export interface StageProps {
  live: LiveModel;
  preview: Preview | null;
  stacked: boolean;
  promoted: ViewKind | null;
  onPromote: (kind: ViewKind) => void;
  nRows: number;
  universe?: { n: number; noun: string };
  rank?: Record<string, number>;
  terms?: Record<string, string>;
  /** Shown in the live bar once a choice is recorded. */
  recorded?: string | null;
}

const CARD_KEY: Record<ViewKind, string> = {
  row_flow: "rows",
  lineage: "lineage",
  relationship: "relationship",
  distribution: "distribution",
  table_focus: "table_focus",
};

function ordered(views: StageView[], promoted: ViewKind | null): StageView[] {
  const i = promoted ? views.findIndex((v) => v.view.kind === promoted) : -1;
  if (i <= 0) return views;
  return [views[i]!, ...views.slice(0, i), ...views.slice(i + 1)];
}

/** A plain function, not a component: the live and preview cards must render the same
 * element types at the same positions so React keeps each chart (and its morph) alive. */
function renderView(sv: StageView, compact: boolean, props: StageProps) {
  const v = sv.view;
  switch (v.kind) {
    case "relationship":
      return <Relationship view={v} compact={compact} evidence={sv.evidence} />;
    case "distribution":
      return <Distribution view={v} stacked={props.stacked} compact={compact} evidence={sv.evidence} />;
    case "lineage":
      return (
        <Lineage lineage={v.after} compact={compact} openLabel={sv.openLabel} emphasis={v.emphasis} />
      );
    case "row_flow":
      return (
        <RowFlow
          steps={v.after}
          compact={compact}
          preview={!sv.openAfter}
          byLevel={sv.byLevel}
          openAfter={sv.openAfter}
          openLabel={sv.openLabel}
        />
      );
    case "table_focus":
      return (
        <TableFocus
          view={v}
          compact={compact}
          rank={props.rank}
          nRows={props.nRows}
          universe={props.universe ?? { n: v.columns_after.length, noun: "columns" }}
        />
      );
  }
}

export function Stage(props: StageProps) {
  const { live, preview, promoted, onPromote } = props;
  const t = useStageTransitions();
  const layout = { layout: t.settle, ...t.arrive };

  const cards: { key: string; area: string; node: React.ReactNode; className?: string }[] = [];
  let template: string;

  if (!preview) {
    template = `"a" auto "b" minmax(0, 1fr) "c" auto / 1fr`;
    const last = live.rows.steps[live.rows.steps.length - 1]!;
    cards.push({
      key: "rows",
      area: "a",
      node: (
        <>
          <header className={s.cardHead}>
            <h3 className={s.cardKicker}>Rows</h3>
            <span className={s.cardChip}>{fmtInt(last.n)} rows</span>
          </header>
          <div className={s.cardBody}>
            <RowFlow
              steps={live.rows.steps}
              compact
              split={live.rows.split}
              openAfter={live.rows.openAfter}
              openLabel="exclusions: this question"
              byLevel={live.rows.byLevel}
            />
          </div>
        </>
      ),
    });
    cards.push({
      key: "lineage",
      area: "b",
      node: (
        <>
          <header className={s.cardHead}>
            <h3 className={s.cardKicker}>Columns</h3>
            <span className={s.cardChip}>raw → model matrix</span>
          </header>
          <div className={s.cardBody}>
            <Lineage lineage={live.lineage} openLabel={live.lineageOpen} />
          </div>
        </>
      ),
    });
    cards.push({
      key: "results",
      area: "c",
      node: (
        <>
          <header className={s.cardHead}>
            <h3 className={s.cardKicker}>Results</h3>
          </header>
          <p className={s.resultsLine}>
            Nothing has been fit yet. It waits on {live.waits.join(", then ")}.
          </p>
        </>
      ),
    });
  } else {
    const views = ordered(preview.views, promoted);
    // A view with an intrinsic height (a flow, a table, a few levels) takes what it needs
    // and gives the rest to the charts below; a chart primary takes the room.
    const first = views[0]!.view;
    const intrinsic =
      first.kind === "row_flow" ||
      first.kind === "table_focus" ||
      (first.kind === "distribution" && !!first.levels);
    const p = intrinsic ? "auto" : "minmax(0, 1fr)";
    const thumbRow = intrinsic ? "minmax(0, 1fr)" : views.length >= 3 ? "226px" : "236px";
    template =
      views.length >= 3
        ? `"p p" ${p} "t1 t2" ${thumbRow} / 1fr 1fr`
        : views.length === 2
          ? `"p" ${p} "t1" ${thumbRow} / 1fr`
          : `"p" ${intrinsic ? "auto" : "minmax(0, 1fr)"} / 1fr`;
    views.forEach((sv, i) => {
      const primary = i === 0;
      const area = primary ? "p" : `t${i}`;
      const v = sv.view;
      cards.push({
        key: CARD_KEY[v.kind],
        area,
        className: primary ? s.primary : s.thumb,
        node: primary ? (
          <>
            <header className={s.cardHead}>
              <h3 className={s.cardTitle}>
                <Rich text={chipify(v.title)} />
              </h3>
            </header>
            <div className={s.cardBody}>
              {renderView(sv, false, props)}
            </div>
            <footer className={s.caption}>
              <p className={s.captionText}>
                <Rich text={chipify(v.caption)} terms={props.terms} />
              </p>
              <p className={s.basis}>{preview.basis}</p>
            </footer>
          </>
        ) : (
          <>
            <header className={s.cardHead}>
              <button
                type="button"
                className={s.promote}
                onClick={() => onPromote(v.kind)}
                aria-label={`Show “${v.title}” large`}
              >
                <span className={s.thumbTitle}>
                  <Rich text={chipify(v.title)} />
                </span>
                <span className={s.promoteIcon} aria-hidden="true" />
              </button>
              {sv.chip ? <span className={s.cardChip}>{sv.chip}</span> : null}
            </header>
            <div className={s.cardBody}>
              {renderView(sv, true, props)}
            </div>
          </>
        ),
      });
    });
  }

  return (
    <div className={s.stage} data-mode={preview ? "preview" : "live"} data-testid="stage">
      <div className={s.stageBar}>
        <AnimatePresence initial={false} mode="popLayout">
          {preview ? (
            <motion.div
              key="preview"
              className={s.stageBarInner}
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              exit={{ opacity: 0, transition: { duration: 0 } }}
              transition={t.arrive}
            >
              <span className={s.previewPill}>{preview.pill ?? "Preview"}</span>
              {preview.label ? (
                <span className={s.stageTitle} data-testid="stage-title">
                  <Rich text={chipify(preview.label)} />
                </span>
              ) : null}
              <span className={s.stageAside}>{preview.aside ?? "nothing is recorded"}</span>
            </motion.div>
          ) : (
            <motion.div
              key="live"
              className={s.stageBarInner}
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              exit={{ opacity: 0, transition: { duration: 0 } }}
              transition={t.arrive}
            >
              <span className={s.stageTitle}>Pipeline</span>
              <span className={s.stageSub}>what is true of the working data now</span>
              {props.recorded ? <span className={s.stageAsideOk}>{props.recorded}</span> : null}
            </motion.div>
          )}
        </AnimatePresence>
      </div>
      {preview?.note ? (
        <p className={s.stageNote}>
          <Rich text={preview.note} terms={props.terms} />
        </p>
      ) : null}
      <LayoutGroup id="stage">
        <div className={s.grid} style={{ gridTemplate: template }}>
          <AnimatePresence initial={false} mode="popLayout">
            {cards.map((c) => (
              <motion.section
                key={c.key}
                layout="position"
                className={`${s.card} ${c.className ?? ""}`}
                style={{ gridArea: c.area }}
                initial={{ opacity: 0 }}
                animate={{ opacity: 1 }}
                exit={{ opacity: 0, transition: { duration: 0 } }}
                transition={layout}
                data-card={c.key}
              >
                {c.node}
              </motion.section>
            ))}
          </AnimatePresence>
        </div>
      </LayoutGroup>
    </div>
  );
}
