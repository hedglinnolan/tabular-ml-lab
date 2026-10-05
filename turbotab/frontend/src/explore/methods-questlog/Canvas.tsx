/**
 * The canvas: the production stage's pieces (StageBar, the transform player, PreviewGrid, the
 * shelf, the Results' Comparison) fed with the server's real preview, evidence and result
 * artifacts. One view here is composed by the prototype from a real artifact because the engine
 * draws none for that question yet (where the exposure enters the model; MODELING_SEQUENCE §3 asks
 * for lineage there); it says so in its own bar.
 */
import { useEffect, useLayoutEffect, useMemo, useState, type ReactNode } from "react";
import { scaleLinear } from "d3-scale";
import type { FitArtifact, LineageView, PreviewResult, ShelfArtifact, SplitArtifact } from "../../api/m1-stage-types";
import { initial } from "../../components/stage/player";
import { PreviewGrid } from "../../components/stage/PreviewGrid";
import { Comparison } from "../../components/stage/results/Comparison";
import { comparisonOf, metricBasis } from "../../components/stage/results/model";
import { Shelf } from "../../components/stage/results/Shelf";
import { PlayerControls, StageBar } from "../../components/stage/StageBar";
import { Rich } from "../../components/stage/text";
import { readoutOf, storyboardOf, trackOf } from "../../components/stage/tracks";
import { createPlayerStore, PlayerContext } from "../../components/stage/usePlayer";
import stage from "../../components/stage/Stage.module.css";
import {
  fmtCI,
  fmtEst,
  matteredAttrs,
  matteredRows,
  t2Attrs,
  table2Rows,
  type MatteredRow,
} from "../methods-shared/results";
import { fmtInt, INF, type SequenceFit } from "./data";
import s from "./questlog.module.css";

const AUTOPLAY_MS = 320;

interface PreviewProps {
  result: PreviewResult;
  pill: "Preview" | "Evidence" | null;
  label: string;
  aside: ReactNode;
  action?: ReactNode;
  /** A line under the bar (e.g. that the picture is composed by the prototype). */
  note?: ReactNode;
  /** Where the player rests for a review capture: a storyboard step, or the result. */
  rest?: number | "with" | "now";
  /** The view drawn large first (a finding's distribution over its table). */
  promote?: string;
}

/** A preview or a finding's evidence, on the production transform player. */
export function PreviewCanvas({ result, pill, label, aside, action, note, rest = "with", promote }: PreviewProps) {
  const [store] = useState(() => createPlayerStore());
  const tracks = useMemo(() => result.views.map((v) => trackOf(v)), [result]);
  const story = useMemo(() => storyboardOf(tracks), [tracks]);
  const readout = useMemo(() => readoutOf(result.views).slice(0, 3), [result]);
  // Evidence is the data as loaded: one state, never a "with this choice" (it is not a choice).
  const evidence = pill === "Evidence";
  const still = evidence || tracks.every((t) => t.still);
  const stillLabel = evidence ? "Your data as loaded" : "With this choice (preview)";
  const [promoted, setPromoted] = useState<string | null>(promote ?? null);

  useLayoutEffect(() => {
    store.reset(initial(story.last, evidence ? "with" : "now"));
    if (still) return;
    // A new scene starts at "your data now" and plays its storyboard forward once its views are in.
    const id = window.setTimeout(() => {
      if (typeof rest === "number") store.dispatch({ type: "seek", step: rest });
      else store.dispatch({ type: "show", side: rest });
    }, AUTOPLAY_MS);
    return () => window.clearTimeout(id);
  }, [store, story.last, still, rest, result, evidence]);

  useEffect(() => {
    // Review captures step the player by hand (as the production stage's hook does).
    const w = window as unknown as { __questlog?: unknown };
    w.__questlog = { store };
    return () => {
      delete w.__questlog;
    };
  }, [store]);

  useEffect(() => {
    if (still) return;
    const onKey = (e: KeyboardEvent) => {
      if (e.key !== " " || e.repeat || e.metaKey || e.ctrlKey || e.altKey) return;
      const el = e.target as HTMLElement | null;
      if (el && (el.isContentEditable || /^(INPUT|TEXTAREA|SELECT|BUTTON)$/.test(el.tagName))) return;
      e.preventDefault();
      store.dispatch({ type: "flip" });
    };
    window.addEventListener("keydown", onKey, true);
    return () => window.removeEventListener("keydown", onKey, true);
  }, [still, store]);

  return (
    <PlayerContext.Provider value={store}>
      <section className={stage.stage} data-testid="stage">
        <StageBar pill={pill} label={label} aside={aside} loading={false} action={action}>
          <PlayerControls story={story} readout={readout} still={still} stillLabel={stillLabel} />
        </StageBar>
        {note}
        <div className={stage.body}>
          <div className={stage.views}>
            {result.note ? (
              <p className={stage.note}>
                <Rich text={result.note} />
              </p>
            ) : null}
            <PreviewGrid
              tracks={tracks}
              story={story}
              promoted={(promoted as never) ?? null}
              onPromote={(k) => setPromoted(k)}
              basis={result.basis}
              provenance={`${pill === "Evidence" ? "Evidence" : "Preview, not recorded"}: ${label}. ${result.basis}`}
            />
          </div>
        </div>
      </section>
    </PlayerContext.Provider>
  );
}

/** A canvas with no picture: what the engine says it can show for this question (often: nothing
 *  yet), in the stage's own frame. */
export function NoteCanvas({ pill, label, aside, children }: { pill: string | null; label: string; aside: ReactNode; children: ReactNode }) {
  return (
    <section className={stage.stage} data-testid="stage">
      <StageBar pill={pill} label={label} aside={aside} loading={false} />
      <div className={s.resCard}>{children}</div>
    </section>
  );
}

/** The shelf of model families, as the production Results draw it. */
export function ShelfCanvas({ shelf, chosen }: { shelf: ShelfArtifact; chosen: string[] }) {
  return (
    <section className={stage.stage} data-testid="stage">
      <StageBar pill={null} label="Model families for this task" aside="nothing is fit until the plan is locked" loading={false} />
      <div className={s.resCard}>
        <Shelf shelf={shelf} chosen={chosen} fit={null} />
      </div>
    </section>
  );
}

// ── a composed lineage (the engine serves no view for this question yet) ───────

type Lineage = NonNullable<LineageView["after"]>;

function rolesLineage(): Lineage {
  const p = INF.previews.roles!.body as PreviewResult;
  const v = p.views.find((x) => x.kind === "lineage") as LineageView;
  return v.after!;
}

export function exposureLineage(exposure: string | null): PreviewResult {
  const view = {
    kind: "lineage",
    title: exposure ? `Where \`${exposure}\` enters the model` : "Which columns enter the model",
    caption: exposure
      ? `\`${exposure}\` is the exposure; every other predictor is there to adjust its estimate.`
      : "Choose an exposure: the canvas marks where it enters the model.",
    emphasis: exposure ? [exposure] : [],
    coach: [],
    before: null,
    after: rolesLineage(),
    story: [],
  } as unknown as LineageView;
  return {
    kind: "set_estimand",
    views: [view],
    basis: "Read from the column names and summaries; no rows were read.",
    note: null,
    caution: null,
  } as unknown as PreviewResult;
}

// ── results: Table 2, its appendix, and which decisions mattered ───────────────

const pText = (p: number | null | undefined) => (p === null || p === undefined ? "" : p < 0.001 ? "< 0.001" : p.toFixed(3));

function adjustedWords(key: string, adjusted: string[]): string {
  if (!adjusted.length) return "nothing";
  if (key === "model_2") return `the declared set (${adjusted.length})`;
  if (key === "model_3") return "the declared set + body size";
  return adjusted.map((c) => `\`${c}\``).join(", ");
}

const VARIES: Record<MatteredRow["varies"], (r: MatteredRow) => string> = {
  primary: () => "as declared",
  adjustment: (r) =>
    r.key === "crude" ? "adjustment set: none" : r.key === "model_3" ? "timing-unknown body size added" : "adjustment set: the field's Model 1",
  rows: (r) => `eligibility: ${fmtInt(r.n)} rows`,
};

function SpecCurve({ rows }: { rows: MatteredRow[] }) {
  const W = 520;
  const left = 196;
  const right = 72;
  const rowH = 30;
  const lo = Math.min(0, ...rows.map((r) => r.lo ?? r.estimate));
  const hi = Math.max(0, ...rows.map((r) => r.hi ?? r.estimate));
  const x = scaleLinear().domain([lo, hi]).nice().range([left, W - right]);
  const ticks = x.ticks(4);
  const H = rows.length * rowH + 26;
  return (
    <svg className={s.spec} viewBox={`0 0 ${W} ${H}`} role="img" aria-label="The exposure's estimate across the declared analyses">
      {ticks.map((t) => (
        <g key={t}>
          <line className={s.specAxis} x1={x(t)} x2={x(t)} y1={4} y2={H - 18} />
          <text className={s.specTick} x={x(t)} y={H - 4} textAnchor="middle">
            {t === 0 ? "0" : t.toFixed(2).replace("-", "−")}
          </text>
        </g>
      ))}
      <line className={s.specZero} x1={x(0)} x2={x(0)} y1={4} y2={H - 18} />
      {rows.map((r, i) => {
        const y = 14 + i * rowH;
        const primary = r.varies === "primary";
        return (
          <g key={r.key} className={primary ? s.specPrimary : undefined} {...matteredAttrs(r)}>
            <title>{`${r.label}: ${fmtEst(r.estimate)} (${fmtCI(r.lo, r.hi)}), ${fmtInt(r.n)} rows`}</title>
            <text className={s.specLabel} x={0} y={y + 1} fontWeight={primary ? 700 : 500}>
              {r.label}
            </text>
            <text className={s.specSub} x={0} y={y + 13}>
              {VARIES[r.varies](r)}
            </text>
            {r.lo !== null && r.hi !== null ? <line className={s.specCi} x1={x(r.lo)} x2={x(r.hi)} y1={y + 4} y2={y + 4} /> : null}
            <circle className={s.specDot} cx={x(r.estimate)} cy={y + 4} r={primary ? 5 : 4.5} />
            <text className={s.specNum} x={W} y={y + 8} textAnchor="end">
              {fmtEst(r.estimate)}
            </text>
          </g>
        );
      })}
    </svg>
  );
}

export function ResultsCanvas({
  appendix,
  onAppendix,
  mattered,
  onMattered,
}: {
  appendix: boolean;
  onAppendix: () => void;
  mattered: boolean;
  onMattered: () => void;
}) {
  const t2 = table2Rows(INF.effects);
  const seq = (INF.effects.families[0]?.sequence ?? []) as SequenceFit[];
  const pOf = (key: string) => seq.find((f) => f.key === key)?.effects.find((e) => e.feature === INF.effects.exposure)?.p;
  const model = INF.fit.models[0]!;
  const rows = matteredRows(INF.effects, INF.sensitivity).sort((a, b) => a.estimate - b.estimate);
  const est = rows.map((r) => r.estimate);
  const primary = rows.find((r) => r.varies === "primary");
  const furthest = primary
    ? rows.reduce((a, b) => (Math.abs(b.estimate - primary.estimate) > Math.abs(a.estimate - primary.estimate) ? b : a))
    : null;
  const allExclude = rows.every((r) => (r.hi !== null && r.hi < 0) || (r.lo !== null && r.lo > 0));
  const exposure = INF.effects.exposure ?? "the exposure";
  return (
    <section className={stage.stage} data-testid="stage">
      <StageBar pill={null} label="Results" aside="the declared plan, fit once" loading={false} />
      <div className={s.resCard} data-purpose="table2" data-testid="table2">
        <div className={s.subhead}>
          <h3 className={s.resTitle}>
            <Rich text={`Table 2 · \`${exposure}\` and \`glucose\``} />
          </h3>
          <span className={s.hint}>exposure only</span>
        </div>
        <table className={s.t2}>
          <thead>
            <tr>
              <th>Model</th>
              <th>Adjusted for</th>
              <th className={s.num}>Difference per unit (95% CI)</th>
              <th className={s.num}>p</th>
            </tr>
          </thead>
          <tbody>
            {t2.map((r) => (
              <tr key={r.key} data-primary={r.primary || undefined} {...t2Attrs(r)}>
                <td>{r.label}</td>
                <td className={s.t2Adj}>
                  <Rich text={adjustedWords(r.key, r.adjustedFor)} />
                </td>
                <td className={s.num}>
                  {fmtEst(r.estimate)} ({fmtCI(r.lo, r.hi)})
                </td>
                <td className={s.num}>{pText(pOf(r.key))}</td>
              </tr>
            ))}
          </tbody>
        </table>
        <p className={s.caption} data-testid="t2-inference">
          <Rich
            text={`${model.inference.caption} Difference in the mean \`glucose\` per unit of \`${exposure}\`, as the header reads; its unit is not settled.`}
          />
        </p>
        <div className={s.resFoot}>
          <span className={s.footNote}>
            <Rich text={`${model.adjustment_terms.length} other coefficients are ${INF.effects.appendix_title}.`} />
          </span>
          <span className={s.footActions}>
            <button type="button" className={s.btn} onClick={onAppendix} aria-expanded={appendix} data-testid="appendix">
              {appendix ? "Hide the appendix" : "Appendix: adjustment terms"}
            </button>
            <button type="button" className={mattered ? s.btn : s.btnPrimary} onClick={onMattered} aria-expanded={mattered} data-testid="show-mattered">
              Which of my decisions mattered?
            </button>
          </span>
        </div>
        {appendix ? (
          <div className={s.appendix}>
            <table className={s.t2}>
              <thead>
                <tr>
                  <th>Adjustment term</th>
                  <th className={s.num}>Coefficient (95% CI)</th>
                  <th>Why it is not an effect</th>
                </tr>
              </thead>
              <tbody>
                {model.adjustment_terms.map((c) => (
                  <tr key={c.feature}>
                    <td>
                      <code className="v">{c.feature}</code>
                    </td>
                    <td className={s.num}>
                      {fmtEst(c.estimate)} ({fmtCI(c.ci_low, c.ci_high)})
                    </td>
                    <td className={s.t2Adj}>{c.why ?? ""}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        ) : null}
      </div>
      {mattered ? (
        <div className={s.resCard} data-purpose="specification_curve" data-testid="mattered">
          <div className={s.subhead}>
            <h3 className={s.resTitle}>Which of my decisions mattered?</h3>
            <span className={s.hint}>sensitivity, never a way to choose</span>
          </div>
          <p className={s.caption}>
            <Rich
              text={`The \`${exposure}\` estimate across the ${rows.length} analyses declared before any estimate was seen: from ${fmtEst(Math.min(...est))} to ${fmtEst(Math.max(...est))}; ${allExclude ? "every 95% interval excludes 0" : "some 95% intervals include 0"}.${furthest ? ` ${furthest.label} moves it furthest from the primary.` : ""}`}
            />
          </p>
          <SpecCurve rows={rows} />
          <p className={s.caption}>
            <Rich text={`Plan SHA-256 \`${INF.plan.plan_sha256.slice(0, 12)}\`, locked ${INF.plan.declared_at.slice(0, 10)}.`} />
          </p>
        </div>
      ) : null}
    </section>
  );
}

export function ComparisonCanvas({ fit, shelf, split }: { fit: FitArtifact; shelf: ShelfArtifact | null; split: SplitArtifact | null }) {
  const data = comparisonOf(fit, shelf);
  const sel = (fit as unknown as { selection?: { text?: string } | null }).selection;
  return (
    <section className={stage.stage} data-testid="stage">
      <StageBar pill={null} label="Results" aside="cross-validated; the held-out rows stay sealed" loading={false} />
      <div className={s.resCard}>
        <Comparison data={data} fit={fit} basis={metricBasis(fit, split)} />
        {sel?.text ? (
          <p className={s.caption}>
            <Rich text={sel.text} />
          </p>
        ) : null}
      </div>
    </section>
  );
}
