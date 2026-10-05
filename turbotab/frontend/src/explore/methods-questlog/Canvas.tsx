/**
 * The canvas: the production stage's pieces (StageBar, the transform player, PreviewGrid, the
 * Results' Comparison) fed with the server's real preview, evidence and result artifacts. Two
 * views here are composed by the prototype from real artifacts because the engine draws none for
 * that question yet (the exposure and the adjustment set; MODELING_SEQUENCE §3 asks for lineage
 * there); each says so in its own bar.
 */
import { useEffect, useLayoutEffect, useMemo, useState, type ReactNode } from "react";
import { scaleLinear } from "d3-scale";
import type { FitArtifact, LineageView, PreviewResult, ShelfArtifact, SplitArtifact } from "../../api/m1-stage-types";
import { initial } from "../../components/stage/player";
import { PreviewGrid } from "../../components/stage/PreviewGrid";
import { Comparison } from "../../components/stage/results/Comparison";
import { comparisonOf, metricBasis } from "../../components/stage/results/model";
import { PlayerControls, StageBar } from "../../components/stage/StageBar";
import { Rich } from "../../components/stage/text";
import { readoutOf, storyboardOf, trackOf } from "../../components/stage/tracks";
import { createPlayerStore, PlayerContext } from "../../components/stage/usePlayer";
import stage from "../../components/stage/Stage.module.css";
import { fmtEst, fmtInt, INF, type Coef } from "./data";
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

// ── composed lineages (the engine serves no view for these questions yet) ───────

type Lineage = NonNullable<LineageView["after"]>;

function rolesLineage(): Lineage {
  const p = INF.previews.roles!.body as PreviewResult;
  const v = p.views.find((x) => x.kind === "lineage") as LineageView;
  return v.after!;
}

/** The recorded roles' lineage with `leave` columns taken out of the model (no path past raw). */
function without(l: Lineage, leave: Set<string>): Lineage {
  const out = new Map<string, string[]>();
  for (const k of l.links) out.set(k.source, [...(out.get(k.source) ?? []), k.target]);
  const reach = new Set<string>();
  const stack = l.nodes.filter((n) => n.lane === "raw" && !leave.has(n.column ?? "")).map((n) => n.id);
  while (stack.length) {
    const id = stack.pop()!;
    for (const t of out.get(id) ?? []) {
      if (reach.has(t)) continue;
      reach.add(t);
      stack.push(t);
    }
  }
  const nodes = l.nodes.filter((n) => n.lane === "raw" || reach.has(n.id));
  const keep = new Set(nodes.map((n) => n.id));
  const links = l.links.filter(
    (k) => keep.has(k.source) && keep.has(k.target) && !leave.has(k.source.replace(/^raw:/, "")),
  );
  return { ...l, nodes, links };
}

export function exposureLineage(exposure: string): PreviewResult {
  const view: LineageView = {
    kind: "lineage",
    title: `Where \`${exposure}\` enters the model`,
    caption: `\`${exposure}\` is the exposure; every other predictor is there to adjust its estimate.`,
    emphasis: [exposure],
    coach: [],
    before: null,
    after: rolesLineage(),
    story: [],
  } as unknown as LineageView;
  return { kind: "set_estimand", views: [view], basis: "Read from the column names and summaries; no rows were read.", note: null, caution: null } as unknown as PreviewResult;
}

export function adjustmentLineage(leave: string[]): PreviewResult {
  const before = rolesLineage();
  const after = without(before, new Set(leave));
  const list = leave.map((c) => `\`${c}\``);
  const named = list.length > 2 ? `${list.slice(0, -1).join(", ")} and ${list.at(-1)}` : list.join(" and ");
  const view: LineageView = {
    kind: "lineage",
    title: "Which columns enter the primary model",
    caption: `With these answers ${named} leave the primary model.`,
    emphasis: leave,
    coach: [],
    before,
    after,
    story: [],
  } as unknown as LineageView;
  return { kind: "set_adjustment", views: [view], basis: "Read from the column names and summaries; no rows were read.", note: null, caution: null } as unknown as PreviewResult;
}

// ── results: Table 2, its appendix, and which decisions mattered ───────────────

const ci = (c: Coef) => `${fmtEst(c.estimate)} (${fmtEst(c.ci_low)} to ${fmtEst(c.ci_high)})`;
const pText = (p: number | null) => (p === null ? "" : p < 0.001 ? "< 0.001" : p.toFixed(3));

function adjustedWords(fitKey: string, adjusted: string[]): string {
  if (!adjusted.length) return "nothing";
  if (fitKey === "model_2") return `the declared set (${adjusted.length})`;
  if (fitKey === "model_3") return "the declared set + body size";
  return adjusted.map((c) => `\`${c}\``).join(", ");
}

export interface Spec {
  key: string;
  label: string;
  differs: string;
  n: number;
  coef: Coef;
  primary: boolean;
}

/** Every declared analysis of the exposure: the model sequence and the sensitivity analysis. */
export function specs(): Spec[] {
  const seq = INF.effects.families[0]!.sequence;
  const out: Spec[] = seq.map((f) => ({
    key: f.key,
    label: f.label,
    differs:
      f.key === "crude"
        ? "adjustment set: none"
        : f.key === "model_1"
          ? "adjustment set: age, gender, kcal"
          : f.key === "model_3"
            ? "timing-unknown body size adjusted"
            : "as declared",
    n: f.n_rows,
    coef: f.effects[0]!,
    primary: f.key === "model_2",
  }));
  for (const f of INF.sensitivity.fits.filter((x) => x.label !== "Primary"))
    out.push({ key: `sens:${f.label}`, label: f.label, differs: `eligibility: ${fmtInt(f.n_rows)} rows`, n: f.n_rows, coef: f.coefficient, primary: false });
  return out.sort((a, b) => a.coef.estimate - b.coef.estimate);
}

function SpecCurve({ rows }: { rows: Spec[] }) {
  const W = 520;
  const left = 196;
  const right = 72;
  const rowH = 30;
  const lo = Math.min(0, ...rows.map((r) => r.coef.ci_low));
  const hi = Math.max(0, ...rows.map((r) => r.coef.ci_high));
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
        return (
          <g key={r.key} className={r.primary ? s.specPrimary : undefined}>
            <title>{`${r.label}: ${ci(r.coef)}, ${fmtInt(r.n)} rows`}</title>
            <text className={s.specLabel} x={0} y={y + 1} fontWeight={r.primary ? 700 : 500}>
              {r.label}
            </text>
            <text className={s.specSub} x={0} y={y + 13}>
              {r.differs}
            </text>
            <line className={s.specCi} x1={x(r.coef.ci_low)} x2={x(r.coef.ci_high)} y1={y + 4} y2={y + 4} />
            <circle className={s.specDot} cx={x(r.coef.estimate)} cy={y + 4} r={r.primary ? 5 : 4.5} />
            <text className={s.specNum} x={W} y={y + 8} textAnchor="end">
              {fmtEst(r.coef.estimate)}
            </text>
          </g>
        );
      })}
    </svg>
  );
}

export function ResultsCanvas({ appendix, onAppendix }: { appendix: boolean; onAppendix: () => void }) {
  const fam = INF.effects.families[0]!;
  const model = INF.fit.models[0]!;
  const rows = specs();
  const est = rows.map((r) => r.coef.estimate);
  const primary = rows.find((r) => r.primary)!;
  const furthest = rows.reduce((a, b) =>
    Math.abs(b.coef.estimate - primary.coef.estimate) > Math.abs(a.coef.estimate - primary.coef.estimate) ? b : a,
  );
  const allExclude = rows.every((r) => r.coef.ci_high < 0 || r.coef.ci_low > 0);
  return (
    <section className={stage.stage} data-testid="stage">
      <StageBar pill={null} label="Results" aside="the declared plan, fit once" loading={false} />
      <div className={s.resCard} data-purpose="table2">
        <div className={s.subhead}>
          <h3 className={s.resTitle}>
            <Rich text="Table 2 · `sugar` and `glucose`" />
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
            {fam.sequence.map((f) => (
              <tr key={f.key} data-primary={f.key === "model_2" || undefined}>
                <td>{f.label}</td>
                <td className={s.t2Adj}>
                  <Rich text={adjustedWords(f.key, f.adjusted_for)} />
                </td>
                <td className={s.num}>{ci(f.effects[0]!)}</td>
                <td className={s.num}>{pText(f.effects[0]!.p)}</td>
              </tr>
            ))}
          </tbody>
        </table>
        <p className={s.caption}>
          <Rich text={`${model.inference.caption} Difference in the mean \`glucose\` per unit of \`sugar\`, as the header reads; its unit is not settled.`} />
        </p>
        <div className={s.footer}>
          <span className={s.footNote}>
            <Rich text={`${model.adjustment_terms.length} other coefficients are ${INF.effects.appendix_title}.`} />
          </span>
          <button type="button" className={s.btn} onClick={onAppendix} aria-expanded={appendix} data-testid="appendix">
            {appendix ? "Hide the appendix" : "Appendix: adjustment terms"}
          </button>
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
                    <td className={s.num}>{ci(c)}</td>
                    <td className={s.t2Adj}>{c.why ?? ""}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        ) : null}
      </div>
      <div className={s.resCard} data-purpose="specification_curve">
        <div className={s.subhead}>
          <h3 className={s.resTitle}>Which of my decisions mattered?</h3>
          <span className={s.hint}>sensitivity, never a way to choose</span>
        </div>
        <p className={s.caption}>
          <Rich
            text={`The \`sugar\` estimate across the ${rows.length} analyses declared before any estimate was seen: from ${fmtEst(Math.min(...est))} to ${fmtEst(Math.max(...est))}; ${allExclude ? "every 95% interval excludes 0" : "some 95% intervals include 0"}. ${furthest.label} moves it furthest from the primary.`}
          />
        </p>
        <SpecCurve rows={rows} />
        <p className={s.caption}>
          <Rich text={`Plan SHA-256 \`${INF.plan.plan_sha256.slice(0, 12)}\`, locked ${INF.plan.declared_at.slice(0, 10)}.`} />
        </p>
      </div>
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
