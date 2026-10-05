/**
 * The map: the analysis drawn left to right as a figure — the table, each decision where it acts,
 * the estimate — with the rows as a ribbon above and the columns as lanes below. Regions are the
 * reporting guideline's sections (STROBE-nut under inference, TRIPOD+AI under prediction).
 *
 * A stated decision is a small solid dot with its phrase; an asked one a hollow ring with its guess
 * (the current objective breathes, gently, unless motion is reduced); a waiting one a dashed ring;
 * a silent one is not drawn, only counted in its region's footer. Clicking any node opens it.
 */
import { useMemo, type KeyboardEvent } from "react";
import { FX, INF, PRED, fmtCi, fmtInt } from "./fixture";
import { specsOf } from "./Results";
import { H, LABEL_W, RAIL_Y, RIBBON_Y, layout } from "./geometry";
import {
  INITIAL,
  MAP_TITLE,
  NODE_TITLE,
  REGIONS,
  answered,
  derive,
  fitFor,
  tierOf,
  unconfirmed,
  type Answers,
  type NodeId,
  type Purpose,
  type Tier,
} from "./model";
import { draw } from "./threads";
import { useSize } from "../../components/stage/views/geometry";
import m from "./map.module.css";

interface Props {
  purpose: Purpose;
  /** What is recorded. */
  answers: Answers;
  /** What the hovered option would record (drawn instead, marked as a preview). */
  preview: Answers | null;
  focus: NodeId | null;
  /** The newcomer's walk: only this node is lit. */
  walking: NodeId | null;
  objective: NodeId | null;
  onFocus: (n: NodeId) => void;
  onSilent: (region: string) => void;
}

const INF_TEACH: Record<string, { value: string; label: string }[]> = Object.fromEntries(
  Object.entries(FX.teaching).map(([k, t]) => [k, t.options]),
);

/** A teaching entry's label for an option (the app's four-word name for it). */
export const teachLabel = (key: string, value: string) =>
  INF_TEACH[key]?.find((o) => o.value === value)?.label ?? value;
const TEACH_LABEL = teachLabel;

/** The phrase under a node: the stated value, the guess, or what it waits on. */
export function phraseOf(n: NodeId, a: Answers, purpose: Purpose): { text: string; kind: "value" | "guess" | "wait" | "none" } {
  switch (n) {
    case "lens":
      return { text: "dietary", kind: "value" };
    case "outcome":
      return { text: "glucose", kind: "value" };
    case "purpose":
      return { text: purpose, kind: "value" };
    case "grain":
      return { text: "one row per unit", kind: "value" };
    case "readings": {
      const left = unconfirmed(a).length + (a.unit ? 0 : 1);
      return answered(a, "readings")
        ? { text: `${INF.readings.items.length + 1} confirmed`, kind: "value" }
        : { text: `${left} to confirm`, kind: "guess" };
    }
    case "exclusions":
      if (purpose === "prediction")
        return a.pExclusions ? { text: "Keep every row", kind: "value" } : { text: "guess: keep every row", kind: "guess" };
      if (a.exclusions) return { text: INF.exclusions.labels.options[a.exclusions === "none" ? "keep_every_row" : a.exclusions]?.label ?? a.exclusions, kind: "value" };
      return { text: "asked · no guess", kind: "guess" };
    case "seal":
      return { text: INF.seal.options.find((o) => o.holdout === 0)!.label, kind: "value" };
    case "clusters":
      return { text: TEACH_LABEL("clusters", "none"), kind: "value" };
    case "exposure":
      return a.exposure ? { text: "sugar · total · substitution", kind: "value" } : { text: "guess: sugar", kind: "guess" };
    case "adjustment": {
      if (!a.exposure) return { text: "waits on the exposure", kind: "wait" };
      const groups = INF.adjustment.groups;
      const done = groups.filter((g) => a.adjustment[g.key]).length;
      if (done < groups.length) {
        const n = groups.reduce((s, g) => s + g.columns.length, 0);
        return done
          ? { text: `${groups.length - done} of ${groups.length} groups left`, kind: "guess" }
          : { text: `${n} covariates, ${groups.length} groups`, kind: "guess" };
      }
      let inM = 0;
      let out = 0;
      let beside = 0;
      for (const g of groups)
        for (const c of g.columns) {
          const d = derive(a.adjustment[g.key]![c]!);
          if (d.adjusted) inM++;
          else if (d.secondary) beside++;
          else out++;
        }
      return { text: `${inM} in · ${beside} beside · ${out} out`, kind: "value" };
    }
    case "energy":
      return { text: TEACH_LABEL("energy_adjustment", a.energy), kind: "value" };
    case "form":
      return { text: INF.form.options.find((o) => o.value === a.form)?.label ?? a.form, kind: "value" };
    case "missing":
      return { text: "waits on the adjustment", kind: "wait" };
    case "model1":
      return a.model1 === "guess"
        ? { text: INF.model_1.guess.join(", "), kind: "value" }
        : a.model1 === "empty"
          ? { text: "none declared", kind: "value" }
          : { text: `guess: ${INF.model_1.guess.join(", ")}`, kind: "guess" };
    case "family":
      return { text: INF.shelf.find((f) => f.key === "linear")!.label, kind: "value" };
    case "lock":
      return a.locked
        ? { text: `locked · ${INF.lock.digests[a.locked.key] ?? ""}`, kind: "value" }
        : { text: "at the first estimate", kind: "wait" };
    case "estimate": {
      const pick = a.locked ? fitFor(a) : null;
      if (pick?.kind === "fit") {
        const m2 = pick.fit.sequence.find((s) => s.key === "model_2")!;
        return pick.fit.tests.length
          ? { text: "a curve: 3 terms", kind: "value" }
          : { text: `n ${fmtInt(m2.n_rows)}`, kind: "value" };
      }
      return { text: !a.locked ? "after the lock" : pick?.kind === "error" ? "the engine could not fit" : "not captured", kind: "wait" };
    }
    case "matter": {
      const pick = a.locked ? fitFor(a) : null;
      if (pick?.kind !== "fit") return { text: "after the lock", kind: "wait" };
      return pick.fit.tests.length
        ? { text: "curves: none to line up", kind: "value" }
        : { text: `${specsOf(pick.fit, a).length} declared`, kind: "value" };
    }
    case "p_missing":
      return a.pMissing
        ? { text: PRED.missing.labels.options.impute!.label, kind: "value" }
        : { text: `guess: ${PRED.missing.labels.options.impute!.label.toLowerCase()}`, kind: "guess" };
    case "p_seal":
      return a.pSeal
        ? { text: PRED.seal.options.find((o) => String(o.holdout) === a.pSeal)!.label, kind: "value" }
        : { text: `guess: ${PRED.seal.options[0]!.label.toLowerCase()}`, kind: "guess" };
    case "p_energy":
    case "p_models":
      return a.pSeal ? { text: "asked next · not captured", kind: "wait" } : { text: "waits on the seal", kind: "wait" };
    case "p_score":
      return { text: a.pSeal && a.pSeal !== "0" ? "opened once, at the end" : "waits on the seal", kind: "wait" };
    default:
      return { text: "", kind: "none" };
  }
}

/** Steps that change no number (not applicable here), by region: the export's silent lines. */
export function silentByRegion(a: Answers, purpose: Purpose): Record<string, { key: string; reason: string }[]> {
  const steps = purpose === "inference" ? INF.steps : PRED.steps;
  const where: Record<string, string> = {
    orientation: "data",
    event: "data",
    follow_up: "data",
    repeat_kind: "data",
    unit: "data",
    aggregation: "data",
    temporal: "data",
    survey: "participants",
    time_varying: purpose === "inference" ? "exposure" : "methods",
    causal: purpose === "inference" ? "methods" : "methods",
    estimand: "methods",
    adjustment: "methods",
  };
  const out: Record<string, { key: string; reason: string }[]> = {};
  for (const s of steps) {
    if (s.status !== "not_applicable" || !s.reason) continue;
    const r = where[s.key];
    if (!r) continue;
    (out[r] ??= []).push({ key: s.key, reason: s.reason });
  }
  if (purpose === "inference" && answered(a, "adjustment"))
    (out.methods ??= []).push({ key: "missing", reason: INF.missing.sentences.complete_case! });
  return out;
}

export function MapView({ purpose, answers, preview, focus, walking, objective, onFocus, onSilent }: Props) {
  const [ref, { w }] = useSize<HTMLDivElement>();
  const width = Math.max(960, w);
  // A silent decision is not drawn: its region's other nodes close the gap.
  const silentNodes = REGIONS[purpose]
    .flatMap((r) => r.nodes)
    .filter((n) => tierOf(answers, n, purpose) === "silent")
    .join();
  const L = useMemo(
    () => layout(width, purpose, new Set(silentNodes ? (silentNodes.split(",") as NodeId[]) : [])),
    [width, purpose, silentNodes],
  );
  const shown = preview ?? answers;
  const drawing = useMemo(() => draw(L, shown, purpose), [L, shown, purpose]);
  const recordedDrawing = useMemo(() => (preview ? draw(L, answers, purpose) : null), [L, answers, preview, purpose]);
  const changed = useMemo(() => {
    if (!recordedDrawing) return new Set<string>();
    const before = new Map(recordedDrawing.segs.map((s) => [s.key, s.d + s.tone + s.dashed]));
    return new Set(drawing.segs.filter((s) => before.get(s.key) !== s.d + s.tone + s.dashed).map((s) => s.key));
  }, [drawing, recordedDrawing]);
  const silent = silentByRegion(answers, purpose);
  const regions = REGIONS[purpose];
  // The estimate, where the lanes end: once the plan is locked, and only for a fit captured.
  const pick = purpose === "inference" && answers.locked ? fitFor(answers) : null;
  const estimate =
    pick?.kind === "fit"
      ? pick.fit.tests.length
        ? "sugar: a curve of 3 terms"
        : `β ${fmtCi(pick.fit.sequence.find((s) => s.key === "model_2")!.effects[0]!)}`
      : null;
  const lit = (n: NodeId) => !walking || walking === n;

  const key = (n: NodeId) => (e: KeyboardEvent) => {
    if (e.key === "Enter" || e.key === " ") {
      e.preventDefault();
      onFocus(n);
    }
  };

  return (
    <div ref={ref} className={m.mapWrap} data-testid="map" data-walking={walking ?? undefined}>
      <svg className={m.map} width={width} height={H} viewBox={`0 0 ${width} ${H}`} role="group" aria-label="The analysis, from the table to the estimate">
        {/* regions: the methods section's parts */}
        {L.regions.map((r, i) => (
          <g key={r.id} className={m.region} data-odd={i % 2 || undefined}>
            <rect x={r.x0} y={0} width={r.x1 - r.x0} height={H} className={m.regionBg} />
            {i > 0 ? <line x1={r.x0} x2={r.x0} y1={6} y2={H - 6} className={m.regionRule} /> : null}
            <text x={r.x0 + 12} y={17} className={m.regionTitle}>
              {r.title}
            </text>
            <text x={r.x0 + 12} y={30} className={m.regionItems}>
              {r.items}
            </text>
            {silent[r.id]?.length ? (
              <g
                className={m.silent}
                role="button"
                tabIndex={0}
                aria-label={`${silent[r.id]!.length} silent decisions in ${r.title}: shown in the export only`}
                onClick={() => onSilent(r.id)}
                onKeyDown={(e) => {
                  if (e.key === "Enter" || e.key === " ") {
                    e.preventDefault();
                    onSilent(r.id);
                  }
                }}
              >
                <text x={r.x1 - 10} y={H - 8} textAnchor="end">
                  {silent[r.id]!.length} silent · export only
                </text>
              </g>
            ) : null}
          </g>
        ))}

        {/* the rows */}
        <g className={m.ribbon} data-preview={recordedDrawing && recordedDrawing.ribbon.d !== drawing.ribbon.d ? true : undefined}>
          <path d={drawing.ribbon.d} className={m.ribbonBand} />
          {drawing.ribbon.sealed ? (
            <>
              <path d={drawing.ribbon.sealed.d} className={m.ribbonSealed} />
              <text x={L.x.p_score! - 12} y={RAIL_Y - 36} textAnchor="end" className={m.ribbonNote}>
                {drawing.ribbon.sealed.label}
              </text>
            </>
          ) : null}
          <text x={LABEL_W + 2} y={RIBBON_Y + 4} textAnchor="end" className={m.ribbonLabel}>
            {drawing.ribbon.start}
          </text>
          {drawing.ribbon.drops.map((d) => (
            <text key={d.x} x={d.x + 22} y={RIBBON_Y + 19} className={m.ribbonDrop}>
              {d.label}
            </text>
          ))}
          <text x={(purpose === "inference" ? L.x.estimate! : L.x.p_score!) + 8} y={RIBBON_Y + 4} className={m.ribbonLabel}>
            {drawing.ribbon.end}
          </text>
        </g>

        {/* the columns */}
        <g className={m.lanes}>
          {drawing.labels.map((l) => (
            <text key={l.key} x={LABEL_W} y={l.y + 3.5} textAnchor="end" className={m.laneLabel} data-tone={l.tone}>
              {l.label}
            </text>
          ))}
          {drawing.segs.map((s) => (
            <path
              key={s.key}
              d={s.d}
              className={m.seg}
              data-tone={s.tone}
              data-dashed={s.dashed || undefined}
              data-faint={s.faint || undefined}
              data-changed={changed.has(s.key) || undefined}
              style={{ strokeWidth: s.width }}
            />
          ))}
          {drawing.caps.map((c) => (
            <g key={c.key} className={m.cap} data-tone={c.tone}>
              {c.key.endsWith("-out") ? <line x1={c.x} x2={c.x} y1={c.y - 4} y2={c.y + 4} /> : null}
              <text x={c.x + (c.key.endsWith("-out") ? 6 : 0)} y={c.key.endsWith("-out") ? c.y + 3.5 : c.y}>
                {c.label}
              </text>
            </g>
          ))}
          {drawing.markers.map((k) => (
            <g key={k.key} className={m.marker} transform={`translate(${k.x},${k.y})`}>
              <rect x={-k.label.length * 3.1 - 5} y={-7} width={k.label.length * 6.2 + 10} height={14} rx={7} />
              <text textAnchor="middle" y={3.5}>
                {k.label}
              </text>
            </g>
          ))}
          {drawing.brackets.map((b) => (
            <g key={b.key} className={m.bracket} data-dashed={b.dashed || undefined}>
              <path d={`M${b.x - 4},${b.y0 - 6}H${b.x}V${b.y1 + 6}H${b.x - 4}`} />
              <text x={b.x + 6} y={b.y0 - 8} className={m.bracketLabel}>
                {b.label}
              </text>
              <text x={b.x + 6} y={b.y0 + 4} className={m.bracketSub}>
                {b.sub}
              </text>
            </g>
          ))}
          {drawing.result ? (
            <>
              <path d={drawing.result.d} className={m.resultLine} data-live={drawing.result.live || undefined} />
              {estimate ? (
                <text x={drawing.result.x0 + 8} y={drawing.result.y + 15} className={m.resultLabel} data-testid="map-estimate">
                  {estimate}
                </text>
              ) : null}
            </>
          ) : null}
        </g>

        {/* the decisions */}
        {regions.flatMap((r) => r.nodes).map((n) => {
          const nx = L.x[n];
          if (nx === undefined) return null;
          if (n === "source") return <Source key={n} x={nx} focus={focus} walking={walking} onFocus={onFocus} purpose={purpose} />;
          const tier = tierOf(answers, n, purpose);
          if (tier === "silent") return null;
          const ph = phraseOf(n, shown, purpose);
          const isObjective = objective === n;
          const previewing = !!preview && phraseOf(n, answers, purpose).text !== ph.text;
          const room = Math.max(60, (L.room[n] ?? 100) - 6);
          return (
            <g
              key={n}
              className={m.node}
              data-tier={tier}
              data-focus={focus === n || undefined}
              data-dim={!lit(n) || undefined}
              data-objective={isObjective || undefined}
              data-preview={previewing || undefined}
              style={{ transform: `translate(${nx}px, ${RAIL_Y}px)` }}
              role="button"
              tabIndex={0}
              aria-label={`${NODE_TITLE[n]}: ${tierWord(tier)}. ${ph.text}`}
              aria-pressed={focus === n}
              onClick={() => onFocus(n)}
              onKeyDown={key(n)}
              data-testid={`node-${n}`}
            >
              <rect x={-room / 2} y={-30} width={room} height={66} className={m.hit} />
              <text y={-17} textAnchor="middle" className={m.nodeKicker}>
                {MAP_TITLE[n]}
              </text>
              <Glyph tier={tier} objective={isObjective} />
              <PhraseText text={ph.text} kind={ph.kind} tier={tier} chars={Math.floor((room - 4) / 6.1)} />
            </g>
          );
        })}
      </svg>
    </div>
  );
}

function tierWord(t: Tier): string {
  return t === "asked" ? "asked of you" : t === "recorded" ? "answered" : t === "stated" ? "written in, changeable" : t;
}

function Glyph({ tier, objective }: { tier: Tier; objective: boolean }) {
  switch (tier) {
    case "asked":
      return (
        <>
          {objective ? <circle r={13} className={m.halo} /> : null}
          <circle r={8} className={m.ring} />
        </>
      );
    case "recorded":
      return (
        <>
          <circle r={7.5} className={m.dotRecorded} />
          <path d="M-3.2,0.2 L-0.8,2.6 L3.4,-2.4" className={m.check} />
        </>
      );
    case "stated":
      return <circle r={5} className={m.dotStated} />;
    case "waiting":
      return <circle r={7} className={m.ringWaiting} />;
    case "gate":
      return <rect x={-6.5} y={-6.5} width={13} height={13} rx={2.5} className={m.gate} />;
    case "result":
      return <path d="M0,-8 L8,0 L0,8 L-8,0Z" className={m.result} />;
    default:
      return <circle r={5} className={m.dotStated} />;
  }
}

/** The phrase under a node, wrapped onto at most two lines of about 18 characters. */
function PhraseText({ text, kind, tier, chars }: { text: string; kind: string; tier: Tier; chars: number }) {
  const lines = wrap(text, Math.max(9, chars));
  return (
    <text y={22} textAnchor="middle" className={m.phrase} data-kind={kind} data-tier={tier}>
      {lines.map((l, i) => (
        <tspan key={i} x={0} dy={i === 0 ? 0 : 13}>
          {l}
        </tspan>
      ))}
    </text>
  );
}

export function wrap(text: string, width: number): string[] {
  const words = text.split(" ");
  const out: string[] = [];
  let cur = "";
  for (const w of words) {
    if (cur && (cur + " " + w).length > width) {
      out.push(cur);
      cur = w;
    } else cur = cur ? `${cur} ${w}` : w;
  }
  if (cur) out.push(cur);
  if (out.length > 2) return [out[0]!, `${out.slice(1).join(" ").slice(0, width - 1)}…`];
  return out;
}

/** The table: the source of every lane, with the four phrases the app wrote in about it. */
function Source({
  x,
  focus,
  walking,
  onFocus,
  purpose,
}: {
  x: number;
  focus: NodeId | null;
  walking: NodeId | null;
  onFocus: (n: NodeId) => void;
  purpose: Purpose;
}) {
  const rows: [NodeId, string][] = [
    ["lens", "dietary"],
    ["outcome", "glucose"],
    ["purpose", purpose],
    ["grain", "one row per unit"],
  ];
  return (
    <g className={m.source} data-dim={walking ? true : undefined} transform={`translate(${x},${RAIL_Y})`}>
      <g
        role="button"
        tabIndex={0}
        aria-label={`The table: ${FX.meta.file}, ${fmtInt(FX.meta.rows)} rows by ${FX.meta.cols} columns`}
        data-focus={focus === "source" || undefined}
        onClick={() => onFocus("source")}
        onKeyDown={(e) => {
          if (e.key === "Enter" || e.key === " ") {
            e.preventDefault();
            onFocus("source");
          }
        }}
        data-testid="node-source"
        className={m.sourceHead}
      >
        <rect x={-58} y={-40} width={116} height={30} rx={7} className={m.sourceBox} />
        <text y={-27} textAnchor="middle" className={m.sourceFile}>
          {FX.meta.file.replace(/\.csv$/, "")}
        </text>
        <text y={-16} textAnchor="middle" className={m.sourceDims}>
          {fmtInt(FX.meta.rows)} × {FX.meta.cols}
        </text>
      </g>
      {rows.map(([n, v], i) => (
        <g
          key={n}
          className={m.sourceRow}
          data-focus={focus === n || undefined}
          role="button"
          tabIndex={0}
          aria-label={`${NODE_TITLE[n]}: ${v}, written in, changeable`}
          onClick={() => onFocus(n)}
          onKeyDown={(e) => {
            if (e.key === "Enter" || e.key === " ") {
              e.preventDefault();
              onFocus(n);
            }
          }}
          transform={`translate(0,${-2 + i * 13.5})`}
          data-testid={`node-${n}`}
        >
          <rect x={-58} y={-9} width={116} height={13} className={m.hit} />
          <text x={-6} textAnchor="end" className={m.sourceKey}>
            {MAP_TITLE[n].toLowerCase()}
          </text>
          <text x={0} className={m.sourceVal}>
            {v}
          </text>
        </g>
      ))}
    </g>
  );
}

export { INITIAL };
