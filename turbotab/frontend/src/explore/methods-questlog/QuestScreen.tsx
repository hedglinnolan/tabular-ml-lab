/**
 * /lab/methods-questlog — design prototype (angle B, "quest log") of the living methods section
 * (BLUEPRINT §11.4). Left: the methods section as an objective list, sections in the reporting
 * guideline's order, each with its open slots counted and finished ones collapsed to their
 * sentence. Center: the current objective as one focused card. Right: the canvas (the production
 * stage's pieces). The whole document is one press away. Above it all, the pipeline banner.
 *
 * Walked by clicks: from the first draft, each card's own control records the shared scenario's
 * answer (methods-shared/SCENARIO.md) and the quest log moves to the next captured moment, until
 * the fit locks the plan and Table 2 and "Which of my decisions mattered?" are on the canvas.
 * Reset returns to the first draft. Every sentence, guess, piece of evidence and number is the
 * real server's, captured on the NHANES export (fixture.json; capture_drive.py, trim.py); nothing
 * is fetched. `?m=<moment>` (with `&doc`, `&mattered`, `&appendix`, `&variant=prediction`, `&shot`)
 * opens a moment for review captures; nothing requires it.
 */
import { useCallback, useEffect, useMemo, useState, type ReactNode } from "react";
import type { FitArtifact, PreviewResult, ShelfArtifact, SplitArtifact } from "../../api/m1-stage-types";
import type { ProjectView } from "../../api/schema";
import { BannerView } from "../../components/banner/Banner";
import { deriveBanner, type BannerInput } from "../../components/banner/derive";
import { Header } from "../../components/Header";
import { ColumnsContext } from "../../components/stage/text";
import { StageFocusProvider } from "../../state/focus";
import { fmtEst, table2Rows } from "../methods-shared/results";
import { ComparisonCanvas, exposureLineage, NoteCanvas, PreviewCanvas, ResultsCanvas, ShelfCanvas } from "./Canvas";
import {
  AdjustmentCard,
  EnergyCard,
  EstimandCard,
  ExclusionsCard,
  Frame,
  ItemCard,
  LockCard,
  LockedCard,
  MissingCard,
  missingPreviewKey,
  ModelsCard,
  ReadingsCard,
  RolesCard,
  SequenceCard,
  SplitCard,
  splitPreviewKey,
  type Mode,
} from "./Cards";
import { FX, INF, isPreview, PREDICTION, type Moment } from "./data";
import { MethodsDoc } from "./Doc";
import {
  ADJUSTMENT_ANSWERS,
  advance,
  askOf,
  idOf,
  isLocked,
  linesOf,
  loadWalk,
  momentOf,
  nextSentence,
  nextSingle,
  objectiveItem,
  saveWalk,
  SCENARIO,
  singlesDone,
  START,
  walkAt,
  type Walk,
} from "./journey";
import { readingSlots } from "./readings";
import { KINDS, sectionsOf, STROBE, TRIPOD, type Extras, type Guideline, type Item } from "./sections";
import { Rail } from "./Rail";
import s from "./questlog.module.css";

type Current = { section: string; item: string } | null;
type Variant = "inference" | "prediction";

/** Review presets, read once: from the query, or from a hash route's own query (#/questlog?m=…). */
function readPreset() {
  const q = new URLSearchParams(window.location.search);
  const hash = window.location.hash;
  const h = hash.includes("?") ? new URLSearchParams(hash.slice(hash.indexOf("?") + 1)) : null;
  const get = (k: string) => q.get(k) ?? h?.get(k) ?? null;
  const has = (k: string) => q.has(k) || !!h?.has(k);
  const rest = get("rest");
  return {
    m: get("m"),
    shot: has("shot"),
    doc: has("doc"),
    appendix: has("appendix"),
    mattered: has("mattered"),
    variant: (get("variant") === "prediction" ? "prediction" : "inference") as Variant,
    rest: (rest === null ? undefined : rest === "now" || rest === "with" ? rest : Number(rest)) as number | "now" | "with" | undefined,
  };
}

const sectionKey = (item: string, guideline: Guideline = "STROBE-nut") =>
  (guideline === "STROBE-nut" ? STROBE : TRIPOD).find((d) => d.items.includes(item))?.key ?? "data";

function objectiveOf(w: Walk): Current {
  const item = objectiveItem(w);
  return item ? { section: sectionKey(item), item } : null;
}

const answered = (m: Moment, key: string) => m.view.interview.find((x) => x.key === key)?.status === "answered";

function analyzedOf(m: Moment): number | null {
  const cohort = m.stages.cohort?.artifact as { n_final?: number } | undefined;
  return answered(m, "missing") && cohort?.n_final ? cohort.n_final : null;
}

/** What the walk recorded locally since its moment was captured, as the items' tiers. */
function overridesOf(w: Walk): Extras["overrides"] {
  const id = idOf(w);
  if (id === "adjustment") return { adjustment: { count: ADJUSTMENT_ANSWERS.length - w.adjusted.length } };
  if (id === "model_sequence")
    return {
      model_sequence: { tier: "asked", count: 1, optional: false },
      ...(w.codes ? { models: { tier: "waiting", waitingOn: ["Model sequence"], count: 0 } } : {}),
    };
  if (id === "models" && w.codes) return { models: { tier: "asked", count: 1, waitingOn: [] } };
  return {};
}

function banner(m: Moment) {
  const st = m.stages as unknown as Record<string, BannerInput["ingest"]>;
  const model = deriveBanner({
    view: m.view as unknown as ProjectView,
    ingest: st.ingest,
    oriented: st.oriented as BannerInput["oriented"],
    working: st.working as BannerInput["working"],
    cohort: st.cohort as BannerInput["cohort"],
    split: st.split as BannerInput["split"],
    design: st.design as BannerInput["design"],
    fit: st.fit as BannerInput["fit"],
    shelf: st.shelf as BannerInput["shelf"],
  });
  // The production banner states a cross-validated MSE under inference, where MODELING_SEQUENCE
  // §1 row 11 shows none: under inference the result is the exposure's primary estimate.
  const result = model.segments[3];
  const primary = table2Rows(INF.effects).find((r) => r.primary);
  if (m.view.state.purpose === "inference" && result.value !== null && primary) {
    model.segments[3] = {
      ...result,
      metric: INF.effects.exposure ?? "the exposure",
      value: primary.estimate,
      basis: "Model 2",
      family: "per unit",
      summary: `Result: the primary estimate for ${INF.effects.exposure ?? "the exposure"}, ${fmtEst(primary.estimate)} per unit.`,
    };
  }
  return model;
}

const KIND_ITEM: Record<string, string> = Object.fromEntries(Object.entries(KINDS).flatMap(([item, kinds]) => kinds.map((k) => [k, item])));

export function QuestScreen() {
  const [preset] = useState(readPreset);
  const [first] = useState<Walk>(() => (preset.m ? walkAt(preset.m) : (loadWalk() ?? START)));
  const [walk, setWalk] = useState<Walk>(first);
  const [variant, setVariant] = useState<Variant>(preset.variant);
  const [current, setCurrent] = useState<Current>(() => objectiveOf(first));
  const [doc, setDoc] = useState(preset.doc || isLocked(first));
  const [peek, setPeek] = useState<string | null>(null);
  const [readingFocus, setReadingFocus] = useState<string | null>(null);
  const [families, setFamilies] = useState<string[]>([]);
  const [appendix, setAppendix] = useState(preset.appendix);
  const [mattered, setMattered] = useState(preset.mattered);

  useEffect(() => {
    if (!preset.m) saveWalk(walk);
  }, [walk, preset.m]);

  const prediction = variant === "prediction";
  const guideline: Guideline = prediction ? "TRIPOD+AI" : "STROBE-nut";
  const moment = prediction ? PREDICTION : momentOf(walk);
  const lines = useMemo(() => (prediction ? PREDICTION.methods.lines : linesOf(walk)), [prediction, walk]);
  const ask = prediction ? null : askOf(walk);
  const locked = !prediction && isLocked(walk);
  const id = idOf(walk);
  const objective = prediction ? null : objectiveItem(walk);

  const sections = useMemo(() => {
    const extras: Extras = {
      readings: { open: ask?.groups.length ?? 0, waiting: !answered(moment, "roles") },
      analyzed: analyzedOf(moment),
      overrides: prediction ? {} : overridesOf(walk),
    };
    return sectionsOf(moment, lines, extras, guideline);
  }, [moment, lines, ask, guideline, prediction, walk]);

  /** Every record moves the quest log: the new moment, its objective in focus. */
  const go = useCallback((next: Walk) => {
    setWalk(next);
    setCurrent(objectiveOf(next));
    setDoc(isLocked(next));
    setPeek(null);
    setReadingFocus(null);
    setFamilies([]);
  }, []);
  const record = useCallback(() => go(advance(walk)), [go, walk]);
  const toObjective = useCallback(() => {
    setCurrent(objectiveOf(walk));
    setDoc(isLocked(walk));
    setPeek(null);
  }, [walk]);
  const reset = () => {
    saveWalk(null);
    setVariant("inference");
    setMattered(false);
    setAppendix(false);
    go(START);
  };
  const onAdjust = (i: number) => {
    const next = { ...walk, adjusted: [...walk.adjusted, i] };
    if (ADJUSTMENT_ANSWERS.every((a) => next.adjusted.includes(a.index))) go(advance(next));
    else {
      setWalk(next);
      setPeek(null);
    }
  };

  const next = useCallback(() => {
    const flat = sections.flatMap((sec) => sec.items.filter((i) => i.tier === "asked" && !i.optional).map((i) => ({ section: sec.key, item: i.key })));
    const obj = objectiveOf(walk);
    if (obj && (doc || current?.item !== obj.item)) {
      setCurrent(obj);
    } else if (flat.length) {
      const at = flat.findIndex((f) => f.item === current?.item);
      setCurrent(flat[(at + 1) % flat.length]!);
    }
    setDoc(false);
    setPeek(null);
  }, [sections, walk, doc, current]);

  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      const el = e.target as HTMLElement | null;
      if (el && /^(INPUT|TEXTAREA|SELECT)$/.test(el.tagName)) return;
      if (e.metaKey || e.ctrlKey || e.altKey) return;
      if (e.key === "m" || e.key === "M") setDoc((d) => !d);
      if ((e.key === "n" || e.key === "N") && !prediction) next();
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [next, prediction]);

  const known = useMemo(() => new Set(Object.keys(moment.view.state.roles ?? {}).concat(["glucose"])), [moment]);
  const sec = prediction ? undefined : sections.find((x) => x.key === current?.section);
  const item = sec?.items.find((i) => i.key === current?.item);

  // ── the center ──
  const stated = (it: Item): Mode => (it.tier === "stated" || it.tier === "engine" ? "stated" : "open");
  const sentenceOf = (kind: string) => lines.find((l) => l.kind === kind && l.in_force)?.sentence ?? nextSentence(walk, kind);
  const slots = readingSlots(ask);
  // The row in focus: the single the walk confirms next, else the first whose column the engine
  // raised a finding about (its evidence is what the canvas shows), else the first.
  const hasEvidence = (cols: string[]) => Object.values(INF.evidence).some((e) => e.columns.some((c) => cols.includes(c)));
  const focusSlot =
    readingFocus ??
    slots.find((r) => r.columns[0] === nextSingle(walk))?.id ??
    slots.find((r) => hasEvidence(r.columns))?.id ??
    slots[0]?.id ??
    null;
  const backLabel = locked ? "Back to the results" : "Go to the open slot";

  function cardFor(it: Item): ReactNode {
    const mode = stated(it);
    switch (it.key) {
      case "roles":
        if (mode === "stated" || id === "draft") return <RolesCard mode={mode} sentence={sentenceOf("set_roles")} onRecord={record} onBack={toObjective} />;
        break;
      case "readings":
        if (it.tier !== "waiting")
          return (
            <ReadingsCard
              ask={ask}
              singles={singlesDone(walk).map((l) => l.sentence)}
              next={nextSingle(walk)}
              focus={focusSlot}
              onFocus={setReadingFocus}
              onSingle={record}
              onBlock={() => (ask?.consumer === "the fit" ? go({ ...walk, codes: true }) : record())}
              stated={lines.filter((l) => KINDS.readings!.includes(l.kind)).map((l) => l.sentence)}
              onBack={toObjective}
            />
          );
        break;
      case "exclusions":
        if (mode === "stated" || id === "exclusions")
          return (
            <ExclusionsCard
              mode={mode}
              sentences={[sentenceOf("set_exclusions"), sentenceOf("set_sensitivity")]}
              onRecord={record}
              onBack={toObjective}
              onPeek={setPeek}
            />
          );
        break;
      case "missing":
        if (mode === "stated" || id === "missing")
          return <MissingCard mode={mode} sentence={sentenceOf("set_missing")} onRecord={record} onBack={toObjective} onPeek={(k) => setPeek(missingPreviewKey(k))} />;
        break;
      case "split":
        if (mode === "stated" || id === "split")
          return <SplitCard mode={mode} sentence={sentenceOf("set_split")} onRecord={record} onBack={toObjective} onPeek={(h) => setPeek(splitPreviewKey(h))} />;
        break;
      case "estimand":
        if (mode === "stated" || id === "estimand")
          return <EstimandCard mode={mode} sentence={sentenceOf("set_estimand")} onRecord={record} onBack={toObjective} onPeek={(c) => setPeek(`exposure:${c}`)} />;
        break;
      case "adjustment":
        if (mode === "stated" || id === "adjustment")
          return <AdjustmentCard mode={mode} adjusted={walk.adjusted} focus={peek} onFocus={setPeek} onAnswer={onAdjust} onBack={toObjective} />;
        break;
      case "energy_adjustment":
        if (mode === "stated" || id === "energy")
          return (
            <EnergyCard
              mode={mode}
              sentence={sentenceOf("set_energy_adjustment")}
              hovered={peek?.startsWith("energy_") ? peek.slice("energy_".length) : null}
              onPeek={(v) => setPeek(`energy_${v}`)}
              onRecord={record}
              onBack={toObjective}
            />
          );
        break;
      case "model_sequence":
        if (mode === "stated" || id === "model_sequence")
          return <SequenceCard mode={mode} sentence={sentenceOf("set_model_sequence")} onRecord={record} onBack={toObjective} />;
        break;
      case "models":
        if (mode === "stated" || id === "models" || id === "model_sequence")
          return (
            <ModelsCard
              mode={mode}
              sentence={sentenceOf("select_models")}
              ready={id === "models" && walk.codes}
              onRecord={record}
              onBack={toObjective}
              onChosen={setFamilies}
            />
          );
        break;
      case "lock":
        if (id === "ready") {
          const declared = ["set_estimand", "set_model_sequence", "set_sensitivity"].map((k) => lines.find((l) => l.kind === k)?.sentence).filter((x): x is string => !!x);
          return <LockCard sentence={nextSentence(walk, "lock_plan")} declared={declared} onRecord={record} />;
        }
        break;
    }
    const isObjective = it.key === objective;
    return (
      <ItemCard
        tier={it.tier}
        title={it.title}
        sentences={it.sentences}
        waitingOn={it.waitingOn}
        ask={it.ask}
        onBack={isObjective ? null : toObjective}
        backLabel={backLabel}
      />
    );
  }

  let center: ReactNode = null;
  const lockSentence = lines.find((l) => l.kind === "lock_plan")?.sentence ?? null;
  if (prediction || doc) {
    center = (
      <MethodsDoc
        guideline={guideline}
        sections={sections}
        lines={lines}
        title={prediction ? "Methods, as drafted: predicting `glucose`" : "Methods, as drafted: `sugar` and `glucose`"}
        locked={locked ? lockSentence : null}
        onPhrase={
          prediction
            ? undefined
            : (kind) => {
                const it = KIND_ITEM[kind];
                if (!it) return;
                setCurrent({ section: sectionKey(it), item: it });
                setDoc(false);
                setPeek(null);
              }
        }
      />
    );
  } else if (sec && item) {
    const position =
      item.key === objective
        ? `objective ${Math.max(1, sec.items.findIndex((i) => i.key === item.key) + 1)} of ${sec.items.length}`
        : stated(item) === "stated"
          ? "a stated phrase: its alternatives play on the canvas"
          : `${item.tier === "waiting" ? "waiting" : "objective"} ${Math.max(1, sec.items.findIndex((i) => i.key === item.key) + 1)} of ${sec.items.length}`;
    const nextAsked = sections
      .flatMap((x) => x.items.filter((i) => i.tier === "asked" && !i.optional).map((i) => ({ sec: x, i })))
      .find((x) => x.i.key !== item.key);
    const then = nextAsked ? (
      <>
        <span className={s.kicker}>Then</span>
        <span>
          <b>{nextAsked.i.title}</b> · {nextAsked.sec.title}
          {nextAsked.i.count > 1 ? ` · ${nextAsked.i.count} open` : ""}
        </span>
      </>
    ) : null;
    center = (
      <Frame section={sec.title} refText={sec.ref} position={position} next={then}>
        {cardFor(item)}
      </Frame>
    );
  } else if (locked) {
    center = (
      <Frame section="Results" refText="STROBE 14–17" position="the plan is locked">
        <LockedCard sentence={lockSentence} onDoc={() => setDoc(true)} />
      </Frame>
    );
  }

  // ── the canvas ──
  const composedNote = (
    <p className={s.composed}>
      Composed by the prototype from your recorded roles; for this question the engine says: “
      {(INF.previews.estimand_sugar_substitution?.body as PreviewResult | undefined)?.note ?? "Nothing about this choice can be shown on your data yet."}”
    </p>
  );
  function previewCanvas(key: string | null, label: string, fallback?: string): ReactNode {
    const p = key ? INF.previews[key] : undefined;
    if (p && !isPreview(p.body))
      return (
        <NoteCanvas pill="Preview" label={label} aside="refused here">
          <p className={s.caption}>{p.body.error.message.replace(/`/g, "")}</p>
        </NoteCanvas>
      );
    if (p && isPreview(p.body)) return <PreviewCanvas result={p.body} pill="Preview" label={label} aside="nothing is recorded" rest={preset.rest} />;
    return fallback && fallback !== key ? previewCanvas(fallback, label) : null;
  }
  const itemByKey = (k: string | undefined) => sections.flatMap((x) => x.items).find((i) => i.key === k);
  function canvasFor(key: string | undefined): ReactNode {
    const it = itemByKey(key);
    const isStated = !!it && stated(it) === "stated";
    switch (key) {
      case "roles":
        return previewCanvas("roles", "Record the roles as proposed");
      case "readings": {
        if (ask?.consumer === "the screens") {
          const exit = ask.exits[0];
          return previewCanvas("unit_kcal_1", exit?.label ?? "`kcal` read in kcal a day");
        }
        const slot = slots.find((r) => r.id === focusSlot);
        const cols = slot ? slot.columns : slots.flatMap((r) => r.columns);
        const ev = Object.values(INF.evidence).find((e) => e.columns.some((c) => cols.includes(c)));
        if (!ev) return <NoteCanvas pill="Evidence" label="Evidence" aside="your data as loaded"><p className={s.caption}>The engine raised no finding about this column; its guess and evidence are on the card.</p></NoteCanvas>;
        return <PreviewCanvas result={ev.evidence} pill="Evidence" label={ev.summary} aside="your data as loaded" promote="distribution" />;
      }
      case "exclusions": {
        const k = peek ?? "exclusions_none";
        const label = k === "exclusions_none" ? "Keep every row" : (INF.cards.exclusions.labels.options.find((o) => `exclusions_${o.key}` === k)?.label ?? k);
        return previewCanvas(k, label, "exclusions_none");
      }
      case "missing": {
        const methods = INF.cards.missing.card.methods as { key: string; label: string }[];
        const k = peek ?? (isStated ? missingPreviewKey(methods.find((m) => m.key === "complete_case")?.key ?? "") : missingPreviewKey(methods[0]?.key ?? ""));
        const m = methods.find((x) => missingPreviewKey(x.key) === k);
        return previewCanvas(k, m?.label ?? "Missing values");
      }
      case "split": {
        const k = peek ?? splitPreviewKey(SCENARIO.holdout ?? 0);
        const o = INF.cards.seal_plan.options.find((x) => splitPreviewKey(x.holdout) === k);
        return previewCanvas(k, o?.label ?? "Held-out rows");
      }
      case "estimand": {
        const exposure = peek?.startsWith("exposure:") ? peek.slice("exposure:".length) : isStated ? (SCENARIO.estimand?.exposure ?? null) : null;
        return (
          <PreviewCanvas
            result={exposureLineage(exposure)}
            pill="Preview"
            label={exposure ? `\`${exposure}\` as the exposure` : "The exposure"}
            aside="nothing is recorded"
            note={composedNote}
            rest={preset.rest}
          />
        );
      }
      case "adjustment": {
        const k = peek ?? "adjust_demographic";
        const g = INF.cards.adjustment.groups.find((x) => `adjust_${x.key}` === k);
        const a = ADJUSTMENT_ANSWERS.find((x) => `adjust_answers_${x.index}` === k);
        const label = g ? `${g.label} as ${g.derived_words ?? "asked"}` : a ? `${a.columns.map((c) => `\`${c}\``).join(", ")} as the scenario answers` : "The adjustment set";
        return previewCanvas(k, label);
      }
      case "energy_adjustment": {
        const v = peek?.startsWith("energy_") ? peek.slice("energy_".length) : (isStated ? SCENARIO.energy : INF.cards.energy.card.ranking.first);
        const label = INF.cards.energy.labels.options.find((o) => o.key === v)?.label ?? v ?? "Energy adjustment";
        return previewCanvas(`energy_${v}`, label);
      }
      case "model_sequence": {
        const p = INF.previews.model_sequence?.body as PreviewResult | undefined;
        return (
          <NoteCanvas pill="Preview" label="The declared model sequence" aside="nothing is recorded">
            <p className={s.caption}>The engine: “{p?.note ?? "Nothing about this choice can be shown on your data yet."}” Each model is fit once, when the plan is locked.</p>
          </NoteCanvas>
        );
      }
      case "models":
        return INF.cards.shelf ? <ShelfCanvas shelf={INF.cards.shelf as unknown as ShelfArtifact} chosen={isStated ? SCENARIO.models : families} /> : null;
      case "lock":
        return (
          <NoteCanvas pill={null} label="The declared plan" aside="nothing is estimated yet">
            <p className={s.caption}>No estimate is computed or shown before the plan is locked; fitting it serves the first one.</p>
          </NoteCanvas>
        );
      default:
        return null;
    }
  }

  let canvas: ReactNode;
  if (prediction) {
    const st = PREDICTION.stages;
    canvas = (
      <ComparisonCanvas
        fit={FX.prediction.fit as unknown as FitArtifact}
        shelf={(st.shelf?.artifact ?? null) as unknown as ShelfArtifact | null}
        split={(st.split?.artifact ?? null) as unknown as SplitArtifact | null}
      />
    );
  } else if (locked && (doc || !item || !peek)) {
    canvas = <ResultsCanvas appendix={appendix} onAppendix={() => setAppendix((a) => !a)} mattered={mattered} onMattered={() => setMattered((m) => !m)} />;
  } else {
    canvas = (!doc && canvasFor(item?.key)) || canvasFor(objective ?? undefined);
  }

  const summary = moment.view.summary;
  return (
    <StageFocusProvider>
      <ColumnsContext.Provider value={known}>
        <div className={s.screen} data-moment={prediction ? "prediction" : id} data-shot={preset.shot || undefined}>
          <Header
            jobs={
              <span className={s.headActions}>
                <button
                  type="button"
                  className={s.btnQuiet}
                  aria-pressed={prediction}
                  onClick={() => {
                    setVariant(prediction ? "inference" : "prediction");
                    setPeek(null);
                  }}
                  data-testid="proto-variant"
                  title="The same opening and readings under prediction, captured to its fit (TRIPOD+AI)"
                >
                  {prediction ? "Back to the inference walk" : "Prediction variant (TRIPOD+AI)"}
                </button>
                <button
                  type="button"
                  className={s.btn}
                  onClick={reset}
                  data-testid="proto-reset"
                  title="Back to the first draft: every answer of this walk is cleared"
                >
                  Reset
                </button>
              </span>
            }
          >
            <span style={{ fontFamily: "var(--serif)", fontSize: 15, fontWeight: 700 }}>{summary.name}</span>
            <span style={{ fontFamily: "var(--mono)", fontSize: 11.5, color: "var(--muted)" }}>
              {summary.n_rows?.toLocaleString("en-US")} rows × {summary.n_cols} columns
            </span>
          </Header>
          <BannerView model={banner(moment)} />
          <div className={s.window} data-doc={doc || prediction || undefined}>
            <Rail
              guideline={guideline}
              sections={sections}
              current={doc || prediction ? null : current}
              objective={objective}
              locked={locked}
              docOpen={doc || prediction}
              onPick={(section, it) => {
                if (prediction) return;
                setCurrent({ section, item: it });
                setDoc(false);
                setPeek(null);
              }}
              onNext={next}
              onDoc={() => !prediction && setDoc((d) => !d)}
            />
            <main className={s.center} aria-label={doc || prediction ? "The methods section" : "The current objective"} data-testid="center">
              {center}
            </main>
            <aside className={s.canvas} aria-label="The canvas">
              {canvas}
            </aside>
          </div>
        </div>
      </ColumnsContext.Provider>
    </StageFocusProvider>
  );
}
