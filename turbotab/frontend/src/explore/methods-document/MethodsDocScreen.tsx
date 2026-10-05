/**
 * /lab/methods-document — design prototype, angle A (document-first): the Record IS the methods
 * section, typeset as a paper (BLUEPRINT §11.4). Header, the pipeline banner on top, then the
 * working window: the document on the left, the canvas on the right showing the focused phrase or
 * slot. Built for review from screenshots; the winner is built for real.
 *
 * Everything on screen is the real server's (capture/drive.py drove the NHANES reference journey
 * through the HTTP API; fixture.json holds its answers). The production banner and stage render
 * from those answers through a query cache seeded per moment, so the canvas is the real canvas.
 *
 * ?m= picks a moment: m1 the draft after the outcome and purpose · m2 the Data section's readings
 * · m3 the exposure and estimand · m4 the adjustment set · m5 a stated phrase opened (the energy
 * model), the canvas playing the hovered option · m6 after the lock (Results, Table 2, the
 * specification curve) · m6b the complete methods · m7 prediction (TRIPOD+AI) · m8a a concept's
 * first encounter · m8b the same concept met again, condensed · m9 the block confirm unlocked.
 */
import { useCallback, useEffect, useMemo, useState } from "react";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import type { RoleProposal } from "../../api/m1-types";
import { keys } from "../../api/queries";
import type { Decision } from "../../api/schema";
import { BannerView } from "../../components/banner/Banner";
import { deriveBanner } from "../../components/banner/derive";
import { Header } from "../../components/Header";
import { decisionKey, stateSeq } from "../../components/stage/data";
import { StageFocusProvider, useStageFocus, type StageFocus } from "../../state/focus";
import { Canvas, type CanvasMode } from "./Canvas";
import { DocumentPane, type DocHandlers } from "./Document";
import {
  artifactOf,
  FX,
  isPreview,
  moment,
  stageOf,
  teaching,
  type AdjustmentCard,
  type EnergyReading,
  type EstimandCard,
  type Moment,
  type MomentId,
  type QuestionLabels,
} from "./fixture";
import {
  buildDoc,
  conceptLearned,
  conceptTeaching,
  openAsk,
  readingLines,
  singleConfirmations,
  unlockedBlock,
  type ConceptKey,
  type Para,
} from "./model";
import { Flow, OtherAnalyses, Performance, TableTwo } from "./Results";
import {
  AdjustmentSlot,
  ConceptCondensed,
  EnergyAlternatives,
  EstimandSlot,
  QuestionSlot,
  ReadingsSlot,
  type EstimandChoice,
} from "./Slots";
import s from "./doc.module.css";

interface Spec {
  fixture: MomentId;
  focus?: string;
  edit?: string;
  hover?: string;
  concept?: ConceptKey;
  spec?: boolean;
  /** An element to bring into view once, as a navigation press would. */
  scroll?: string;
  row?: string;
  group?: string;
}

const SPECS: Record<string, Spec> = {
  m1: { fixture: "m1" },
  m2: { fixture: "m2", focus: "readings" },
  m3: { fixture: "m3", focus: "estimand" },
  m4: { fixture: "m4", focus: "adjustment" },
  m5: { fixture: "m5", edit: "energy_adjustment", hover: "residual", scroll: "para-energy_adjustment" },
  m6: { fixture: "m6", spec: true, scroll: "para-results-table2" },
  m6b: { fixture: "m6" },
  m7: { fixture: "m7" },
  m8a: { fixture: "m3", focus: "estimand", scroll: "concept" },
  m8b: { fixture: "m5", concept: "substitution", scroll: "para-energy_adjustment" },
  m9: { fixture: "m9", focus: "readings" },
};

function param(name: string): string | null {
  return new URLSearchParams(window.location.search).get(name);
}

/** A query cache holding the moment's server answers under the keys the production banner and
 *  stage read, so they render the real artifacts and previews with no server. */
function seededClient(pid: string, m: Moment, labels: Map<string, string>): QueryClient {
  const qc = new QueryClient({
    defaultOptions: {
      queries: { staleTime: Infinity, gcTime: Infinity, retry: false, refetchOnWindowFocus: false, refetchOnMount: false },
    },
  });
  qc.setQueryData(keys.view(pid), m.view);
  for (const name of Object.keys(m.stages)) qc.setQueryData(keys.stage(pid, name), stageOf(m, name));
  // A stage waiting on an answer has no artifact: the server answers its read with none.
  for (const [name, st] of Object.entries(m.view.stages)) {
    if (m.stages[name] || st.status === "idle") continue;
    qc.setQueryData(keys.stage(pid, name), { stage: name, key: st.key, fresh: false, status: st.status, artifact: null });
  }
  const seq = stateSeq(m.view);
  for (const p of Object.values(m.previews)) {
    const key = decisionKey(p.decision);
    const label = labels.get(key) ?? p.decision.kind;
    const answer = isPreview(p.body)
      ? { key, seq, label, decision: p.decision, result: p.body, refusal: null }
      : { key, seq, label, decision: p.decision, result: null, refusal: p.body };
    qc.setQueryData([pid, "preview", seq, key], answer);
  }
  return qc;
}

export function MethodsDocScreen() {
  const id = param("m") ?? "m1";
  const spec = SPECS[id] ?? SPECS.m1!;
  return <Proto key={id} spec={spec} />;
}

function previewLabels(m: Moment): Map<string, string> {
  const labels = artifactOf<{ labels: QuestionLabels | null }>(m, "proposals")?.labels;
  const out = new Map<string, string>();
  for (const [k, p] of Object.entries(m.previews)) {
    const opt = labels?.energy_adjustment?.options.find((o) => o.key === k);
    out.set(decisionKey(p.decision), opt?.label ?? k);
  }
  return out;
}

function Proto({ spec }: { spec: Spec }) {
  const m = moment(spec.fixture);
  const pid = `doc-${spec.fixture}`;
  const [client] = useState(() => seededClient(pid, m, previewLabels(m)));
  return (
    <QueryClientProvider client={client}>
      <StageFocusProvider>
        <Loaded spec={spec} m={m} pid={pid} />
      </StageFocusProvider>
    </QueryClientProvider>
  );
}

function Loaded({ spec, m, pid }: { spec: Spec; m: Moment; pid: string }) {
  const view = m.view;
  const doc = useMemo(() => buildDoc(m), [m]);
  const { focus: stageFocus, setFocus: setStageFocus, reset } = useStageFocus();
  const [focus, setFocus] = useState<string | null>(spec.focus ?? null);
  const [edit, setEdit] = useState<string | null>(spec.edit ?? null);
  const [hover, setHover] = useState<string | null>(spec.hover ?? null);
  const [concept, setConcept] = useState<string | null>(spec.concept ?? null);
  const [specOn, setSpecOn] = useState(!!spec.spec);
  const [pointerMoved, setPointerMoved] = useState(!!spec.hover);
  // A moment that opens on a hovered option arrives as a user would: the page first, then the
  // pointer on the option, so the stage meets the preview as a new scene and plays it.
  const [armed, setArmed] = useState(!spec.hover);
  useEffect(() => {
    if (armed) return;
    const id = window.setTimeout(() => setArmed(true), 450);
    return () => window.clearTimeout(id);
  }, [armed]);
  const ask = openAsk(view);
  const roles = useMemo(() => artifactOf<{ columns: RoleProposal[] }>(m, "roles")?.columns ?? [], [m]);
  const block = unlockedBlock(view);
  const family = ask ? (readingLines(ask, roles).find((l) => l.family) ?? null) : null;
  const [row, setRow] = useState<string | null>(block ? null : (family?.key ?? null));
  const [group, setGroup] = useState<string | null>(spec.group ?? null);
  const estimandCard = artifactOf<{ estimand: EstimandCard | null }>(m, "proposals")?.estimand ?? null;
  const [choice, setChoice] = useState<EstimandChoice>({ exposure: "sugar", effect: "total", contrast: "substitution" });

  const scrollTo = useCallback((target: string) => {
    window.requestAnimationFrame(() => {
      const el = document.getElementById(target) ?? document.querySelector(`[data-testid="${target}"]`);
      el?.scrollIntoView({ block: "start" });
    });
  }, []);

  // A moment opened by URL arrives where its review capture looks, once (a navigation press).
  useEffect(() => {
    const target = spec.scroll === "concept" ? "contrast" : (spec.scroll ?? (spec.focus ? `para-${spec.focus}` : null));
    if (target) scrollTo(target);
  }, [spec, scrollTo]);

  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if (e.key !== "Escape") return;
      setFocus(null);
      setEdit(null);
      setConcept(null);
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, []);

  const labels = artifactOf<{ labels: QuestionLabels | null }>(m, "proposals")?.labels ?? null;
  const energy = artifactOf<{ energy: EnergyReading | null }>(m, "proposals")?.energy ?? null;
  const currentEnergy = (view.state.energy_adjustment as { method?: string } | null)?.method ?? "";

  // ── what the canvas shows ──
  let mode: CanvasMode = "live";
  let canvasFocus: StageFocus = stageFocus;
  if (edit === "energy_adjustment" && hover && armed && m.previews[hover]) {
    mode = "option";
    const p = m.previews[hover]!;
    const label = labels?.energy_adjustment?.options.find((o) => o.key === hover)?.label ?? hover;
    canvasFocus = { kind: "option", decision: p.decision as Decision, label };
  } else if (focus === "readings") mode = "evidence";
  else if (focus === "estimand") mode = "exposure";
  else if (focus === "adjustment") mode = "adjustment";
  else if (specOn) mode = "spec";

  const evidence = useMemo(() => {
    if (!ask) return null;
    if (block && row === null) {
      const items = (block.decision as { items: { column: string; value: string }[] }).items;
      return { label: `What the block settles: ${items.length} readings`, title: `Each with the value it takes; ${ask.consumer} reads them.`, items };
    }
    const lines = readingLines(ask, roles);
    const g = lines.find((x) => x.key === row) ?? lines.find((x) => x.family) ?? lines[0];
    if (!g) return null;
    const items = g.columns.map((c) => ({ column: c, value: g.guess ?? "" }));
    const label = g.columns.length > 1 ? `\`${g.columns[0]}\` and ${g.columns.length - 1} more` : `\`${g.columns[0]}\``;
    return { label, title: `${g.columns.length > 1 ? `${g.columns.length} columns` : `\`${g.columns[0]}\``}: ${g.guessWords}?`, items };
  }, [ask, block, row, roles]);

  const concepts = {
    substitution: conceptTeaching(m, "substitution"),
    mediator: conceptTeaching(m, "mediator"),
  };

  const handlers: DocHandlers = {
    focus,
    edit,
    concept,
    onNext: () => {
      const order = doc.objectives;
      if (!order.length) return;
      const next = order[(order.indexOf(focus ?? "") + 1) % order.length]!;
      setEdit(null);
      setFocus(next);
      scrollTo(`para-${next}`);
    },
    onExit: () => {
      setFocus(null);
      setEdit(null);
    },
    onFocus: (id) => {
      setEdit(null);
      setFocus(id);
    },
    onPhrase: (id) => {
      setFocus(null);
      setEdit((e) => (e === id ? null : id));
    },
    onConcept: (key) => setConcept((c) => (c === key ? null : key)),
    onGo: (id) => scrollTo(id),
    renderSlot: (p: Para) => {
      if (p.slot === "readings" && ask)
        return (
          <ReadingsSlot
            ask={ask}
            proposals={roles}
            singles={singleConfirmations(view)}
            block={block}
            focusRow={row}
            onFocusRow={setRow}
            readFromData={m.readings.read_from_data.length}
          />
        );
      if (p.slot === "estimand" && estimandCard) {
        const c = concepts.substitution;
        return (
          <EstimandSlot
            card={estimandCard}
            entry={teaching("estimand")}
            choice={choice}
            concept={c && !conceptLearned(m, c) ? c : null}
            onExposure={(col) => setChoice((x) => ({ ...x, exposure: col }))}
          />
        );
      }
      if (p.slot === "adjustment") {
        const card = artifactOf<{ adjustment: AdjustmentCard | null }>(m, "proposals")?.adjustment;
        const c = concepts.mediator;
        if (card)
          return (
            <AdjustmentSlot
              card={card}
              entry={teaching("adjustment")}
              focusGroup={group}
              onFocusGroup={setGroup}
              concept={c && !conceptLearned(m, c) ? c : null}
            />
          );
      }
      return <QuestionSlot entry={p.question ? teaching(p.question) : null} />;
    },
    renderEdit: (p: Para) =>
      p.question === "energy_adjustment" ? (
        <EnergyAlternatives
          labels={labels?.energy_adjustment ?? null}
          energy={energy}
          entry={teaching("energy_adjustment")}
          previews={m.previews}
          current={currentEnergy}
          hover={hover}
          onHover={(k) => {
            setHover(k);
            setPointerMoved(true);
          }}
          keys={pointerMoved}
        />
      ) : null,
    renderResults: (p: Para) => {
      if (p.id === "results-flow") return <Flow m={m} />;
      if (p.id === "results-table2") return <TableTwo m={m} />;
      if (p.id === "results-performance") return <Performance m={m} />;
      if (p.id === "results-other")
        return (
          <OtherAnalyses
            m={m}
            specOn={specOn}
            onSpec={() => {
              setFocus(null);
              setEdit(null);
              setSpecOn((v) => !v);
              reset();
            }}
          />
        );
      return null;
    },
    slotLabel: (p: Para) => {
      if (p.slot === "readings" && ask) {
        const n = ask.groups.reduce((k, g) => k + g.columns.length, 0);
        return `${n} readings to confirm`;
      }
      if (p.slot === "estimand")
        return focus === "estimand" ? `${choice.exposure} · ${choice.effect} effect · ${choice.contrast}` : "choose the exposure and its effect";
      if (p.slot === "adjustment") {
        const card = artifactOf<{ adjustment: AdjustmentCard | null }>(m, "proposals")?.adjustment;
        const n = card?.groups.reduce((k, g) => k + g.columns.length, 0) ?? 0;
        return `${n} covariates to place`;
      }
      return p.question ? (teaching(p.question)?.title ?? p.head) : p.head;
    },
    renderConcept: (_p: Para, key: string) => {
      const c = concepts[key as ConceptKey];
      return c && conceptLearned(m, c) ? <ConceptCondensed concept={c} open /> : null;
    },
  };

  const banner = deriveBanner({
    view,
    ingest: stageOf(m, "ingest") as never,
    oriented: stageOf(m, "oriented") as never,
    working: stageOf(m, "working") as never,
    cohort: stageOf(m, "cohort") as never,
    split: stageOf(m, "split") as never,
    design: stageOf(m, "design") as never,
    fit: stageOf(m, "fit") as never,
    shelf: stageOf(m, "shelf") as never,
  });

  return (
    <div className={s.screen}>
      <Header>
        <span className={s.name}>{view.summary.name}</span>
        <span className={s.size}>
          {view.summary.n_rows?.toLocaleString("en-US")} rows × {view.summary.n_cols} columns
        </span>
        <span className={s.protoTag} title={FX.captured}>
          prototype · real server answers
        </span>
      </Header>
      <BannerView model={banner} />
      <div className={s.window}>
        <main className={s.docCol} aria-label="The record: the methods section">
          <DocumentPane doc={doc} h={handlers} title="Methods" />
        </main>
        <aside className={s.canvasCol} aria-label="The canvas">
          <Canvas
            mode={mode}
            pid={pid}
            m={m}
            focus={canvasFocus}
            onFocus={setStageFocus}
            evidence={evidence}
            exposure={estimandCard ? { column: choice.exposure, contrast: choice.contrast, effect: choice.effect } : null}
            adjustmentGroup={group}
          />
        </aside>
      </div>
    </div>
  );
}
