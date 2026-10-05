/**
 * /lab/methods-document — design prototype, angle A (document-first): the Record IS the methods
 * section, typeset as a paper (BLUEPRINT §11.4). Header, the pipeline banner on top, then the
 * working window: the document on the left, the canvas on the right showing the focused phrase or
 * slot.
 *
 * Walked by clicks: from the first draft, each open slot's own control records the shared
 * scenario's answer (methods-shared/SCENARIO.md) and the document becomes the server's next
 * captured moment (walk.ts), to the locked Table 2 and "Which of my decisions mattered?". "Next"
 * walks the open slots; Reset returns to the first draft; the prediction variant is one press.
 *
 * Everything on screen is the real server's (capture/drive.py drove the scenario through the HTTP
 * API; fixture.json holds its answers). The production banner and stage render from those answers
 * through a query cache seeded with every moment, so the canvas is the real canvas, and nothing is
 * ever fetched: the page runs with no server and no mock worker.
 *
 * ?m= opens a moment for review (a moment id, or the review captures' m1 … m9); nothing needs it.
 */
import { useCallback, useEffect, useState } from "react";
import { MutationCache, QueryClient, QueryClientProvider } from "@tanstack/react-query";
import type { RoleProposal, ShelfFamily } from "../../api/m1-types";
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
  PATH,
  stageOf,
  teaching,
  type AdjustmentCard,
  type Derivation,
  type EnergyReading,
  type EstimandCard,
  type Moment,
  type MomentId,
  type ModelSequenceCard,
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
  ADJ_FIELDS,
  AdjustmentSlot,
  ChoiceSlot,
  ConceptCondensed,
  EnergyAlternatives,
  EstimandSlot,
  ExclusionsSlot,
  LabeledSlot,
  LockSlot,
  ModelSequenceSlot,
  PhraseOptions,
  QuestionSlot,
  ReadingsSlot,
  RolesSlot,
  shelfChoices,
  type AdjAnswers,
  type AdjField,
  type EstimandChoice,
} from "./Slots";
import type { Lane } from "./views/AdjustmentLanes";
import { ACTS, captured, nextSingle, sameSet, stepOf } from "./walk";
import s from "./doc.module.css";

// ── review presets (?m=) ─────────────────────────────────────────────────────

interface Preset {
  moment: MomentId;
  focus?: string;
  edit?: string;
  hover?: string;
  concept?: ConceptKey;
  spec?: boolean;
  /** An element to bring into view once, as a navigation press would. */
  scroll?: string;
  /** The estimand chosen in its slot (a review capture of the slot mid-choice). */
  choose?: boolean;
}

const PRESETS: Record<string, Preset> = {
  m1: { moment: "draft" },
  m2: { moment: "readings", focus: "readings" },
  m3: { moment: "estimand", focus: "estimand", choose: true },
  m4: { moment: "adjustment", focus: "adjustment" },
  m5: { moment: "model_sequence", edit: "energy_adjustment", hover: "residual", scroll: "para-energy_adjustment" },
  m6: { moment: "locked", spec: true, scroll: "para-results-table2" },
  m6b: { moment: "locked" },
  m7: { moment: "prediction" },
  m8a: { moment: "estimand", focus: "estimand", scroll: "concept", choose: true },
  m8b: { moment: "model_sequence", concept: "substitution", scroll: "para-energy_adjustment" },
  m9: { moment: "single-cycle_begin_year", focus: "readings" },
};

const ALL: MomentId[] = [...PATH, "prediction"];

/** ?m= in the query, or after a hash route (#/document?m=…, the static build). */
function presetFromUrl(): Preset | null {
  const query = window.location.search || (window.location.hash.split("?")[1] ?? "");
  const m = new URLSearchParams(query).get("m");
  if (!m) return null;
  if (PRESETS[m]) return PRESETS[m];
  return (ALL as string[]).includes(m) ? { moment: m as MomentId } : null;
}

// ── the query cache: every moment's server answers, under the keys production reads ──

const pidOf = (id: MomentId) => `doc-${id}`;

function optionLabels(m: Moment): Map<string, string> {
  const labels = artifactOf<{ labels: QuestionLabels | null }>(m, "proposals")?.labels;
  const out = new Map<string, string>();
  for (const [k, p] of Object.entries(m.previews)) {
    const opt = [...(labels?.energy_adjustment?.options ?? []), ...(labels?.exclusions?.options ?? [])].find(
      (o) => o.key === k,
    );
    out.set(decisionKey(p.decision), opt?.label ?? k);
  }
  return out;
}

/**
 * Put a moment's answers in the cache under its own project id, as the production banner and stage
 * read them: the view, each captured stage, and each captured preview. Only what is missing is put,
 * so a call is idempotent; a preview the stage let go of (its own cache time) is put back before it
 * is asked again, so the stage never asks a server.
 */
function seed(qc: QueryClient, m: Moment): void {
  const pid = pidOf(m.id);
  const put = (key: readonly unknown[], value: unknown) => {
    if (qc.getQueryData(key) === undefined) qc.setQueryData(key, value);
  };
  put(keys.view(pid), m.view);
  for (const name of Object.keys(m.stages)) put(keys.stage(pid, name), stageOf(m, name));
  for (const [name, st] of Object.entries(m.view.stages)) {
    if (m.stages[name] || st.status === "idle") continue;
    // A stage with no captured answer reads as the server answers it: no artifact. One the view
    // reports fresh is an estimate stage before the lock (reading it is what locks the plan), held
    // fresh and empty so the canvas is never what locks it; the lock is the slot's press.
    put(keys.stage(pid, name), { stage: name, key: st.key, fresh: st.status === "fresh", status: st.status, artifact: null });
  }
  const seq = stateSeq(m.view);
  const labels = optionLabels(m);
  for (const p of Object.values(m.previews)) {
    const key = decisionKey(p.decision);
    const label = labels.get(key) ?? p.decision.kind;
    const answer = isPreview(p.body)
      ? { key, seq, label, decision: p.decision, result: p.body, refusal: null }
      : { key, seq, label, decision: p.decision, result: null, refusal: p.body };
    put([pid, "preview", seq, key], answer);
  }
}

/** What a control on the canvas that would record says instead: the walk records in the paper. */
const RECORDS_IN_THE_PAPER =
  "This prototype records in the paper, through each slot's own controls; nothing is sent to a server.";

function seededClient(): QueryClient {
  const qc = new QueryClient({
    // A production control that would record (a substitution pair on the Results, a refusal's way
    // out) never reaches a server: the mutation fails before it runs, and the control says why.
    mutationCache: new MutationCache({
      onMutate: () => {
        throw new Error(RECORDS_IN_THE_PAPER);
      },
    }),
    defaultOptions: {
      queries: {
        staleTime: Infinity,
        gcTime: Infinity,
        retry: false,
        refetchOnWindowFocus: false,
        refetchOnMount: false,
        refetchOnReconnect: false,
      },
    },
  });
  for (const id of ALL) seed(qc, moment(id));
  return qc;
}

// ── the screen ───────────────────────────────────────────────────────────────

export function MethodsDocScreen() {
  const [run, setRun] = useState(0);
  const [preset] = useState(presetFromUrl);
  return <Proto key={run} preset={run === 0 ? preset : null} onReset={() => setRun((r) => r + 1)} />;
}

function Proto({ preset, onReset }: { preset: Preset | null; onReset: () => void }) {
  const [client] = useState(seededClient);
  return (
    <QueryClientProvider client={client}>
      <StageFocusProvider>
        <Walk preset={preset} onReset={onReset} client={client} />
      </StageFocusProvider>
    </QueryClientProvider>
  );
}

const CAPTURED = captured();
/** The sentence the lock records (the server's, captured when the scenario locked the plan). */
const LOCK_SENTENCE = moment("locked").view.decisions.find((d) => d.decision.kind === "lock_plan")?.sentence ?? null;
const NO_CHOICE: EstimandChoice = { exposure: "", effect: "", contrast: "" };
const ANSWER_WORDS: Record<string, string> = { yes: "yes", no: "no", unknown: "don't know" };

function Walk({ preset, onReset, client }: { preset: Preset | null; onReset: () => void; client: QueryClient }) {
  const [id, setId] = useState<MomentId>(preset && preset.moment !== "prediction" ? preset.moment : "draft");
  const [pred, setPred] = useState(preset?.moment === "prediction");
  const m = moment(pred ? "prediction" : id);
  const pid = pidOf(m.id);
  const view = m.view;
  const doc = buildDoc(m);
  const act = pred ? null : (ACTS[id] ?? null);
  // The slot the walk records next comes first; "Next" walks the rest after it.
  const objectives =
    act && doc.objectives.includes(act.slot) ? [act.slot, ...doc.objectives.filter((x) => x !== act.slot)] : doc.objectives;
  const shown = { ...doc, objectives };

  const { focus: stageFocus, setFocus: setStageFocus, reset: resetStage } = useStageFocus();
  const [focus, setFocus] = useState<string | null>(preset?.focus ?? null);
  const [edit, setEdit] = useState<string | null>(preset?.edit ?? null);
  const [hover, setHover] = useState<string | null>(preset?.hover ?? null);
  const [concept, setConcept] = useState<string | null>(preset?.concept ?? null);
  const [specOn, setSpecOn] = useState(!!preset?.spec);
  const [pointerMoved, setPointerMoved] = useState(!!preset?.hover);
  // A moment that opens on a hovered option arrives as a user would: the page first, then the
  // pointer on the option, so the stage meets the preview as a new scene and plays it.
  const [armed, setArmed] = useState(!preset?.hover);
  useEffect(() => {
    if (armed) return;
    const t = window.setTimeout(() => setArmed(true), 450);
    return () => window.clearTimeout(t);
  }, [armed]);

  // What each slot holds before it is recorded (nothing here is recorded; the record is the fixture's).
  const [row, setRow] = useState<string | null>(null);
  const [group, setGroup] = useState<string | null>(null);
  const [choice, setChoice] = useState<EstimandChoice>(preset?.choose ? CAPTURED.estimand : NO_CHOICE);
  const [peek, setPeek] = useState<string | null>(null);
  const [rows, setRows] = useState<string | null>(null);
  const [beside, setBeside] = useState<string[]>([]);
  const [missing, setMissing] = useState<string | null>(null);
  const [split, setSplit] = useState<string | null>(null);
  const [energyPick, setEnergyPick] = useState<string | null>(null);
  const [model1, setModel1] = useState<string[]>([]);
  const [families, setFamilies] = useState<string[]>([]);
  const [confirmed, setConfirmed] = useState<string[]>([]);
  const [answers, setAnswers] = useState<AdjAnswers>({});

  const scrollTo = useCallback((target: string, block: ScrollLogicalPosition = "start") => {
    window.requestAnimationFrame(() => {
      const el = document.getElementById(target) ?? document.querySelector(`[data-testid="${target}"]`);
      el?.scrollIntoView({ block });
    });
  }, []);

  // A moment opened by URL arrives where its review capture looks, once (a navigation press).
  useEffect(() => {
    if (!preset) return;
    const target =
      preset.scroll === "concept" ? "contrast" : (preset.scroll ?? (preset.focus ? `para-${preset.focus}` : null));
    if (target) scrollTo(target);
  }, [preset, scrollTo]);

  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if (e.key !== "Escape") return;
      setFocus(null);
      setEdit(null);
      setConcept(null);
      setHover(null);
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, []);

  // ── the server's cards at this moment ──
  const proposals = artifactOf<{
    labels: QuestionLabels | null;
    energy: EnergyReading | null;
    estimand: EstimandCard | null;
    adjustment: AdjustmentCard | null;
    model_sequence?: ModelSequenceCard | null;
    exclusions?: { key: string; affected: number }[];
  }>(m, "proposals");
  const labels = proposals?.labels ?? null;
  const energy = proposals?.energy ?? null;
  const estimandCard = proposals?.estimand ?? null;
  const adjustmentCard = proposals?.adjustment ?? null;
  const currentEnergy = (view.state.energy_adjustment as { method?: string } | null)?.method ?? "";
  const ask = openAsk(view);
  const roles = artifactOf<{ columns: RoleProposal[] }>(m, "roles")?.columns ?? [];
  const block = unlockedBlock(view);
  const lockSentence = LOCK_SENTENCE;

  // ── the walk ──
  const headOf = (slot: string) =>
    doc.sections.flatMap((x) => x.paras).find((p) => p.id === slot)?.head.toLowerCase() ?? slot;
  const waitFor = (slot: string): string | null => {
    if (pred) return "The prediction variant is shown as fitted; the walk is the inference path (press Prediction variant to go back).";
    if (!act) return "This prototype's captured path ends here, at the locked plan.";
    if (slot === act.slot) return null;
    return `This prototype walks one captured path: the ${headOf(act.slot)} first.`;
  };
  const offPath = (what: string) => `This prototype captured one path: here it records ${what}.`;

  const advance = () => {
    if (!act) return;
    const same = ACTS[act.next]?.slot === act.slot;
    setId(act.next);
    setHover(null);
    setEdit(null);
    setConcept(null);
    resetStage();
    // One reading after another keeps the readings open; any other record shows the paper with its
    // new sentence, and "Next" names the slot that opened.
    if (same) return;
    setFocus(null);
    scrollTo(`para-${act.slot}`, "center");
  };

  const onHover = (key: string) => {
    seed(client, m);
    setHover(key);
    setPointerMoved(true);
  };

  const togglePrediction = () => {
    setPred((v) => !v);
    setFocus(null);
    setEdit(null);
    setHover(null);
    setConcept(null);
    setSpecOn(false);
    resetStage();
  };

  // ── what the canvas shows ──
  let mode: CanvasMode = "live";
  let canvasFocus: StageFocus = stageFocus;
  const hovered = hover && armed && (focus || edit) ? m.previews[hover] : undefined;
  if (hovered) {
    mode = "option";
    canvasFocus = { kind: "option", decision: hovered.decision as Decision, label: optionLabels(m).get(decisionKey(hovered.decision)) ?? hover! };
  } else if (focus === "readings") mode = "evidence";
  else if (focus === "estimand") mode = "exposure";
  else if (focus === "adjustment") mode = "adjustment";
  else if (specOn) mode = "spec";

  const evidence = (() => {
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
  })();

  const concepts = {
    substitution: conceptTeaching(m, "substitution"),
    mediator: conceptTeaching(m, "mediator"),
  };

  // ── the adjustment slot's answers ──
  const deriveOf = (c: string): Derivation | null => {
    const a = answers[c];
    if (!a || ADJ_FIELDS.some((f) => !a[f])) return null;
    return FX.derive[ADJ_FIELDS.map((f) => a[f]).join(",")] ?? null;
  };
  const placed: Record<string, Lane> = {};
  for (const c of Object.keys(answers)) {
    const d = deriveOf(c);
    if (d) placed[c] = d.adjusted ? "confounder" : d.secondary ? "timing_unknown" : "mediator";
  }
  const adjustmentHold = (): string | null => {
    const wait = waitFor("adjustment");
    if (wait || !adjustmentCard) return wait;
    const unconfirmed = adjustmentCard.groups.filter((g) => g.guess && !confirmed.includes(g.key));
    if (unconfirmed.length) return "Confirm each group the pack guessed first.";
    const unguessed = adjustmentCard.groups.filter((g) => !g.guess).flatMap((g) => g.columns);
    if (unguessed.some((c) => !deriveOf(c))) return "Answer the three questions for each covariate without a guess.";
    const off = unguessed.find((c) => ADJ_FIELDS.some((f, i) => answers[c]?.[f] !== CAPTURED.adjustment[c]?.[i]));
    if (off)
      return offPath(
        `\`${off}\` answered ${(CAPTURED.adjustment[off] ?? []).map((a) => ANSWER_WORDS[a] ?? a).join(", ")}`,
      );
    return null;
  };

  // ── each slot's hold: the walk's order first, then the one captured choice ──
  const pickHold = (slot: string, picked: string | null, want: string, wantLabel: string) =>
    waitFor(slot) ?? (!picked ? "Choose one first." : picked !== want ? offPath(`“${wantLabel}”`) : null);
  const labelIn = (opts: { key: string; label: string }[] | undefined, key: string) =>
    opts?.find((o) => o.key === key)?.label ?? key;
  const splitOptions = (teaching("split")?.options ?? []).map((o) => ({ key: o.value, label: o.label, what: o.consequence, tag: null }));
  const familyChoices = shelfChoices(artifactOf<{ families: ShelfFamily[] }>(m, "shelf")?.families ?? []);

  const handlers: DocHandlers = {
    focus,
    edit,
    concept,
    onNext: () => {
      if (!objectives.length) return;
      const next = objectives[(objectives.indexOf(focus ?? "") + 1) % objectives.length]!;
      setEdit(null);
      setHover(null);
      setFocus(next);
      scrollTo(`para-${next}`);
    },
    onExit: () => {
      setFocus(null);
      setEdit(null);
      setHover(null);
    },
    onFocus: (pId) => {
      setEdit(null);
      setHover(null);
      setFocus(pId);
    },
    onPhrase: (pId) => {
      setFocus(null);
      setHover(null);
      setEdit((e) => (e === pId ? null : pId));
    },
    onConcept: (key) => setConcept((c) => (c === key ? null : key)),
    onGo: (target) => scrollTo(target),
    renderSlot: (p: Para) => {
      if (p.slot === "readings" && ask) {
        const single = act?.slot === "readings" ? nextSingle(id) : null;
        return (
          <ReadingsSlot
            ask={ask}
            proposals={roles}
            singles={singleConfirmations(view)}
            block={block}
            focusRow={row}
            onFocusRow={setRow}
            readFromData={m.readings.read_from_data.length}
            next={single}
            onConfirm={(c) => {
              if (c === single) advance();
            }}
            onBlock={() => {
              if (!waitFor("readings") && !single) advance();
            }}
            hold={waitFor("readings")}
          />
        );
      }
      if (p.slot === "estimand" && estimandCard) {
        const c = concepts.substitution;
        const want = CAPTURED.estimand;
        const wantLabel = `the ${(estimandCard.effects.find((e) => e.effect === want.effect)?.label ?? want.effect).toLowerCase()} of \`${want.exposure}\`, ${(estimandCard.contrasts.find((x) => x.contrast === want.contrast)?.label ?? want.contrast).toLowerCase()}`;
        const exposure = estimandCard.exposures.find((e) => e.column === choice.exposure);
        const complete = !!choice.exposure && !!choice.effect && (!exposure?.energy_contrast || !!choice.contrast);
        const hold =
          waitFor("estimand") ??
          (!complete
            ? "Choose the exposure, its effect and, for an energy source, which energy question."
            : choice.exposure !== want.exposure || choice.effect !== want.effect || choice.contrast !== want.contrast
              ? offPath(wantLabel)
              : null);
        return (
          <EstimandSlot
            card={estimandCard}
            entry={teaching("estimand")}
            choice={choice}
            concept={c && !conceptLearned(m, c) ? c : null}
            onChoose={(patch) => setChoice((x) => ({ ...x, ...patch }))}
            onPeek={setPeek}
            onRecord={() => {
              if (!hold) advance();
            }}
            hold={hold}
          />
        );
      }
      if (p.slot === "adjustment" && adjustmentCard) {
        const c = concepts.mediator;
        const hold = adjustmentHold();
        return (
          <AdjustmentSlot
            card={adjustmentCard}
            entry={teaching("adjustment")}
            focusGroup={group}
            onFocusGroup={setGroup}
            concept={c && !conceptLearned(m, c) ? c : null}
            confirmed={confirmed}
            onConfirmGroup={(k) => setConfirmed((x) => (x.includes(k) ? x.filter((y) => y !== k) : [...x, k]))}
            answers={answers}
            onAnswer={(col: string, f: AdjField, v: string) =>
              setAnswers((x) => ({ ...x, [col]: { ...x[col], [f]: v } }))
            }
            derive={deriveOf}
            onRecord={() => {
              if (!hold) advance();
            }}
            hold={hold}
          />
        );
      }
      if (p.slot === "model_sequence" && proposals?.model_sequence) {
        const card = proposals.model_sequence;
        const hold =
          waitFor("model_sequence") ??
          (!model1.length
            ? "Choose Model 1's columns first."
            : !sameSet(model1, CAPTURED.model1)
              ? offPath(`Model 1 adjusted for ${CAPTURED.model1.map((x) => `\`${x}\``).join(", ")}`)
              : null);
        return (
          <ModelSequenceSlot
            card={card}
            picked={model1}
            onToggle={(col) => setModel1((x) => (x.includes(col) ? x.filter((y) => y !== col) : [...x, col]))}
            onGuess={() => setModel1([...card.guess])}
            onRecord={() => {
              if (!hold) advance();
            }}
            hold={hold}
          />
        );
      }
      if (p.slot === "lock") {
        const hold = waitFor("results-table2");
        return (
          <LockSlot
            sentence={lockSentence}
            onLock={() => {
              if (!hold) advance();
            }}
            hold={hold}
          />
        );
      }
      switch (p.question) {
        case "roles": {
          const hold = waitFor("roles");
          return (
            <RolesSlot
              entry={teaching("roles")}
              proposals={roles}
              onRecord={() => {
                if (!hold) advance();
              }}
              hold={hold}
            />
          );
        }
        case "exclusions": {
          const options = labels?.exclusions?.options;
          const wantLabel = `${labelIn(options, CAPTURED.rows)}, with ${CAPTURED.beside.map((k) => `“${labelIn(options, k)}”`).join(" and ")} beside it`;
          const hold =
            waitFor("exclusions") ??
            (!rows
              ? "Choose the primary rows first."
              : rows !== CAPTURED.rows || !sameSet(beside, CAPTURED.beside)
                ? offPath(wantLabel)
                : null);
          const affected = Object.fromEntries((proposals?.exclusions ?? []).map((x) => [x.key, x.affected]));
          return (
            <ExclusionsSlot
              entry={teaching("exclusions")}
              labels={labels?.exclusions ?? null}
              previews={m.previews}
              affected={affected}
              chosen={rows}
              beside={beside}
              hover={hover}
              onHover={onHover}
              onChoose={setRows}
              onBeside={(k) => setBeside((x) => (x.includes(k) ? x.filter((y) => y !== k) : [...x, k]))}
              onRecord={() => {
                if (!hold) advance();
              }}
              hold={hold}
            />
          );
        }
        case "missing": {
          const hold = pickHold("missing", missing, CAPTURED.missing, labelIn(labels?.missing?.options, CAPTURED.missing));
          return (
            <LabeledSlot
              slot="missing"
              entry={teaching("missing")}
              labels={labels?.missing ?? null}
              previews={m.previews}
              ranking={null}
              chosen={missing}
              hover={hover}
              onHover={onHover}
              onChoose={setMissing}
              onRecord={() => {
                if (!hold) advance();
              }}
              hold={hold}
            />
          );
        }
        case "energy_adjustment": {
          const hold = pickHold(
            "energy_adjustment",
            energyPick,
            CAPTURED.energy,
            labelIn(labels?.energy_adjustment?.options, CAPTURED.energy),
          );
          return (
            <LabeledSlot
              slot="energy_adjustment"
              entry={teaching("energy_adjustment")}
              labels={labels?.energy_adjustment ?? null}
              previews={m.previews}
              ranking={energy?.ranking ?? null}
              chosen={energyPick}
              hover={hover}
              onHover={onHover}
              onChoose={setEnergyPick}
              onRecord={() => {
                if (!hold) advance();
              }}
              hold={hold}
            />
          );
        }
        case "split": {
          const hold = pickHold("split", split, CAPTURED.split, labelIn(splitOptions, CAPTURED.split));
          const pick = splitOptions.find((o) => o.key === split);
          return (
            <ChoiceSlot
              slot="split"
              entry={teaching("split")}
              options={splitOptions}
              chosen={split ? [split] : []}
              multi={false}
              onChoose={setSplit}
              onRecord={() => {
                if (!hold) advance();
              }}
              hold={hold}
              recordLabel={pick ? `Record: ${pick.label}` : "Choose one to record"}
            />
          );
        }
        case "models": {
          const wantLabel = CAPTURED.models.map((k) => `“${labelIn(familyChoices, k)}”`).join(" and ");
          const hold =
            waitFor("models") ??
            (!families.length ? "Choose at least one family." : !sameSet(families, CAPTURED.models) ? offPath(wantLabel) : null);
          return (
            <ChoiceSlot
              slot="models"
              entry={teaching("models")}
              options={familyChoices}
              chosen={families}
              multi
              onChoose={(k) => setFamilies((x) => (x.includes(k) ? x.filter((y) => y !== k) : [...x, k]))}
              onRecord={() => {
                if (!hold) advance();
              }}
              hold={hold}
              recordLabel={
                families.length
                  ? `Fit ${families.map((k) => labelIn(familyChoices, k).toLowerCase()).join(" and ")}`
                  : "Choose the families to fit"
              }
            />
          );
        }
        default:
          return <QuestionSlot entry={p.question ? teaching(p.question) : null} hold={waitFor(p.id)} />;
      }
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
          onHover={onHover}
          keys={pointerMoved}
          onKeep={() => {
            setEdit(null);
            setHover(null);
          }}
        />
      ) : (
        <PhraseOptions entry={p.question ? teaching(p.question) : null} onKeep={() => setEdit(null)} />
      ),
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
              setHover(null);
              setSpecOn((v) => !v);
              resetStage();
            }}
          />
        );
      return null;
    },
    slotLabel: (p: Para) => {
      if (p.slot === "readings" && ask) {
        const n = ask.groups.reduce((k, g) => k + g.columns.length, 0);
        return `${n} ${n === 1 ? "reading" : "readings"} to confirm`;
      }
      if (p.slot === "estimand")
        return focus === "estimand" && choice.exposure
          ? [choice.exposure, choice.effect && `${choice.effect} effect`, choice.contrast].filter(Boolean).join(" · ")
          : "choose the exposure and its effect";
      if (p.slot === "adjustment") {
        const n = adjustmentCard?.groups.reduce((k, g) => k + g.columns.length, 0) ?? 0;
        return `${n} covariates to place`;
      }
      if (p.slot === "model_sequence") return "declare Model 1's columns";
      if (p.slot === "lock") return "show the estimates (this locks the plan)";
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
  const step = stepOf(id);

  return (
    <div className={s.screen}>
      <Header
        jobs={
          <>
            <span className={s.walkStep} data-testid="proto-step">
              {pred ? "prediction variant, fitted" : `moment ${step.at} of ${step.of}`}
            </span>
            <button
              type="button"
              className={s.headBtn}
              aria-pressed={pred}
              onClick={togglePrediction}
              data-testid="proto-prediction"
              title="The same table under prediction (TRIPOD+AI), as fitted"
            >
              Prediction variant
            </button>
            <button type="button" className={s.headBtn} onClick={onReset} data-testid="proto-reset">
              Reset to the first draft
            </button>
          </>
        }
      >
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
          <DocumentPane doc={shown} h={handlers} title="Methods" />
        </main>
        <aside className={s.canvasCol} aria-label="The canvas">
          {/* One canvas per moment: a preview still settling from the last moment's slot is never
              asked again under this one's answers (it would be a request no server answers). */}
          <Canvas
            key={pid}
            mode={mode}
            pid={pid}
            m={m}
            focus={canvasFocus}
            onFocus={setStageFocus}
            evidence={evidence}
            exposure={
              estimandCard && (peek ?? choice.exposure)
                ? { column: peek ?? choice.exposure, contrast: choice.contrast, effect: choice.effect }
                : null
            }
            adjustmentGroup={group}
            placed={placed}
            confirmed={confirmed}
          />
        </aside>
      </div>
    </div>
  );
}
