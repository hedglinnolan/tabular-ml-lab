/**
 * /lab/methods-map — design prototype C, "pipeline map first" (dev only).
 *
 * The analysis is a living map from the raw table to the estimate; every decision is a node on it,
 * and the methods text reads alongside as the derived record. Click a node to open its slot or
 * phrase; hover an option to play its consequence on the canvas and on the map; record it and the
 * engine's sentence lands in the record. A newcomer walks the asked nodes with "Next asked"; an
 * expert clicks anywhere.
 *
 * Every sentence and number is the real engine's, captured from the server on the NHANES export
 * (capture.py). State is client-side and kept in this browser; Reset clears it. URL parameters pick
 * a state for review captures: ?demo=draft|unlock|exposure|adjust|energy|locked|matter|after|prediction.
 */
import { useCallback, useEffect, useMemo, useReducer, useRef, useState, type ReactNode } from "react";
import { Header } from "../../components/Header";
import { Rich } from "../../components/stage/text";
import { Canvas, type Scene } from "./Canvas";
import { NodeCard, sceneOf, type Ctx } from "./Cards";
import { INF, PRED } from "./fixture";
import { MapView, silentByRegion } from "./MapView";
import { Methods } from "./Methods";
import {
  ASKED,
  INITIAL,
  NODE_TERMS,
  NODE_TITLE,
  REGIONS,
  fitFor,
  guessTriple,
  objectives,
  previewFor,
  record as recordOf,
  reduce,
  truthTriple,
  type Action,
  type Answers,
  type NodeId,
  type Purpose,
  type Triple,
} from "./model";
import { Mattered, Table2 } from "./Results";
import c from "./screen.module.css";

const STORE = "turbotab.methods-map.v1";

function param(name: string): string | null {
  return new URLSearchParams(window.location.search).get(name);
}

function load(): { a: Answers; purpose: Purpose } | null {
  try {
    const raw = localStorage.getItem(STORE);
    if (!raw) return null;
    const v = JSON.parse(raw) as { a: Answers; purpose: Purpose };
    return v.a && v.purpose ? { a: { ...INITIAL, ...v.a }, purpose: v.purpose } : null;
  } catch {
    return null;
  }
}

// ── review presets: each is a real path through the reducer ─────────────────

const ROLE_KEYS = INF.readings.items.map((i) => i.key);

function play(actions: Action[]): Answers {
  return actions.reduce(reduce, INITIAL);
}

const READ3: Action[] = ROLE_KEYS.slice(0, 3).map((key) => ({ type: "reading", key }));
const READ_ALL: Action[] = [...READ3, { type: "block" }, { type: "unit" }];
/** The fixture's answers for a group: the pack's guess where it has one, else the declared truth. */
const TRUTH = (g: string): Record<string, Triple> => {
  const grp = INF.adjustment.groups.find((x) => x.key === g)!;
  return Object.fromEntries(grp.columns.map((col) => [col, guessTriple(g) ?? truthTriple(col)!]));
};
const ADJ_ALL: Action[] = INF.adjustment.groups.map((g) => ({ type: "adjust", group: g.key, answers: TRUTH(g.key) }));
const ALL: Action[] = [
  ...READ_ALL,
  { type: "exclusions", key: "none" },
  { type: "sensitivity", key: "willett_2013_by_sex" },
  { type: "sensitivity", key: "nhs_hpfs_by_sex" },
  { type: "sensitivity", key: "sex_neutral_500_5000" },
  { type: "exposure" },
  ...ADJ_ALL,
  { type: "model1", value: "guess" },
];

const WALKED: NodeId[] = ["readings", "exclusions", "exposure", "adjustment", "model1"];

const PRESETS: Record<
  string,
  { a: Answers; focus: NodeId | null; purpose?: Purpose; walking?: boolean; visited?: NodeId[] }
> = {
  draft: { a: INITIAL, focus: null },
  readings: { a: INITIAL, focus: "readings", walking: true },
  unlock: { a: play(READ3), focus: "readings" },
  exposure: { a: play([...READ_ALL, { type: "exclusions", key: "none" }]), focus: "exposure", walking: true, visited: ["readings", "exclusions"] },
  adjust: {
    a: play([...READ_ALL, { type: "exclusions", key: "none" }, { type: "exposure" }, ...ADJ_ALL.slice(0, 3)]),
    focus: "adjustment",
    visited: ["readings", "exclusions", "exposure"],
  },
  energy: { a: play(ALL), focus: "energy", visited: WALKED },
  locked: { a: play([...ALL, { type: "lock" }]), focus: "estimate", visited: WALKED },
  matter: { a: play([...ALL, { type: "lock" }]), focus: "matter", visited: WALKED },
  after: { a: play([...ALL, { type: "lock" }, { type: "energy", method: "residual_energy_dropped" }]), focus: "energy", visited: WALKED },
  prediction: { a: play(READ_ALL), focus: "p_seal", purpose: "prediction", visited: ["readings"] },
};

export function MethodsMapScreen() {
  const demo = param("demo");
  const preset = demo ? PRESETS[demo] : undefined;
  const saved = preset ? null : load();
  const [purpose, setPurpose] = useState<Purpose>(preset?.purpose ?? saved?.purpose ?? "inference");
  const [a, dispatch] = useReducer(reduce, preset?.a ?? saved?.a ?? INITIAL);
  const [focus, setFocusRaw] = useState<NodeId | null>(preset?.focus ?? null);
  const [walking, setWalking] = useState<NodeId | null>(preset?.walking ? (preset.focus ?? null) : null);
  const [opened, setOpened] = useState<Record<string, number>>(preset?.focus ? { [preset.focus]: 1 } : {});
  const [firstMet, setFirstMet] = useState<Record<string, NodeId>>(() =>
    preset ? [...(preset.visited ?? []), ...(preset.focus ? [preset.focus] : [])].reduce(meet, {}) : {},
  );
  const [shown, setShown] = useState<{ node: NodeId; key: string | null; scene: Scene | null; preview: Answers | null } | null>(null);
  const [receipt, setReceipt] = useState<{ node: NodeId; texts: string[] } | null>(null);
  const [pointer, setPointer] = useState(0);
  const [silent, setSilent] = useState<string | null>(null);

  useEffect(() => {
    if (preset) return;
    try {
      localStorage.setItem(STORE, JSON.stringify({ a, purpose }));
    } catch {
      /* storage unavailable: the state lasts for this page only */
    }
  }, [a, purpose, preset]);

  const cardPane = useRef<HTMLDivElement>(null);
  // Opening a node is a press whose purpose is to go there: its card starts at its top.
  useEffect(() => {
    if (cardPane.current) cardPane.current.scrollTop = 0;
  }, [focus]);
  const setFocus = useCallback((n: NodeId | null) => {
    setFocusRaw(n);
    setShown(null);
    setSilent(null);
    if (!n) return;
    setOpened((o) => ({ ...o, [n]: (o[n] ?? 0) + 1 }));
    // A concept is first met at the first card that uses it; set once, never moved.
    setFirstMet((m) => meet(m, n));
  }, []);

  const objs = objectives(a, purpose);
  const nextAsked = useMemo(() => {
    const order = ASKED[purpose];
    const from = focus ? order.indexOf(focus) : -1;
    const open = (n: NodeId) => !objs.find((o) => o.node === n)?.done;
    return order.slice(from + 1).find(open) ?? order.find(open) ?? null;
  }, [purpose, focus, objs]);
  const lockNext: NodeId | null = purpose === "inference" && !nextAsked && !a.locked ? "lock" : null;

  const goNext = useCallback(() => {
    const n = nextAsked ?? lockNext;
    if (!n) return;
    setFocus(n);
    setWalking(n);
  }, [nextAsked, lockNext, setFocus]);

  const onFocus = useCallback(
    (n: NodeId) => {
      setFocus(n);
      setWalking((w) => (w && w !== n ? null : w));
    },
    [setFocus],
  );

  const doRecord = useCallback(
    (e: Action, node: NodeId) => {
      const before = recordOf(a, purpose).flatMap((s) => s.lines.map((l) => l.text));
      const next = reduce(a, e);
      dispatch(e);
      // The receipt quotes what this answer recorded: the new lines of its own node.
      const after = recordOf(next, purpose).flatMap((s) =>
        s.lines.filter((l) => l.text && l.node === node).map((l) => l.text!),
      );
      const fresh = after.filter((t) => !before.includes(t));
      if (e.type === "undo") setReceipt(null);
      else if (fresh.length) setReceipt({ node, texts: fresh });
      setShown(null);
    },
    [a, purpose],
  );

  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if (e.key === "Escape") {
        setWalking(null);
        setShown(null);
      }
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, []);

  const reset = () => {
    dispatch({ type: "reset" });
    setFocus(null);
    setWalking(null);
    setReceipt(null);
    setFirstMet({});
    setOpened({});
    setPointer(0);
    try {
      localStorage.removeItem(STORE);
    } catch {
      /* nothing kept */
    }
    if (demo) window.history.replaceState(null, "", window.location.pathname);
  };

  const switchPurpose = (p: Purpose) => {
    setPurpose(p);
    setFocus(null);
    setWalking(null);
    setReceipt(null);
  };

  // ── the canvas: the hovered option, else what the focused node shows by default ──
  const scene: Scene | null = useMemo(() => {
    if (shown && shown.node === focus && shown.scene) return shown.scene;
    if (!focus) return null;
    if (purpose === "inference" && (focus === "estimate" || focus === "matter" || (focus === "lock" && a.locked))) {
      if (!a.locked) return null;
      const pick = fitFor(a);
      const label = focus === "matter" ? "Which of my decisions mattered?" : "Table 2";
      if (pick.kind !== "fit") {
        return {
          kind: "panel",
          group: `results-${focus}`,
          key: "none",
          label,
          body: (
            <div className={c.results}>
              {pick.kind === "error" ? (
                <p className={c.refusalLike}>
                  <Rich text={`The engine could not fit this plan: ${pick.message}`} />
                </p>
              ) : (
                <p className={c.empty}>
                  This prototype captured the fits for the fixture's adjustment answers, with every row kept or Willett 2013's
                  screen; {pick.reason} selects a fit it did not take.
                </p>
              )}
            </div>
          ),
        };
      }
      return {
        kind: "panel",
        group: `results-${focus}`,
        key: pick.key,
        label,
        aside: a.after.length ? "changed after the estimates were seen" : "as locked",
        body: focus === "matter" ? <Mattered fit={pick.fit} a={a} /> : <Table2 fit={pick.fit} a={a} />,
      };
    }
    return defaultScene(focus, a, purpose);
  }, [shown, focus, a, purpose]);

  const ctx: Ctx | null = focus
    ? {
        purpose,
        a,
        record: doRecord,
        show: (key, sc, preview) => setShown({ node: focus, key, scene: sc, preview: preview ?? null }),
        shown: shown?.node === focus ? shown.key : null,
        first: (opened[focus] ?? 0) <= 1,
        firstMet,
        receipt,
        onPointerRecord: () => setPointer((p) => p + 1),
        keysUnlocked: pointer >= 2,
        next: nextAsked ?? lockNext,
        goNext,
        focus: onFocus,
      }
    : null;

  const regionsSilent = silentByRegion(a, purpose);
  const doneCount = objs.filter((o) => o.done).length;
  const freshText = receipt?.texts[receipt.texts.length - 1] ?? null;

  return (
    <div className={c.page} data-purpose={purpose}>
      <Header
        jobs={
          <div className={c.headerRight}>
            <span className={c.progress} aria-label={`${doneCount} of ${objs.length} asked decisions answered`}>
              {objs.map((o) => (
                <button
                  key={o.node}
                  type="button"
                  className={c.pip}
                  data-done={o.done || undefined}
                  data-at={focus === o.node || undefined}
                  onClick={() => onFocus(o.node)}
                  title={`${NODE_TITLE[o.node]}: ${o.done ? "answered" : "asked"}`}
                  aria-label={`${NODE_TITLE[o.node]}: ${o.done ? "answered" : "asked"}`}
                />
              ))}
              <span className={c.progressText}>
                {doneCount} of {objs.length} asked
              </span>
            </span>
            <button type="button" className={c.next} onClick={goNext} disabled={!nextAsked && !lockNext} data-testid="next">
              {nextAsked ? `Next asked: ${NODE_TITLE[nextAsked]} →` : lockNext ? "Next: lock the plan →" : a.locked ? "The plan is locked" : "All asked are answered"}
            </button>
            {walking ? (
              <button type="button" className={c.ghost} onClick={() => setWalking(null)} data-testid="show-all">
                Whole map
              </button>
            ) : null}
            <button type="button" className={c.ghost} onClick={reset} data-testid="reset">
              Reset
            </button>
          </div>
        }
      >
        <span className={c.proto}>Prototype C · the map</span>
        <div className={c.purpose} role="radiogroup" aria-label="Purpose">
          {(["inference", "prediction"] as Purpose[]).map((p) => (
            <button key={p} type="button" role="radio" aria-checked={purpose === p} className={c.purposeOpt} onClick={() => switchPurpose(p)} data-testid={`purpose-${p}`}>
              {p === "inference" ? "Inference · STROBE-nut" : "Prediction · TRIPOD+AI"}
            </button>
          ))}
        </div>
      </Header>
      <main className={c.main}>
        <div className={c.mapBand}>
          <MapView
            purpose={purpose}
            answers={a}
            preview={shown?.node === focus ? (shown?.preview ?? null) : null}
            focus={focus}
            walking={walking}
            objective={nextAsked}
            onFocus={onFocus}
            onSilent={(r) => {
              setFocusRaw(null);
              setSilent(r);
            }}
          />
        </div>
        <div className={c.lower}>
          <div className={c.cardPane} ref={cardPane}>
            {silent ? (
              <SilentCard region={silent} items={regionsSilent[silent] ?? []} purpose={purpose} />
            ) : ctx && focus ? (
              <NodeCard key={`${purpose}-${focus}`} node={focus} ctx={ctx} />
            ) : (
              <Intro objs={objs} onGo={goNext} onFocus={onFocus} purpose={purpose} />
            )}
          </div>
          <div className={c.canvasPane}>
            <Canvas
              scene={scene}
              empty={<EmptyCanvas focus={focus} locked={!!a.locked} />}
              onRecord={null}
            />
          </div>
          <Methods a={a} purpose={purpose} focus={focus} walking={walking} onFocus={onFocus} fresh={freshText} />
        </div>
      </main>
    </div>
  );
}

/** The concepts node `n` teaches that no earlier card has: they are first met here. */
function meet(m: Record<string, NodeId>, n: NodeId): Record<string, NodeId> {
  const terms = NODE_TERMS[n] ?? [];
  if (!terms.some((t) => !m[t.term])) return m;
  const next = { ...m };
  for (const t of terms) if (!next[t.term]) next[t.term] = n;
  return next;
}

function defaultScene(focus: NodeId, a: Answers, purpose: Purpose): Scene | null {
  if (purpose === "prediction") {
    if (focus === "p_seal") return sceneOf("p-seal", "0.2", PRED.seal.options[0]!.label, PRED.seal.previews["0.2"]);
    if (focus === "p_missing") return sceneOf("p-missing", "impute", PRED.missing.labels.options.impute!.label, PRED.missing.previews.impute);
    if (focus === "exclusions") return sceneOf("p-excl", "keep_every_row", PRED.exclusions.labels.options.keep_every_row!.label, PRED.exclusions.previews.none, a.pExclusions);
    return null;
  }
  switch (focus) {
    case "exclusions": {
      const k = a.exclusions ?? "none";
      return sceneOf("exclusions", k, INF.exclusions.labels.options[k === "none" ? "keep_every_row" : k]?.label ?? k, previewFor("exclusions", k, a), !!a.exclusions);
    }
    case "energy":
      return sceneOf("energy", a.energy, FXlabel(a.energy), previewFor("energy", a.energy, a), true);
    case "seal":
      return sceneOf("seal", "0", INF.seal.options.find((o) => o.holdout === 0)!.label, previewFor("split", "0.0", a), true);
    case "missing":
      return sceneOf("missing", "complete_case", INF.missing.labels.options.complete_case!.label, previewFor("missing", "complete_case", a), true);
    case "readings":
      return { kind: "note", group: "readings", key: "unit", label: `${INF.readings.unit.column}: ${INF.readings.unit.guess_words}?`, note: INF.readings.unit.previews.confirm.result?.note ?? "", basis: INF.readings.unit.previews.confirm.result?.basis, aside: "the map shows which columns the readings send on" };
    case "exposure":
      return { kind: "note", group: "estimand", key: "estimand", label: "The exposure and its estimand", note: INF.estimand.preview_note, basis: INF.estimand.preview_basis, aside: "the map shows the exposure's lane" };
    case "adjustment":
      return { kind: "note", group: "adjustment", key: "adj", label: "The adjustment set", note: INF.adjustment.preview_note, aside: "the map shows where each column goes" };
    case "form":
      return { kind: "note", group: "form", key: a.form, label: INF.form.options.find((o) => o.value === a.form)!.label, note: INF.form.preview_note, aside: "as recorded" };
    default:
      return null;
  }
}

function FXlabel(method: string): string {
  return INF.energy.labels.options[method]?.label ?? method;
}

function Intro({
  objs,
  onGo,
  onFocus,
  purpose,
}: {
  objs: { node: NodeId; done: boolean }[];
  onGo: () => void;
  onFocus: (n: NodeId) => void;
  purpose: Purpose;
}) {
  const left = objs.filter((o) => !o.done);
  return (
    <article className={c.card} data-testid="intro">
      <header className={c.cardHead}>
        <p className={c.cardKicker}>
          <span>The methods, drafted</span>
          <span className={c.items}>{purpose === "inference" ? "STROBE-nut" : "TRIPOD+AI"}</span>
        </p>
        <h1 className={c.cardTitle}>{left.length ? `${left.length} decisions are asked of you` : "Every asked decision is answered"}</h1>
      </header>
      <p className={c.hintLead}>
        Solid dots are written in and can be changed; rings are asked. Click any node, or walk the asked ones in order.
      </p>
      <ol className={c.objectiveList}>
        {objs.map((o) => (
          <li key={o.node} data-done={o.done || undefined}>
            <button type="button" className={c.link} onClick={() => onFocus(o.node)}>
              {NODE_TITLE[o.node]}
            </button>
            {o.done ? <span className={c.done}>answered</span> : null}
          </li>
        ))}
      </ol>
      {left.length ? (
        <button type="button" className={c.primary} onClick={onGo} data-testid="intro-start">
          Start with {NODE_TITLE[left[0]!.node]} →
        </button>
      ) : null}
    </article>
  );
}

function SilentCard({ region, items, purpose }: { region: string; items: { key: string; reason: string }[]; purpose: Purpose }) {
  const r = REGIONS[purpose].find((x) => x.id === region)!;
  return (
    <article className={c.card} data-testid="silent">
      <header className={c.cardHead}>
        <p className={c.cardKicker}>
          <span>{r.title}</span>
          <span className={c.items}>{r.items}</span>
        </p>
        <h1 className={c.cardTitle}>Silent here · in the export only</h1>
      </header>
      <ul className={c.readList}>
        {items.map((x) => (
          <li key={x.key}>
            <span className={c.silentKey}>{x.key.replace(/_/g, " ")}</span>
            <span className={c.evidence}>
              <Rich text={x.reason} />
            </span>
          </li>
        ))}
      </ul>
    </article>
  );
}

function EmptyCanvas({ focus, locked }: { focus: NodeId | null; locked: boolean }): ReactNode {
  return (
    <div className={c.emptyCanvas}>
      <p>
        {focus === "lock" || focus === "estimate" || focus === "matter"
          ? locked
            ? ""
            : "The estimates appear here once the plan is locked."
          : focus
            ? "This decision has no picture on your data; its effect is drawn on the map above."
            : "Open a node on the map to see what its decision does to your data."}
      </p>
    </div>
  );
}
