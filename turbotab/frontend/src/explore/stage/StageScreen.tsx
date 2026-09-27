/**
 * /lab/explore/stage — "the consequence stage", a design prototype for BLUEPRINT §11.
 *
 * Left, the Record stays minimal: the question, option names, one consequence line each.
 * Right, the pipeline panel. Hovering or focusing an option turns the panel into that
 * option's consequence on the user's own data; arrow keys flip between options and the
 * panel morphs; leaving the options (pointer and focus) restores the live pipeline.
 *
 * Four scenarios, all from the real-data fixture (docs/turbotab-next/m1/explore):
 *   energy      S1 six energy-adjustment methods, one refused and still on the shelf
 *   exclusions  S2 three implausible-intake rules: row flow + kcal with the cuts marked
 *   findings    S3 the 13 NHANES findings: 3 pushed, paged, claim + lever, evidence staged
 *   transform   S4 log2(x + 1) over 495 genomics count columns
 */
import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { LayoutGroup } from "motion/react";
import { Header } from "../../components/Header";
import { DecisionSentence, QuestionBlock } from "../../components/record/blocks";
import { B } from "./fixture";
import { FindingsRecord } from "./FindingsRecord";
import { ALL_FINDING_IDS, FINDINGS_BASIS, PUSHED, REST, REST_LINE } from "./findings";
import { fmtInt } from "./format";
import { OptionList } from "./OptionList";
import { DATASETS, FACTS, QUESTIONS, type Preview, type ScenarioId } from "./scenarios";
import { Stage } from "./Stage";
import { Rich } from "./text";
import type { ViewKind } from "./types";
import s from "./StageScreen.module.css";

const SCENARIOS: { id: ScenarioId; label: string }[] = [
  { id: "energy", label: "Energy" },
  { id: "exclusions", label: "Exclusions" },
  { id: "findings", label: "Findings" },
  { id: "transform", label: "Wide" },
];

function readScenario(): ScenarioId {
  const v = new URLSearchParams(window.location.search).get("s");
  return SCENARIOS.some((x) => x.id === v) ? (v as ScenarioId) : "energy";
}

const RANK = Object.fromEntries(B.most_changed.map((m) => [m.column, m.shape_change]));

/** How long the preview survives the pointer crossing the gap between Record and stage. */
const LEAVE_MS = 220;

export function StageScreen() {
  const [scenario, setScenario] = useState<ScenarioId>(readScenario);
  const [hover, setHover] = useState<string | null>(null);
  const [focus, setFocus] = useState<string | null>(null);
  const [recorded, setRecorded] = useState<Partial<Record<ScenarioId, string>>>({});
  const [open, setOpen] = useState<Partial<Record<ScenarioId, boolean>>>({});
  const [promoted, setPromoted] = useState<Partial<Record<ScenarioId, ViewKind>>>({});
  const [pages, setPages] = useState<Record<string, number>>({});
  const leaveTimer = useRef<number | null>(null);
  const zoneRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    const url = new URL(window.location.href);
    url.searchParams.set("s", scenario);
    window.history.replaceState(null, "", url);
  }, [scenario]);

  const cancelLeave = () => {
    if (leaveTimer.current !== null) window.clearTimeout(leaveTimer.current);
    leaveTimer.current = null;
  };
  const scheduleLeave = () => {
    cancelLeave();
    leaveTimer.current = window.setTimeout(() => setHover(null), LEAVE_MS);
  };

  const go = useCallback((to: ScenarioId) => {
    setHover(null);
    setFocus(null);
    setScenario(to);
  }, []);

  const q = scenario === "findings" ? null : QUESTIONS[scenario];
  const key = hover ?? focus;
  const isOpen = q ? (open[scenario] ?? !recorded[scenario]) : false;
  const rec = recorded[scenario] ?? null;

  // What the stage shows.
  let preview: Preview | null = null;
  if (q && isOpen && key) {
    const opt = [...q.options, ...(q.extra ?? [])].find((o) => o.key === key);
    preview = opt?.preview ?? null;
  }
  if (!q && key) {
    const card = [...PUSHED, ...REST].find((c) => c.id === key);
    const page = card?.pages[pages[card.id] ?? 0];
    if (page)
      preview = {
        pill: "Evidence",
        aside: "your data as loaded",
        label: "",
        basis: page.stage.basis,
        views: page.stage.views,
      };
  }

  const live = q ? (rec ? q.liveAfter(rec) : q.live) : FACTS.exLive;
  const recordedLabel = q && rec ? [...q.options, ...(q.extra ?? [])].find((o) => o.key === rec)?.label : null;

  const record = (k: string) => {
    setRecorded((r) => ({ ...r, [scenario]: k }));
    setOpen((o) => ({ ...o, [scenario]: false }));
    setHover(null);
    setFocus(null);
  };

  const onZoneBlur = (e: React.FocusEvent<HTMLDivElement>) => {
    const next = e.relatedTarget as Node | null;
    if (!next || !zoneRef.current?.contains(next)) setFocus(null);
  };

  const ds = DATASETS[scenario];
  const stageProps = useMemo(
    () => ({
      universe: scenario === "transform" ? { n: B.count_columns.n, noun: "count columns" } : undefined,
      rank: scenario === "transform" ? RANK : undefined,
      nRows: ds.rows,
    }),
    [scenario, ds.rows],
  );

  return (
    <div className={s.screen}>
      <Header>
        <span className={s.dsName}>{ds.name}</span>
        <span className={s.dsSize}>
          {fmtInt(ds.rows)} rows × {fmtInt(ds.cols)} columns
        </span>
        <nav className={s.scenarios} aria-label="Prototype scenarios">
          {SCENARIOS.map((x, i) => (
            <button
              key={x.id}
              type="button"
              className={s.scenario}
              aria-current={scenario === x.id ? "page" : undefined}
              onClick={() => go(x.id)}
              data-scenario={x.id}
            >
              <span className={s.scenarioNum}>S{i + 1}</span> {x.label}
            </button>
          ))}
        </nav>
      </Header>

      <div className={s.layout} ref={zoneRef} onBlur={onZoneBlur}>
        <main className={s.record} key={scenario}>
          {q ? (
            <LayoutGroup id={`q-${scenario}`}>
              {isOpen ? (
                <QuestionBlock
                  layoutId={`q-${scenario}`}
                  kicker={q.kicker}
                  title={q.question}
                  why={<Rich text={q.why} terms={q.terms} />}
                  testId="question"
                >
                  <div
                    onPointerEnter={cancelLeave}
                    onPointerLeave={scheduleLeave}
                  >
                    <OptionList
                      options={q.options}
                      previewKey={key}
                      recordedKey={rec}
                      onFocusOption={(k) => setFocus(k)}
                      onHoverOption={(k) => {
                        cancelLeave();
                        setHover(k);
                      }}
                      onKeyboard={() => setHover(null)}
                      onEscape={() => {
                        setHover(null);
                        setFocus(null);
                      }}
                      onRecord={record}
                      terms={q.terms}
                    />
                  </div>
                </QuestionBlock>
              ) : (
                <DecisionSentence
                  layoutId={`q-${scenario}`}
                  subject={q.kicker.toLowerCase()}
                  onChange={() => setOpen((o) => ({ ...o, [scenario]: true }))}
                  testId="decision"
                >
                  <Rich
                    text={
                      [...q.options, ...(q.extra ?? [])].find((o) => o.key === rec)?.sentence ?? ""
                    }
                  />
                </DecisionSentence>
              )}
            </LayoutGroup>
          ) : (
            <div onPointerEnter={cancelLeave} onPointerLeave={scheduleLeave}>
              <FindingsRecord
                pushed={PUSHED}
                rest={REST}
                restLine={REST_LINE}
                total={ALL_FINDING_IDS.length}
                basis={FINDINGS_BASIS}
                activeCard={key}
                pages={pages}
                onHover={(id) => {
                  cancelLeave();
                  setHover(id);
                }}
                onFocusCard={(id) => setFocus(id)}
                onKeyboard={() => setHover(null)}
                onEscape={() => {
                  setHover(null);
                  setFocus(null);
                }}
                onPage={(id, p) => setPages((x) => ({ ...x, [id]: p }))}
                onRoute={go}
              />
            </div>
          )}
        </main>

        <aside
          className={s.stageCol}
          aria-label="Pipeline"
          onPointerEnter={cancelLeave}
          onPointerLeave={() => (hover ? scheduleLeave() : undefined)}
        >
          <Stage
            live={live}
            preview={preview}
            stacked={q?.stacked ?? false}
            promoted={promoted[scenario] ?? null}
            onPromote={(k) => setPromoted((p) => ({ ...p, [scenario]: k }))}
            terms={q?.terms}
            recorded={recordedLabel ? `recorded: ${recordedLabel}` : null}
            {...stageProps}
          />
        </aside>
      </div>
    </div>
  );
}
