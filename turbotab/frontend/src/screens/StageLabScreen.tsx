/**
 * /lab/stage — the production `<Stage>` on the mock's NHANES project, beside a stand-in for the
 * Record (the record agent owns the real one). It is a review surface and the Playwright journey's
 * page: hover, focus or arrow through options to preview them, tap once to preview and again to
 * record, press a finding for its evidence, or a banner segment for its full view.
 *
 * Works only under `npm run dev:mock` (the project `nhanes-m1` lives in the mock).
 */
import { useCallback, useEffect, useRef, useState, type KeyboardEvent } from "react";
import { useProjectEvents } from "../api/events";
import type { BannerSegment, StageFocus } from "../state/focus";
import { useDecide, useProjectView } from "../api/queries";
import type { Decision } from "../api/schema";
import { Header } from "../components/Header";
import { Stage } from "../components/stage/Stage";
import { Rich } from "../components/stage/text";
import { DEMO_PID } from "../mocks/m1-stage";
import raw from "../mocks/m1-stage-fixture.json";
import s from "./StageLabScreen.module.css";

interface Captured {
  decision: Decision & Record<string, unknown>;
  label?: string;
}
interface FixtureShape {
  previews: Record<string, Captured[]>;
  findings: { findings: { id: string; summary: string; lever_label: string | null; severity: string }[] };
}
const F = raw as unknown as FixtureShape;

interface Option {
  key: string;
  label: string;
  decision: Decision;
}
interface Question {
  key: string;
  kicker: string;
  question: string;
  options: Option[];
}

const ENERGY_LABEL: Record<string, string> = {
  residual: "Willett residual model",
  density: "Nutrient density alone",
  density_multivariate: "Multivariate nutrient density",
  standard: "Standard (multivariate) model",
  partition: "Energy partition model",
  none: "No energy adjustment",
};

function energyOptions(): Option[] {
  const out: Option[] = F.previews.energy_adjustment!.map((c, i) => {
    const m = String(c.decision.method);
    const subset = (c.decision.nutrients as string[]).length < 6;
    return {
      key: `energy-${i}`,
      label: subset ? "Partition protein, carb, fat_total" : (ENERGY_LABEL[m] ?? m),
      decision: c.decision,
    };
  });
  const partition = F.previews.energy_adjustment!.find((c) => c.decision.method === "partition")!.decision;
  out.push({
    key: "energy-sugar",
    label: "Partition, `sugar` included",
    decision: { ...partition, nutrients: [...(partition.nutrients as string[]), "sugar"] } as Decision,
  });
  return out;
}

const QUESTIONS: Question[] = [
  {
    key: "energy_adjustment",
    kicker: "Energy adjustment",
    question: "How should nutrients be adjusted for total energy?",
    options: energyOptions(),
  },
  {
    key: "exclusions",
    kicker: "Eligibility",
    question: "Which rows should be excluded as implausible energy intakes?",
    options: F.previews.exclusions!.map((c, i) => ({ key: `ex-${i}`, label: c.label ?? `Rule ${i + 1}`, decision: c.decision })),
  },
  {
    key: "missing",
    kicker: "Missing values",
    question: "What happens to rows with a blank predictor?",
    options: F.previews.missing!.map((c, i) => ({
      key: `missing-${i}`,
      label: c.decision.strategy === "complete_case" ? "Complete cases" : "Fill the blanks",
      decision: c.decision,
    })),
  },
  {
    key: "split",
    kicker: "Held-out rows",
    question: "How many rows are sealed for the final check?",
    options: F.previews.split!.map((c, i) => ({
      key: `split-${i}`,
      label: Number(c.decision.holdout) ? `Hold out ${Math.round(Number(c.decision.holdout) * 100)}%` : "Cross-validation only",
      decision: c.decision,
    })),
  },
  {
    key: "models",
    kicker: "Models",
    question: "Which model families should be fitted?",
    options: F.previews.models!.map((c, i) => {
      const m = c.decision.models as string[];
      return {
        key: `models-${i}`,
        label: m.length > 1 ? "All three families" : m[0]!.replace(/_/g, " "),
        decision: c.decision,
      };
    }),
  },
];

const SEGMENTS: { segment: BannerSegment; label: string }[] = [
  { segment: "rows", label: "Rows" },
  { segment: "columns", label: "Columns" },
  { segment: "models", label: "Models" },
  { segment: "result", label: "Result" },
];

/** How long a preview survives the pointer crossing from the Record to the stage. */
const LEAVE_MS = 220;

export function StageLabScreen() {
  const stream = useProjectEvents(DEMO_PID);
  const viewQ = useProjectView(DEMO_PID);
  const decide = useDecide(DEMO_PID);
  const [focus, setFocus] = useState<StageFocus>({ kind: "live" });
  const [qi, setQi] = useState(0);
  const [tapped, setTapped] = useState<string | null>(null);
  const leave = useRef(0);
  const q = QUESTIONS[qi]!;

  const preview = useCallback((o: Option) => {
    window.clearTimeout(leave.current);
    setFocus({ kind: "option", decision: o.decision, label: o.label });
  }, []);
  const toLive = useCallback(() => {
    window.clearTimeout(leave.current);
    leave.current = window.setTimeout(() => setFocus({ kind: "live" }), LEAVE_MS);
  }, []);
  useEffect(() => () => window.clearTimeout(leave.current), []);

  const record = (o: Option) => decide.mutate(o.decision, { onSuccess: () => setFocus({ kind: "live" }) });

  const onKey = (e: KeyboardEvent<HTMLLIElement>, i: number) => {
    const opts = q.options;
    if (e.key === "ArrowDown" || e.key === "ArrowUp") {
      e.preventDefault();
      const next = (i + (e.key === "ArrowDown" ? 1 : opts.length - 1)) % opts.length;
      const el = e.currentTarget.parentElement?.querySelectorAll<HTMLElement>("[role=option]")[next];
      el?.focus();
    } else if (e.key === "Enter") {
      e.preventDefault();
      record(opts[i]!);
    } else if (e.key === "Escape") {
      setFocus({ kind: "live" });
    }
  };

  if (viewQ.isPending) return <Header />;
  if (viewQ.isError || !viewQ.data) {
    return (
      <>
        <Header />
        <main className={s.message}>
          The stage lab runs on the mock server only: <code className="v">npm run dev:mock</code>.
        </main>
      </>
    );
  }
  const view = viewQ.data;
  const sel = focus.kind === "option" ? JSON.stringify(focus.decision) : null;

  return (
    <div className={s.screen}>
      <Header>
        <span className={s.name}>_tt_tmp_nhanes</span>
        <span className={s.size}>21,849 rows × 29 columns · stage lab</span>
        {stream === "reconnecting" ? <span className={s.size}>reconnecting…</span> : null}
      </Header>
      <nav className={s.banner} aria-label="Pipeline banner (stand-in)">
        {SEGMENTS.map((x) => (
          <button
            key={x.segment}
            type="button"
            className={s.segment}
            aria-pressed={focus.kind === "banner" && focus.segment === x.segment}
            onClick={() => setFocus({ kind: "banner", segment: x.segment })}
            data-segment={x.segment}
          >
            {x.label}
          </button>
        ))}
        <button type="button" className={s.segment} aria-pressed={focus.kind === "live"} onClick={() => setFocus({ kind: "live" })}>
          Nothing focused
        </button>
      </nav>
      <div className={s.layout}>
        <main className={s.record} aria-label="The record (stand-in)">
          <div className={s.tabs} role="tablist" aria-label="Questions">
            {QUESTIONS.map((x, i) => (
              <button
                key={x.key}
                type="button"
                role="tab"
                aria-selected={i === qi}
                className={s.tab}
                onClick={() => setQi(i)}
                data-question={x.key}
              >
                {x.kicker}
              </button>
            ))}
          </div>
          <section className={s.question} onPointerLeave={toLive}>
            <p className={s.kicker}>{q.kicker}</p>
            <h2 className={s.title}>{q.question}</h2>
            <ul className={s.options} role="listbox" aria-label={q.question}>
              {q.options.map((o, i) => (
                <li
                  key={o.key}
                  role="option"
                  tabIndex={0}
                  aria-selected={sel === JSON.stringify(o.decision)}
                  className={s.option}
                  data-option={o.key}
                  onPointerEnter={(e) => e.pointerType === "mouse" && preview(o)}
                  onFocus={() => preview(o)}
                  onClick={() => {
                    // Touch: the first tap previews, a second tap on the same option records.
                    if (tapped === o.key && sel === JSON.stringify(o.decision)) record(o);
                    else {
                      setTapped(o.key);
                      preview(o);
                    }
                  }}
                  onKeyDown={(e) => onKey(e, i)}
                >
                  <Rich text={o.label} chips={false} />
                </li>
              ))}
            </ul>
            <p className={s.keys}>↑ ↓ preview · Space flip · Enter record · Esc pipeline</p>
          </section>
          <section className={s.findings} aria-label="Change an earlier answer">
            <p className={s.kicker}>Change an earlier answer</p>
            <div className={s.answers}>
              {(
                [
                  ["Energy: residual", { kind: "set_energy_adjustment", method: "residual" }, view.state.energy_adjustment?.method === "residual"],
                  ["Energy: density", { kind: "set_energy_adjustment", method: "density" }, view.state.energy_adjustment?.method === "density"],
                  ["Purpose: prediction", { kind: "set_purpose", purpose: "prediction" }, view.state.purpose === "prediction"],
                  ["Purpose: inference", { kind: "set_purpose", purpose: "inference" }, view.state.purpose === "inference"],
                ] as [string, Record<string, unknown>, boolean][]
              ).map(([label, d, on]) => (
                <button
                  key={label}
                  type="button"
                  className={s.segment}
                  aria-pressed={on}
                  disabled={on || decide.isPending}
                  data-answer={label}
                  onClick={() => {
                    const full =
                      d.kind === "set_energy_adjustment"
                        ? { ...view.state.energy_adjustment, ...d }
                        : d;
                    decide.mutate(full as unknown as Decision);
                  }}
                >
                  {label}
                </button>
              ))}
            </div>
          </section>
          <section className={s.findings}>
            <p className={s.kicker}>Noticed in this table</p>
            <ul className={s.findingList}>
              {F.findings.findings.map((f) => (
                <li key={f.id}>
                  <button
                    type="button"
                    className={s.finding}
                    aria-pressed={focus.kind === "finding" && focus.findingId === f.id}
                    onClick={() => setFocus({ kind: "finding", findingId: f.id })}
                    onFocus={() => setFocus({ kind: "finding", findingId: f.id })}
                    data-finding={f.id}
                  >
                    <Rich text={f.summary} />
                  </button>
                </li>
              ))}
            </ul>
          </section>
        </main>
        <aside className={s.stageCol} onPointerEnter={() => window.clearTimeout(leave.current)}>
          <Stage pid={DEMO_PID} view={view} focus={focus} onFocus={setFocus} />
        </aside>
      </div>
    </div>
  );
}
