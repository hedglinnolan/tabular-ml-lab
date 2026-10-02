/**
 * /lab/stage/m2 — the production `<Stage>` on M2's scenarios (M2_CONTRACT §11), beside a stand-in
 * for the Record (the record agent owns the real one). A review surface and the m2-stage
 * Playwright spec's page: each scenario is a mock project replaying the real server's answers on a
 * sample fixture (src/mocks/m2-stage.ts) — the reshape, the orientation turn, the seal in each
 * basis, the Results sealed → opened → changed after the opening, the repairs, and the lens /
 * outcome / purpose previews. Hover, focus or arrow through options to preview them; Enter (or a
 * second tap) records the ones the capture recorded.
 *
 * Works only under `npm run dev:mock`. `?p=<project>` opens a scenario.
 */
import { useCallback, useEffect, useMemo, useRef, useState, type KeyboardEvent } from "react";
import { useProjectEvents } from "../api/events";
import { useDecide, useProjectView } from "../api/queries";
import type { Decision } from "../api/schema";
import { Header } from "../components/Header";
import { Stage } from "../components/stage/Stage";
import { Rich } from "../components/stage/text";
import { M2_STAGE, decisionKey, type M2Project } from "../mocks/m2-stage";
import type { BannerSegment, StageFocus } from "../state/focus";
import s from "./StageLabScreen.module.css";

const GROUP_LABEL: Record<string, [string, string]> = {
  lens: ["Lens", "Which research lens does this table belong to?"],
  target: ["Outcome", "What are you predicting or explaining?"],
  purpose: ["Purpose", "Is the model for prediction or for inference?"],
  aggregation: ["Combining rows", "How should each person's rows be combined?"],
  orientation: ["Orientation", "Which way round is this table?"],
  exclusions: ["Eligibility", "Who is eligible for this study?"],
  missing: ["Missing values", "What happens to rows with a blank predictor?"],
  split: ["Held-out rows · the seal", "How many rows should be held out for one final, untouched score?"],
  energy_adjustment: ["Energy adjustment", "How should nutrients be adjusted for total energy?"],
  open_seal: ["Open the seal", "Open the held-out rows, once?"],
  repairs: ["Repairs", "How should these findings be repaired?"],
};

const SEGMENTS: { segment: BannerSegment; label: string }[] = [
  { segment: "rows", label: "Rows" },
  { segment: "columns", label: "Columns" },
  { segment: "models", label: "Models" },
  { segment: "result", label: "Result" },
];

const PIDS = Object.keys(M2_STAGE.projects);
const LEAVE_MS = 220;

function initialPid(): string {
  const p = new URLSearchParams(window.location.search).get("p");
  return p && PIDS.includes(p) ? p : PIDS[0]!;
}

interface Option {
  key: string;
  label: string;
  decision: Decision;
  recordable: boolean;
}

function questionsOf(p: M2Project): { group: string; options: Option[] }[] {
  const records = new Set(p.records.map((r) => decisionKey(r.decision)));
  const groups: { group: string; options: Option[] }[] = [];
  for (const [i, pv] of p.previews.entries()) {
    let g = groups.find((x) => x.group === pv.group);
    if (!g) groups.push((g = { group: pv.group, options: [] }));
    g.options.push({
      key: `${pv.group}-${i}`,
      label: pv.label,
      decision: pv.decision,
      recordable: records.has(decisionKey(pv.decision)),
    });
  }
  return groups;
}

export function StageLabM2Screen() {
  const [pid, setPid] = useState(initialPid);
  const choose = (next: string) => {
    const url = new URL(window.location.href);
    url.searchParams.set("p", next);
    window.history.replaceState(null, "", url);
    setPid(next);
  };
  return <Lab key={pid} pid={pid} onProject={choose} />;
}

function Lab({ pid, onProject }: { pid: string; onProject: (pid: string) => void }) {
  const project = M2_STAGE.projects[pid]!;
  const stream = useProjectEvents(pid);
  const viewQ = useProjectView(pid);
  const decide = useDecide(pid);
  const questions = useMemo(() => questionsOf(project), [project]);
  const [focus, setFocus] = useState<StageFocus>({ kind: "live" });
  const [qi, setQi] = useState(0);
  const [tapped, setTapped] = useState<string | null>(null);
  const leave = useRef(0);
  const q = questions[qi];

  const preview = useCallback((o: Option) => {
    window.clearTimeout(leave.current);
    setFocus({ kind: "option", decision: o.decision, label: o.label.replace(/`/g, "") });
  }, []);
  const toLive = useCallback(() => {
    window.clearTimeout(leave.current);
    leave.current = window.setTimeout(() => setFocus({ kind: "live" }), LEAVE_MS);
  }, []);
  useEffect(() => () => window.clearTimeout(leave.current), []);

  const record = (o: Option) => decide.mutate(o.decision, { onSuccess: () => setFocus({ kind: "live" }) });

  const onKey = (e: KeyboardEvent<HTMLLIElement>, i: number) => {
    const opts = q?.options ?? [];
    if (e.key === "ArrowDown" || e.key === "ArrowUp") {
      e.preventDefault();
      const next = (i + (e.key === "ArrowDown" ? 1 : opts.length - 1)) % opts.length;
      e.currentTarget.parentElement?.querySelectorAll<HTMLElement>("[role=option]")[next]?.focus();
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
          The M2 stage lab runs on the mock server only: <code className="v">npm run dev:mock</code>.
        </main>
      </>
    );
  }
  const view = viewQ.data;
  const sel = focus.kind === "option" ? decisionKey(focus.decision) : null;
  // Answers recorded after the scenario starts that are not one of its questions' options (the
  // change made after the seal was opened).
  const optionKeys = new Set(project.previews.map((pv) => decisionKey(pv.decision)));
  const changes = project.records.filter((r) => !optionKeys.has(decisionKey(r.decision)));
  const findings = Object.keys(project.evidence);
  const summaries = new Map(
    (
      (project.snapshots[project.start]?.artifacts.findings as { findings?: { id: string; summary: string }[] } | undefined)
        ?.findings ?? []
    ).map((f) => [f.id, f.summary]),
  );

  return (
    <div className={s.screen}>
      <Header>
        <span className={s.name}>{project.source}</span>
        <span className={s.size}>
          {view.summary.n_rows?.toLocaleString("en-US")} rows × {view.summary.n_cols?.toLocaleString("en-US")} columns · M2 stage lab
        </span>
        {stream === "reconnecting" ? <span className={s.size}>reconnecting…</span> : null}
      </Header>
      <nav className={s.banner} aria-label="Scenarios">
        {PIDS.map((id) => (
          <button
            key={id}
            type="button"
            className={s.segment}
            aria-pressed={id === pid}
            onClick={() => onProject(id)}
            data-project={id}
          >
            {id.replace(/^m2-/, "").replace(/-/g, " ")}
          </button>
        ))}
      </nav>
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
          <p className={s.kicker}>{project.label}</p>
          {questions.length ? (
            <>
              <div className={s.tabs} role="tablist" aria-label="Questions">
                {questions.map((x, i) => (
                  <button
                    key={x.group}
                    type="button"
                    role="tab"
                    aria-selected={i === qi}
                    className={s.tab}
                    onClick={() => setQi(i)}
                    data-question={x.group}
                  >
                    {GROUP_LABEL[x.group]?.[0] ?? x.group}
                  </button>
                ))}
              </div>
              {q ? (
                <section className={s.question} onPointerLeave={toLive}>
                  <p className={s.kicker}>{GROUP_LABEL[q.group]?.[0] ?? q.group}</p>
                  <h2 className={s.title}>{GROUP_LABEL[q.group]?.[1] ?? q.group}</h2>
                  <ul className={s.options} role="listbox" aria-label={GROUP_LABEL[q.group]?.[1] ?? q.group}>
                    {q.options.map((o, i) => (
                      <li
                        key={o.key}
                        role="option"
                        tabIndex={0}
                        aria-selected={sel === decisionKey(o.decision)}
                        className={s.option}
                        data-option={o.key}
                        data-recordable={o.recordable || undefined}
                        onPointerEnter={(e) => e.pointerType === "mouse" && preview(o)}
                        onFocus={() => preview(o)}
                        onClick={() => {
                          if (tapped === o.key && sel === decisionKey(o.decision)) record(o);
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
              ) : null}
            </>
          ) : null}
          {changes.length ? (
            <section className={s.findings} aria-label="Change an earlier answer">
              <p className={s.kicker}>Change an earlier answer</p>
              <div className={s.answers}>
                {changes.map((c) => (
                  <button
                    key={c.label}
                    type="button"
                    className={s.segment}
                    disabled={decide.isPending}
                    data-answer={c.to}
                    onClick={() => decide.mutate(c.decision)}
                  >
                    {c.label}
                  </button>
                ))}
              </div>
            </section>
          ) : null}
          {findings.length ? (
            <section className={s.findings}>
              <p className={s.kicker}>Noticed in this table</p>
              <ul className={s.findingList}>
                {findings.map((id) => (
                  <li key={id}>
                    <button
                      type="button"
                      className={s.finding}
                      aria-pressed={focus.kind === "finding" && focus.findingId === id}
                      onClick={() => setFocus({ kind: "finding", findingId: id })}
                      onFocus={() => setFocus({ kind: "finding", findingId: id })}
                      data-finding={id}
                    >
                      <Rich text={summaries.get(id) ?? id} />
                    </button>
                  </li>
                ))}
              </ul>
            </section>
          ) : null}
        </main>
        <aside className={s.stageCol} onPointerEnter={() => window.clearTimeout(leave.current)}>
          <Stage pid={pid} view={view} focus={focus} onFocus={setFocus} />
        </aside>
      </div>
    </div>
  );
}
