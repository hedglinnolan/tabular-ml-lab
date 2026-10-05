/**
 * #/kit: every part of the calm kit and every canvas layout, on the scenario's real data, so the
 * structure builders (and a reviewer) see the parts they share before they see a structure. Light
 * and dark: the switch at the top, or the system's setting.
 */
import { useState, type ReactNode } from "react";
import {
  Canvas,
  CanvasFrame,
  Card,
  Chain,
  Footer,
  Mattered,
  ManuscriptBody,
  ManuscriptColumn,
  OptionList,
  SCENARIO_ANSWERS,
  STEP_BY_ID,
  Table2,
  ThemeSwitch,
  initial,
  kit as k,
  reduce,
  results,
  route,
  useWalk,
  type Flip,
  type Layout,
  type WalkState,
} from "./index";

function walkTo(count: number): WalkState {
  let s = initial();
  for (const [step, option] of SCENARIO_ANSWERS.slice(0, count)) s = reduce(s, { type: "record", step, option });
  return s;
}

const LOCKED = walkTo(SCENARIO_ANSWERS.length);
const MID = walkTo(SCENARIO_ANSWERS.findIndex(([id]) => id === "energy"));

/** One layout on real data: the step, the option, and the layout the router picks for it. */
const LAYOUTS: { title: string; step: string; option: string }[] = [
  { title: "Focus: one column's values", step: "unit", option: "kj_1" },
  { title: "Strip: several columns change", step: "energy", option: "residual" },
  { title: "Flow: rows leave", step: "exclusions", option: "willett_2013_by_sex" },
  { title: "Routing: what feeds where", step: "model1", option: "guess" },
  { title: "Angles: the declared tradeoffs", step: "contrast", option: "addition" },
  { title: "Angles: total or direct", step: "effect", option: "direct" },
  { title: "Nothing changes: one line", step: "missing", option: "complete_case" },
  { title: "Not available: the engine's refusal", step: "exclusions", option: "goldberg_schofield" },
];

function Demo({ title, children, note }: { title: string; children: ReactNode; note?: ReactNode }) {
  return (
    <section style={{ display: "grid", gap: 10, minWidth: 0 }}>
      <h2 style={{ margin: 0, fontSize: 19 }}>{title}</h2>
      {note ? <p style={{ margin: 0, color: "var(--muted)", maxWidth: "70ch" }}>{note}</p> : null}
      {children}
    </section>
  );
}

function LayoutDemo({ title, step, option }: { title: string; step: string; option: string }) {
  const s = STEP_BY_ID[step]!;
  const o = s.options.find((x) => x.id === option)!;
  const [flip, setFlip] = useState<Flip>("after");
  const [frame, setFrame] = useState<number | null>(null);
  const layout: Layout = route(o.preview, { disabled: o.disabled });
  return (
    <Demo title={title} note={`${s.question.replaceAll("`", "")} · ${o.name.replaceAll("`", "")} · layout: ${layout}`}>
      <div data-testid={`demo-${layout}`}>
        <CanvasFrame step={s} option={o} flip={flip} setFlip={setFlip} frame={frame} setFrame={setFrame} testid={`canvas-${step}-${option}`} />
      </div>
    </Demo>
  );
}

function OptionStates() {
  const s = STEP_BY_ID.exclusions!;
  const [chosen, setChosen] = useState<string | null>("none");
  const [pointed, setPointed] = useState<string | null>("willett_2013_by_sex");
  return (
    <Demo title="Options" note="Pointed (hover tint), chosen (filled), a quiet label each at most, and a disabled option that says why. Arrow keys move through them.">
      <div style={{ maxWidth: 440 }}>
        <OptionList step={s} chosen={chosen} pointed={pointed} onPoint={setPointed} onChoose={setChosen} />
      </div>
    </Demo>
  );
}

function CardDemo() {
  const walk = useWalk(walkTo(1));
  return (
    <Demo title="Card and canvas, wired to the walk" note="The stage and step atop the card, the question, its options, the disclosure and Continue; the canvas answers the option pointed at.">
      <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(min(100%, 380px), 1fr))", gap: 28, alignItems: "start" }}>
        <Card walk={walk} />
        <Canvas walk={walk} />
      </div>
    </Demo>
  );
}

function ChainDemo() {
  const walk = useWalk(MID);
  return (
    <Demo title="Chain" note="Done stages are marked and can be revisited; the current one is named; later ones wait.">
      <Chain walk={walk} />
    </Demo>
  );
}

function ManuscriptDemo() {
  const walk = useWalk(MID);
  const [open, setOpen] = useState(true);
  return (
    <Demo
      title="Manuscript"
      note="Sentences by STROBE-nut section: stated ones written in, recorded ones clickable to change, asked ones as blanks. As a rail whose overlay lies over the card column, and as a column."
    >
      <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(min(100%, 360px), 1fr))", gap: 28, alignItems: "start" }}>
        <div style={{ display: "grid", gridTemplateColumns: "44px minmax(0, 1fr)", gap: 16, alignItems: "start" }}>
          <button type="button" className={k.rail} aria-expanded={open} onClick={() => setOpen(!open)}>
            Manuscript <span className={k.railCount}>{walk.sentences}</span>
          </button>
          <div style={{ position: "relative", minHeight: 420 }}>
            <p className={k.note}>The card column. The overlay opens over it, never over the canvas.</p>
            {open ? (
              <aside className={k.overlay} aria-label="Manuscript (overlay)">
                <ManuscriptBody sections={walk.manuscript} onOpen={walk.open} />
              </aside>
            ) : null}
          </div>
        </div>
        <ManuscriptColumn walk={walk} />
      </div>
    </Demo>
  );
}

function FooterDemo() {
  const walk = useWalk(walkTo(1));
  return (
    <Demo title="Footer" note="On screens 900 px and narrower: the live readout and Continue, fixed at the bottom.">
      <Footer walk={walk} inline />
    </Demo>
  );
}

function ResultsDemo() {
  const r = results(LOCKED)!;
  return (
    <Demo title="Results after the lock" note="Table 2 (the declared models) and which decisions mattered, in the canvas; never before the plan is locked.">
      <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(min(100%, 460px), 1fr))", gap: 28, alignItems: "start" }}>
        <section className={k.canvas}>
          <div className={k.chead}>
            <h2>Table 2</h2>
          </div>
          <Table2 results={r} />
        </section>
        <section className={k.canvas}>
          <div className={k.chead}>
            <h2>Which of my decisions mattered?</h2>
          </div>
          <Mattered results={r} />
        </section>
      </div>
    </Demo>
  );
}

export function KitDemo() {
  return (
    <div className={k.page} data-testid="kit">
      <header className={k.top}>
        <a className={k.brand} href="#/">
          TurboTab
        </a>
        <div className={k.topright}>
          <span style={{ color: "var(--muted)", fontSize: 15 }}>The calm kit</span>
          <ThemeSwitch />
        </div>
      </header>
      <main style={{ display: "grid", gap: 40, paddingTop: 20, maxWidth: 1240 }}>
        <ChainDemo />
        <CardDemo />
        <OptionStates />
        <Demo title="The canvas grammar" note="The router picks each layout from the option's footprint, measured from the engine's own views.">
          <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(min(100%, 520px), 1fr))", gap: 28, alignItems: "start" }}>
            {LAYOUTS.map((l) => (
              <LayoutDemo key={`${l.step}-${l.option}`} {...l} />
            ))}
          </div>
        </Demo>
        <ManuscriptDemo />
        <FooterDemo />
        <ResultsDemo />
      </main>
    </div>
  );
}
