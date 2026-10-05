/**
 * The page's parts (FOUNDATION §3): the chain, the card with its options, the footer, the
 * manuscript and the shell that holds the zones. Each matches calm-screen.html and
 * color-study.html; a structure composes and configures them, it does not restyle them.
 */
import { useEffect, useId, useRef, useState, type ReactNode } from "react";
import { STEP_BY_ID, type Option, type Step } from "./fixture";
import { plain, Plain } from "./text";
import type { WalkApi } from "./useWalk";
import { readoutOf } from "./canvas/Canvas";
import type { Section } from "./walk";
import k from "./kit.module.css";

// ── Chain ────────────────────────────────────────────────────────────────────

export function Chain({ walk }: { walk: WalkApi }) {
  return (
    <nav aria-label="The analysis, stage by stage">
      <ol className={k.chain} data-testid="chain">
        {walk.chain.map((c) => (
          <li key={c.id} data-status={c.status}>
            <button
              type="button"
              className={k.chainBtn}
              data-status={c.status}
              data-testid={`chain-${c.id}`}
              aria-current={c.status === "current" ? "step" : undefined}
              disabled={c.status === "waiting"}
              onClick={() => walk.open(c.first)}
            >
              <span className={k.chainName}>{c.label}</span>
            </button>
          </li>
        ))}
      </ol>
    </nav>
  );
}

// ── OptionList ───────────────────────────────────────────────────────────────

export interface OptionListProps {
  step: Step;
  chosen: string | null;
  pointed: string | null;
  onPoint: (id: string | null) => void;
  onChoose: (id: string) => void;
}

/** How long the canvas keeps the last pointed option after the pointer leaves it. Crossing the gap
 *  between two options then goes straight from one preview to the next instead of flashing back
 *  to "Your data now" in between (Nolan, 2026-10-05: "it will glitch out back and forth"). */
export const POINT_GRACE_MS = 160;

/** The options: hover tints, choosing fills, disabled ones say why in their one line; arrow keys
 *  move through them (a native radio group). Each speaks plainly; its technical name (`term`, the
 *  second register) sits on its top edge while it is pointed at or focused (FOUNDATION §2). */
export function OptionList({ step, chosen, pointed, onPoint, onChoose }: OptionListProps) {
  const name = useId();
  const clearTimer = useRef<ReturnType<typeof setTimeout> | null>(null);
  const cancelClear = () => {
    if (clearTimer.current !== null) {
      clearTimeout(clearTimer.current);
      clearTimer.current = null;
    }
  };
  const pointAt = (id: string) => {
    cancelClear();
    if (id !== pointed) onPoint(id);
  };
  const clearSoon = () => {
    cancelClear();
    clearTimer.current = setTimeout(() => {
      clearTimer.current = null;
      onPoint(null);
    }, POINT_GRACE_MS);
  };
  useEffect(() => cancelClear, []);
  return (
    <fieldset className={k.opts} data-testid="options">
      <legend className={k.legend}>{step.legend}</legend>
      {step.options.map((o: Option) => (
        <label
          key={o.id}
          className={k.opt}
          data-off={o.disabled || undefined}
          data-pointed={pointed === o.id || undefined}
          data-testid={`opt-${o.id}`}
          onPointerEnter={() => (o.disabled ? clearSoon() : pointAt(o.id))}
          onPointerLeave={clearSoon}
        >
          <input
            type="radio"
            name={name}
            value={o.id}
            checked={chosen === o.id}
            disabled={o.disabled}
            onChange={() => onChoose(o.id)}
            onFocus={() => !o.disabled && pointAt(o.id)}
            onBlur={clearSoon}
          />
          <span className={k.optName}>
            <Plain text={o.name} />
          </span>
          <span className={k.optTag}>{o.label ?? ""}</span>
          <span className={k.optWhat}>
            <Plain text={o.what} />
          </span>
          {o.term ? (
            <span className={k.optTerm} data-testid={`term-${o.id}`}>
              Known as <Plain text={o.term} />
            </span>
          ) : null}
        </label>
      ))}
    </fieldset>
  );
}

// ── Card ─────────────────────────────────────────────────────────────────────

export function Why({ text }: { text: string }) {
  return (
    <details className={k.why}>
      <summary>Why does this matter?</summary>
      <p>
        <Plain text={text} />
      </p>
    </details>
  );
}

export function Continue({ walk, label = "Continue" }: { walk: WalkApi; label?: string }) {
  return (
    <button type="button" className={k.primary} disabled={!walk.canProceed} onClick={walk.proceed} data-testid="continue">
      {label}
    </button>
  );
}

/** The hint beside Continue: what is chosen, and that nothing is recorded until Continue. */
export function Hint({ walk }: { walk: WalkApi }) {
  const step = walk.step;
  if (!step) return <span className={k.hint}>{walk.state.open === "table2" ? "Next: which of your decisions mattered." : "The plan is locked."}</span>;
  if (walk.blocked.length) return <span className={k.hint}>Answer the earlier question again to continue.</span>;
  const o = step.options.find((x) => x.id === walk.chosen);
  if (!o) return <span className={k.hint}>Choose an option to continue.</span>;
  if (walk.state.answers[step.id] === o.id) return <span className={k.hint}>Recorded. Continue to the next question.</span>;
  return (
    <span className={k.hint}>
      You picked <b>{plain(o.name)}</b>. Nothing is recorded until you continue.
    </span>
  );
}

/** What blocks a step, in words: the earlier answers its captured previews depend on. */
function Blocked({ walk }: { walk: WalkApi }) {
  if (!walk.blocked.length) return null;
  return (
    <p className={k.note} data-testid="blocked">
      This walk holds the engine's answers for{" "}
      {walk.blocked.map((b, i) => {
        const s = STEP_BY_ID[b.step]!;
        const want = s.options.find((o) => o.id === (s.scenario ?? ""))!;
        return (
          <span key={b.step}>
            {i ? "; " : ""}
            <button type="button" className={k.linkish} onClick={() => walk.open(b.step)}>
              {plain(want.name)}
            </button>
          </span>
        );
      })}{" "}
      only. Change that answer to go on.
    </p>
  );
}

function LockNote({ walk }: { walk: WalkApi }) {
  if (walk.step?.id !== "lock" || walk.plan.ok) return null;
  const p = walk.plan;
  return (
    <p className={k.note} data-testid="lock-blocked">
      {p.missing.length
        ? `Answer the open questions first (${p.missing.length} left).`
        : "No fit was captured for this plan. It differs from the scenario in: "}
      {!p.missing.length
        ? p.differs.map((d, i) => (
            <span key={d.step}>
              {i ? ", " : ""}
              <button type="button" className={k.linkish} onClick={() => walk.open(d.step)}>
                {STEP_BY_ID[d.step]!.head.toLowerCase()}
              </button>
            </span>
          ))
        : null}
    </p>
  );
}

export interface CardProps {
  walk: WalkApi;
  /** Replace the stage label (a structure may name the step its own way). */
  kicker?: ReactNode;
  /** Shown above the actions (a structure's own note). */
  children?: ReactNode;
  continueLabel?: string;
}

/** The open question: its stage and step, the question, its options, the disclosure, Continue. */
export function Card({ walk, kicker, children, continueLabel }: CardProps) {
  const step = walk.step;
  const heading = useRef<HTMLHeadingElement>(null);
  const opened = useRef(walk.state.open);
  useEffect(() => {
    if (opened.current !== walk.state.open) heading.current?.focus();
    opened.current = walk.state.open;
  }, [walk.state.open]);
  if (!step) return <ResultCard walk={walk} kicker={kicker} />;
  return (
    <section className={k.card} data-testid="card" data-step={step.id}>
      <p className={k.stageLabel}>{kicker ?? walk.label}</p>
      <h1 className={k.question} ref={heading} tabIndex={-1}>
        <Plain text={step.question} />
      </h1>
      <p className={k.lede}>
        <Plain text={step.lede} />
      </p>
      <Why text={step.why} />
      <Blocked walk={walk} />
      <OptionList step={step} chosen={walk.chosen} pointed={walk.pointed} onPoint={walk.point} onChoose={walk.choose} />
      <LockNote walk={walk} />
      {children}
      <div className={k.actRow}>
        <Hint walk={walk} />
        <Continue walk={walk} label={continueLabel ?? (step.id === "lock" ? "Lock and fit" : "Continue")} />
      </div>
    </section>
  );
}

function ResultCard({ walk, kicker }: { walk: WalkApi; kicker?: ReactNode }) {
  const t2 = walk.state.open === "table2";
  return (
    <section className={k.card} data-testid="card" data-step={walk.state.open}>
      <p className={k.stageLabel}>{kicker ?? walk.label}</p>
      <h1 className={k.question} tabIndex={-1}>
        {t2 ? "Table 2: the estimate in each declared model" : "Which of my decisions mattered?"}
      </h1>
      <p className={k.lede}>
        {t2
          ? "The plan was locked before any estimate was shown; Model 2 is the reported estimate."
          : "The estimate across the alternatives declared before any estimate was shown."}
      </p>
      {t2 ? (
        <div className={k.actRow}>
          <Hint walk={walk} />
          <Continue walk={walk} label="Which decisions mattered?" />
        </div>
      ) : (
        <p className={k.note}>Change any answer from the chain or the manuscript; the record marks it as made after the estimates were seen.</p>
      )}
    </section>
  );
}

// ── Footer ───────────────────────────────────────────────────────────────────

/** The live readout and Continue, fixed at the bottom on narrow screens (≤ 900 px). */
export function Footer({ walk, inline = false }: { walk: WalkApi; inline?: boolean }) {
  const readout = readoutOf(walk.active);
  return (
    <div className={k.footer} data-testid="footer" style={inline ? { position: "static", display: "block", borderRadius: 10, border: "1px solid var(--line)" } : undefined}>
      <div className={k.actRow}>
        <div style={{ minWidth: 0 }}>
          {readout.length ? (
            <div className={k.footerReadout}>
              {readout.map((r) => (
                <span key={r.label}>
                  {r.label}{" "}
                  <b>
                    {r.now}
                    {walk.flip === "after" && r.after ? ` → ${r.after}` : ""}
                  </b>
                </span>
              ))}
            </div>
          ) : (
            <Hint walk={walk} />
          )}
        </div>
        <button type="button" className={k.primary} disabled={!walk.canProceed} onClick={walk.proceed} data-testid="continue-footer">
          {walk.step?.id === "lock" ? "Lock and fit" : walk.state.open === "table2" ? "What mattered?" : "Continue"}
        </button>
      </div>
    </div>
  );
}

// ── Manuscript ───────────────────────────────────────────────────────────────

export function ManuscriptBody({
  sections,
  onOpen,
  title = "Manuscript",
  action,
}: {
  sections: Section[];
  onOpen: (step: string) => void;
  title?: string;
  action?: ReactNode;
}) {
  return (
    <div data-testid="manuscript">
      <div className={k.msHead}>
        <h2>{title}</h2>
        {action}
      </div>
      {sections.map((sec) => (
        <section key={sec.id} className={k.msSection}>
          <h3>
            {sec.title} <small>{sec.item}</small>
          </h3>
          <ul className={k.msList}>
            {sec.entries.map((e) => (
              <li key={e.id} className={k.msEntry} data-kind={e.kind} data-newest={e.newest || undefined} data-testid={`ms-${e.id}`}>
                <span className={k.msHeadLine}>{e.head}</span>
                {e.kind === "recorded" && e.step ? (
                  <button type="button" className={k.msPhrase} onClick={() => onOpen(e.step!)} title="Change this answer">
                    <Plain text={e.sentence} />
                  </button>
                ) : e.kind === "stated" ? (
                  <Plain text={e.sentence} />
                ) : e.kind === "blank" && e.step ? (
                  <button type="button" className={k.msBlank} onClick={() => onOpen(e.step!)} data-testid={`blank-${e.id}`}>
                    Choose: <Plain text={STEP_BY_ID[e.step]?.question ?? ""} />
                  </button>
                ) : (
                  <span className={k.msWaiting}>{e.kind === "recorded" ? <Plain text={e.sentence} /> : "Waits for an earlier answer."}</span>
                )}
                {e.afterLock ? <span className={k.msAfter}>Changed after the estimates were seen.</span> : null}
              </li>
            ))}
          </ul>
        </section>
      ))}
    </div>
  );
}

/** The manuscript as a slim rail at the far left; opened, it lies over the card column, never
 *  over the canvas. Render `<ManuscriptRail>` in the rail zone and `<ManuscriptOverlay>` inside
 *  the card zone (the Shell does both). */
export function useManuscriptRail() {
  const [open, setOpen] = useState(false);
  useEffect(() => {
    if (!open) return;
    const esc = (e: KeyboardEvent) => e.key === "Escape" && setOpen(false);
    window.addEventListener("keydown", esc);
    return () => window.removeEventListener("keydown", esc);
  }, [open]);
  return { open, setOpen };
}

export function ManuscriptRail({ walk, open, setOpen }: { walk: WalkApi; open: boolean; setOpen: (b: boolean) => void }) {
  return (
    <button type="button" className={k.rail} aria-expanded={open} onClick={() => setOpen(!open)} data-testid="manuscript-rail">
      Manuscript <span className={k.railCount}>{walk.sentences}</span>
    </button>
  );
}

export function ManuscriptOverlay({ walk, open, setOpen, onPin }: { walk: WalkApi; open: boolean; setOpen: (b: boolean) => void; onPin?: () => void }) {
  if (!open) return null;
  return (
    <aside className={k.overlay} aria-label="Manuscript" data-testid="manuscript-overlay">
      <span style={{ float: "right", display: "inline-flex", gap: 14 }}>
        {onPin ? (
          <button type="button" className={`${k.linkish} ${k.pinOnly}`} onClick={onPin} data-testid="manuscript-pin">
            Pin as a column
          </button>
        ) : null}
        <button type="button" className={k.linkish} onClick={() => setOpen(false)}>
          Close
        </button>
      </span>
      <ManuscriptBody
        sections={walk.manuscript}
        onOpen={(id) => {
          walk.open(id);
          setOpen(false);
        }}
      />
    </aside>
  );
}

/** The manuscript as an always-visible column. */
export function ManuscriptColumn({ walk, title, action }: { walk: WalkApi; title?: string; action?: ReactNode }) {
  return (
    <aside className={k.msColumn} aria-label="Manuscript">
      <ManuscriptBody sections={walk.manuscript} onOpen={walk.open} title={title} action={action} />
    </aside>
  );
}

// ── Shell ────────────────────────────────────────────────────────────────────

export interface ShellProps {
  walk: WalkApi;
  /** The structure's name, beside the brand. */
  name?: string;
  /** "rail" (the default), "column" (always visible) or "none" (the structure shows it itself). */
  manuscript?: "rail" | "column" | "none";
  chain?: ReactNode | false;
  card: ReactNode;
  canvas: ReactNode;
  /** Replaces the column-variant manuscript (a structure's own). */
  manuscriptNode?: ReactNode;
  home?: string;
  /** Extra controls in the top bar. */
  tools?: ReactNode;
}

/** The page zones of FOUNDATION §3: chain on top; manuscript rail, card and canvas below; the
 *  footer on narrow screens. */
export function Shell({ walk, name, manuscript: asked = "rail", chain, card, canvas, manuscriptNode, home = "#/", tools }: ShellProps) {
  const rail = useManuscriptRail();
  // On screens 1680 px and wider the rail's manuscript can be pinned open as a third column.
  const [pinned, setPinned] = useState(false);
  const wide = useWide(1680);
  const manuscript = asked === "rail" && pinned && wide ? "column" : asked;
  return (
    <div className={k.page} data-testid="shell">
      <header className={k.top}>
        <span>
          <a className={k.brand} href={home}>
            TurboTab
          </a>
          {name ? <span style={{ color: "var(--muted)", marginLeft: 10, fontSize: 15 }}>{name}</span> : null}
        </span>
        <div className={k.topright}>
          {tools}
          <button type="button" className={k.linkish} onClick={walk.reset} data-testid="proto-reset">
            Start over
          </button>
          <ThemeSwitch />
        </div>
      </header>
      {chain === false ? null : (chain ?? <Chain walk={walk} />)}
      <div className={k.zones} data-manuscript={manuscript}>
        {manuscript === "rail" ? (
          <div className={k.zoneRail}>
            <ManuscriptRail walk={walk} open={rail.open} setOpen={rail.setOpen} />
          </div>
        ) : null}
        {manuscript === "column" ? (
          <div className={k.zoneManuscript}>
            {manuscriptNode ?? (
              <ManuscriptColumn
                walk={walk}
                action={
                  asked === "rail" ? (
                    <button type="button" className={k.linkish} onClick={() => setPinned(false)}>
                      Unpin
                    </button>
                  ) : undefined
                }
              />
            )}
          </div>
        ) : null}
        <div className={k.zoneCard}>
          {card}
          {manuscript === "rail" ? (
            <ManuscriptOverlay
              walk={walk}
              open={rail.open}
              setOpen={rail.setOpen}
              onPin={() => {
                setPinned(true);
                rail.setOpen(false);
              }}
            />
          ) : null}
        </div>
        <div className={k.zoneCanvas}>{canvas}</div>
      </div>
      <Footer walk={walk} />
    </div>
  );
}

function useWide(px: number): boolean {
  const query = `(min-width: ${px}px)`;
  const [wide, setWide] = useState(() => typeof window !== "undefined" && !!window.matchMedia?.(query).matches);
  useEffect(() => {
    const mq = window.matchMedia?.(query);
    if (!mq) return;
    const on = () => setWide(mq.matches);
    mq.addEventListener("change", on);
    return () => mq.removeEventListener("change", on);
  }, [query]);
  return wide;
}

// ── theme ────────────────────────────────────────────────────────────────────

const THEME_KEY = "turbotab.theme";

/** Light and dark, both first-class: the system decides until the viewer chooses. */
export function ThemeSwitch() {
  const [theme, setTheme] = useState<string | null>(() => document.documentElement.dataset.theme ?? null);
  const set = (t: "light" | "dark") => {
    document.documentElement.dataset.theme = t;
    try {
      localStorage.setItem(THEME_KEY, t);
    } catch {
      /* storage unavailable: the choice lasts for this page */
    }
    setTheme(t);
  };
  const dark = theme ? theme === "dark" : window.matchMedia?.("(prefers-color-scheme: dark)").matches;
  return (
    <button type="button" className={k.linkish} onClick={() => set(dark ? "light" : "dark")} data-testid="theme">
      {dark ? "Light" : "Dark"}
    </button>
  );
}
