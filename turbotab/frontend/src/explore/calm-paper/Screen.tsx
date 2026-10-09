/**
 * The paper (#/paper; dev /lab/calm/paper): the manuscript is the interface. The card column is
 * the manuscript itself, by its STROBE-nut sections; the open question stands where its sentence
 * will stand and opens there into the kit's card, with Continue at its foot. A recorded sentence
 * is the way back to its question; a blank is the way forward. The canvas keeps its place and its
 * width on the right (FOUNDATION §3), the chain stays on top, and the rail position stays empty,
 * because the paper is always on screen.
 *
 * Everything shown is the kit's: its parts, its classes, its copy and the walk's manuscript. This
 * file decides only where things sit and how a person moves through them (FOUNDATION §6).
 */
import { useEffect, useRef, useState, type ReactNode, type RefObject } from "react";
import { Canvas, Card, Chain, Footer, Plain, RESULT_STEPS, STEP_BY_ID, ThemeSwitch, kit as k, useWalk, type Entry, type WalkApi } from "../calm-kit";

/** The manuscript's results paragraph: after the lock, the results card opens beneath it. */
const RESULTS_ENTRY = "results:table2";

export function Screen() {
  const walk = useWalk();
  // On narrow screens the zones stack; the canvas then follows the open question inside the
  // paper instead of waiting below the whole manuscript.
  const narrow = useMedia("(max-width: 900px)");
  const paper = useRef<HTMLElement>(null);
  useOpenSlotInView(walk.state.open, paper);
  const canvas = <Canvas walk={walk} />;
  return (
    <div className={k.page} data-testid="shell">
      <header className={k.top}>
        <span>
          <a className={k.brand} href="#/">
            TurboTab
          </a>
          <span style={{ color: "var(--muted)", marginLeft: 10, fontSize: 15 }}>The paper</span>
        </span>
        <div className={k.topright}>
          <button type="button" className={k.linkish} onClick={walk.reset} data-testid="proto-reset">
            Start over
          </button>
          <ThemeSwitch />
        </div>
      </header>
      <Chain walk={walk} />
      {/* The kit's three zones with the rail position left empty: the canvas keeps the width the
          reference layout gives it. */}
      <div className={k.zones} data-manuscript="rail">
        <div className={k.zoneCard}>
          <Paper walk={walk} paperRef={paper} inlineCanvas={narrow ? canvas : null} />
        </div>
        {narrow ? null : <div className={k.zoneCanvas}>{canvas}</div>}
      </div>
      <Footer walk={walk} />
    </div>
  );
}

// ── the paper ────────────────────────────────────────────────────────────────

function Paper({ walk, paperRef, inlineCanvas }: { walk: WalkApi; paperRef: RefObject<HTMLElement | null>; inlineCanvas: ReactNode }) {
  const open = walk.state.open;
  const atResults = (RESULT_STEPS as readonly string[]).includes(open);
  const openSlot = walk.step?.slot ?? null;
  const card = (key: string) => (
    <li key={key} data-open="true" data-testid="open-slot" style={{ paddingBlock: "10px 18px" }}>
      <div style={{ display: "grid", gap: 24 }}>
        <Card walk={walk} />
        {inlineCanvas}
      </div>
    </li>
  );
  return (
    <article aria-label="Manuscript" data-testid="manuscript" ref={paperRef}>
      <div className={k.msHead}>
        <h2>Manuscript</h2>
      </div>
      {walk.manuscript.map((sec) => (
        <section key={sec.id} className={k.msSection} data-section={sec.id}>
          <h3>
            {sec.title} <small>{sec.item}</small>
          </h3>
          <ul className={k.msList}>
            {paragraphs(sec.entries).map((e) =>
              e.id === openSlot ? (
                card(e.id)
              ) : (
                <Paragraph key={e.id} entry={e} walk={walk} current={atResults && e.id === RESULTS_ENTRY} />
              ),
            )}
            {sec.id === "results" && atResults ? card("results-card") : null}
          </ul>
        </section>
      ))}
    </article>
  );
}

/** The paper reads as a paper: consecutive paragraphs that wait on the same earlier answer under
 *  the same head ("Adjustment set", "Column role") stand as one line until they can be written. */
function paragraphs(entries: Entry[]): Entry[] {
  return entries.filter((e, i) => {
    const prev = entries[i - 1];
    return !(prev && e.kind === "waiting" && prev.kind === "waiting" && prev.head === e.head);
  });
}

/** One paragraph of the manuscript, as the kit's manuscript draws it: a recorded phrase reopens its
 *  question in place, a blank opens it, a waiting one says so. */
function Paragraph({ entry: e, walk, current }: { entry: Entry; walk: WalkApi; current: boolean }) {
  let body: ReactNode;
  if (e.kind === "recorded" && e.step && !current) {
    body = (
      <button type="button" className={k.msPhrase} onClick={() => walk.open(e.step!)} title="Change this answer">
        <Plain text={e.sentence} />
      </button>
    );
  } else if (e.kind === "recorded" || e.kind === "stated") {
    body = <Plain text={e.sentence} />;
  } else if (e.kind === "blank" && e.step) {
    body = (
      <button type="button" className={k.msBlank} onClick={() => walk.open(e.step!)} data-testid={`blank-${e.id}`}>
        Choose: <Plain text={STEP_BY_ID[e.step]?.question ?? ""} />
      </button>
    );
  } else {
    body = <span className={k.msWaiting}>Waits for an earlier answer.</span>;
  }
  return (
    <li className={k.msEntry} data-kind={e.kind} data-newest={e.newest || undefined} data-testid={`ms-${e.id}`}>
      <span className={k.msHeadLine}>{e.head}</span>
      {body}
      {e.afterLock ? <span className={k.msAfter}>Changed after the estimates were seen.</span> : null}
    </li>
  );
}

// ── moving through the paper ─────────────────────────────────────────────────

/** The walk's next question is often in another section of the paper: bring the open slot into
 *  view (with its section's heading when that is close above it) unless it already sits in the
 *  upper part of the screen, and move focus to its question. On first load only the scroll. */
function useOpenSlotInView(open: string, root: RefObject<HTMLElement | null>) {
  const last = useRef<string | null>(null);
  useEffect(() => {
    if (last.current === open) return;
    const first = last.current === null;
    const place = () => {
      last.current = open;
      const slot = root.current?.querySelector<HTMLElement>('[data-open="true"]');
      if (!slot) return;
      if (!first) slot.querySelector<HTMLElement>("h1")?.focus({ preventScroll: true });
      const vh = window.innerHeight;
      const top = slot.getBoundingClientRect().top;
      if (top >= 8 && top <= vh * 0.4) return;
      const head = slot.closest("section")?.querySelector("h3");
      const headTop = head ? head.getBoundingClientRect().top : top;
      const anchor = top - headTop <= vh * 0.25 ? headTop : top;
      const still = first || !!window.matchMedia?.("(prefers-reduced-motion: reduce)").matches;
      window.scrollTo({ top: Math.max(0, window.scrollY + anchor - 16), behavior: still ? "auto" : "smooth" });
    };
    // On first load the page's own scroll-to-top runs after this effect; place the slot after it.
    if (!first) {
      place();
      return;
    }
    const frame = window.requestAnimationFrame(place);
    return () => window.cancelAnimationFrame(frame);
  }, [open, root]);
}

function useMedia(query: string): boolean {
  const [on, setOn] = useState(() => typeof window !== "undefined" && !!window.matchMedia?.(query).matches);
  useEffect(() => {
    const mq = window.matchMedia?.(query);
    if (!mq) return;
    const change = () => setOn(mq.matches);
    mq.addEventListener("change", change);
    return () => mq.removeEventListener("change", change);
  }, [query]);
  return on;
}
