/**
 * The teaching layers (BLUEPRINT §11.5, M1_CONTRACT §5) that are not the question itself:
 *
 *   Taught   the app's words with data in chips (backticks) and terms that define themselves:
 *            a dotted underline, and a one-sentence card on hover or keyboard focus (§11.8)
 *   Drawer   layer 3, the concept drawer: the pack's sections, each with its evidence badge.
 *            Never needed to answer; it opens beside the Record, over the stage.
 */
import {
  Fragment,
  createContext,
  useContext,
  useEffect,
  useId,
  useLayoutEffect,
  useRef,
  useState,
  type ReactNode,
} from "react";
import { createPortal } from "react-dom";
import type { TeachingEntry, TeachingTerm } from "../../api/m1-types";
import s from "./teach.module.css";

const TermsContext = createContext<readonly TeachingTerm[]>([]);

export function TermsProvider({
  terms,
  children,
}: {
  terms: readonly TeachingTerm[] | undefined;
  children: ReactNode;
}) {
  return <TermsContext.Provider value={terms ?? []}>{children}</TermsContext.Provider>;
}

const CARD_WIDTH = 270;
const GUTTER = 12;

/**
 * A term's one-sentence card. It renders in a portal (never clipped by the Record's scroll box)
 * at fixed coordinates beside the term: on its preferred side when it fits, else the other side,
 * and always inside the viewport. "Above" is the default: under a question's lines sit its
 * options, and a card must never cover a control (DRIVE_RUBRIC §5.11).
 */
function Term({
  text,
  definition,
  placement,
}: {
  text: string;
  definition: string;
  placement: "above" | "below";
}) {
  const [open, setOpen] = useState(false);
  const id = useId();
  const anchor = useRef<HTMLSpanElement>(null);
  const card = useRef<HTMLSpanElement>(null);

  // Placed straight on the card's DOM node (measured after it renders), not through state.
  useLayoutEffect(() => {
    const el = card.current;
    if (!open || !el) return;
    const place = () => {
      const a = anchor.current?.getBoundingClientRect();
      if (!a) return;
      const h = el.offsetHeight || 96;
      const vw = window.innerWidth;
      const vh = window.innerHeight;
      const roomAbove = a.top - GUTTER;
      const roomBelow = vh - a.bottom - GUTTER;
      let side = placement;
      if (side === "above" && roomAbove < h + 6 && roomBelow > roomAbove) side = "below";
      if (side === "below" && roomBelow < h + 6 && roomAbove > roomBelow) side = "above";
      const top = side === "above" ? a.top - 6 - h : a.bottom + 6;
      const left = Math.max(GUTTER, Math.min(a.left, vw - CARD_WIDTH - GUTTER));
      el.style.top = `${Math.max(GUTTER, top)}px`;
      el.style.left = `${left}px`;
      el.dataset.side = side;
    };
    place();
    window.addEventListener("scroll", place, true);
    window.addEventListener("resize", place);
    return () => {
      window.removeEventListener("scroll", place, true);
      window.removeEventListener("resize", place);
    };
  }, [open, placement]);

  return (
    <span className={s.termWrap}>
      <span
        ref={anchor}
        className={s.term}
        tabIndex={0}
        aria-describedby={id}
        onMouseEnter={() => setOpen(true)}
        onMouseLeave={() => setOpen(false)}
        onFocus={() => setOpen(true)}
        onBlur={() => setOpen(false)}
        onKeyDown={(e) => {
          if (e.key === "Escape" && open) {
            e.stopPropagation();
            setOpen(false);
          }
        }}
        data-term={text.toLowerCase()}
      >
        {text}
      </span>
      {typeof document !== "undefined"
        ? createPortal(
            <span
              ref={card}
              id={id}
              role="tooltip"
              className={s.termCard}
              data-open={open || undefined}
              style={{ top: -9999, left: -9999, width: CARD_WIDTH }}
            >
              {definition}
            </span>,
            document.body,
          )
        : null}
    </span>
  );
}

const escape = (t: string) => t.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");

/**
 * Text in the app's voice. Backticked spans become data chips; the first appearance of each
 * of the question's terms becomes a self-defining term.
 */
export function Taught({
  text,
  placement = "above",
  terms: own,
}: {
  text: string;
  placement?: "above" | "below";
  terms?: readonly TeachingTerm[];
}) {
  const ctx = useContext(TermsContext);
  const terms = own ?? ctx;
  const defs = new Map(terms.map((t) => [t.term.toLowerCase(), t.definition]));
  const pattern =
    terms.length > 0
      ? new RegExp(
          `\\b(?:${[...terms]
            .sort((a, b) => b.term.length - a.term.length)
            .map((t) => escape(t.term))
            .join("|")})\\b`,
          "gi",
        )
      : null;
  const seen = new Set<string>();
  const out: ReactNode[] = [];
  text.split("`").forEach((part, i) => {
    if (i % 2 === 1) {
      out.push(
        <code key={i} className="v">
          {part}
        </code>,
      );
      return;
    }
    let last = 0;
    let k = 0;
    for (const m of pattern ? part.matchAll(pattern) : []) {
      const key = m[0].toLowerCase();
      const definition = defs.get(key);
      // Each term defines itself once per text; later mentions are plain words.
      if (definition === undefined || seen.has(key)) continue;
      seen.add(key);
      if (m.index > last)
        out.push(<Fragment key={`${i}-${k++}`}>{part.slice(last, m.index)}</Fragment>);
      out.push(
        <Term key={`${i}-${k++}`} text={m[0]} definition={definition} placement={placement} />,
      );
      last = m.index + m[0].length;
    }
    if (last < part.length) out.push(<Fragment key={`${i}-${k}`}>{part.slice(last)}</Fragment>);
  });
  return <>{out}</>;
}

/** A pack's badge: where the field stands (SETTLED · CONVENTION · DISPUTED). */
export function EvidenceBadge({ status, source }: { status: string; source?: string }) {
  return (
    <span className={s.badge} data-status={status.toLowerCase()} title={source}>
      {status.toUpperCase()}
    </span>
  );
}

/** Layer 3: the concept drawer. Instant (disclosure is not a consequence, §05.2). */
export function ConceptDrawer({ entry, onClose }: { entry: TeachingEntry; onClose: () => void }) {
  const ref = useRef<HTMLElement>(null);
  const headingId = useId();
  useEffect(() => {
    const opener = document.activeElement as HTMLElement | null;
    ref.current?.focus();
    return () => opener?.focus?.();
  }, []);
  if (!entry.drawer) return null;
  return (
    <aside
      ref={ref}
      className={s.drawer}
      role="dialog"
      aria-modal="false"
      aria-labelledby={headingId}
      tabIndex={-1}
      data-testid="concept-drawer"
      onKeyDown={(e) => {
        if (e.key === "Escape") {
          e.stopPropagation();
          onClose();
        }
      }}
    >
      <header className={s.drawerHead}>
        <div>
          <div className={s.drawerKicker}>Concept · never needed to answer</div>
          <h2 id={headingId} className={s.drawerTitle}>
            {entry.title}
          </h2>
        </div>
        <button type="button" className={s.close} onClick={onClose}>
          Close
        </button>
      </header>
      <div className={s.drawerBody}>
        {entry.drawer.sections.map((sec) => (
          <section key={sec.heading} className={s.section}>
            <h3 className={s.sectionHead}>
              {sec.heading}
              {sec.evidence ? <EvidenceBadge {...sec.evidence} /> : null}
            </h3>
            {sec.body.split(/\n{2,}/).map((para, i) => (
              <p key={i} className={s.sectionBody}>
                <Taught text={para} terms={entry.terms} />
              </p>
            ))}
            {sec.evidence ? <p className={s.source}>{sec.evidence.source}</p> : null}
          </section>
        ))}
      </div>
    </aside>
  );
}
