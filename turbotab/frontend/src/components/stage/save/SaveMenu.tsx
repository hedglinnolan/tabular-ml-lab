/**
 * Save any plot (BLUEPRINT §11.1.3): a labeled real state — now, the current step, with this
 * choice, or the before/after pair — as SVG or PNG (2×), journal-styled, captioned with the choice
 * that produced it. The press is acknowledged at the control with the file's name.
 */
import { useEffect, useId, useRef, useState } from "react";
import { fileName, saveFigure, type SaveFormat, type SaveWhich } from "./export";
import s from "./SaveMenu.module.css";

export interface SaveChoice {
  which: SaveWhich;
  label: string;
}

interface Props {
  /** Names the plot (aria, file name). */
  title: string;
  /** The states this view can save; null when it shows one recorded state only. */
  choices: SaveChoice[] | null;
  /** The initial choice: the state on screen. */
  current?: SaveWhich;
  build: (which: SaveWhich | "as-shown") => string;
}

export function SaveMenu({ title, choices, current, build }: Props) {
  const [open, setOpen] = useState(false);
  const [which, setWhich] = useState<SaveWhich>(current ?? "with");
  const [done, setDone] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);
  const root = useRef<HTMLDivElement>(null);
  const id = useId();

  useEffect(() => {
    if (!open) return;
    const close = (e: PointerEvent) => {
      if (!root.current?.contains(e.target as Node)) setOpen(false);
    };
    document.addEventListener("pointerdown", close);
    return () => document.removeEventListener("pointerdown", close);
  }, [open]);

  useEffect(() => {
    if (!done) return;
    const t = window.setTimeout(() => setDone(null), 2600);
    return () => window.clearTimeout(t);
  }, [done]);

  const toggle = () => {
    if (!open && current) setWhich(current);
    setOpen((o) => !o);
    setError(null);
  };

  const save = async (format: SaveFormat) => {
    const w = choices ? which : "as-shown";
    const name = fileName(title, w, format);
    try {
      await saveFigure(build(w), name, format);
      setDone(name);
      setOpen(false);
    } catch (e) {
      setError(e instanceof Error ? e.message : "The figure could not be saved.");
    }
  };

  return (
    <div
      className={s.root}
      ref={root}
      onKeyDown={(e) => {
        if (e.key === "Escape" && open) {
          e.stopPropagation();
          setOpen(false);
        }
      }}
    >
      <button
        type="button"
        className={s.button}
        aria-expanded={open}
        aria-controls={id}
        aria-label={`Save “${title.replace(/`/g, "")}”`}
        onClick={toggle}
        data-testid="save-button"
      >
        <span className={s.icon} aria-hidden="true" />
        Save
      </button>
      {done ? (
        <span className={s.done} role="status">
          saved {done}
        </span>
      ) : null}
      {open ? (
        <div className={s.menu} id={id} role="dialog" aria-label="Save this plot">
          {choices ? (
            <fieldset className={s.group}>
              <legend className={s.legend}>Which state</legend>
              {choices.map((c) => (
                <label key={c.which} className={s.choice}>
                  <input
                    type="radio"
                    name={`${id}-which`}
                    value={c.which}
                    checked={which === c.which}
                    onChange={() => setWhich(c.which)}
                  />
                  <span>{c.label}</span>
                </label>
              ))}
            </fieldset>
          ) : null}
          <div className={s.formats}>
            <button type="button" className={s.format} onClick={() => void save("svg")}>
              SVG
            </button>
            <button type="button" className={s.format} onClick={() => void save("png")}>
              PNG 2×
            </button>
          </div>
          <p className={s.hint}>Journal style: serif, grayscale, captioned with its provenance.</p>
          {error ? <p className={s.error}>{error}</p> : null}
        </div>
      ) : null}
    </div>
  );
}
