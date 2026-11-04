/**
 * G12 (the owner's Q3): a way forward that names no decision names the card it opens (`target`,
 * core/exit_target.py), and in the quest shell, which opens cards (`OpenCard`), pressing it opens
 * that card instead of only dismissing the note. A way forward with no target, and every way
 * forward where no shell opens cards (Classic), is the caller's press, as before.
 *
 * The refusal is in the served shape (core/decisions.py `Refusal.to_dict`): a target is the
 * engine's `{stage, key}`, and a way forward that names none is served without one.
 */
import { fireEvent, render, screen } from "@testing-library/react";
import type { Refusal } from "../../api/schema";
import { RefusalNote } from "./Refusal";
import { OpenCard, targetOf } from "./target";

const TARGETED = {
  label: "Answer the purpose question first",
  decision: null,
  target: { stage: "question", key: "purpose" },
};
const STAGE_ONLY = { label: "Triage the open noticings in Results", decision: null, target: { stage: "results", key: null } };
const KEEP = { label: "Keep the answers as they are", decision: null };
const REFUSAL = {
  error: { code: "not_yet", message: "Your goal comes before this one.", exits: [TARGETED, STAGE_ONLY, KEEP] },
} as unknown as Refusal;

const button = (label: string) => screen.getByRole("button", { name: label });

describe("targetOf", () => {
  it("reads the served target, and none where the way forward names none or names it in no served shape", () => {
    expect(targetOf(TARGETED)).toEqual(TARGETED.target);
    expect(targetOf(STAGE_ONLY)).toEqual(STAGE_ONLY.target);
    expect(targetOf(KEEP)).toBeNull();
    expect(targetOf({ ...KEEP, target: null })).toBeNull();
    expect(targetOf({ ...KEEP, target: "purpose" })).toBeNull();
    expect(targetOf({ ...KEEP, target: { key: "purpose" } })).toBeNull();
    expect(targetOf(null)).toBeNull();
  });
});

describe("RefusalNote's ways forward that name no decision", () => {
  it("opens the card each names where the shell opens cards, and closes the note", () => {
    const open = vi.fn();
    const onExit = vi.fn();
    const onDismiss = vi.fn();
    render(
      <OpenCard.Provider value={open}>
        <RefusalNote refusal={REFUSAL} onExit={onExit} onDismiss={onDismiss} />
      </OpenCard.Provider>,
    );
    fireEvent.click(button(TARGETED.label));
    expect(open).toHaveBeenCalledWith(TARGETED.target);
    expect(onDismiss).toHaveBeenCalledTimes(1);
    fireEvent.click(button(STAGE_ONLY.label));
    expect(open).toHaveBeenLastCalledWith(STAGE_ONLY.target);
    expect(onExit).not.toHaveBeenCalled();
    // Each says where it goes, as the engine names it.
    expect(button(TARGETED.label)).toHaveAttribute("data-target", "question:purpose");
    expect(button(STAGE_ONLY.label)).toHaveAttribute("data-target", "results");
  });

  it("leaves a way forward with no target to the caller's press", () => {
    const open = vi.fn();
    const onExit = vi.fn();
    render(
      <OpenCard.Provider value={open}>
        <RefusalNote refusal={REFUSAL} onExit={onExit} onDismiss={vi.fn()} />
      </OpenCard.Provider>,
    );
    fireEvent.click(button(KEEP.label));
    expect(onExit).toHaveBeenCalledWith(KEEP);
    expect(open).not.toHaveBeenCalled();
    expect(button(KEEP.label)).not.toHaveAttribute("data-target");
  });

  it("leaves every way forward to the caller's press where no shell opens cards", () => {
    const onExit = vi.fn();
    render(<RefusalNote refusal={REFUSAL} onExit={onExit} onDismiss={vi.fn()} />);
    fireEvent.click(button(TARGETED.label));
    expect(onExit).toHaveBeenCalledWith(TARGETED);
    expect(button(TARGETED.label)).not.toHaveAttribute("data-target");
  });
});
