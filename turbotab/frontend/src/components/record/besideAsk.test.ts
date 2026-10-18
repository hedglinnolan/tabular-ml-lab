/**
 * Where a refusal's readings are the card's to say (W1-W round 1, P4): the stage keeps quiet about a
 * refused option only where the card that records it asks those readings beside it, its question
 * the open step with the ledger's ask card (`askedBeside`). Which kinds each such question records
 * is the engine's (core/ask.py `CONSUMERS`, core/sequence.py `question_of`), written into the
 * shell's contract fixture by turbotab/core/tests/test_quest_renderer_contract.py; `ASKED_BY` is
 * pinned to it here.
 */
import { describe, expect, it } from "vitest";
import type { Decision, ProjectView } from "../../api/schema";
import { CONTRACT } from "../../quest/fixtures/contract";
import { ASKED_BY, askedBeside } from "./besideAsk";

const step = (key: string, status: string, ask: boolean) =>
  ({ key, status, ask: ask ? { question: key } : null }) as unknown as ProjectView["interview"][number];
const families = { kind: "select_models", models: ["ridge"] } as Decision;
const use = { kind: "set_intended_use", use: "risk_estimation", subgroups: ["bp_sys"] } as unknown as Decision;

describe("the questions whose card asks a refusal's readings", () => {
  it("are the engine's, each with the kinds it records", () => {
    expect(ASKED_BY).toEqual(CONTRACT.asked_by);
  });

  it("ask them only where the decision's question is the open step with the ask card", () => {
    expect(askedBeside({ interview: [step("models", "open", true)] }, families)).toBe(true);
    // Its question answered, or open with no ask card: no card beside it asks those readings.
    expect(askedBeside({ interview: [step("models", "answered", true)] }, families)).toBe(false);
    expect(askedBeside({ interview: [step("models", "open", false)] }, families)).toBe(false);
    // A declaration's card asks none, even while another question's ask card is open.
    expect(askedBeside({ interview: [step("models", "open", true)] }, use)).toBe(false);
  });
});
