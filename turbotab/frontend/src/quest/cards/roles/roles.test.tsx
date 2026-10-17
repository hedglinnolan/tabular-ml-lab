/**
 * A column's own role card (the walkthrough of 2026-10-17, finding 1; BLUEPRINT §14 rule 2), on the
 * engine's own lines (fixture.json, written by turbotab/core/tests/test_quest_unconfirmed_roles.py):
 * one press records one `confirm_role` of the line's column, the role as proposed or another the
 * engine offers, and no press confirms several columns; how many columns wait is the quest log's.
 */
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { fireEvent, render, screen, waitFor } from "@testing-library/react";
import { afterEach, beforeEach, vi, type Mock } from "vitest";
import { keys } from "../../../api/queries";
import type { QuestLine, QuestLog, ReadItem, Refusal } from "../../../api/schema";
import { CONTRACT } from "../../fixtures/contract";
import FIXTURE from "./fixture.json";
import { RoleCard } from "./RoleCard";
import { isRoleLine, ownOptions, ownRefusal, roleItem, ROLE_LINE_ID, waitingRoles, waitingSentence } from "./roles";

const LINES = FIXTURE.lines as unknown as QuestLine[];
const lineFor = (column: string) => LINES.find((l) => roleItem(l)?.column === column)!;
const PID = "roles-test";

/** The Predict log the contract fixture holds, with the engine's role lines in Your data. */
function logWith(lines: QuestLine[]): QuestLog {
  const log = structuredClone(CONTRACT.logs.models_open.log) as QuestLog;
  const data = log.stages.find((s) => s.key === "data")!;
  data.lines = [...data.lines, ...lines];
  return log;
}

let fetchMock: Mock;
beforeEach(() => {
  fetchMock = vi.fn(() => new Promise(() => {}));
  vi.stubGlobal("fetch", fetchMock);
});
afterEach(() => {
  vi.unstubAllGlobals();
});

function mount(line: QuestLine, log: QuestLog = logWith(LINES)) {
  const qc = new QueryClient({ defaultOptions: { queries: { retry: false, staleTime: Infinity } } });
  qc.setQueryData(keys.quest(PID), log);
  return render(
    <QueryClientProvider client={qc}>
      <RoleCard pid={PID} line={line} />
    </QueryClientProvider>,
  );
}

/** Every decision the page posted, as sent. */
const posted = () =>
  fetchMock.mock.calls
    .filter(([url, init]) => String(url).includes("/decisions") && (init as RequestInit | undefined)?.method === "POST")
    .map(([, init]) => JSON.parse(String((init as RequestInit).body)));

describe("a column's role line", () => {
  it("is the engine's: its lines carry the card the dispatcher reads", () => {
    expect(FIXTURE.place.id).toBe(ROLE_LINE_ID);
    expect(LINES.length).toBeGreaterThan(1);
    for (const l of LINES) expect(isRoleLine(l), l.key).toBe(true);
    expect(isRoleLine({ source: "reading", id: "default:read-from-data" })).toBe(false);
    expect(isRoleLine({ source: "question", id: ROLE_LINE_ID })).toBe(false);
  });

  it("offers only its own column's confirm_role, the role as proposed first, never a block or another column", () => {
    const item = roleItem(lineFor("bp_sys"))!;
    const own = ownOptions(item);
    expect(own.map((o) => o.role)).toEqual(["covariate", ...FIXTURE.choices.covariate]);
    expect(own[0]!.confirms).toBe(true);
    const crowded: ReadItem = {
      ...item,
      options: [
        { label: "Confirm each of the 3 as shown", decision: { kind: "confirm_readings", items: [{ reading: "role", column: "bp_sys", value: "covariate" }, { reading: "role", column: "hdl", value: "covariate" }] } },
        { label: "`hdl` is a covariate", decision: { kind: "confirm_role", column: "hdl", role: "covariate" } },
        { label: "Change the roles", decision: null },
        ...item.options,
      ],
    };
    expect(ownOptions(crowded)).toEqual(own);
  });

  it("answers a refused press with no way forward that confirms several columns or another", () => {
    const refusal = {
      error: {
        code: "replay",
        message: "The capture answered otherwise here.",
        exits: [
          { label: "Confirm each of the 16 as shown", decision: { kind: "confirm_readings", items: [{ reading: "role", column: "bp_sys", value: "covariate" }, { reading: "role", column: "waist", value: "covariate" }] } },
          { label: "`waist` is a covariate", decision: { kind: "confirm_role", column: "waist", role: "covariate" } },
          { label: "`bp_sys` is left out of the models", decision: { kind: "confirm_role", column: "bp_sys", role: "excluded" } },
          { label: "Change the roles", decision: null },
        ],
      },
    } as unknown as Refusal;
    expect(ownRefusal(refusal, "bp_sys").error.exits.map((e) => e.label)).toEqual(["`bp_sys` is left out of the models", "Change the roles"]);
    expect(ownRefusal(refusal, "bp_sys").error.message).toBe(refusal.error.message);
  });

  it("says how many columns wait, from the served lines, and nothing once none does", () => {
    const log = logWith(LINES);
    expect(waitingRoles(log).map((l) => roleItem(l)!.column)).toEqual(["bp_sys", "imputed_waist", "cycle_begin_year"]);
    expect(waitingSentence(waitingRoles(log).length)).toBe("3 columns wait for their own confirmation and are not predictors yet.");
    expect(waitingSentence(1)).toBe("1 column waits for its own confirmation and is not a predictor yet.");
    expect(waitingSentence(0)).toBeNull();
  });
});

describe("the role card", () => {
  it("shows the column, its proposed role, the evidence and why it waits, and records one confirm_role per press", async () => {
    const line = lineFor("bp_sys");
    mount(line);
    const card = screen.getByTestId("role-bp_sys");
    expect(card).toHaveTextContent("The role of bp_sys");
    expect(card).toHaveTextContent("bp_sys proposed as a covariate");
    expect(screen.getByTestId("role-evidence")).toHaveTextContent(roleItem(line)!.evidence.replaceAll("`", ""));
    expect(screen.getByTestId("role-reason")).toHaveTextContent(line.reason!.replaceAll("`", ""));
    expect(screen.getByTestId("roles-waiting")).toHaveTextContent("3 columns wait for their own confirmation and are not predictors yet.");
    const buttons = screen.getAllByRole("button");
    expect(buttons.map((b) => b.textContent)).toEqual(["Confirm bp_sys as a covariate", "bp_sys is what you study", "bp_sys is left out of the models"]);
    fireEvent.click(screen.getByRole("button", { name: "Confirm bp_sys as a covariate" }));
    await waitFor(() => expect(posted()).toEqual([{ kind: "confirm_role", column: "bp_sys", role: "covariate" }]));
  });

  it("changes the role with one press too", async () => {
    mount(lineFor("cycle_begin_year"));
    fireEvent.click(screen.getByRole("button", { name: "cycle_begin_year is a covariate" }));
    await waitFor(() => expect(posted()).toEqual([{ kind: "confirm_role", column: "cycle_begin_year", role: "covariate" }]));
  });

  it("holds a confirmed column as it is, with the other roles still offered, and no count once none waits", () => {
    const waist = lineFor("waist");
    expect(waist.status).toBe("answered");
    mount(waist, logWith([waist]));
    expect(screen.getByTestId("role-waist")).toHaveTextContent("waist is a covariate");
    expect(screen.queryByRole("button", { name: /^Confirm/ })).toBeNull();
    expect(screen.getAllByRole("button")).toHaveLength(FIXTURE.choices.covariate.length);
    expect(screen.queryByTestId("roles-waiting")).toBeNull();
  });
});
