import { act, render } from "@testing-library/react";
import {
  HOVER_GRACE_MS,
  INITIAL_FOCUS,
  LIVE,
  StageFocusProvider,
  effectiveFocus,
  focusReducer,
  useStageFocus,
  type FocusAction,
  type FocusState,
  type StageFocus,
} from "./focus";

const residual: StageFocus = {
  kind: "option",
  decision: {
    kind: "set_energy_adjustment",
    method: "residual",
    energy_column: "kcal",
    nutrients: ["protein"],
    log_transform: false,
    strata: null,
  },
  label: "Residual method",
};
const density: StageFocus = {
  kind: "option",
  decision: {
    kind: "set_energy_adjustment",
    method: "density",
    energy_column: "kcal",
    nutrients: ["protein"],
    log_transform: false,
    strata: null,
  },
  label: "Density alone",
};
const rows: StageFocus = { kind: "banner", segment: "rows" };
const finding: StageFocus = { kind: "finding", findingId: "energy" };

function run(...actions: FocusAction[]): FocusState {
  return actions.reduce(focusReducer, INITIAL_FOCUS);
}

describe("focusReducer", () => {
  it("starts live", () => {
    expect(effectiveFocus(INITIAL_FOCUS)).toEqual(LIVE);
  });

  it("shows a pointer preview over what is held, and falls back when the pointer leaves", () => {
    const s = run({ type: "hold", focus: residual }, { type: "hover", focus: density });
    expect(effectiveFocus(s)).toEqual(density);
    expect(effectiveFocus(focusReducer(s, { type: "unhover" }))).toEqual(residual);
  });

  it("lets a deliberate act replace a lingering pointer preview", () => {
    const s = run({ type: "hover", focus: density }, { type: "hold", focus: finding });
    expect(s.hover).toBeNull();
    expect(effectiveFocus(s)).toEqual(finding);
  });

  it("releases only what is still held", () => {
    const held = run({ type: "hold", focus: residual });
    expect(effectiveFocus(focusReducer(held, { type: "release", focus: residual }))).toEqual(LIVE);
    // Focus moved to another option before the first one's blur arrived: nothing to release.
    const moved = run({ type: "hold", focus: residual }, { type: "hold", focus: density });
    expect(effectiveFocus(focusReducer(moved, { type: "release", focus: residual }))).toEqual(
      density,
    );
  });

  it("toggles a banner segment on, and off again with a second press", () => {
    const on = run({ type: "toggle", focus: rows });
    expect(effectiveFocus(on)).toEqual(rows);
    expect(effectiveFocus(focusReducer(on, { type: "toggle", focus: rows }))).toEqual(LIVE);
    // From another focus, a press switches straight to the segment.
    const from = run({ type: "hold", focus: residual }, { type: "toggle", focus: rows });
    expect(effectiveFocus(from)).toEqual(rows);
  });

  it("compares options by their decision, not by object identity", () => {
    const copy: StageFocus = JSON.parse(JSON.stringify(residual)) as StageFocus;
    const s = run({ type: "hold", focus: residual });
    expect(focusReducer(s, { type: "hold", focus: copy })).toBe(s);
  });

  it("resets to live", () => {
    const s = run({ type: "hold", focus: residual }, { type: "hover", focus: density });
    expect(effectiveFocus(focusReducer(s, { type: "reset" }))).toEqual(LIVE);
  });
});

describe("StageFocusProvider", () => {
  it("keeps a pointer preview through the grace period, then drops it", () => {
    vi.useFakeTimers();
    let api: ReturnType<typeof useStageFocus> | null = null;
    function Probe() {
      api = useStageFocus();
      return <span data-testid="kind">{api.focus.kind}</span>;
    }
    const { getByTestId } = render(
      <StageFocusProvider>
        <Probe />
      </StageFocusProvider>,
    );
    act(() => api!.preview(residual));
    expect(getByTestId("kind").textContent).toBe("option");
    act(() => api!.endPreview());
    act(() => vi.advanceTimersByTime(HOVER_GRACE_MS - 20));
    expect(getByTestId("kind").textContent).toBe("option");
    // The pointer reached the stage in time: the preview stays.
    act(() => api!.keepPreview());
    act(() => vi.advanceTimersByTime(HOVER_GRACE_MS * 2));
    expect(getByTestId("kind").textContent).toBe("option");
    act(() => api!.endPreview());
    act(() => vi.advanceTimersByTime(HOVER_GRACE_MS + 1));
    expect(getByTestId("kind").textContent).toBe("live");
    vi.useRealTimers();
  });
});
