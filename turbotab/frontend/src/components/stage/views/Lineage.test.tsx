/**
 * The lineage's narrowest clear width (P6): below `lineageMinWidth` the adjusted lane's pills are
 * squeezed narrower than their labels and the labels run into their neighbors, so the live Columns
 * card scrolls instead (stage/LiveScenes.tsx). The truth is the lineage drawn with all the room it
 * wants: at the minimum width every pill is as wide as there, and one pixel narrower it is not.
 * The lineage is the server's NHANES residual preview (src/mocks/m1-stage-fixture.json).
 */
import { act, render } from "@testing-library/react";
import { afterEach, beforeEach, vi } from "vitest";
import type { LineageView, PreviewResult } from "../../../api/m1-stage-types";
import stageFixture from "../../../mocks/m1-stage-fixture.json";
import { Lineage, lineageMinWidth } from "./Lineage";

const energy = (stageFixture as unknown as { previews: { energy_adjustment: { decision: { method?: string }; body: PreviewResult }[] } })
  .previews.energy_adjustment;
const LINEAGE = (energy.find((x) => x.decision.method === "residual")!.body.views.find((v) => v.kind === "lineage") as LineageView).after;

let observed: ResizeObserverCallback[] = [];
beforeEach(() => {
  observed = [];
  vi.stubGlobal(
    "ResizeObserver",
    class {
      constructor(private cb: ResizeObserverCallback) {}
      observe() {
        observed.push(this.cb);
      }
      unobserve() {}
      disconnect() {}
    },
  );
});
afterEach(() => vi.unstubAllGlobals());

/** The adjusted lane's pill widths, drawn in a box `width` wide. */
function pills(width: number): number[] {
  const r = render(<Lineage lineage={LINEAGE} />);
  act(() => {
    for (const cb of observed) cb([{ contentRect: { width, height: 420 } } as ResizeObserverEntry], {} as ResizeObserver);
  });
  const widths = [...r.container.querySelectorAll('rect[rx="5"]')].map((x) => Number(x.getAttribute("width")));
  r.unmount();
  observed = [];
  return widths;
}

describe("the lineage's narrowest clear width", () => {
  it("draws every pill as wide as it wants at the minimum width, and squeezes them one pixel below", () => {
    const min = lineageMinWidth(LINEAGE);
    const free = pills(4000);
    expect(free.length).toBeGreaterThan(0);
    expect(pills(min)).toEqual(free);
    const squeezed = pills(min - 1);
    expect(squeezed.length).toBe(free.length);
    for (let i = 0; i < free.length; i++) expect(squeezed[i]!).toBeLessThan(free[i]!);
  });
});
