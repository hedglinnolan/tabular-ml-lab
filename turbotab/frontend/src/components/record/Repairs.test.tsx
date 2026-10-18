/**
 * A repair still to decide carries no SETTLED badge (the walkthrough of 2026-10-17, minor 19): the
 * pack's evidence status, "settled", read as the repair being settled on a card that asks for it.
 * A convention or a dispute in the field is still badged. The finding is the real server's (the
 * clinical journey's impossible-versus-extreme values, src/mocks/fixtures), served with its
 * evidence SETTLED and three repairs, none recorded.
 */
import { render, screen, within } from "@testing-library/react";
import type { Finding } from "../../api/schema";
import type { M3Fixture } from "../../mocks/m3";
import { StageFocusProvider } from "../../state/focus";
import { RepairsSection } from "./Repairs";

const FIXTURES = import.meta.glob<M3Fixture>("../../mocks/fixtures/m3-clinical.json", { eager: true, import: "default" });
const ID = "pack::clinical::impossible_vs_extreme";

/** The first finding served under `id`, wherever the capture holds it. */
function served(o: unknown): Finding | null {
  if (Array.isArray(o)) {
    for (const v of o) {
      const hit = served(v);
      if (hit) return hit;
    }
  } else if (o && typeof o === "object") {
    const rec = o as Record<string, unknown>;
    if (rec.id === ID && Array.isArray(rec.repairs)) return rec as unknown as Finding;
    for (const v of Object.values(rec)) {
      const hit = served(v);
      if (hit) return hit;
    }
  }
  return null;
}
const FINDING = served(Object.values(FIXTURES)[0])!;

function mount(finding: Finding) {
  render(
    <StageFocusProvider>
      <RepairsSection
        findings={[finding]}
        decisions={[]}
        entry={undefined}
        deferTarget={() => null}
        subject={(k) => k}
        record={() => {}}
        pending={false}
        answerAt={null}
      />
    </StageFocusProvider>,
  );
  return screen.getByTestId(`repair-${finding.id}`);
}

describe("a repair still to decide (minor 19)", () => {
  it("carries no SETTLED badge", () => {
    expect(FINDING.evidence?.status.toLowerCase()).toBe("settled");
    expect(FINDING.repairs.length).toBeGreaterThan(0);
    const card = mount(FINDING);
    expect(within(card).queryByText(/^settled$/i)).toBeNull();
  });

  it("keeps the badge of a convention or a dispute in the field", () => {
    for (const status of ["CONVENTION", "DISPUTED"]) {
      const card = mount({ ...FINDING, id: `${ID}-${status}`, evidence: { ...FINDING.evidence!, status } } as Finding);
      expect(within(card).getByText(status)).toBeInTheDocument();
    }
  });
});
