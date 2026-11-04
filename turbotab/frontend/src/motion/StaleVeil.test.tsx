/**
 * The stale veil's tag (StaleVeil.tsx): each veiled state says its own words, unchanged for every
 * caller that gives no tag; a caller's `tag` replaces those words, in the tag and in the compact
 * dot alike, and only while veiled.
 */
import { render, screen } from "@testing-library/react";
import { StaleVeil, type VeilState } from "./StaleVeil";

const tagOf = (testId: string) => screen.getByTestId(testId).querySelector("[aria-live]")!.textContent;

describe("StaleVeil's tag", () => {
  it.each<[Exclude<VeilState, "fresh">, string]>([
    ["stale", "stale — an earlier answer changed"],
    ["recomputing", "recomputing"],
    ["stopped", "stale — you stopped the recompute"],
    ["failed", "stale — the recompute did not finish"],
  ])("says the %s state's own words where no tag is given", (state, words) => {
    render(
      <>
        <StaleVeil state={state} testId="veil">
          <p>body</p>
        </StaleVeil>
        <StaleVeil state={state} compact testId="dot">
          <p>body</p>
        </StaleVeil>
      </>,
    );
    expect(tagOf("veil")).toBe(words);
    expect(tagOf("dot")).toBe(words);
    expect(screen.getByTestId("dot").querySelector("[title]")).toHaveAttribute("title", words);
  });

  it("says the tag it is given in place of its state's words, veiled and inert as the state is", () => {
    const tag = "as the caller says it";
    render(
      <>
        <StaleVeil state="failed" tag={tag} testId="veil">
          <p>body</p>
        </StaleVeil>
        <StaleVeil state="failed" tag={tag} compact testId="dot">
          <p>body</p>
        </StaleVeil>
      </>,
    );
    expect(tagOf("veil")).toBe(tag);
    expect(tagOf("dot")).toBe(tag);
    expect(screen.getByTestId("dot").querySelector("[title]")).toHaveAttribute("title", tag);
    expect(screen.getByTestId("veil")).toHaveAttribute("data-veil", "failed");
    expect(screen.getByText("body", { selector: "[data-testid=veil] p" }).closest("[inert]")).not.toBeNull();
  });

  it("says no tag while fresh, given one or not", () => {
    render(
      <StaleVeil state="fresh" tag="as the caller says it" testId="veil">
        <p>body</p>
      </StaleVeil>,
    );
    expect(tagOf("veil")).toBe("");
    expect(screen.getByText("body").closest("[inert]")).toBeNull();
  });
});
