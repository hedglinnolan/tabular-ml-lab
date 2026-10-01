import { keyAction, nextIndex } from "./optionNav";

describe("nextIndex", () => {
  it("moves one option per arrow key and stops at the ends", () => {
    expect(nextIndex("ArrowDown", 0, 4)).toBe(1);
    expect(nextIndex("ArrowDown", 3, 4)).toBe(3);
    expect(nextIndex("ArrowUp", 0, 4)).toBe(0);
    expect(nextIndex("ArrowUp", 2, 4)).toBe(1);
    expect(nextIndex("ArrowRight", 1, 4)).toBe(2);
    expect(nextIndex("ArrowLeft", 1, 4)).toBe(0);
  });

  it("jumps to the first and last options", () => {
    expect(nextIndex("Home", 3, 6)).toBe(0);
    expect(nextIndex("End", 0, 6)).toBe(5);
    expect(nextIndex("PageDown", 2, 6)).toBe(5);
  });

  it("recovers from an index past the end (the list shrank)", () => {
    expect(nextIndex("ArrowUp", 9, 4)).toBe(2);
    expect(nextIndex("ArrowDown", -3, 4)).toBe(1);
  });

  it("ignores keys that are not movements, and empty lists", () => {
    expect(nextIndex("Enter", 1, 4)).toBeNull();
    expect(nextIndex("a", 1, 4)).toBeNull();
    expect(nextIndex("ArrowDown", 0, 0)).toBeNull();
  });
});

describe("keyAction", () => {
  it("records on Enter; Space is the stage's flip for a single choice", () => {
    expect(keyAction("Enter", "single")).toBe("record");
    expect(keyAction(" ", "single")).toBeNull();
  });

  it("toggles on Space and records on Enter for a multiple choice", () => {
    expect(keyAction(" ", "multi")).toBe("toggle");
    expect(keyAction("Enter", "multi")).toBe("record");
  });

  it("returns to your data on Escape", () => {
    expect(keyAction("Escape", "single")).toBe("escape");
    expect(keyAction("ArrowDown", "single")).toBeNull();
  });
});
