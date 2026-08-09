# L56 — continuation, after the interim report

**A and B are accepted.** Verified independently at `db41853` in an isolated worktree: 890 / 371,
register 182, six gates green, `partition_is_exhaustive()` **356 / 0 / 9 = 365** matching
`status==FIXED` exactly, `missing == 0`, and **93 passed / 0 skipped** across the five repaired-skip
files.

**Your divergence section corrected the prompt in three places and all three corrections are upheld.
Every one of them was my error, not a discovery you should have priced.**

- **A1's stated input.** `import-graph.json` holds **87** keys and `reverse-deps.json` **74**, all
  `ml/` · `utils/` · `pages/` · `models/`, with **zero** `turbotab/` entries — confirmed. I checked
  that those two files *existed* and inferred that they covered this package. That is asserting a
  presence without reading, which is the fifth entry in `PM_TRANSITION.md` §07 — an entry I had added
  one commit earlier. Building the walker was the right call.
- **A2 was a leading question.** There is one drive and it is already `scope="module"`. The real cause
  was one field, found by bisection. My framing presupposed a structure I had not opened.
- **B2 rested on a claim I quoted forward.** *"Exactly one fixture has a remainder"* is false, and
  `bound=2` was unreachable regardless — `_attention_stack(findings, decisions)` takes no bound
  argument, confirmed. Two lenses at the shipping bound is the correct repair, and narrowing to one
  would have been `GUIDED-097`'s rule reduced to one and reported green.

**Two things I checked myself rather than take on report.** Your probe: I planted a row in the
`file.py::same_stem` shape — RED naming it; total revert of the resolver **and its test file** — GREEN,
defect reproduced. (My first attempt reverted only the module and died on an `ImportError`, which
§08.1 does not accept; the total revert is the one that counts.) And `--pytest-args` on the unscopable
case returns **empty stdout and exit 2**, so a caller that ignores the exit status runs zero tests
rather than a confident subset. **That is the best single decision in A1.**

**Two notes, neither a defect.** The `354 / 363` partition in your report is a B1-era snapshot — I
measure 355 / 364 at `18a4cc1` and 356 / 365 at HEAD; the assertion holds at HEAD, which is what
matters, but stamp a partition count with the commit it was taken at. And you are right that
`not_pytest` at **9 against a cap of 10** must not be raised in the loop that trips it.

---

## The three rulings

### 1 · C1 — split it, and **not this loop**

**Split `predictions_for`.** Two consumers with opposite arity behind one accessor is *how*
`GUIDED-236` happened; scoping C1 to ROC alone leaves the decision curve reading a single-model
accessor, which is a second live instance of the defect **inside the loop that fixes the first**.

- The single-model path keeps its meaning under a name that says what it returns — it is *the
  best-calibratable run*, not *predictions*.
- A new multi-model accessor serves ROC and the decision curve.
- **The gate is that calibration's payload is byte-identical before and after, asserted as a test.**
  Not reviewed — tested.

**It is a Part-B-sized build with three figure consumers, so it becomes L57's Part B**, arriving with
this decision already made rather than as a question. Pricing it as a tail-end addition is how L54
lost a part.

### 2 · C2 — edit the prototype first and re-carry. Do not work around the carry

**Confirmed:** the first `<style>` block in `docs/turbotab/prototypes/interview-feed.html` and in
`turbotab/web/index.html` are **30,968 characters and byte-identical**, and `test_skeleton.py:698`
reads the prototype. The carry is the thing that keeps the two surfaces from diverging.

**One addition to the row's `act`: the validator asserts `--c1 != --accent` in BOTH files**, not only
the app — otherwise the next carry silently reintroduces it, which is this project's most-repeated
failure shape.

### 3 · Budget — **C2 + C3, then the full sweep, then the final report**

Take **C3 first if you prefer**; you called it independent and cheapest and I agree.

**C2 is free right now precisely because nothing draws multi-series yet.** `FIG_DRAW` holds two kinds
and `WEBC` is read at two lines in the multi-series path, so the ramp's distinguishing function is
currently reached by nothing. Doing C2 **before** C1 is correct sequencing, not a compromise — it is
the whole argument in `DRIVE-015`, and after C1 lands the same change costs a visual regression
surface it does not cost today.

**D, E and F go to L57 — filed and dated, not silently dropped.** F's reconnaissance in particular
should arrive in L57 with `TEST-063`'s guard pinned first, exactly as the prompt specifies.

**Then the full `turbotab/` sweep, once, on a quiet machine, with `ps` checked.** Treat ~1h35m as an
estimate: the only measured full sweep is **2:01:04**, so whatever this run reports becomes the new
baseline and should be recorded beside `AUDIT-040` as such.

---

## Standing for the rest of the loop

- **You still own `findings.json` and `register.json`.** I have written nothing to either since your
  report, and will not until you close the loop. Docs-only commits from me will say so.
- **A scoped run is never reported as a full run** — you have held this throughout; hold it in the
  final report too, per part.
- **`TEST-072` and `TEST-073` stay `OPEN` and are correctly filed.** `TEST-072`'s reasoning is
  accepted in full: an instrumented coverage run that takes four hours is not an improvement on a
  two-hour one, and the blind spot printing on **every** run is worth more than the row.
- **`AUDIT-040` stays `PARTIAL`.** 1,175s → 738s is an improvement, not a closure, and you said so
  first.
- **Report once more, in `AGENT_ONBOARD.md` §10's shape**, with the sweep number and what C2/C3
  actually landed. Lead with the divergence section again — it has now corrected the adjudicator in
  every loop it has been written for, and this loop it did it three times in one report.
