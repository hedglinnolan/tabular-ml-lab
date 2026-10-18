"""W1-C · the full tier runs the light real-server browser journeys (plan §5 W1, acceptance 6).

The real-server Playwright specs (m1-journey, m2-journeys, m3-no-dead-end) skip against the mock,
so the mock job never ran them: they had stopped pressing Fit without anyone seeing. The full tier
(.github/workflows/v2-full.yml) now has a job that builds the frontend, starts the real server
on a fresh TURBOTAB_HOME for each spec, and runs the light ones. The heavy journeys (metabolomics
and survey in m3-no-dead-end, about 27 and 31 minutes) are marked and stay out of it: they run only
with E2E_HEAVY=1, which the job never sets.
"""
from __future__ import annotations

import re
from pathlib import Path

import yaml  # installed with uvicorn[standard] (turbotab/server/requirements.txt)

REPO = Path(__file__).resolve().parents[3]
FULL = REPO / ".github" / "workflows" / "v2-full.yml"
E2E = REPO / "turbotab" / "frontend" / "e2e"
LIGHT_SPECS = ("m1-journey", "m2-journeys", "m3-no-dead-end")


def _job() -> dict:
    jobs = yaml.safe_load(FULL.read_text("utf-8"))["jobs"]
    found = [job for job in jobs.values()
             if any("E2E_BASE_URL" in (s.get("run") or "") for s in job["steps"])]
    assert len(found) == 1, list(jobs)
    return found[0]


def test_the_full_tier_runs_the_light_real_server_journeys_each_on_a_fresh_home():
    job = _job()
    script = "\n".join(s.get("run") or "" for s in job["steps"])
    assert "npm run build" in script  # the server serves turbotab/frontend/dist
    assert "python -m turbotab.server" in script
    assert re.search(r"TURBOTAB_HOME=\"?\$\(mktemp -d\)", script), script
    for spec in LIGHT_SPECS:
        assert spec in script, spec
    # The heavy journeys stay out: nothing in the job asks for them.
    assert "E2E_HEAVY" not in script and "E2E_HEAVY" not in str(job.get("env", {}))
    assert job.get("timeout-minutes", 0) >= 60


def test_the_heavy_journeys_are_marked_and_skip_without_e2e_heavy():
    spec = (E2E / "m3-no-dead-end.spec.ts").read_text("utf-8")
    heavy = re.findall(r'name: "([^"]+)",\n\s+file: [^\n]+\n\s+heavy: true', spec)
    assert sorted(heavy) == ["metabolomics", "survey"], heavy
    assert "process.env.E2E_HEAVY" in spec


def test_every_real_server_journey_presses_fit():
    """The real server serves no score before Fit (SIZING P0.8): each journey that fits presses it,
    through the Results' one button (e2e/fit.ts)."""
    helper = (E2E / "fit.ts").read_text("utf-8")
    assert 'getByTestId("press-fit")' in helper
    for spec in LIGHT_SPECS:
        text = (E2E / f"{spec}.spec.ts").read_text("utf-8")
        assert re.search(r"\b(pressFit|fitOnceComputed)\(page", text), spec


SRC = REPO / "turbotab" / "frontend" / "src"
# The frontend reads the scheduler's hold on the fit (StageStatus.held, W1-B or the shell).
HELD_READ = re.compile(r"""\.held\s*===\s*["']fit["']|data-testid=["']fit-held["']""")


def _renders_a_held_fit() -> list[str]:
    return sorted(str(path.relative_to(SRC)) for path in SRC.rglob("*.ts*")
                  if ".test." not in path.name and "mocks" not in path.parts
                  and HELD_READ.search(path.read_text("utf-8")))


def test_a_held_fit_is_pressed_never_skipped():
    """A journey whose fit the scheduler holds presses the held Fit the stage renders (W1-B's
    FitHeld, plan §1 B2): the frontend reads ``StageStatus.held``, and e2e/fit.ts no longer skips a
    held journey, so the light real-server journeys walk through a held fit."""
    helper = (E2E / "fit.ts").read_text("utf-8")
    assert "test.skip" not in helper and "HELD_WITHOUT_FIT" not in helper
    assert 'getByTestId("press-fit")' in helper
    rendering = _renders_a_held_fit()
    assert rendering, "nothing in the frontend renders a held fit"
    held = (SRC / "components" / "stage" / "results" / "Results.tsx").read_text("utf-8")
    # The held Fit is the same FitPress the helper presses, under its own wrapper.
    assert 'data-testid="fit-held"' in held and "<FitPress" in held


def test_the_real_server_defects_m1_measures_stay_visible():
    """The M1 journey's guarantees stay visible, never only annotated. Since the journey went
    Predict (b20fc210) it asks no energy question, whose preview showed the two stage defects the
    real server had (a term's card over an option, the distribution card's Save over the
    primary's): the journey says so, and asks the energy question nowhere. What it measures now is
    held as strictly as before or more: a screen over 300 KB is a known-defect fixme that names it
    on the real server and a failure on the mock; the primary's Save is measured on both backends,
    asserted uncovered on both (no longer only on the mock) and pressed by pointer on both."""
    spec = (E2E / "m1-journey.spec.ts").read_text("utf-8")
    assert 'test.describe.configure({ mode: "serial" })' in spec
    start = spec.index('test("every review screen stays under 300 KB"')
    body = spec[start:spec.index("\n});", start)]
    assert "test.fixme(over.length > 0" in body and "expect(" in body
    # The mock still fails on an oversized screen, as before.
    assert "if (!REAL) expect(bytes" in spec
    # The Save: measured, and asserted on both backends.
    assert "measurements.save_covered_by = saveCoveredBy;" in spec
    held = 'expect(saveCoveredBy, "nothing covers the primary\'s Save").toBeNull();'
    line = next(l for l in spec.splitlines() if held in l)
    assert line.strip() == held, "the Save's check is no longer guarded by a backend"
    assert "const openSave = async () => primarySave.click();" in spec
    # The term card's defect is named as not measured, because Predict asks no energy question,
    # which the journey asserts is not on the screen.
    assert "a term's card covering an option" in spec and "prediction does not ask that question" \
        in " ".join(spec.split())
    assert 'getByTestId("question-energy_adjustment")).toHaveCount(0)' in spec
