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
