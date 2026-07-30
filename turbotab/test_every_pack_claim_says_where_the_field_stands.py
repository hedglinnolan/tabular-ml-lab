"""`GUIDED-047` — the evidence badge.

`DOMAIN_SCIENCE.md` §01.1. All four research threads arrived at the same
recommendation without being asked for it, and the clinical one says why:

> *"That single design decision is what would make TurboTab trustworthy to a
> methodologist, because it makes the tool's epistemic position legible rather
> than uniformly confident."*

Every advisory a pack produces now carries `evidence_status` — SETTLED,
CONVENTION or DISPUTED — and a `source` naming a research file and a section in
it.

## It sharpens the three markers rather than replacing them

`derived`/`convention`/`offered` describe **the app's** confidence.
SETTLED/CONVENTION/DISPUTED describe **the field's**, and the second is the one
a reviewer can check.

They are a compatibility table, not a translation, and the case that makes that
necessary is `offered`. Building this, the first version of `MARKER_STATUS`
forbade SETTLED with `offered` — and it was wrong, in a way §01.2 names
directly: *there is a class of thing the app must detect and must not act on.*
Pooled QC rows are not participants; that is settled, not a convention, and the
app still only *offers* the exclusion, because acting on a high-confidence
detection whose consequence is irreversible if wrong is what every pack's
`hard_stops` list forbids. **Settled science and a withheld hand are
compatible.**

## The three rendering obligations, and where each is enforced

* **SETTLED** may pre-select with its reason shown.
* **CONVENTION** may pre-select and must be stated *as* convention.
* **DISPUTED** is never defaulted silently; both positions stated.

The third is enforced at construction — a DISPUTED prior whose marker
pre-selects raises — because the badge saying one thing while the interface does
another is worse than no badge. The user acts on the interface.

## What the gate does not check

`docs/turbotab/tools/evidence.py check` resolves the file and the section. It
does **not** check that the claim is faithful to what it cites. That limit is
exactly `ledger.py check`'s — it enforces that a `FIXED` row *names* a test and
cannot tell whether the test is any good — and it is stated in the tool, in the
design language, and here, because a gate whose limit is not stated gets read as
a guarantee.
"""
from __future__ import annotations

import os
import re
import sys
from pathlib import Path

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from turbotab import packs as P                                       # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
RESEARCH = ROOT / "docs" / "turbotab" / "research"

ALL_PRIORS = [(key, prior) for key, pack in P.PACKS.items()
              for prior in pack.priors]


def _ids(pair):
    return f"{pair[0]}/{pair[1].question}"


# ── every claim is badged, and every badge resolves ─────────────────────────

@pytest.mark.parametrize("pair", ALL_PRIORS, ids=_ids)
def test_every_pack_prior_carries_a_status_and_a_source(pair):
    _, prior = pair
    assert prior.evidence is not None
    assert prior.evidence.status in P.EVIDENCE_STATUSES
    assert prior.evidence.source


@pytest.mark.parametrize("pair", ALL_PRIORS, ids=_ids)
def test_every_source_resolves_to_a_real_section_of_a_real_file(pair):
    """A citation nobody can follow is a citation nobody can check."""
    _, prior = pair
    filename, _, section = prior.evidence.source.partition("#")
    path = ROOT / "docs" / "turbotab" / filename
    assert path.exists(), f"{filename} does not exist"
    headings = {m.group(1).strip() for m in
                re.finditer(r"^#{1,6}\s+(.*?)\s*$", path.read_text(), re.M)}
    assert section in headings, (
        f"{filename} has no section {section!r}")


def test_the_priors_span_all_three_statuses():
    """A badge that only ever reads SETTLED is a badge nobody is reading.

    `GUIDED-045`'s axis applied here: a check whose output never varies has a
    pass set no wider than "the field exists". Two DISPUTED claims and one
    CONVENTION are the honest state of these ten priors, and the day they all
    read SETTLED somebody has been rounding.
    """
    statuses = {p.evidence.status for _, p in ALL_PRIORS}
    assert statuses == set(P.EVIDENCE_STATUSES), (
        f"the ten priors span only {sorted(statuses)}; a badge with one value "
        f"is not a badge")


# ── the rendering obligations ───────────────────────────────────────────────

def test_a_disputed_claim_may_never_be_pre_selected():
    """The obligation that matters, enforced at construction.

    A DISPUTED badge beside a pre-selected default is the app saying the field
    disagrees and choosing a side anyway — and the user acts on the choice, not
    on the badge.
    """
    for marker in ("derived", "convention"):
        with pytest.raises(P.PackError, match="never defaulted silently"):
            P.Prior(question="q", marker=marker,
                    reason="a reason long enough to satisfy the other rule here",
                    evidence=P.Evidence(
                        status=P.DISPUTED,
                        source="research/GENOMICS_PACK.md#04 · Normalization — no default asserted",
                        both_sides="one side; and the other side"))


def test_a_disputed_claim_must_state_both_positions():
    """One side stated under a DISPUTED badge is the app picking a side while
    wearing a badge that says it has not."""
    with pytest.raises(P.EvidenceError, match="must state both positions"):
        P.Evidence(status=P.DISPUTED,
                   source="research/GENOMICS_PACK.md#04 · Normalization — no default asserted")


def test_both_sides_belongs_to_disputed_alone():
    """On a settled claim it invents a controversy."""
    with pytest.raises(P.EvidenceError, match="belongs to DISPUTED only"):
        P.Evidence(status=P.SETTLED,
                   source="research/METABOLOMICS_PACK.md#03 · Missing data",
                   both_sides="a controversy nobody is having")


def test_the_disputed_priors_actually_state_both_sides():
    """Not just that the field is required — that it says something."""
    disputed = [p for _, p in ALL_PRIORS if p.evidence.status == P.DISPUTED]
    assert disputed, "no DISPUTED prior; the obligation is untested on real data"
    for prior in disputed:
        assert len(prior.evidence.both_sides) > 80, (
            f"{prior.question}: both positions in "
            f"{len(prior.evidence.both_sides)} characters is one position with "
            f"a caveat")


def test_settled_and_offered_are_compatible():
    """§01.2's class, asserted so the first version's mistake cannot return.

    *There is a class of thing the app must detect and must not act on.* Pooled
    QC rows are the case: settled science, withheld hand.
    """
    qc = next(p for k, p in ALL_PRIORS if p.question == "qc_rows_excluded")
    assert qc.marker == "offered" and qc.evidence.status == P.SETTLED
    assert P.SETTLED in P.MARKER_STATUS["offered"]


def test_a_derived_prior_cannot_rest_on_a_convention():
    """The other direction. `derived` is the engine being certain, and it may
    only be certain where the field is."""
    with pytest.raises(P.PackError, match="disagree"):
        P.Prior(question="q", marker="derived",
                reason="a reason long enough to satisfy the other rule here",
                evidence=P.Evidence(
                    status=P.CONVENTION_STATUS,
                    source="research/METABOLOMICS_PACK.md#03 · Missing data"))


# ── it reaches a reader ─────────────────────────────────────────────────────

def test_the_badge_travels_to_the_interface():
    """`DRIVE-001`'s class applied to content: built, correct, and unreachable
    by a reader is the same as not built — and the entire argument for the
    badge is that it reaches a reader."""
    from fastapi.testclient import TestClient

    from turbotab import api
    client = TestClient(api.app)
    fixture = Path(__file__).parent / "sample_data" / "metabolomics_untargeted.csv"
    with open(fixture, "rb") as fh:
        pid = client.post("/project", files={
            "file": ("m.csv", fh, "text/csv")}).json()["id"]
    client.post(f"/project/{pid}/decision",
                json={"kind": "set_target", "payload": {"column": "responder"}})
    client.post(f"/project/{pid}/decision",
                json={"kind": "set_lens", "payload": {"lens": ["metabolomics"]}})

    plan = client.get(f"/project/{pid}/interview?step=preprocess").json()
    badged = [q for q in plan["questions"] if q.get("evidence_status")]
    assert badged, (
        "the metabolomics lens settles 300 columns and no question carries the "
        "badge for the claim it rests on")
    for q in badged:
        assert q["evidence_source"], f"{q['key']} is badged with no source"
        # THE OBLIGATION, over HTTP: nothing DISPUTED arrives pre-settled.
        if q["evidence_status"] == "DISPUTED":
            assert q["status"] == "asked", (
                f"{q['key']} is DISPUTED and was settled without asking")


def test_the_page_renders_the_badge_and_never_invents_one():
    """Rendered from what the server sends. A page that decided a claim was
    settled would be the interface taking the epistemic position the badge
    exists to make legible."""
    page = (ROOT / "turbotab" / "web" / "index.html").read_text(encoding="utf-8")
    assert len(page) > 20_000 and "renderAll" in page      # positive control
    body = page[page.index("function evidenceBadge"):]
    body = body[:body.index("\n  }")]
    assert "evidence_status" in body and "(q || {})" in body, (
        "the badge is not read from the question the server sent")
    for invented in ("SETTLED\"", "'SETTLED'"):
        assert invented not in body.replace('esc(status)', ''), (
            "the page composes a status of its own")
    assert ".badge.disputed" in page and "var(--stop)" not in page[
        page.index(".badge{"):page.index(".badge{") + 700], (
        "DISPUTED wears --stop, which §02 reserves for the blocker band alone")


# ── the gate ────────────────────────────────────────────────────────────────

def test_the_gate_runs_and_states_its_own_limit():
    """A gate whose limit is not stated gets read as a guarantee."""
    tool = (ROOT / "docs" / "turbotab" / "tools" / "evidence.py").read_text()
    assert "does not check the claim is faithful to it" in tool, (
        "the tool does not say what it cannot check")
    hook = (ROOT / ".githooks" / "pre-commit").read_text()
    assert "evidence.py check" in hook, (
        "the gate has no trigger, which is `GUIDED-019` — a check nothing "
        "triggers does not exist")


def test_no_verify_at_build_number_ships_as_a_constant():
    """The packs' own worst failure mode, named by the packs.

    All four threads hit an egress proxy and marked the numbers they could not
    read from primary text. Shipping one as a literal is how a wrong number
    reaches a manuscript.
    """
    marked = set()
    for path in RESEARCH.glob("*.md"):
        for m in re.finditer(r"\[verify-at-build:?\s*([^\]]*)\]", path.read_text()):
            marked.update(re.findall(r"\b(\d+(?:\.\d+)?)\s*%?", m.group(1)))
    assert marked, "no [verify-at-build] numbers found; the research changed"

    source = (ROOT / "turbotab" / "packs.py").read_text(encoding="utf-8")
    code = "\n".join(l.split("#", 1)[0] for l in source.split("\n")
                     if not l.strip().startswith("#"))
    code = re.sub(r'"(?:[^"\\]|\\.)*"', '""', code)
    code = re.sub(r"'(?:[^'\\]|\\.)*'", "''", code)
    leaked = [n for n in sorted(marked)
              if re.search(rf"(?<![\w.]){re.escape(n)}(?![\w.])", code)]
    assert not leaked, (
        f"these are marked [verify-at-build] in the research and appear as "
        f"literals in packs.py: {leaked}")
