"""The decoy queue (R1_PLAN.md R1-G, item 5): decoys generated from real candidates, planted beside
them in a grading queue, and the false-promotion rate of the two-referee gate over the verdicts.

* Each generator changes what its kind says and nothing else, and returns None where it does not
  apply; the truth is the construction.
* The queue holds every candidate and every decoy once, under neutral ids, shows a referee only the
  merits fields, keeps each decoy out of its sibling's batch, and seals its key by its SHA-256.
* The score's rate is the share of false decoys the gate promoted, its Wilson interval
  statsmodels' own; the gate is the registry's (``turbotab.core.claims.referee_gate``), so an
  author's or a fork's verdict does not count, and one referee who does not uphold blocks. The
  rate is given only once every false decoy is decided (two referees who count upheld it, or one
  who counts did not): while any is undecided it is refused, naming them, and the command exits 1.
* Scored on the claims backtest's own key and verdicts, it gives what the backtest's
  ``score.py`` reported on 2026-10-15 (``notes/claims-backtest/key/score_report.json``).
* The tool reads and writes only the paths it is given.
"""
from __future__ import annotations

import hashlib
import json
import random
from pathlib import Path

import pytest
from statsmodels.stats.proportion import proportion_confint

from research.registry import decoy_queue, decoys
from turbotab.core.claims import Referee

REPO = Path(__file__).resolve().parents[3]
BACKTEST = REPO / "docs" / "turbotab-next" / "notes" / "claims-backtest"


def candidate(id, statement, **over):
    base = {"id": id, "statement": statement, "goal": "estimate",
            "regime": "a cluster-randomized trial, a yes/no outcome", "decision": f"{id}.decision",
            "app_action": "noted", "claimed_basis": f"{id}'s own basis (Li & Redden 2015)",
            "author": "theorist A"}
    return {**base, **over}


RNG = random.Random(7)


# ── the generators ───────────────────────────────────────────────────────────────────────────

def test_constant_moves_one_number_outside_a_citation_and_leaves_its_year():
    c = candidate("c.one", "Below 10 clusters, intervals are too narrow (Li & Redden 2015); "
                           "at 10 or more they hold.")
    d = decoys.constant(c, random.Random(1), [c])
    assert d is not None and d["type"] == "constant" and d["sibling"] == "c.one"
    new = d["statement"].split()[1]
    assert new != "10" and d["statement"] == c["statement"].replace("10", new)
    assert "2015" in d["statement"] and "10" in d["why_false"] and new in d["why_false"]
    assert {k: d[k] for k in ("regime", "decision", "claimed_basis")} == {
        k: c[k] for k in ("regime", "decision", "claimed_basis")}


def test_constant_moves_a_number_written_with_thousands_separators_whole():
    c = candidate("c.one", "Above 10,000 rows, a band refit on 10,000 resampled rows holds.")
    for seed in range(6):
        d = decoys.constant(c, random.Random(seed), [c])
        new = d["statement"].split()[1].rstrip(",")
        assert new != "10,000" and int(new.replace(",", "")) in {2000, 5000, 20000, 50000}
        assert d["statement"] == c["statement"].replace("10,000", new)
        assert d["why_false"] == f"it states {new} where its candidate states 10,000"


@pytest.mark.parametrize("statement", [
    "Boosting fits stagewise (Friedman 2001).",  # a number only in a citation
    "The guard was ruled in 2026 and is a convention.",  # a year
    "A plain sentence with no number in it.",
    "It follows finding A-001.",  # a number inside an id
])
def test_constant_does_not_apply_where_no_number_is_stated(statement):
    assert decoys.constant(candidate("c.one", statement), RNG, []) is None


@pytest.mark.parametrize("statement, turned", [
    ("A gain above 0.01 counts.", "A gain below 0.01 counts."),
    ("Above 30 units the sandwich holds.", "Below 30 units the sandwich holds."),
    ("With fewer clusters, intervals run narrower.", None),  # two pairs: one is turned
    ("It needs at least two referees.", "It needs at most two referees."),
])
def test_direction_turns_a_word_to_its_opposite(statement, turned):
    d = decoys.direction(candidate("c.one", statement), random.Random(3), [])
    assert d is not None and d["type"] == "direction" and d["statement"] != statement
    if turned is not None:
        assert d["statement"] == turned
    else:
        assert d["statement"] in ("With more clusters, intervals run narrower.",
                                  "With fewer clusters, intervals run wider.")


@pytest.mark.parametrize("statement", ["A plain sentence.", "The widest net is widely cast."])
def test_direction_does_not_apply_where_no_word_has_an_opposite(statement):
    """"widest" and "widely" are not "wide": a word is turned whole."""
    assert decoys.direction(candidate("c.one", statement), RNG, []) is None


def test_overreach_widens_the_regime_and_says_so_in_the_statement():
    c = candidate("c.one", "Intervals are too narrow below ten clusters.")
    d = decoys.overreach(c, random.Random(2), [])
    assert d["regime"] == decoys.WIDE and d["type"] == "overreach"
    assert d["statement"].endswith("intervals are too narrow below ten clusters.")
    assert c["regime"] in d["why_false"]


@pytest.mark.parametrize("regime", ["", decoys.WIDE])
def test_overreach_does_not_apply_to_a_regime_that_is_blank_or_already_wide(regime):
    assert decoys.overreach(candidate("c.one", "It holds.", regime=regime), RNG, []) is None


def test_citation_takes_the_basis_of_a_candidate_on_another_decision():
    a = candidate("c.a", "One thing holds.")
    same = candidate("c.same", "Another holds.", decision=a["decision"])
    other = candidate("c.b", "A third holds.")
    for seed in range(5):
        d = decoys.citation(a, random.Random(seed), [a, same, other])
        assert d["claimed_basis"] == other["claimed_basis"] and "c.b" in d["why_false"]
        assert d["statement"] == a["statement"]


def test_citation_does_not_apply_where_no_other_decision_gives_a_basis():
    a = candidate("c.a", "One thing holds.")
    pool = [a, candidate("c.same", "x", decision=a["decision"]),
            candidate("c.blank", "y", claimed_basis="")]
    assert decoys.citation(a, RNG, pool) is None


def test_mechanism_swaps_the_reason_after_because():
    a = candidate("c.a", "Fold differences are correlated because they share training rows.")
    b = candidate("c.b", "The sandwich is biased low because few clusters estimate it.")
    d = decoys.mechanism(a, RNG, [a, b])
    assert d["statement"] == "Fold differences are correlated because few clusters estimate it."
    assert "c.b" in d["why_false"]


def test_mechanism_does_not_apply_without_a_reason_to_swap():
    a = candidate("c.a", "Fold differences are correlated.")
    b = candidate("c.b", "The sandwich is biased low because few clusters estimate it.")
    assert decoys.mechanism(a, RNG, [a, b]) is None
    assert decoys.mechanism(b, RNG, [a, b]) is None  # no other candidate gives a reason


def _pool():
    return [candidate(f"c.n{i:02d}", f"Below {10 + i} clusters intervals are narrower, because "
                                     f"reason {i} holds.") for i in range(12)]


def test_generate_takes_the_kinds_in_turn_from_distinct_candidates_and_repeats_by_seed():
    pool = _pool()
    made = decoys.generate(pool, 10, seed=11)
    assert len(made) == 10 and [d["key"] for d in made] == [f"G{i:03d}" for i in range(1, 11)]
    assert [d["type"] for d in made] == [*decoys.KINDS, *decoys.KINDS]
    assert len({d["sibling"] for d in made}) == 10, "no candidate is used twice while one is free"
    assert all(d["author"] == "theorist A" and d["why_false"] for d in made)
    assert decoys.generate(pool, 10, seed=11) == made
    assert decoys.generate(pool, 10, seed=12) != made


def test_generate_makes_fewer_where_no_kind_applies():
    plain = [candidate("c.a", "A plain sentence.", regime="", claimed_basis="")]
    assert decoys.generate(plain, 3, seed=1) == []


# ── the queue ────────────────────────────────────────────────────────────────────────────────

def _queue(tmp_path, n_decoys=6, batches=3):
    pool = _pool()
    key = decoy_queue.build(pool, tmp_path / "queue", decoys=n_decoys, seed=5, batches=batches)
    return pool, key


def test_the_queue_holds_each_candidate_and_decoy_once_under_neutral_ids(tmp_path):
    pool, key = _queue(tmp_path)
    rows = [r for path in sorted((tmp_path / "queue" / "batches").glob("batch_*.jsonl"))
            for r in decoy_queue.read_jsonl(path)]
    assert len(rows) == len(pool) + 6 == len(key["items"])
    assert sorted(r["id"] for r in rows) == [f"K-{i:03d}" for i in range(1, 19)]
    assert all(set(r) == {"id", *decoys.MERITS_FIELDS} for r in rows), "a referee sees no key field"
    statements = {r["statement"] for r in rows}
    assert {c["statement"] for c in pool} <= statements
    assert {d["statement"] for d in key["decoys"]} <= statements


def test_no_decoy_shares_a_batch_with_its_sibling(tmp_path):
    pool, key = _queue(tmp_path, n_decoys=8, batches=4)
    batch = {it["orig"]: it["batch"] for it in key["items"]}
    decoy_items = [it for it in key["items"] if it["kind"] == "decoy"]
    assert len(decoy_items) == 8
    assert all(batch[it["sibling"]] != it["batch"] for it in decoy_items)


def test_the_key_is_sealed_by_its_sha_256_and_the_queue_repeats_by_seed(tmp_path):
    _, key = _queue(tmp_path)
    folder = tmp_path / "queue"
    digest = hashlib.sha256((folder / "key.json").read_bytes()).hexdigest()
    assert (folder / "key.sha256").read_text(encoding="utf-8") == f"{digest}  key.json\n"
    _, again = _queue(tmp_path / "again")
    assert again == key


def test_the_tool_writes_only_under_the_folder_it_is_given(tmp_path):
    before = set(Path.cwd().iterdir())
    _queue(tmp_path)
    written = {p.relative_to(tmp_path).as_posix() for p in tmp_path.rglob("*") if p.is_file()}
    assert written == {"queue/key.json", "queue/key.sha256", "queue/batches/batch_01.jsonl",
                       "queue/batches/batch_02.jsonl", "queue/batches/batch_03.jsonl"}
    assert set(Path.cwd().iterdir()) == before


# ── the score ────────────────────────────────────────────────────────────────────────────────

def _key(n_false, n_real=4, vacuous=0, author="theorist A"):
    items = [{"rid": f"R-{i}", "orig": f"c.{i}", "kind": "real", "sibling": None,
              "author": author} for i in range(n_real)]
    kinds = ["constant"] * n_false + ["vacuous"] * vacuous
    items += [{"rid": f"D-{i}", "orig": f"G{i:03d}", "kind": "decoy", "sibling": "c.0",
               "author": author} for i in range(len(kinds))]
    return {"items": items, "decoys": [{"key": f"G{i:03d}", "type": t} for i, t in
                                       enumerate(kinds)]}


def _votes(key, upheld_by, rids=None):
    """Each named referee upholds the items ``rids`` (every item when None) and no other."""
    out = {}
    for it in key["items"]:
        rid = it["rid"]
        out[rid] = [Referee(name, "upheld" if rids is None or rid in rids else "not_upheld")
                    for name in upheld_by]
    return out


@pytest.mark.parametrize("k, n", [(3, 10), (1, 26), (7, 7)])
def test_the_rate_is_the_share_of_false_decoys_promoted_and_its_interval_statsmodels(k, n):
    key = _key(n)
    promoted = {f"D-{i}" for i in range(k)} | {f"R-{i}" for i in range(4)}
    report = decoy_queue.score(key, _votes(key, ["referee A", "referee B"], promoted))
    fp = report["false_promotion"]
    assert (fp["promoted"], fp["decoys"], fp["decided"], fp["rate"]) == (k, n, n, k / n)
    assert fp["undecided"] == [] and fp["refused"] is None
    lo, hi = proportion_confint(k, n, alpha=0.05, method="wilson")
    assert fp["wilson_95"] == pytest.approx([lo, hi], abs=1e-12)
    assert fp["rule_of_three_95"] is None
    assert report["real"] == {"candidates": 4, "promoted": 4, "undecided": 0}


def test_none_promoted_reports_the_rule_of_three_and_statsmodels_upper_bound():
    key = _key(26)
    report = decoy_queue.score(key, _votes(key, ["referee A", "referee B"], set()))
    fp = report["false_promotion"]
    assert fp["promoted"] == 0 and fp["rule_of_three_95"] == pytest.approx(3 / 26)
    assert fp["wilson_95"] == pytest.approx(list(proportion_confint(0, 26, method="wilson")),
                                            abs=1e-12)


GATE_CASES = {
    # verdicts on one false decoy, and whether the gate promotes it
    "two fresh referees uphold": ([("referee A", "upheld", ""), ("referee B", "upheld", "")], 1),
    "one of them is its author": ([("referee A", "upheld", ""), ("theorist A", "upheld", "")], 0),
    "one of them is a fork": ([("referee A", "upheld", ""), ("referee B", "upheld", "theorist A")],
                              0),
    "a third does not uphold": ([("referee A", "upheld", ""), ("referee B", "upheld", ""),
                                 ("referee C", "not_upheld", "")], 0),
    "one referee only": ([("referee A", "upheld", "")], 0),
    "the author's dissent does not count": ([("referee A", "upheld", ""),
                                            ("referee B", "upheld", ""),
                                            ("theorist A", "not_upheld", "")], 1),
}


@pytest.mark.parametrize("case", sorted(GATE_CASES))
def test_the_score_promotes_through_the_registry_s_two_referee_gate(case):
    votes, promoted = GATE_CASES[case]
    key = _key(1, n_real=0)
    report = decoy_queue.score(key, {"D-0": [Referee(*v) for v in votes]})
    assert report["false_promotion"]["promoted"] == promoted


def test_vacuous_decoys_are_left_out_of_the_rate_and_missing_verdicts_are_named():
    key = _key(2, vacuous=2)
    votes = _votes(key, ["referee A", "referee B"])
    votes["D-1"] = votes["D-1"][:1]
    report = decoy_queue.score(key, votes)
    assert report["queue"]["false_decoys"] == 2 and report["queue"]["vacuous_decoys"] == 2
    fp = report["false_promotion"]
    assert (fp["decoys"], fp["decided"], fp["promoted"]) == (2, 1, 1)
    assert fp["undecided"] == ["D-1"], "D-1 has one referee, so is undecided, not unpromoted"
    assert fp["rate"] is None and fp["wilson_95"] is None and fp["refused"]
    assert report["vacuous"] == {"decoys": 2, "promoted": 2, "undecided": 0}
    assert report["missing_verdicts"] == ["D-1"]


# The verifier's case (RG-3): ten false decoys, of which referees A and B both upheld two and
# gave no verdict on the rest. Each case is the verdicts on the other eight.
REST = {
    "none": [],
    "one referee's uphold": [("referee A", "upheld", "")],
    "two upholds, one of them a fork's": [("referee A", "upheld", ""),
                                          ("referee B", "upheld", "theorist A")],
    "two upholds, one of them the author's": [("referee A", "upheld", ""),
                                              ("theorist A", "upheld", "")],
}


@pytest.mark.parametrize("case", sorted(REST))
def test_the_rate_is_refused_while_any_false_decoy_is_undecided(case):
    key = _key(10, n_real=0)
    votes = {f"D-{i}": [Referee("referee A", "upheld"), Referee("referee B", "upheld")]
             for i in range(2)}
    votes.update({f"D-{i}": [Referee(*v) for v in REST[case]] for i in range(2, 10)})
    report = decoy_queue.score(key, votes)
    fp = report["false_promotion"]
    assert (fp["decoys"], fp["decided"], fp["promoted"]) == (10, 2, 2)
    assert fp["undecided"] == [f"D-{i}" for i in range(2, 10)]
    assert fp["rate"] is None and fp["wilson_95"] is None and fp["rule_of_three_95"] is None
    lines = decoy_queue.summary(report)
    rate = next(x for x in lines if x.startswith("false-promotion rate"))
    assert rate.startswith("false-promotion rate refused: 8 of 10 false decoys are undecided")
    assert "%" not in rate and "D-2" in rate


DECIDING = {
    # the verdicts on each of the eight, and how many of the ten the gate then promotes
    "two referees who count uphold": ([("referee A", "upheld", ""), ("referee B", "upheld", "")],
                                      10),
    "two referees who count do not": ([("referee A", "not_upheld", ""),
                                       ("referee B", "not_upheld", "")], 2),
    "one who counts does not, alone": ([("referee A", "not_upheld", "")], 2),
    "one who counts does not, beside a fork's uphold": ([("referee A", "not_upheld", ""),
                                                         ("referee B", "upheld", "theorist A")],
                                                        2),
}


@pytest.mark.parametrize("case", sorted(DECIDING))
def test_the_rate_is_given_once_every_false_decoy_is_decided(case):
    """The other side: two upholds that count decide a decoy promoted, and one dissent that counts
    decides it not, whatever else is missing; with all ten decided, the rate is k of 10."""
    rest, k = DECIDING[case]
    key = _key(10, n_real=0)
    votes = {f"D-{i}": [Referee("referee A", "upheld"), Referee("referee B", "upheld")]
             for i in range(2)}
    votes.update({f"D-{i}": [Referee(*v) for v in rest] for i in range(2, 10)})
    fp = decoy_queue.score(key, votes)["false_promotion"]
    assert (fp["decided"], fp["undecided"], fp["refused"]) == (10, [], None)
    assert (fp["promoted"], fp["rate"]) == (k, k / 10)
    lo, hi = proportion_confint(k, 10, alpha=0.05, method="wilson")
    assert fp["wilson_95"] == pytest.approx([lo, hi], abs=1e-12)


def test_a_real_candidate_left_undecided_is_counted_apart(tmp_path):
    key = _key(1, n_real=3)
    votes = _votes(key, ["referee A", "referee B"])
    votes["R-2"] = votes["R-2"][:1]
    report = decoy_queue.score(key, votes)
    assert report["real"] == {"candidates": 3, "promoted": 2, "undecided": 1}
    assert "real candidates promoted: 2 of the 2 decided; 1 undecided" in decoy_queue.summary(
        report)


def test_the_backtest_s_own_key_and_verdicts_give_the_backtest_s_reported_gate():
    key = json.loads((BACKTEST / "key" / "decoy_key.json").read_text(encoding="utf-8"))
    reported = json.loads((BACKTEST / "key" / "score_report.json").read_text(encoding="utf-8"))
    votes = decoy_queue.read_verdicts([
        f"opus={BACKTEST}/verdicts/merits_*_opus.jsonl",
        f"sonnet={BACKTEST}/verdicts/merits_*_sonnet.jsonl"])
    report = decoy_queue.score(key, votes)
    gate, fp = reported["gate"], report["false_promotion"]
    assert (fp["decoys"], fp["promoted"]) == (gate["false_decoys"], gate["passed_gate"])
    assert [round(x, 3) for x in fp["wilson_95"]] == gate["wilson_passed"]
    assert round(fp["rule_of_three_95"], 3) == gate["upper95_rule_of_three"]
    assert {t: v["decoys"] for t, v in report["by_type"].items()} == {
        t: sum(v.values()) // 2 for t, v in gate["by_type"].items()}
    for name in ("opus", "sonnet"):
        assert report["single_referee"][name]["upheld_false_decoys"] == gate[f"single_{name}_passed"]
    both_true = sum(reported[f"real_{s}"]["both_true"] for s in ("random", "tier2", "targeted"))
    assert report["real"] == {"candidates": 117, "promoted": both_true, "undecided": 0}
    assert (fp["decided"], fp["undecided"], fp["refused"]) == (gate["false_decoys"], [], None)
    assert report["vacuous"]["decoys"] == len(reported["vacuous"])
    assert report["missing_verdicts"] == reported["missing_merits"] == []
    assert report["authors_recorded"] is False


# ── the candidates ───────────────────────────────────────────────────────────────────────────

def test_a_candidate_gives_the_parts_its_claim_asserts_and_not_one_retired():
    from turbotab.core.claims import Basis, Part, Registry
    from turbotab.core.tests.test_claims_fixtures import rule

    claim = rule(parts=(Part("value", "judgment", Basis(reason="a planted judgment")),
                        Part("scope", "retired", graded_by="a domain reviewer, 2026-10-19")))
    [row] = decoy_queue.registry_candidates(Registry([claim]))
    assert row["claimed_basis"] == "value (judgment): a planted judgment"


# ── the command line, end to end ─────────────────────────────────────────────────────────────

def test_the_command_line_builds_a_queue_from_the_registry_and_prints_its_rate(tmp_path, capsys):
    cands, out = tmp_path / "candidates.jsonl", tmp_path / "queue"
    assert decoy_queue.main(["candidates", "--out", str(cands)]) == 0
    real = decoy_queue.read_jsonl(cands)
    assert len(real) == 27 and {c["origin"] for c in real} == {"engine", "audit"}
    assert decoy_queue.main(["build", "--candidates", str(cands), "--out", str(out),
                             "--decoys", "6", "--seed", "3"]) == 0
    key = json.loads((out / "key.json").read_text(encoding="utf-8"))
    upheld = {it["rid"] for it in key["items"] if it["kind"] == "real"}
    upheld.add(next(it["rid"] for it in key["items"] if it["kind"] == "decoy"))
    for name in ("referee A", "referee B"):
        (tmp_path / f"{name[-1]}.jsonl").write_text("".join(
            json.dumps({"id": it["rid"], "truth": "true" if it["rid"] in upheld else "false"})
            + "\n" for it in key["items"]), encoding="utf-8")
    capsys.readouterr()
    assert decoy_queue.main(["score", "--key", str(out / "key.json"),
                             "--verdicts", f"referee A={tmp_path}/A.jsonl",
                             "--verdicts", f"referee B={tmp_path}/B.jsonl",
                             "--report", str(tmp_path / "report.json")]) == 0
    printed = capsys.readouterr().out
    assert "false-promotion rate: 1 of 6 false decoys promoted (16.7%)" in printed
    assert "real candidates promoted: 27 of 27" in printed
    saved = json.loads((tmp_path / "report.json").read_text(encoding="utf-8"))
    assert saved["false_promotion"]["rate"] == pytest.approx(1 / 6)


def test_the_command_line_refuses_the_rate_and_exits_1_while_a_false_decoy_is_undecided(
        tmp_path, capsys):
    """Referee A upholds one false decoy and referee B gives it no verdict, so nothing decides
    it: the report is written, with its rate refused, and the score exits 1."""
    cands, out = tmp_path / "candidates.jsonl", tmp_path / "queue"
    assert decoy_queue.main(["candidates", "--out", str(cands)]) == 0
    assert decoy_queue.main(["build", "--candidates", str(cands), "--out", str(out),
                             "--decoys", "6", "--seed", "3"]) == 0
    key = json.loads((out / "key.json").read_text(encoding="utf-8"))
    skipped = next(it["rid"] for it in key["items"] if it["kind"] == "decoy")
    for name in ("A", "B"):
        (tmp_path / f"{name}.jsonl").write_text("".join(
            json.dumps({"id": it["rid"], "truth": "true" if it["rid"] == skipped else "false"})
            + "\n" for it in key["items"] if not (name == "B" and it["rid"] == skipped)),
            encoding="utf-8")
    capsys.readouterr()
    assert decoy_queue.main(["score", "--key", str(out / "key.json"),
                             "--verdicts", f"referee A={tmp_path}/A.jsonl",
                             "--verdicts", f"referee B={tmp_path}/B.jsonl",
                             "--report", str(tmp_path / "report.json")]) == 1
    printed = capsys.readouterr().out
    assert f"false-promotion rate refused: 1 of 6 false decoys are undecided ({skipped})" in printed
    saved = json.loads((tmp_path / "report.json").read_text(encoding="utf-8"))
    assert saved["false_promotion"]["rate"] is None
    assert saved["false_promotion"]["undecided"] == [skipped]
