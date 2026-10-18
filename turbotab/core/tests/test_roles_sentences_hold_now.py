"""W1-R round 2 · the roles' sentences say what holds now (the walkthrough of 2026-10-17, findings 2
and 3c; W1-R R2's verifier, V1 and V4).

* **Finding 2.** The roles answer said which columns "wait for their own confirmation", and the
  methods kept saying it after each was confirmed. The methods now restate that clause on the
  answers as they stand: a column confirmed since is no longer listed, and with none left the
  clause is gone. The Record keeps each sentence as said.
* **Finding 3c.** Under Predict nothing is studied and nothing adjusted for, so the roles answer
  and each confirmation of a role call an exposure and a covariate a predictor
  (``readings.role_words``). The roles are recorded in Your data, often before the goal is
  answered, so the methods word them by the goal as it stands, not as it stood then.

Every expectation is from construction: the answers, which columns the roles answer recorded below
high, and which were confirmed since, are written here.
"""
from __future__ import annotations

import pytest

from turbotab.core import voice
from turbotab.core.decisions import (ConfirmReading, ConfirmReadings, ConfirmRole, DecisionLog,
                                     ReadingItem, SetLens, SetPurpose, SetRoles, SetTarget, SetTask)
from turbotab.core.provenance import methods_text

ROLES = {"protein_g": "exposure", "fat_g": "exposure", "bp_sys": "covariate", "waist": "covariate",
         "age": "covariate", "energy_kcal": "energy"}
WAITING = ["bp_sys", "waist", "protein_g"]
HEAD = (SetLens(lenses=["dietary"]), SetTarget(column="y"), SetTask(column="y", task="regression"))
ESTIMATE_BODY = ("Column roles were set for `6` columns: energy `energy_kcal`; exposures "
                 "`protein_g` and `fat_g`; covariates `bp_sys`, `waist` and `age`")
# Under Predict the exposures and covariates are one group, listed as every group is (four names
# at most, ``voice.listing``).
PREDICT_BODY = ("Column roles were set for `6` columns: energy `energy_kcal`; predictors "
                "`protein_g`, `fat_g`, `bp_sys` and 2 more")


def _waits(columns: list[str]) -> str:
    """The clause the roles sentence ends with while ``columns`` wait (as said before this round)."""
    one = len(columns) == 1
    return (f"; {voice.listing(columns, limit=3)} {'was' if one else 'were'} proposed below high "
            f"confidence and {'waits' if one else 'wait'} for {'its' if one else 'their'} own "
            f"confirmation before any default reads {'it' if one else 'them'}.")


def _log(tmp_path, *decisions) -> DecisionLog:
    log = DecisionLog(tmp_path / "decisions.jsonl")
    for d in decisions:
        log.append(d, sentence=lambda x, before: voice.sentence_for(x, before, None))
    return log


def _said(log: DecisionLog, kind: str) -> list[str]:
    return [line.sentence for line in methods_text(log.records()).lines if line.kind == kind]


def _recorded(log: DecisionLog, kind: str) -> list[str]:
    return [r.sentence for r in log.records() if r.decision.kind == kind]


# ── finding 2: the columns that wait are the ones that wait now ────────────────


@pytest.mark.parametrize("purpose, body", [("inference", ESTIMATE_BODY),
                                           ("prediction", PREDICT_BODY)])
def test_the_roles_sentence_lists_only_the_columns_that_still_wait(tmp_path, purpose, body):
    log = _log(tmp_path, *HEAD, SetPurpose(purpose=purpose),
               SetRoles(roles=ROLES, unconfirmed=WAITING))
    first = f"{body}{_waits(WAITING)}"
    assert _recorded(log, "set_roles") == [first] == _said(log, "set_roles")
    # two of three confirmed on their own: only the third is listed
    for column in ("bp_sys", "waist"):
        log.append(ConfirmRole(column=column, role=ROLES[column]),
                   sentence=lambda x, before: voice.sentence_for(x, before, None))
    assert _said(log, "set_roles") == [f"{body}{_waits(['protein_g'])}"]
    # the third confirmed: no column waits, so nothing is said of one
    log.append(ConfirmReading(reading="role", column="protein_g", value="exposure"),
               sentence=lambda x, before: voice.sentence_for(x, before, None))
    assert _said(log, "set_roles") == [f"{body}."]
    # the Record keeps the sentence as said
    assert _recorded(log, "set_roles") == [first]


def test_a_column_confirmed_as_another_role_no_longer_waits(tmp_path):
    log = _log(tmp_path, *HEAD, SetPurpose(purpose="inference"),
               SetRoles(roles=ROLES, unconfirmed=WAITING),
               ConfirmRole(column="waist", role="excluded"))
    [said] = _said(log, "set_roles")
    assert said.endswith(_waits(["bp_sys", "protein_g"]))


def test_with_nothing_waiting_the_roles_sentence_names_no_wait(tmp_path):
    log = _log(tmp_path, *HEAD, SetPurpose(purpose="inference"), SetRoles(roles=ROLES))
    assert _said(log, "set_roles") == [f"{ESTIMATE_BODY}."]


# ── finding 3c: a role in the goal's words, as the goal stands ─────────────────


def test_under_predict_each_confirmation_names_its_group(tmp_path):
    """W1-T's V2 (U8b): under Predict a role's confirmation names the roles card's group for it in
    the project, as its role line's press does (the roles ruling of 2026-10-17: under the dietary
    lens a nutrient and a characteristic)."""
    log = _log(tmp_path, *HEAD, SetPurpose(purpose="prediction"),
               SetRoles(roles=ROLES, unconfirmed=WAITING),
               ConfirmRole(column="bp_sys", role="covariate"),
               ConfirmReading(reading="role", column="protein_g", value="exposure"),
               ConfirmReadings(items=[ReadingItem(reading="role", column="waist",
                                                  value="covariate")]))
    assert _recorded(log, "confirm_role") == [
        "`bp_sys` was confirmed as a characteristic on its own, after the evidence for its "
        "proposal was read."]
    assert _recorded(log, "confirm_reading") == [
        "`protein_g` was confirmed as a nutrient, on its own, after the evidence for its reading "
        "was read."]
    assert _recorded(log, "confirm_readings") == [
        "Confirmed together, each as the question showed it: `waist` was confirmed as a "
        "characteristic."]


def test_under_estimate_each_confirmation_names_the_role_as_before(tmp_path):
    log = _log(tmp_path, *HEAD, SetPurpose(purpose="inference"),
               SetRoles(roles=ROLES, unconfirmed=WAITING),
               ConfirmRole(column="bp_sys", role="covariate"),
               ConfirmReading(reading="role", column="protein_g", value="exposure"))
    assert _recorded(log, "confirm_role") == [
        "`bp_sys` was confirmed as a covariate on its own, after the evidence for its proposal was "
        "read."]
    assert _recorded(log, "confirm_reading") == [
        "`protein_g` was confirmed as an exposure, on its own, after the evidence for its reading "
        "was read."]


def test_roles_recorded_before_the_goal_are_worded_by_the_goal_in_the_methods(tmp_path):
    """Your data comes before Your question: the roles and their confirmations are recorded with no
    goal, in Estimate's words. Once the goal is prediction the methods say "predictor", and a
    confirmation names the roles card's group (U8b); the Record keeps each sentence as said."""
    log = _log(tmp_path, *HEAD, SetRoles(roles=ROLES, unconfirmed=WAITING),
               ConfirmRole(column="bp_sys", role="covariate"))
    as_said = _recorded(log, "set_roles") + _recorded(log, "confirm_role")
    assert as_said == [f"{ESTIMATE_BODY}{_waits(WAITING)}",
                       "`bp_sys` was confirmed as a covariate on its own, after the evidence for "
                       "its proposal was read."]
    assert _said(log, "set_roles") == [f"{ESTIMATE_BODY}{_waits(['waist', 'protein_g'])}"]
    log.append(SetPurpose(purpose="prediction"),
               sentence=lambda x, before: voice.sentence_for(x, before, None))
    assert _said(log, "set_roles") == [f"{PREDICT_BODY}{_waits(['waist', 'protein_g'])}"]
    assert _said(log, "confirm_role") == [
        "`bp_sys` was confirmed as a characteristic on its own, after the evidence for its "
        "proposal was read."]
    assert _recorded(log, "set_roles") + _recorded(log, "confirm_role") == as_said
    # back under Estimate, the words are Estimate's again
    log.append(SetPurpose(purpose="inference"),
               sentence=lambda x, before: voice.sentence_for(x, before, None))
    assert _said(log, "confirm_role") == [as_said[1]]


# ── finding 11 (ruling R-c): the readings the values settled are a sentence of their own ──────

READ = "Read from the values, no question asked: `age` is a covariate (ages in years)."


def _families(task: str | None = None) -> str:
    """The families answer as the families declare their own labels for ``task``
    (``methods_label``)."""
    import turbotab.core.models  # noqa: F401 - registers the families
    from turbotab.core.models.base import get_family

    a, b = (get_family(k).methods_label(task) for k in ("linear", "ridge"))
    return f"Two model families were chosen: {a} and {b}."


def _families_said(purpose: str, read: str | None) -> str:
    from turbotab.core.decisions import ProjectState, SelectModels

    return voice.sentence_for(SelectModels(models=["linear", "ridge"]),
                              ProjectState(purpose=purpose), {"read_from_values": read})


@pytest.mark.parametrize("purpose", ["prediction", "inference"])
def test_the_families_answer_names_the_families_and_nothing_else(purpose):
    assert _families_said(purpose, READ) == _families_said(purpose, None) == _families()


def _sections(tmp_path, purpose: str, read: str | None) -> dict[str, list]:
    from types import SimpleNamespace

    from turbotab.core.decisions import SelectModels
    from turbotab.core.export.methods import methods_section

    tmp_path.mkdir(parents=True, exist_ok=True)
    log = _log(tmp_path, *HEAD, SetPurpose(purpose=purpose), SetRoles(roles=ROLES),
               SelectModels(models=["linear", "ridge"]))
    source = SimpleNamespace(name="diet", purpose=purpose, methods=methods_text(log.records()),
                             artifact=lambda stage: None, read_from_values=read)
    doc = methods_section(source, {"engine": {}, "inputs": [], "decisions": {"n": 1},
                                   "analysis_plan": {}})
    return {s.key: s.entries for s in doc.sections}


@pytest.mark.parametrize("purpose, section", [("prediction", "predictors"),
                                              ("inference", "variables")])
def test_the_readings_the_values_settled_are_their_own_sentence_among_the_predictors(
        tmp_path, purpose, section):
    """TRIPOD+AI 9 (under Predict) and STROBE 7 (under Estimate) are where a predictor's
    definition and measurement are said; the families answer is the analysis's."""
    sections = _sections(tmp_path, purpose, READ)
    found = [(key, e) for key, entries in sections.items() for e in entries if READ in e.text]
    assert [(key, e.source, e.text) for key, e in found] == [(section, "analysis", READ)]
    [families] = [e for entries in sections.values() for e in entries if e.kind == "select_models"]
    assert families.text == _families("regression")
    # silent: nothing the values settled, no such sentence
    quiet = _sections(tmp_path / "quiet", purpose, None)
    assert not [e for entries in quiet.values() for e in entries if "Read from the values" in e.text]


def _sections_of(log: DecisionLog, purpose: str, read: str | None) -> dict[str, list]:
    from types import SimpleNamespace

    from turbotab.core.export.methods import methods_section

    source = SimpleNamespace(name="diet", purpose=purpose, methods=methods_text(log.records()),
                             artifact=lambda stage: None, read_from_values=read)
    doc = methods_section(source, {"engine": {}, "inputs": [], "decisions": {"n": 1},
                                   "analysis_plan": {}})
    return {s.key: s.entries for s in doc.sections}


@pytest.mark.parametrize("purpose, section", [("prediction", "predictors"),
                                              ("inference", "variables")])
def test_a_families_answer_recorded_with_the_readings_states_them_once(tmp_path, purpose,
                                                                      section):
    """W1-R round 3 (SE-m1): a families answer recorded before ruling R-c ended with the readings
    the values settled ("… ridge regression. Read from the values, no question asked: …", the
    sentence as that version wrote it, built here). The methods restate it as the families alone,
    so the readings are said once, in their own sentence among the predictors; the Record keeps the
    sentence as said."""
    from turbotab.core.decisions import SelectModels

    legacy = f"{_families('regression')[:-1]}. {READ}"
    log = _log(tmp_path, *HEAD, SetPurpose(purpose=purpose), SetRoles(roles=ROLES))
    log.append(SelectModels(models=["linear", "ridge"]), sentence=lambda x, before: legacy)
    sections = _sections_of(log, purpose, READ)
    found = [(key, e.source, e.text) for key, entries in sections.items() for e in entries
             if "Read from the values" in e.text]
    assert found == [(section, "analysis", READ)]
    [families] = [e for entries in sections.values() for e in entries if e.kind == "select_models"]
    assert families.text == _families("regression")
    assert _recorded(log, "select_models") == [legacy]
    # silent: a families answer recorded as the families alone is said as recorded
    fresh = _log(tmp_path / "fresh", *HEAD, SetPurpose(purpose=purpose), SetRoles(roles=ROLES),
                 SelectModels(models=["linear", "ridge"]))
    assert _said(fresh, "select_models") == _recorded(fresh, "select_models") == [
        _families("regression")]


# ── W1-R round 3 (SE-B2): under Predict a move between the predictor roles is a change ──
#
# Under Predict an exposure and a covariate are each a predictor, but the engine still treats them
# apart: the screened elastic net (prediction only) screens the Predictor columns and passes every
# other column through, and the omics normalization and reference ComBat act on Predictor columns
# alone. So a move between the two is a change, said in the Predict roles card's own labels for the
# two roles.

MOVED = {**ROLES, "bp_sys": "exposure"}


def _labels() -> dict[str, str]:
    """The Predict roles card's own labels for its groups, by role, in this dietary project
    (``readings.predictor_groups``; the roles ruling of 2026-10-17)."""
    from turbotab.core.readings import predictor_groups

    return {role: g.head[0] for role, g in predictor_groups(["dietary"]).items()}


def test_under_predict_a_covariate_made_a_predictor_column_is_said_as_a_move(tmp_path):
    label = _labels()
    # the ruling's names for the two groups of a dietary project
    assert (label["exposure"], label["covariate"]) == ("Nutrient", "Characteristic")
    log = _log(tmp_path, *HEAD, SetPurpose(purpose="prediction"), SetRoles(roles=ROLES),
               SetRoles(roles=MOVED))
    want = (f"`bp_sys` moved from the {label['covariate']} role to the {label['exposure']} role; "
            f"every other role is unchanged.")
    assert _recorded(log, "set_roles")[-1] == want
    assert want in _said(log, "set_roles")
    assert not any("unchanged." in s and "every other" not in s for s in _said(log, "set_roles"))


def test_under_predict_each_move_is_listed_beside_the_others(tmp_path):
    """With up to three moves, a move inside the predictors is listed with the rest."""
    label = _labels()
    log = _log(tmp_path, *HEAD, SetPurpose(purpose="prediction"), SetRoles(roles=ROLES),
               SetRoles(roles={**ROLES, "protein_g": "covariate", "age": "excluded"}))
    assert _recorded(log, "set_roles")[-1] == (
        f"`protein_g` moved from the {label['exposure']} role to the {label['covariate']} role "
        f"and `age` became excluded; every other role is unchanged.")


def test_under_predict_the_same_roles_again_are_still_unchanged(tmp_path):
    log = _log(tmp_path, *HEAD, SetPurpose(purpose="prediction"), SetRoles(roles=ROLES),
               SetRoles(roles=ROLES))
    assert _recorded(log, "set_roles")[-1] == ("The roles of all `6` columns were recorded "
                                               "unchanged.")


def test_under_estimate_the_same_move_is_said_as_before(tmp_path):
    log = _log(tmp_path, *HEAD, SetPurpose(purpose="inference"), SetRoles(roles=ROLES),
               SetRoles(roles=MOVED))
    assert _recorded(log, "set_roles")[-1] == ("`bp_sys` became an exposure; every other role is "
                                               "unchanged.")


def test_a_move_recorded_before_the_goal_is_said_as_a_move_once_the_goal_is_prediction(tmp_path):
    log = _log(tmp_path, *HEAD, SetRoles(roles=ROLES), SetRoles(roles=MOVED))
    as_said = "`bp_sys` became an exposure; every other role is unchanged."
    assert _recorded(log, "set_roles")[-1] == as_said
    log.append(SetPurpose(purpose="prediction"),
               sentence=lambda x, before: voice.sentence_for(x, before, None))
    label = _labels()
    assert _said(log, "set_roles")[-1] == (
        f"`bp_sys` moved from the {label['covariate']} role to the {label['exposure']} role; "
        f"every other role is unchanged.")
    assert _recorded(log, "set_roles")[-1] == as_said


def test_under_predict_the_omics_screen_reads_the_move():
    """Why the move is a change under Predict: the screened elastic net screens the Predictor
    columns by their correlation with the outcome and passes every other column through. Keeping
    one, the screen keeps `bp_sys` beside the best Predictor column while it is a covariate, and
    drops it once it is a Predictor column that numpy's correlation ranks below another."""
    from types import SimpleNamespace

    import numpy as np
    import pandas as pd

    import turbotab.core.methods.omics  # noqa: F401 - registers the screened family
    from turbotab.core.models.base import get_family

    rng = np.random.default_rng(7)
    n = 400
    X = pd.DataFrame({"g1": rng.normal(size=n), "g2": rng.normal(size=n),
                      "bp_sys": rng.normal(size=n)})
    y = 2.0 * X["g1"] + 0.2 * X["bp_sys"] + rng.normal(size=n)
    r = {c: abs(float(np.corrcoef(X[c], y)[0, 1])) for c in X}
    family = get_family("screened_elastic_net")

    def kept(roles: dict[str, str]) -> list[str]:
        [(_name, screen)] = family.preprocess(SimpleNamespace(predictors=list(X), roles=roles))
        screen.set_params(keep=1)
        return list(screen.fit(X, y).transform(X).columns)

    best = max(("g1", "g2"), key=r.get)
    assert r["bp_sys"] < r[best]
    assert kept({"g1": "exposure", "g2": "exposure", "bp_sys": "covariate"}) == [best, "bp_sys"]
    assert kept({"g1": "exposure", "g2": "exposure", "bp_sys": "exposure"}) == [best]
