"""MC-2b-4 · the omics methods paragraph reads a family through its declarations, never its key.

``methods/omics.py`` held four switches: ``model_clause`` named the elastic net and its screened
form, and ``chain_choices``, ``methods_paragraph`` and ``fit_methods`` each named the feature-wise
family. Each is now a predicate over the family's declarations (:func:`tunes_a_mix`: ``tuning.kind``
is ``"path"`` over a lasso-ridge mix; :func:`tests_each_feature`: an inference table and no
predictions). The old switches are copied below as the reference: each predicate selects exactly
the families they named, and a family that is not registered under those keys but declares the
same is read the same way.
"""
from __future__ import annotations

from types import SimpleNamespace

import pytest

import turbotab.core.models  # noqa: F401 - registers the families
from turbotab.core.contracts import contracts
from turbotab.core.methods import omics
from turbotab.core.models.base import families

contracts()  # the omics chain registers the screened elastic net

VALIDATIONS = (None, "holdout", "kfold", "repeated_kfold")


# The reference: the switch as it stood on turbotab-next 0b0b10ed.
def old_model_clause(family, validation):
    if family in ("elastic_net", "screened_elastic_net") and validation in ("kfold", "repeated_kfold"):
        return "elastic-net parameters were tuned in an inner CV nested in an outer CV"
    if family in ("elastic_net", "screened_elastic_net"):
        return "elastic-net parameters were tuned by an inner CV within the rows each fit was given"
    return None


OLD_MIX = {"elastic_net", "screened_elastic_net"}
OLD_TESTS = {"featurewise"}  # chain_choices, methods_paragraph and fit_methods each tested for it


def test_the_clause_is_written_by_exactly_the_families_the_old_switch_named():
    assert {f.key for f in families() if omics.tunes_a_mix(f)} == OLD_MIX


def test_the_feature_wise_tests_are_exactly_the_family_the_old_switch_named():
    assert {f.key for f in families() if omics.tests_each_feature(f)} == OLD_TESTS


@pytest.mark.parametrize("validation", VALIDATIONS)
def test_every_family_writes_the_clause_it_wrote(validation):
    for family in families():
        assert omics.model_clause(family.key, validation) == old_model_clause(family.key, validation), \
            family.key


def test_an_unregistered_key_writes_nothing_and_tests_nothing():
    assert omics.model_clause("not_a_family", "kfold") is None
    assert omics.tests_each_feature("not_a_family") is False


def test_multiplicity_is_a_choice_of_the_tests_only_family(monkeypatch):
    monkeypatch.setattr(omics, "multiplicity_policy", lambda state: {"method": "bh"})
    state = SimpleNamespace()
    for family in families():
        choices = omics.chain_choices(state, [], family.key)
        assert ("multiplicity" in choices) == (family.key in OLD_TESTS), family.key
    assert "multiplicity" not in omics.chain_choices(state, [])


def test_a_family_under_another_key_is_read_by_what_it_declares():
    """The key is nothing: a copy of the elastic net, or of the feature-wise family, under another
    key is read the same, and the old keys without the declaration are not."""
    from turbotab.core.models import get_family

    net, tests = type(get_family("elastic_net")), type(get_family("featurewise"))
    probe_net = type("Net", (net,), {"key": "probe_net"})()
    probe_tests = type("Tests", (tests,), {"key": "probe_tests"})()
    assert omics.tunes_a_mix(probe_net) and omics.model_clause(probe_net, "kfold")
    assert omics.tests_each_feature(probe_tests)
    assert not omics.tunes_a_mix(probe_tests) and not omics.tests_each_feature(probe_net)
    flat = type("Flat", (net,), {"key": "elastic_net", "tuning": None})()
    assert not omics.tunes_a_mix(flat) and omics.model_clause(flat, "kfold") is None
    predicting = type("Predicting", (tests,), {"key": "featurewise", "predicts": True})()
    assert not omics.tests_each_feature(predicting)


def test_ridge_declares_a_path_without_a_mix_and_writes_no_clause():
    """Ridge is a path family too, but its one penalty is not a mix: it was not named, so it is
    not read as the elastic net."""
    from turbotab.core.models import get_family

    assert get_family("ridge").tuning.kind == "path"
    assert not omics.tunes_a_mix("ridge") and omics.model_clause("ridge", "kfold") is None
