from __future__ import annotations

import numpy as np
import pytest

import bbob_features as features
from pflacco_feature import PAPER_FEATURE_NAMES


@pytest.mark.parametrize("function_id", range(1, 25))
def test_pflacco_outputs_use_same_100_samples_and_exact_objective_budget(
    function_id, monkeypatch
):
    get_problem = features.ioh.get_problem
    problems, outputs, samples = [], {}, []

    def capture_problem(*args):
        objective = get_problem(*args)
        problems.append(objective)
        return objective

    monkeypatch.setattr(features.ioh, "get_problem", capture_problem)
    for name in (
        "calculate_ela_meta",
        "calculate_dispersion",
        "calculate_information_content",
        "calculate_nbc",
    ):
        original = getattr(features, name)

        def capture(x, y, _original=original, **kwargs):
            assert x.shape == (100, 10)
            assert y.shape == (100,)
            samples.append((x.to_numpy().copy(), y.copy()))
            result = _original(x, y, **kwargs)
            outputs.update(result)
            return result

        monkeypatch.setattr(features, name, capture)
    vector = features.extract_features(function_id, 10, 1)
    assert len(problems) == 1
    assert problems[0].state.evaluations == 100
    assert features.FEATURE_NAMES == PAPER_FEATURE_NAMES
    assert len(samples) == 4
    for x, y in samples[1:]:
        np.testing.assert_array_equal(x, samples[0][0])
        np.testing.assert_array_equal(y, samples[0][1])
    for name, actual in zip(features.FEATURE_NAMES, vector):
        assert actual == (0.0 if name.endswith("costs_runtime") else outputs[name])
    assert np.isfinite(vector).all()


def test_feature_budget_cannot_be_overridden_and_global_rng_is_preserved():
    with pytest.raises(ValueError, match="100"):
        features.extract_features(1, 10, 1, sample_count=500)
    with pytest.raises(ValueError, match="100"):
        features.FeatureBank(sample_count=99)
    state = np.random.get_state()
    features.extract_features(1, 10, 1)
    after = np.random.get_state()
    assert state[0] == after[0]
    np.testing.assert_array_equal(state[1], after[1])
    assert state[2:] == after[2:]


def test_feature_bank_caches_explicit_sampling_seeds_separately():
    bank = features.FeatureBank(seed=0)
    first = bank.sample((1, 10, 1, 0))
    second = bank.sample((1, 10, 1, 1))

    assert (1, 10, 1, 0) in bank.raw
    assert (1, 10, 1, 1) in bank.raw
    assert not np.array_equal(first, second)

    with pytest.raises(ValueError, match="Feature keys"):
        bank.sample((1, 10))


def test_singular_quadratic_condition_number_is_capped(monkeypatch):
    original = features.calculate_ela_meta

    def singular_condition(*args, **kwargs):
        result = original(*args, **kwargs)
        result["ela_meta.quad_simple.cond"] = np.inf
        return result

    monkeypatch.setattr(features, "calculate_ela_meta", singular_condition)
    vector = features.extract_features(1, 10, 1)
    index = features.FEATURE_NAMES.index("ela_meta.quad_simple.cond")

    assert vector[index] == features.QUADRATIC_CONDITION_CAP
    assert np.isfinite(vector).all()
