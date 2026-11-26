from __future__ import annotations

import json

import ioh
import numpy as np
import pytest
import torch
from autooptlib.aldes import EvaluationConfig

import train as training
import train_bbob
from bbob_backend import BBOBEvaluator, BBOBObjective, BBOBProtocol, decode_bbob

GAUSSIAN = [17, 0, 29, 8, 29, 12, 29, 18]
RESTART = [17, 0, 29, 11, 29, 12, 29, 18]


def _config(seed=17):
    return EvaluationConfig(population_size=4, evaluations=20, runs=2, seed=seed)


@pytest.mark.parametrize("function_id", range(1, 25))
@pytest.mark.parametrize("dimension", [5, 40])
def test_all_bbob_functions_use_real_coordinates_and_minimization(
    function_id, dimension
):
    evaluator = BBOBEvaluator(function_id, [(dimension, 1)], config=_config())
    point = np.resize(np.array([0.125, 0.25, -0.5, -1.75, 2.5]), dimension)
    actual, _, _ = evaluator.problems[0].evaluate(evaluator.data[0], point)
    reference = ioh.get_problem(function_id, 1, dimension, ioh.ProblemClass.BBOB)
    assert actual == pytest.approx(reference(point))
    assert actual != pytest.approx(reference(point.astype(int)))
    values = evaluator.evaluate(GAUSSIAN)
    assert values.shape == (1, 2)
    assert np.isfinite(values).all()
    assert np.all(values >= evaluator.data[0].optimum.y - 1e-8)


@pytest.mark.parametrize(
    "search",
    [
        [8, 29],
        [9, 23, 29],
        [10, 23, 29],
        [11, 29],
        [4, 29, 8, 29],
        [5, 29, 8, 29],
        [6, 23, 29, 8, 29],
        [7, 23, 29, 8, 29],
        [8, 30, 21],
    ],
)
def test_continuous_search_operators_execute_within_budget(search):
    evaluator = BBOBEvaluator(3, [(5, 1)], config=_config())
    sequence = [17, 0, 29, *search, 12, 29, 18]
    design = decode_bbob(sequence, evaluator.problems, evaluator.setting)
    design.evaluate(
        evaluator.problems, evaluator.data, evaluator.setting, seed_instance=[0]
    )
    assert np.isfinite(design.performance).all()
    # The objective is called exactly once per evaluation, including initialization.
    assert evaluator.data[0].state.evaluations <= 2 * 20


@pytest.mark.parametrize("choose", range(4))
@pytest.mark.parametrize("update", range(12, 17))
def test_choose_and_update_execute_on_bbob(choose, update):
    sequence = [17, choose, 29, 8, 29, update]
    if update == 16:
        sequence.append(23)
    sequence += [29, 18]
    evaluator = BBOBEvaluator(1, [(5, 1)], config=_config())
    assert np.isfinite(evaluator.evaluate(sequence)).all()


@pytest.mark.parametrize("fork_token", [21, 22])
def test_forks_and_real_mutations_execute(fork_token):
    sequence = [17, 0, 31, fork_token, 7, 23, 29, 8, 29, 12, 29, 18]
    evaluator = BBOBEvaluator(1, [(5, 1)], config=_config())
    assert np.isfinite(evaluator.evaluate(sequence)).all()


def test_protocol_split_and_candidate_randomness():
    protocol = BBOBProtocol(
        population_size=4,
        train_evaluations=20,
        test_evaluations=20,
        train_runs=1,
        test_runs=1,
    )
    objective = BBOBObjective(protocol, seed=5)
    costs, records = objective.evaluate([GAUSSIAN, RESTART], 1)
    assert records[0].instances == [(5, 1), (10, 1), (20, 1)]
    _, reversed_records = BBOBObjective(protocol, seed=5).evaluate(
        [RESTART, GAUSSIAN], 1
    )
    np.testing.assert_array_equal(records[0].raw, reversed_records[1].raw)
    assert costs[0] == pytest.approx(np.log1p(records[0].gaps).mean())
    first_test = objective.evaluate([GAUSSIAN], 1, test=True)[1][0]
    assert first_test.instances == [(40, 1)]
    objective.evaluate([GAUSSIAN], 1)
    repeated_test = objective.evaluate([GAUSSIAN], 1, test=True)[1][0]
    np.testing.assert_array_equal(first_test.raw, repeated_test.raw)


def test_protocol_rejects_overlap():
    with pytest.raises(ValueError, match="must differ"):
        BBOBProtocol(test_dimensions=(10,))


def test_cli_trains_and_tests_real_bbob(tmp_path, monkeypatch):
    original = training.get_model
    captured = []

    def get_model(mode):
        model, optimizer = original(mode)
        captured.append((model, [p.detach().clone() for p in model.parameters()]))
        return model, optimizer

    monkeypatch.setattr(training, "get_model", get_model)
    train_bbob.main(
        [
            "--problems",
            "1,24",
            "--epochs",
            "1",
            "--ppo-epochs",
            "1",
            "--batch-size",
            "8",
            "--population-size",
            "4",
            "--train-evaluations",
            "80",
            "--train-runs",
            "1",
            "--test-evaluations",
            "80",
            "--test-runs",
            "1",
            "--evaluate-test",
            "--output-dir",
            str(tmp_path),
        ]
    )
    for function_id in (1, 24):
        result = json.loads((tmp_path / f"bbob_f{function_id}_seed1.json").read_text())
        assert result["protocol"]["train_dimensions"] == [5, 10, 20]
        assert result["test"]["instances"] == [[40, 1]]
        assert np.isfinite(result["test"]["mean_gap"])
    for model, before in captured:
        assert any(
            not torch.equal(old, new) for old, new in zip(before, model.parameters())
        )
