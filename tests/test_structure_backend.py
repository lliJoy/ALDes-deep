from __future__ import annotations

import json

import numpy as np
import pytest
import torch

import train_bbob_joint as joint
from bbob_backend import BBOBProtocol
from models.model.transformer import Transformer
from structure_backend import StructureBBOBObjective, StructureLibrary, run_sequence
from structure_grammar import StructureGrammar


@pytest.fixture(scope="module")
def library():
    return StructureLibrary.load()


def test_new_grammar_blocks_consecutive_repeats_and_enforces_length():
    grammar = StructureGrammar(10, 3)
    assert grammar.decode([10, 0, 4, 0, 11, 11]) == [0, 4, 0]
    assert not grammar.allowed_next_tokens([10, 0])[0]
    assert grammar.allowed_next_tokens([10, 0])[4]
    assert grammar.allowed_next_tokens([10, 0, 4, 0]).sum() == 1
    for invalid in (
        [10, 11],
        [10, 0],
        [10, 0, 0, 11],
        [10, 0, 11, 4],
        [10, 0, 1, 2, 3, 11],
    ):
        with pytest.raises(ValueError):
            grammar.decode(invalid)


def test_structure_length_range_can_be_variable_or_fixed():
    ranged = StructureGrammar(10, max_structures=4, min_structures=2)
    assert not ranged.allowed_next_tokens([10, 0])[11]
    assert ranged.allowed_next_tokens([10, 0, 1])[11]
    assert ranged.decode([10, 0, 1, 11, 11, 11]) == [0, 1]

    fixed = StructureGrammar(10, max_structures=3, min_structures=3)
    assert not fixed.allowed_next_tokens([10, 0, 1])[11]
    assert fixed.allowed_next_tokens([10, 0, 1, 2])[11]
    assert fixed.decode([10, 0, 1, 2, 11]) == [0, 1, 2]


def test_exported_structure_configs_and_parameters_are_preserved(library):
    assert [e["structure_id"] for e in library.entries] == [1, 2, 3, 4]
    from aldes_graphs.NEUROEA.support.rng import NumpyTorchRNG

    from structure_backend import GraphOperator

    for entry in library.entries:
        operator = GraphOperator(entry["structure_config"], NumpyTorchRNG(0))
        for spec, block in zip(entry["structure_config"]["blocks"], operator.blocks):
            np.testing.assert_array_equal(spec["parameters"], block.parameter.numpy())
    assert StructureLibrary.from_metadata(library.metadata()).digest == library.digest


def test_structure_library_can_exclude_structure_and_rebuild_tokens(library):
    filtered = library.excluding({2})
    ids = [entry["structure_id"] for entry in filtered.entries]
    assert 2 not in ids
    assert len(ids) == 3
    assert "G002" not in filtered.metadata()["token_mapping"].values()
    with pytest.raises(ValueError, match="Unknown excluded"):
        library.excluding({999})


@pytest.mark.parametrize("token", range(4))
def test_every_graph_runs_as_repeatable_operator_with_strict_budget(library, token):
    other = (token + 1) % 4
    result = run_sequence(
        library, [4, token, other, token, 5], 3, 10, 1, 1200, 0
    )
    assert result.evaluations == 1200
    assert result.initializations == 1
    assert result.completed_steps >= 1
    assert np.isfinite(result.best)


def test_population_is_passed_from_previous_structure_not_reinitialized(library):
    calls = []

    def observe(index, before, after, problem):
        calls.append((index, before, after))
        assert len(before) == len(after) == 100
        assert problem.initializations == 1
        if len(calls) > 1:
            assert before is calls[-2][2]

    result = run_sequence(
        library, [4, 0, 2, 0, 5], 1, 10, 1, 2500, 0, observer=observe
    )
    assert [c[0] for c in calls[:3]] == [0, 2, 0]
    assert result.evaluations == 2500


def test_decoder_replay_likelihood_with_repetition_and_variable_length(library):
    grammar = StructureGrammar(10, 3)
    model = Transformer(
        dec_voc_size=12,
        d_model=32,
        n_head=4,
        max_len=5,
        ffn_hidden=64,
        n_layers=1,
        drop_prob=0.0,
        device=torch.device("cpu"),
        condition_on_features=True,
        grammar=grammar,
    )
    src, target, attention = (
        torch.randn(2, 32),
        torch.full((2, 1), 10),
        torch.zeros((2, 27)),
    )
    actions = torch.tensor([[10, 0, 11, 11, 11], [10, 4, 5, 4, 11]])
    returned, _, logp = model(src, target, attention, actions)
    torch.testing.assert_close(returned, actions)
    assert logp[0, 2:].count_nonzero() == 0  # Padding EOS only; first EOS is a choice.
    (-logp.sum()).backward()
    film = model.decoder.layers[0].attention_film.modulation
    assert film.weight.grad.count_nonzero()
    generated, _, _ = model(src, target, attention)
    for row in generated:
        grammar.decode(row.detach().numpy())


def test_graph_candidates_share_randomness_and_test_does_not_shift_training(library):
    protocol = BBOBProtocol(
        train_dimensions=(10,),
        test_dimensions=(10,),
        allow_instance_overlap=True,
        population_size=100,
        train_evaluations=800,
        test_evaluations=800,
        train_runs=1,
        test_runs=1,
    )
    actions = [[4, 0, 5, 5], [4, 2, 0, 5]]
    objective = StructureBBOBObjective(protocol, 1, library)
    first = objective.evaluate(actions, 1)
    reversed_first = StructureBBOBObjective(protocol, 1, library).evaluate(
        actions[::-1], 1
    )
    np.testing.assert_array_equal(first[0], reversed_first[0][::-1])
    objective.evaluate(actions, 1, test=True)
    second = objective.evaluate(actions, 1)
    reference = StructureBBOBObjective(protocol, 1, library)
    reference.evaluate(actions, 1)
    np.testing.assert_array_equal(second[0], reference.evaluate(actions, 1)[0])


def test_structure_training_saves_and_reloads_shared_weights(tmp_path, monkeypatch):
    captured = []
    original = joint.save_checkpoint

    def save(path, model, metadata):
        captured.append(model)
        original(path, model, metadata)

    monkeypatch.setattr(joint, "save_checkpoint", save)
    ppo = joint.training.PPO
    updates = []

    def capture_update(model, *args, **kwargs):
        film = model.decoder.layers[0].attention_film.modulation.weight
        before = film.detach().clone()
        result = ppo(model, *args, **kwargs)
        updates.append(not torch.equal(before, film))
        return result

    monkeypatch.setattr(joint.training, "PPO", capture_update)
    joint.main(
        [
            "--epochs",
            "2",
            "--ppo-epochs",
            "1",
            "--batch-size",
            "4",
            "--prediction-length",
            "4",
            "--train-evaluations",
            "1000",
            "--train-runs",
            "1",
            "--test-evaluations",
            "500",
            "--test-runs",
            "1",
            "--evaluate-test",
            "--output-dir",
            str(tmp_path),
        ]
    )
    summary = json.loads((tmp_path / "bbob_joint_seed1.json").read_text())
    assert updates == [True]
    assert summary["operator_space"] == "structures"
    assert summary["excluded_structures"] == []
    assert summary["min_structures"] == 4
    assert summary["max_structures"] == 4
    assert list(summary["structure_library"]["token_mapping"].values()) == [
        "G001",
        "G002",
        "G003",
        "G004",
    ]
    assert summary["protocol"]["population_size"] == 100
    model, _metadata = joint.load_checkpoint(tmp_path / "bbob_joint_seed1.pt")
    assert model.decoder.vocab_size == 6
    assert model.decoder.grammar.min_structures == 4
    assert model.decoder.grammar.max_structures == 4
    assert model.structure_library.digest == captured[0].structure_library.digest
    for a, b in zip(model.parameters(), captured[0].parameters()):
        torch.testing.assert_close(a, b, rtol=0, atol=0)
    assert set(map(int, summary["test"]["functions"])) == set(
        joint.split_functions(0)[1]
    )
    assert len(summary["training_history"]) == 2
    joint.main(
        [
            "--evaluate-only",
            str(tmp_path / "bbob_joint_seed1.pt"),
            "--output-dir",
            str(tmp_path / "reloaded"),
        ]
    )
    restored = json.loads((tmp_path / "reloaded/evaluation.json").read_text())
    assert restored["test"] == summary["test"]
