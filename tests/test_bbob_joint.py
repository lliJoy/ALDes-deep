from __future__ import annotations

import json

import numpy as np
import torch

import train_bbob_joint as joint
from bbob_backend import BBOBProtocol
from bbob_features import FeatureBank, extract_features


def test_seed_zero_split_is_fixed_and_disjoint():
    train_ids, test_ids = joint.split_functions(0)
    assert train_ids == [1, 2, 3, 4, 5, 6, 7, 8, 11, 13, 14, 15, 16, 18, 19, 23]
    assert test_ids == [9, 10, 12, 17, 20, 21, 22, 24]
    assert sorted(train_ids + test_ids) == list(range(1, 25))


def test_continuous_features_are_deterministic_and_test_scaling_is_frozen():
    first = extract_features(1, 10, 1)
    np.testing.assert_array_equal(first, extract_features(1, 10, 1))
    assert first.shape == (32,)
    assert np.isfinite(first).all()
    bank = FeatureBank()
    keys = [(f, 10, 1) for f in joint.split_functions(0)[0]]
    bank.fit(keys)
    assert set(bank.raw) == set(keys)
    expected_mean = np.mean([bank.sample(key) for key in keys], axis=0)
    np.testing.assert_allclose(bank.mean, expected_mean)
    metadata = bank.metadata()
    bank.transform((9, 10, 1))
    assert bank.metadata() == metadata
    restored = FeatureBank.from_metadata(metadata)
    np.testing.assert_array_equal(
        bank.transform((9, 10, 1)), restored.transform((9, 10, 1))
    )


def test_shuffled_sampling_covers_only_training_problems_and_uses_one_objective(
    monkeypatch,
):
    protocol = BBOBProtocol(
        train_dimensions=(10,), test_dimensions=(10,), allow_instance_overlap=True
    )
    train_ids, _ = joint.split_functions(0)
    bank = FeatureBank()
    bank.fit([(fid, 10, 1) for fid in train_ids])
    calls = []

    class Objective:
        def __init__(self, *_args):
            pass

        def evaluate(self, actions, fid):
            calls.append(fid)
            records = [type("Record", (), {"raw": np.ones((1, 1))}) for _ in actions]
            return [1.0] * len(actions), records

    monkeypatch.setattr(joint, "BBOBObjective", Objective)
    objective = joint.SampledObjective(protocol, 1, train_ids, bank, 2)
    selected = []
    for epoch in range(32):
        src, key = objective.batch(epoch)
        assert src.shape == (2, 32)
        np.testing.assert_array_equal(src.cpu().numpy()[0], bank.transform(key))
        selected.append(key[0])
        objective([[], []], key)
    assert calls == selected
    assert set(selected[:16]) == set(train_ids)
    assert set(selected[16:]) == set(train_ids)


def test_shuffled_sampling_balances_instances_and_feature_seeds(monkeypatch):
    protocol = BBOBProtocol(
        train_dimensions=(10,),
        test_dimensions=(10,),
        train_instances=(1, 2),
        test_instances=(1,),
        allow_instance_overlap=True,
    )
    train_ids = [1, 2]
    feature_seeds = (3, 7)
    bank = FeatureBank()
    cases = [
        (fid, 10, instance, feature_seed)
        for fid in train_ids
        for instance in protocol.train_instances
        for feature_seed in feature_seeds
    ]
    bank.fit(cases)

    objective = joint.SampledObjective(
        protocol, 1, train_ids, bank, 2, feature_seeds=feature_seeds
    )
    selected = []
    for epoch in range(len(cases)):
        src, key = objective.batch(epoch)
        selected.append(objective.selected_case)
        assert key == objective.selected_case[:3]
        np.testing.assert_array_equal(
            src.cpu().numpy()[0], bank.transform(objective.selected_case)
        )

    assert set(selected) == set(cases)


def test_conditioned_weights_update_and_reload_without_test_training(
    tmp_path, monkeypatch
):
    original = joint.training.get_model
    captured = []

    def get_model(mode):
        assert mode == "continual"
        model, optimizer = original(mode)
        film = model.decoder.layers[0].attention_film.modulation.weight
        captured.append((model, film.detach().clone()))
        return model, optimizer

    monkeypatch.setattr(joint.training, "get_model", get_model)
    joint.main(
        [
            "--operator-space",
            "primitive",
            "--epochs",
            "2",
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
            "40",
            "--test-runs",
            "1",
            "--feature-samples",
            "100",
            "--evaluate-test",
            "--output-dir",
            str(tmp_path),
        ]
    )
    assert len(captured) == 1
    model, before = captured[0]
    film = model.decoder.layers[0].attention_film.modulation.weight
    assert not torch.equal(before, film)
    summary = json.loads((tmp_path / "bbob_joint_seed1.json").read_text())
    train_ids, heldout_ids = joint.split_functions(0)
    assert summary["protocol"]["train_dimensions"] == [10]
    assert summary["protocol"]["test_dimensions"] == [10]
    assert len(summary["training_history"]) == 2
    assert all(e["function_id"] in train_ids for e in summary["training_history"])
    assert set(map(int, summary["test"]["functions"])) == set(heldout_ids)
    assert len(list(tmp_path.glob("*.pt"))) == 1
    checkpoint = tmp_path / "bbob_joint_seed1.pt"
    restored, metadata = joint.load_checkpoint(checkpoint)
    bank = FeatureBank.from_metadata(metadata["features"])
    vector = bank.transform((9, 10, 1))
    torch.testing.assert_close(
        joint.infer(restored, vector), joint.infer(model, vector), rtol=0, atol=0
    )
    # Identical weights can produce different distributions when features change.
    target = torch.full((1, 1), 17, device=joint.conf.device)
    attention = torch.arange(27, device=joint.conf.device).unsqueeze(0)
    action = joint.infer(model, vector)
    with torch.no_grad():
        _, _, first = model(
            torch.as_tensor(vector, device=joint.conf.device).unsqueeze(0),
            target,
            attention,
            action,
        )
        _, _, second = model(
            torch.zeros((1, 32), device=joint.conf.device), target, attention, action
        )
    assert not torch.allclose(first, second)
    frozen = [p.detach().clone() for p in restored.parameters()]
    joint.evaluate_heldout(
        restored,
        BBOBProtocol(
            **{
                k: tuple(v) if isinstance(v, list) else v
                for k, v in metadata["protocol"].items()
            }
        ),
        metadata["seed"],
        heldout_ids,
        bank,
    )
    assert all(torch.equal(a, b) for a, b in zip(frozen, restored.parameters()))
    joint.main(
        ["--evaluate-only", str(checkpoint), "--output-dir", str(tmp_path / "reloaded")]
    )
    assert len(captured) == 1
    result = json.loads((tmp_path / "reloaded/evaluation.json").read_text())
    assert result["test"] == summary["test"]
