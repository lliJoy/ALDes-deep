import math

import numpy as np
import pytest
import torch

import train as training
from models.model.transformer import Transformer
from structure_grammar import StructureGrammar
from train_bbob_joint import build_parser, initialize_structure_policy


def policy():
    model = Transformer(
        dec_voc_size=4,
        d_model=8,
        n_head=2,
        max_len=4,
        ffn_hidden=16,
        n_layers=1,
        drop_prob=0,
        device=torch.device("cpu"),
        grammar=StructureGrammar(2, 2),
    )
    with torch.no_grad():
        model.decoder.linear.weight.zero_()
        model.decoder.linear.bias.zero_()
    return model


def test_exact_masked_entropy_and_finite_gradients():
    model = policy()
    actions = torch.tensor([[2, 0, 1, 3], [2, 1, 3, 3]])
    model(None, actions[:, :1], None, actions)
    entropy = model.decoder.last_entropy.squeeze(-1)
    assert entropy[0].tolist() == pytest.approx([math.log(2), math.log(2), 0])
    assert entropy[1].tolist() == pytest.approx([math.log(2), math.log(2), 0])
    assert model.decoder.last_entropy_mask.squeeze(-1).tolist() == [
        [True, True, False],
        [True, True, False],
    ]
    (-entropy.mean()).backward()
    assert all(
        torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None
    )


def test_ppo_entropy_flattens_policy_when_costs_are_equal(monkeypatch):
    class MemoryLog:
        def write_log(self, message):
            pass

        def dump_log(self, experience):
            pass

    torch.manual_seed(3)
    model = policy()
    with torch.no_grad():
        model.decoder.linear.bias[0] = 2
    initial = model.decoder.linear.bias[0].item()
    monkeypatch.setattr(training, "logs", MemoryLog())
    monkeypatch.setattr(training, "EWC_", None)
    monkeypatch.setattr(training, "init_lr", 0.1)
    training.PPO(
        model,
        torch.optim.SGD(model.parameters(), lr=0.1),
        1.0,
        1,
        1,
        None,
        0.2,
        None,
        torch.full((4, 1), 2),
        None,
        1,
        evaluator=lambda actions, _: (
            [1.0] * len(actions),
            [np.ones(1)] * len(actions),
        ),
        entropy_coef=0.1,
    )
    assert model.decoder.linear.bias[0].item() < initial
    assert all(torch.isfinite(p).all() for p in model.parameters())


def test_joint_entropy_cli_default_and_disable():
    assert build_parser().parse_args([]).entropy_coef == 0.01
    assert build_parser().parse_args(["--entropy-coef", "0"]).entropy_coef == 0
    assert build_parser().parse_args([]).exclude_structures is None
    exact = build_parser().parse_args(["--prediction-length", "4"])
    assert exact.prediction_length == 4
    fixed = build_parser().parse_args(
        [
            "--exclude-structures",
            "G015",
            "--min-structures",
            "3",
            "--max-structures",
            "3",
        ]
    )
    assert fixed.exclude_structures == "G015"
    assert fixed.min_structures == fixed.max_structures == 3


def test_structure_policy_starts_near_uniform_with_trainable_output():
    torch.manual_seed(1)
    model = Transformer(
        dec_voc_size=6,
        d_model=32,
        n_head=4,
        max_len=7,
        ffn_hidden=64,
        n_layers=1,
        drop_prob=0,
        device=torch.device("cpu"),
        condition_on_features=True,
        grammar=StructureGrammar(4, 5),
    )
    initialize_structure_policy(model)
    model.eval()
    captured = []
    hook = model.decoder.linear.register_forward_hook(
        lambda _module, _inputs, output: captured.append(output.detach())
    )
    with torch.no_grad():
        model(
            torch.randn(1, 32),
            torch.full((1, 1), 4),
            torch.zeros((1, 27)),
            reference=True,
        )
    hook.remove()
    probabilities = torch.softmax(captured[0][0, 0, :4], dim=0)
    assert probabilities.tolist() == pytest.approx([0.25] * 4, abs=0.002)
    assert model.decoder.linear.weight.count_nonzero()
