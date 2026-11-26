"""Train independent ALDes policies on IOH's continuous BBOB F1--F24."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

import torch

import conf
import train as training
from aldes_setting import begin_index
from bbob_backend import BBOBObjective, BBOBProtocol, component_names
from util.my_util import seed_torch


def _integers(text):
    try:
        values = tuple(dict.fromkeys(int(item.strip()) for item in text.split(",")))
    except ValueError as exc:
        raise argparse.ArgumentTypeError("Expected comma-separated integers.") from exc
    if not values:
        raise argparse.ArgumentTypeError("At least one integer is required.")
    return values


def _positive(text):
    value = int(text)
    if value < 1:
        raise argparse.ArgumentTypeError("Expected a positive integer.")
    return value


def build_parser(*, include_problems=True):
    parser = argparse.ArgumentParser(description=__doc__)
    if include_problems:
        parser.add_argument(
            "--problems", type=_integers, default=(1,), help="BBOB IDs 1..24"
        )
    parser.add_argument("--seeds", type=_integers, default=(1,))
    parser.add_argument("--train-dimensions", type=_integers, default=(5, 10, 20))
    parser.add_argument("--test-dimensions", type=_integers, default=(40,))
    parser.add_argument("--train-instances", type=_integers, default=(1,))
    parser.add_argument("--test-instances", type=_integers, default=(1,))
    parser.add_argument("--epochs", type=_positive, default=conf.total_epoch)
    parser.add_argument("--ppo-epochs", type=_positive, default=conf.ppo_epoch)
    parser.add_argument("--batch-size", type=_positive, default=conf.batch_size_src)
    parser.add_argument("--population-size", type=_positive, default=100)
    parser.add_argument("--train-evaluations", type=_positive, default=10000)
    parser.add_argument("--test-evaluations", type=_positive, default=10000)
    parser.add_argument("--train-runs", type=_positive, default=5)
    parser.add_argument("--test-runs", type=_positive, default=30)
    parser.add_argument("--evaluate-test", action="store_true")
    parser.add_argument("--output-dir", type=Path, default=None)
    return parser


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    if any(p < 1 or p > 24 for p in args.problems):
        parser.error("BBOB problem IDs must be in 1..24.")
    if any(seed < 0 or seed > 2**32 - 1 for seed in args.seeds):
        parser.error("Seeds must be in 0..2**32-1.")
    try:
        protocol = BBOBProtocol(
            **{
                name: getattr(args, name, field.default)
                for name, field in BBOBProtocol.__dataclass_fields__.items()
            }
        )
    except ValueError as exc:
        parser.error(str(exc))
    output = args.output_dir or training.logs.log_dir / "bbob"
    output.mkdir(parents=True, exist_ok=True)
    training.EWC_ = None
    print(
        f"BBOB train dimensions={protocol.train_dimensions}; test={protocol.test_dimensions}"
    )
    print(
        "Continuous operators; [-5,5] box; serial evaluation; cost=mean(log1p(f-f_opt))"
    )
    for seed in args.seeds:
        for problem_id in args.problems:
            seed_torch(seed)
            training.logs.seed, training.logs.problem_id = seed, problem_id
            objective = BBOBObjective(protocol, seed)
            training.logs.write_log({"suite": "BBOB", "protocol": asdict(protocol)})
            print(f"Train BBOB F{problem_id}, seed={seed}")
            model, optimizer = training.get_model("single")
            target = torch.full((args.batch_size, 1), begin_index, device=conf.device)
            attention = torch.arange(27, device=conf.device).repeat(args.batch_size, 1)
            action = training.PPO(
                model,
                optimizer,
                conf.clip,
                args.epochs,
                args.ppo_epochs,
                None,
                0.2,
                None,
                target,
                attention,
                problem_id,
                evaluator=objective,
            )
            sequence = action[0].detach().cpu().tolist()
            result = {
                "suite": "BBOB",
                "problem_id": problem_id,
                "seed": seed,
                "protocol": asdict(protocol),
                "epochs": args.epochs,
                "ppo_epochs": args.ppo_epochs,
                "batch_size": args.batch_size,
                "boundary_handling": "clip to [-5,5]",
                "cost": "mean(log1p(max(f-f_opt,0)))",
                "action_tokens": sequence,
                "components": component_names(sequence),
            }
            print(f"Inferred continuous algorithm: {result['components']}")
            training.logs.write_log(result)
            if args.evaluate_test:
                costs, records = objective.evaluate(action, problem_id, test=True)
                record = records[0]
                result["test"] = {
                    "cost": costs[0],
                    "instances": record.instances,
                    "raw_objectives": record.raw.tolist(),
                    "optima": record.optimum[:, 0].tolist(),
                    "gaps": record.gaps.tolist(),
                    "mean_gap": float(record.gaps.mean()),
                }
                print(
                    f"Test mean gap={result['test']['mean_gap']:.6g}; log-gap cost={costs[0]:.6g}"
                )
            destination = output / f"bbob_f{problem_id}_seed{seed}.json"
            destination.write_text(
                json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8"
            )
            print(f"Saved {destination}")


if __name__ == "__main__":
    main()
