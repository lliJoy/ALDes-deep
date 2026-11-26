"""Shared feature-conditioned BBOB policy: one sampled problem per epoch."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np
import torch

import conf
import train as training
from aldes_setting import total_index
from bbob_backend import BBOBObjective, BBOBProtocol, component_names
from bbob_features import FeatureBank
from models.model.transformer import Transformer
from structure_backend import DEFAULT_LIBRARY, StructureBBOBObjective, StructureLibrary
from structure_grammar import StructureGrammar
from train_bbob import _integers, _positive
from train_bbob import build_parser as independent_parser
from util.my_util import seed_torch


def split_functions(seed=0):
    ids = sorted(
        int(v)
        for v in np.random.default_rng(seed).choice(np.arange(1, 25), 16, replace=False)
    )
    return ids, sorted(set(range(1, 25)) - set(ids))


class SampledObjective:
    def __init__(
        self,
        protocol,
        seed,
        function_ids,
        features,
        batch_size,
        library=None,
        max_structures=6,
        min_structures=1,
        feature_seeds=(0,),
    ):
        self.protocol, self.seed = protocol, seed
        self.features, self.batch_size = features, batch_size
        self.cases = [
            (f, d, i, feature_seed)
            for f in function_ids
            for d in protocol.train_dimensions
            for i in protocol.train_instances
            for feature_seed in feature_seeds
        ]
        self.keys = sorted({case[:3] for case in self.cases})
        self.rng = np.random.default_rng(np.random.SeedSequence([seed, 731]))
        self.order, self.objectives, self.history = [], {}, []
        self.selected = None
        self.selected_case = None
        self.library = library
        self.max_structures, self.min_structures = max_structures, min_structures

    def batch(self, epoch):
        if not self.order:
            self.order = self.rng.permutation(len(self.cases)).tolist()
        self.selected_case = self.cases[self.order.pop()]
        self.selected = self.selected_case[:3]
        vector = self.features.transform(self.selected_case)
        src = torch.as_tensor(vector, device=conf.device).repeat(self.batch_size, 1)
        print(
            f"Epoch {epoch + 1}: sampled "
            f"(function, dimension, instance, feature_seed)={self.selected_case}"
        )
        return src, self.selected

    def __call__(self, actions, key):
        if key != self.selected or key not in self.keys:
            raise ValueError("Only the sampled training problem can be evaluated.")
        fid, dimension, instance = key
        if key not in self.objectives:
            protocol = replace(
                self.protocol,
                train_dimensions=(dimension,),
                train_instances=(instance,),
            )
            self.objectives[key] = (
                BBOBObjective(protocol, self.seed)
                if self.library is None
                else StructureBBOBObjective(
                    protocol,
                    self.seed,
                    self.library,
                    self.max_structures,
                    self.min_structures,
                )
            )
        costs, records = self.objectives[key].evaluate(actions, fid)
        self.history.append(
            {
                "function_id": fid,
                "dimension": dimension,
                "instance_id": instance,
                "feature_seed": self.selected_case[3],
                "mean_cost": float(np.mean(costs)),
            }
        )
        return costs, [record.raw for record in records]


def model_config(library=None, max_structures=6):
    return {
        "dec_voc_size": total_index if library is None else len(library.entries) + 2,
        "d_model": conf.d_model,
        "n_head": conf.n_heads,
        "max_len": conf.max_len if library is None else max_structures + 2,
        "ffn_hidden": conf.ffn_hidden,
        "n_layers": conf.n_layers,
        "drop_prob": conf.drop_prob,
        "condition_on_features": True,
        "conditioning_mode": "film",
        "feature_dim": 32,
        "context_hidden_dim": 128,
    }


def initialize_structure_policy(model, output_std=5e-4):
    """Initialize a trainable structure policy with near-uniform token logits."""

    model.apply(training.initialize_weights)
    model.reset_conditioning_identity()
    with torch.no_grad():
        torch.nn.init.normal_(model.decoder.linear.weight, mean=0.0, std=output_std)
        if model.decoder.linear.bias is not None:
            model.decoder.linear.bias.zero_()


def save_checkpoint(path, model, metadata):
    temporary = path.with_suffix(".tmp")
    torch.save(
        {
            "format": "aldes-bbob-structures-film-v5"
            if hasattr(model, "structure_library")
            else "aldes-bbob-pflacco-film-v4",
            "model_config": getattr(model, "architecture_config", model_config()),
            "model_state_dict": {
                k: v.detach().cpu() for k, v in model.state_dict().items()
            },
            "metadata": metadata,
        },
        temporary,
    )
    temporary.replace(path)


def load_checkpoint(path):
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if payload.get("format") not in {
        "aldes-bbob-pflacco-v3",
        "aldes-bbob-structures-v4",
        "aldes-bbob-pflacco-film-v4",
        "aldes-bbob-structures-film-v5",
    }:
        raise ValueError(
            "Expected PFLACCO BBOB v3 weights; retrain weights made with older features."
        )
    library, grammar = None, None
    if payload["format"] in {
        "aldes-bbob-structures-v4",
        "aldes-bbob-structures-film-v5",
    }:
        library = StructureLibrary.from_metadata(
            payload["metadata"]["structure_library"]
        )
        grammar = StructureGrammar(
            len(library.entries),
            payload["metadata"]["max_structures"],
            payload["metadata"].get("min_structures", 1),
        )
    config = dict(payload["model_config"])
    if payload["format"] in {
        "aldes-bbob-pflacco-v3",
        "aldes-bbob-structures-v4",
    }:
        config.setdefault("conditioning_mode", "prefix")
    model = Transformer(**config, device=conf.device, grammar=grammar).to(conf.device)
    if library is not None:
        model.structure_library = library
        model.architecture_config = config
    model.load_state_dict(payload["model_state_dict"], strict=True)
    return model, payload["metadata"]


def infer(model, vector):
    model.eval()
    with torch.no_grad():
        actions, _, _ = model(
            torch.as_tensor(vector, device=conf.device).reshape(1, -1),
            torch.full((1, 1), model.decoder.begin_index, device=conf.device),
            torch.arange(27, device=conf.device).unsqueeze(0),
            reference=True,
        )
    return actions


def evaluate_heldout(model, protocol, seed, heldout_ids, features):
    results = {}
    for fid in heldout_ids:
        cases = []
        for dimension in protocol.test_dimensions:
            for instance in protocol.test_instances:
                vector = features.transform((fid, dimension, instance))
                action = infer(model, vector)
                selected = replace(
                    protocol, test_dimensions=(dimension,), test_instances=(instance,)
                )
                library = getattr(model, "structure_library", None)
                objective = (
                    BBOBObjective(selected, seed)
                    if library is None
                    else StructureBBOBObjective(
                        selected,
                        seed,
                        library,
                        model.decoder.grammar.max_structures,
                        model.decoder.grammar.min_structures,
                    )
                )
                costs, records = objective.evaluate(action, fid, test=True)
                record = records[0]
                sequence = action[0].cpu().tolist()
                cases.append(
                    {
                        "dimension": dimension,
                        "instance_id": instance,
                        "action_tokens": sequence,
                        "components": (
                            component_names(sequence)
                            if library is None
                            else library.names(
                                sequence,
                                model.decoder.grammar.max_structures,
                                model.decoder.grammar.min_structures,
                            )
                        ),
                        "cost": costs[0],
                        "raw_objectives": record.raw.tolist(),
                        "gaps": record.gaps.tolist(),
                        "optima": record.optimum[:, 0].tolist(),
                        "mean_gap": float(record.gaps.mean()),
                        "features": vector.tolist(),
                        "feature_evaluations": features.sample_count,
                    }
                )
                print(f"Held-out F{fid}, {dimension}D: log-gap={costs[0]:.6g}")
        results[str(fid)] = {
            "cases": cases,
            "mean_log_gap": float(np.mean([c["cost"] for c in cases])),
        }
    return {
        "mean_log_gap": float(np.mean([v["mean_log_gap"] for v in results.values()])),
        "functions": results,
    }


def build_parser():
    parser = argparse.ArgumentParser(
        description=__doc__,
        parents=[independent_parser(include_problems=False)],
        add_help=False,
    )
    parser.set_defaults(train_dimensions=(10,), test_dimensions=(10,))
    parser.set_defaults(population_size=100)
    parser.add_argument(
        "--operator-space", choices=("structures", "primitive"), default="structures"
    )
    parser.add_argument("--structure-library", type=Path, default=DEFAULT_LIBRARY)
    parser.add_argument(
        "--exclude-structures",
        default=None,
        help=(
            "comma-separated structure IDs to remove, e.g. G002,G004 "
            "(all structures are available by default)"
        ),
    )
    parser.add_argument("--min-structures", type=_positive, default=1)
    parser.add_argument("--max-structures", type=_positive, default=6)
    parser.add_argument(
        "--prediction-length",
        type=_positive,
        help="exact predicted structure count; overrides both length limits",
    )
    parser.add_argument("--split-seed", type=int, default=0)
    parser.add_argument(
        "--feature-samples", type=_positive, choices=(100,), default=100
    )
    parser.add_argument("--feature-seed", type=int, default=0)
    parser.add_argument(
        "--train-feature-seeds",
        type=_integers,
        default=None,
        help=(
            "comma-separated ELA sampling seeds used for training augmentation; "
            "defaults to --feature-seed"
        ),
    )
    parser.add_argument(
        "--entropy-coef",
        type=float,
        default=0.01,
        help="PPO entropy bonus coefficient (default: 0.01; use 0 to disable)",
    )
    parser.add_argument(
        "--evaluate-only",
        type=Path,
        help="load shared weights and frozen scaling; test only",
    )
    return parser


def _write_json(path, payload):
    path.write_text(
        json.dumps(payload, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    if not np.isfinite(args.entropy_coef) or args.entropy_coef < 0:
        parser.error("--entropy-coef must be finite and nonnegative.")
    if args.prediction_length is not None:
        args.min_structures = args.prediction_length
        args.max_structures = args.prediction_length
    if args.min_structures > args.max_structures:
        parser.error("--min-structures cannot exceed --max-structures.")
    excluded_value = args.exclude_structures
    if excluded_value is None:
        excluded_value = ""
    try:
        excluded_structures = {
            int(value.strip().upper().removeprefix("G"))
            for value in excluded_value.split(",")
            if value.strip()
        }
    except ValueError:
        parser.error("--exclude-structures expects IDs such as G015,G065.")
    if any(
        s < 0 or s > 2**32 - 1
        for s in [
            args.split_seed,
            args.feature_seed,
            *args.seeds,
            *(args.train_feature_seeds or ()),
        ]
    ):
        parser.error("Seeds must be in 0..2**32-1.")
    output = args.output_dir or training.logs.log_dir / "bbob_conditioned"
    if args.evaluate_only:
        model, metadata = load_checkpoint(args.evaluate_only)
        protocol = BBOBProtocol(
            **{
                k: tuple(v) if isinstance(v, list) else v
                for k, v in metadata["protocol"].items()
            }
        )
        features = FeatureBank.from_metadata(metadata["features"])
        test = evaluate_heldout(
            model, protocol, metadata["seed"], metadata["heldout_functions"], features
        )
        output.mkdir(parents=True, exist_ok=True)
        _write_json(
            output / "evaluation.json",
            {"checkpoint": str(args.evaluate_only), "metadata": metadata, "test": test},
        )
        return
    try:
        values = {
            name: getattr(args, name, field.default)
            for name, field in BBOBProtocol.__dataclass_fields__.items()
        }
        # Same dimension/instance IDs are allowed: training/test function IDs differ.
        values["allow_instance_overlap"] = True
        protocol = BBOBProtocol(**values)
    except ValueError as exc:
        parser.error(str(exc))
    train_ids, heldout_ids = split_functions(args.split_seed)
    train_feature_seeds = args.train_feature_seeds or (args.feature_seed,)
    library = (
        StructureLibrary.load(args.structure_library)
        if args.operator_space == "structures"
        else None
    )
    if library is not None and excluded_structures:
        try:
            library = library.excluding(excluded_structures)
        except ValueError as exc:
            parser.error(str(exc))
    if library is None and excluded_structures:
        parser.error("--exclude-structures requires --operator-space structures.")
    if library is not None and protocol.population_size != 100:
        parser.error("Exported structures require --population-size 100.")
    assert not set(train_ids) & set(heldout_ids)
    print(f"Training functions: {train_ids}; held out: {heldout_ids}")
    print(
        f"Train dimensions={protocol.train_dimensions}; test dimensions={protocol.test_dimensions}"
    )
    print(
        "Extracting training-only continuous features; one sampled problem per epoch."
    )
    if conf.d_model != 32:
        parser.error("The continuous feature schema currently requires d_model=32.")
    features = FeatureBank(sample_count=args.feature_samples, seed=args.feature_seed)
    keys = [
        (f, d, i, feature_seed)
        for f in train_ids
        for d in protocol.train_dimensions
        for i in protocol.train_instances
        for feature_seed in train_feature_seeds
    ]
    features.fit(keys)
    output.mkdir(parents=True, exist_ok=True)
    training.EWC_ = None
    for seed in args.seeds:
        seed_torch(seed)
        training.logs.seed, training.logs.problem_id = seed, "bbob_conditioned"
        if library is None:
            model, optimizer = training.get_model("continual")
        else:
            config = model_config(library, args.max_structures)
            model = Transformer(
                **config,
                device=conf.device,
                grammar=StructureGrammar(
                    len(library.entries), args.max_structures, args.min_structures
                ),
            ).to(conf.device)
            initialize_structure_policy(model)
            model.structure_library, model.architecture_config = library, config
            optimizer = torch.optim.Adam(
                model.parameters(),
                lr=conf.init_lr,
                weight_decay=conf.weight_decay,
                eps=conf.adam_eps,
            )
            print(
                f"Shared structure policy: {len(library.entries)} operators, "
                f"repetitions allowed, length={args.min_structures}..{args.max_structures}"
            )
        objective = SampledObjective(
            protocol,
            seed,
            train_ids,
            features,
            args.batch_size,
            library,
            args.max_structures,
            args.min_structures,
            train_feature_seeds,
        )
        metadata = {
            "suite": "BBOB",
            "mode": "sampled_feature_conditioned",
            "seed": seed,
            "split_seed": args.split_seed,
            "split_method": "numpy.default_rng.choice(1..24,16,replace=False), sorted",
            "train_functions": train_ids,
            "heldout_functions": heldout_ids,
            "protocol": asdict(protocol),
            "epochs": args.epochs,
            "ppo_epochs": args.ppo_epochs,
            "batch_size": args.batch_size,
            "entropy_coef": args.entropy_coef,
            "features": features.metadata(),
            "train_feature_seeds": train_feature_seeds,
            "training_raw_features": [
                {
                    "function_id": f,
                    "dimension": d,
                    "instance_id": i,
                    "feature_seed": feature_seed,
                    "values": features.sample((f, d, i, feature_seed)).tolist(),
                    "feature_evaluations": features.sample_count,
                }
                for f, d, i, feature_seed in keys
            ],
            "sampling": (
                "balanced shuffled cycles over training "
                "(function, dimension, instance, feature_seed); one per epoch"
            ),
            "cost": "mean(log1p(max(f-f_opt,0))) on sampled problem; baseline per problem configuration",
            "boundary_handling": "clip to [-5,5]",
        }
        if library is not None:
            metadata.update(
                {
                    "operator_space": "structures",
                    "structure_library": library.metadata(),
                    "excluded_structures": [
                        f"G{value:03d}" for value in sorted(excluded_structures)
                    ],
                    "min_structures": args.min_structures,
                    "max_structures": args.max_structures,
                    "schedule": "repeat generated sequence until FE budget; one graph population-update cycle per token; initialize once; trained graph parameters frozen",
                }
            )
        _write_json(output / f"config_seed{seed}.json", metadata)
        training.logs.write_log(metadata)
        training.PPO(
            model,
            optimizer,
            conf.clip,
            args.epochs,
            args.ppo_epochs,
            None,
            0.2,
            None,
            torch.full(
                (args.batch_size, 1), model.decoder.begin_index, device=conf.device
            ),
            torch.arange(27, device=conf.device).repeat(args.batch_size, 1),
            0,
            evaluator=objective,
            batch_provider=objective.batch,
            entropy_coef=args.entropy_coef,
        )
        checkpoint = output / f"bbob_joint_seed{seed}.pt"
        save_checkpoint(checkpoint, model, metadata)
        result = {
            **metadata,
            "checkpoint": str(checkpoint),
            "training_history": objective.history,
        }
        destination = output / f"bbob_joint_seed{seed}.json"
        _write_json(destination, result)
        print(f"Saved shared conditioned weights: {checkpoint}")
        if args.evaluate_test:
            result["test"] = evaluate_heldout(
                model, protocol, seed, heldout_ids, features
            )
            _write_json(destination, result)
        print(f"Saved {destination}")


if __name__ == "__main__":
    main()
