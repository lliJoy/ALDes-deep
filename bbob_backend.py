"""Continuous BBOB execution for ALDes's 32-token program grammar.

The grammar topology is retained, but tokens have the continuous meanings below.
This is a new search space, not the paper's discrete PBO experiment.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from types import SimpleNamespace

import ioh
import numpy as np
from autooptlib.aldes import AutoOptEvaluator, EvaluationConfig
from autooptlib.aldes.codec import _branches, _parse, _termination
from autooptlib.aldes.vocabulary import PARAMETER_INDICES, TOKEN_BY_INDEX
from autooptlib.components import get_component
from autooptlib.problems.base import make_problem
from autooptlib.utils.design import Design
from autooptlib.utils.design._helpers import (
    Pathway,
    PathwayParam,
    SearchParam,
    SearchStep,
)

# Token arity and crossover+mutation syntax stay identical to the trained model.
# Coordinate crossovers preserve floats; all discrete mutations are replaced.
BBOB_COMPONENTS = {
    **{i: TOKEN_BY_INDEX[i].name for i in range(17)},
    6: "cross_arithmetic",
    7: "cross_sim_binary",
    8: "search_mu_gaussian",
    9: "search_mu_polynomial",
    10: "search_mu_uniform",
    11: "reinit_continuous",
}


def component_names(sequence):
    return [BBOB_COMPONENTS[item.token] for item in _parse(sequence)]


def make_bbob_problem(function_id):
    """Adapt (dimension, IOH instance ID) pairs; enforce the [-5, 5] box."""
    if not 1 <= int(function_id) <= 24:
        raise ValueError("BBOB function ID must be in 1..24.")

    def data_factory(instance):
        dimension, instance_id = instance
        if dimension < 2 or instance_id < 1:
            raise ValueError("BBOB requires dimension >= 2 and instance ID >= 1.")
        return ioh.get_problem(
            int(function_id), instance_id, dimension, ioh.ProblemClass.BBOB
        )

    return make_problem(
        lambda decision, objective: float(objective(np.asarray(decision, dtype=float))),
        bounds=lambda instance: np.tile([[-5.0], [5.0]], (1, instance[0])),
        data_factory=data_factory,
        problem_type="continuous",
        name=f"ioh_bbob_f{function_id}",
    )


def _parameter(instruction, problems):
    if instruction.parameter_token is None:
        return None
    bounds, _ = get_component(BBOB_COMPONENTS[instruction.token])(problems, "parameter")
    if bounds is None:
        return None
    bounds = np.asarray(bounds, dtype=float).reshape(-1, 2)
    fraction = PARAMETER_INDICES.index(instruction.parameter_token) / 9.0
    return bounds[:, 0] + fraction * (bounds[:, 1] - bounds[:, 0])


def decode_bbob(sequence, problems, setting):
    """Decode pointers/branches using AutoOptLib's existing execution engine."""
    pathways, parameters = [], []
    for branch in _branches(_parse(sequence)):
        searches, search_parameters = [], []
        position = 1
        while position < len(branch) - 1:
            primary = branch[position]
            secondary = None
            if primary.token in {4, 5, 6, 7}:
                secondary = branch[position + 1]
                position += 1
            termination = _termination(primary, setting)
            if secondary is not None and secondary.pointer == "forward":
                termination = np.array([-np.inf, 1.0])
            searches.append(
                SearchStep(
                    primary=BBOB_COMPONENTS[primary.token],
                    secondary=(BBOB_COMPONENTS[secondary.token] if secondary else None),
                    termination=termination,
                )
            )
            search_parameters.append(
                SearchParam(
                    primary=_parameter(primary, problems),
                    secondary=_parameter(secondary, problems) if secondary else None,
                )
            )
            position += 1
        pathways.append(
            Pathway(
                choose=BBOB_COMPONENTS[branch[0].token],
                search=searches,
                update=BBOB_COMPONENTS[branch[-1].token],
                archive=[],
            )
        )
        parameters.append(
            PathwayParam(
                choose=_parameter(branch[0], problems),
                search=search_parameters,
                update=_parameter(branch[-1], problems),
            )
        )
    design = Design()
    design.construct([pathways], [parameters])
    return design


class BBOBEvaluator(AutoOptEvaluator):
    def __init__(self, function_id, instances, *, config=None):
        super().__init__(make_bbob_problem(function_id), instances, config=config)

    def evaluate(self, sequence, *, instance_indices=None):
        indices = (
            list(range(len(self.instances)))
            if instance_indices is None
            else list(instance_indices)
        )
        if not indices or any(i < 0 or i >= len(self.instances) for i in indices):
            raise ValueError("Invalid BBOB instance indices.")
        for objective in self.data:
            objective.reset()
        algorithm = decode_bbob(sequence, self.problems, self.setting)
        algorithm.evaluate(
            self.problems, self.data, self.setting, seed_instance=indices
        )
        values = np.asarray(algorithm.performance[indices, :], dtype=float).copy()
        if not np.isfinite(values).all():
            raise FloatingPointError("Non-finite BBOB objective value.")
        return values


@dataclass(frozen=True)
class BBOBProtocol:
    train_dimensions: tuple[int, ...] = (5, 10, 20)
    test_dimensions: tuple[int, ...] = (40,)
    train_instances: tuple[int, ...] = (1,)
    test_instances: tuple[int, ...] = (1,)
    population_size: int = 50
    train_evaluations: int = 5000
    test_evaluations: int = 50000
    train_runs: int = 5
    test_runs: int = 30
    allow_instance_overlap: bool = False

    def __post_init__(self):
        for dimensions in (self.train_dimensions, self.test_dimensions):
            if not dimensions or any(d < 2 for d in dimensions):
                raise ValueError("BBOB dimensions must be >= 2.")
        for instances in (self.train_instances, self.test_instances):
            if not instances or any(i < 1 for i in instances):
                raise ValueError("IOH instance IDs must be >= 1.")
        training = {(d, i) for d in self.train_dimensions for i in self.train_instances}
        testing = {(d, i) for d in self.test_dimensions for i in self.test_instances}
        if training & testing and not self.allow_instance_overlap:
            raise ValueError(
                "Training and test (dimension, instance) pairs must differ."
            )
        for test in (False, True):
            EvaluationConfig(
                population_size=self.population_size,
                evaluations=self.test_evaluations if test else self.train_evaluations,
                runs=self.test_runs if test else self.train_runs,
            )


class BBOBObjective:
    """PPO callback with isolated evaluation seeds and common candidate randomness.

    Cost is mean(log1p(max(f - f_opt, 0))). The known optimum is used only as
    a scalar offset, never as a search location. Raw objectives remain in logs.
    Evaluations are serial, avoiding Windows process-pool teardown failures.
    """

    def __init__(self, protocol, seed):
        self.protocol = protocol
        self.seed = seed
        self.round = 0

    def evaluate(self, actions, function_id, *, test=False):
        protocol = self.protocol
        seed = int(
            np.random.SeedSequence(
                [self.seed, int(test), 0 if test else self.round]
            ).generate_state(1)[0]
        )
        if not test:
            self.round += 1
        dimensions = protocol.test_dimensions if test else protocol.train_dimensions
        ids = protocol.test_instances if test else protocol.train_instances
        instances = [(d, i) for d in dimensions for i in ids]
        evaluator = BBOBEvaluator(
            function_id,
            instances,
            config=EvaluationConfig(
                population_size=protocol.population_size,
                evaluations=(
                    protocol.test_evaluations if test else protocol.train_evaluations
                ),
                runs=protocol.test_runs if test else protocol.train_runs,
                seed=seed,
            ),
        )
        if hasattr(actions, "detach"):
            actions = actions.detach().cpu().numpy()
        initial_state = deepcopy(evaluator.setting.rng.bit_generator.state)
        optimum = np.asarray([objective.optimum.y for objective in evaluator.data])[
            :, None
        ]
        costs, records = [], []
        for sequence in actions:
            evaluator.setting.rng.bit_generator.state = deepcopy(initial_state)
            raw = evaluator.evaluate(sequence)
            gaps = np.maximum(raw - optimum, 0.0)
            costs.append(float(np.log1p(gaps).mean()))
            records.append(
                SimpleNamespace(
                    raw=raw, gaps=gaps, optimum=optimum, instances=instances
                )
            )
        return costs, records

    def __call__(self, actions, function_id):
        costs, records = self.evaluate(actions, function_id)
        return costs, [record.raw for record in records]
