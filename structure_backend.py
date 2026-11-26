"""Execute complete exported graphs as repeatable population transformations."""

from __future__ import annotations

import hashlib
import json
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import ioh
import numpy as np
import torch
from aldes_graphs.NEUROEA.BLOCK import Block
from aldes_graphs.NEUROEA.Block_Crossover import Block_Crossover
from aldes_graphs.NEUROEA.Block_Exchange import Block_Exchange
from aldes_graphs.NEUROEA.Block_Mutation import Block_Mutation
from aldes_graphs.NEUROEA.Block_Population import Block_Population
from aldes_graphs.NEUROEA.Block_Selection import Block_Selection
from aldes_graphs.NEUROEA.Block_Tournament import Block_Tournament
from aldes_graphs.NEUROEA.support.population import Population
from aldes_graphs.NEUROEA.support.rng import NumpyTorchRNG

from structure_grammar import StructureGrammar

BLOCK_TYPES = {
    c.__name__.removeprefix("Block_"): c
    for c in (
        Block_Population,
        Block_Tournament,
        Block_Exchange,
        Block_Crossover,
        Block_Mutation,
        Block_Selection,
    )
}
DEFAULT_LIBRARY = Path(__file__).with_name("trained_structures")


class StructureLibrary:
    def __init__(self, entries):
        self.entries = deepcopy(
            sorted(entries, key=lambda entry: entry["structure_id"])
        )
        if not self.entries or len({e["structure_id"] for e in self.entries}) != len(
            self.entries
        ):
            raise ValueError("Expected distinct exported structure IDs.")
        for entry in self.entries:
            config = entry["structure_config"]
            blocks = config["blocks"]
            if (
                blocks[0]["type"] != "Population"
                or sum(b["type"] == "Population" for b in blocks) != 1
            ):
                raise ValueError(
                    "A macro operator requires exactly one Population node at index zero."
                )
            if any(b["type"] not in BLOCK_TYPES for b in blocks):
                raise ValueError("Unknown block type in exported structure.")
            selections = [b for b in blocks if b["type"] == "Selection"]
            if (
                len(selections) != 1
                or selections[0].get("kwargs", {}).get("n_solutions", 100) != 100
            ):
                raise ValueError(
                    "Exported structures must have one Selection retaining 100 solutions."
                )
            build_graph(
                config, NumpyTorchRNG(0)
            )  # Check graph and parameter contracts now.
        canonical = json.dumps(self.entries, sort_keys=True, separators=(",", ":"))
        self.digest = hashlib.sha256(canonical.encode()).hexdigest()

    @classmethod
    def load(cls, directory=DEFAULT_LIBRARY):
        entries = []
        directory = Path(directory)
        paths = sorted(directory.glob("graph_*_trained.json"))
        if not paths:
            paths = sorted(directory.glob("structure_*.json"))
        for path in paths:
            payload = json.loads(path.read_text(encoding="utf-8"))
            if "structure_config" in payload:
                structure_id = payload["structure_id"]
                structure_config = payload["structure_config"]
            else:
                try:
                    structure_id = int(path.stem.removeprefix("structure_"))
                except ValueError as exc:
                    raise ValueError(
                        f"Cannot infer structure ID from {path.name}."
                    ) from exc
                structure_config = payload
            entries.append(
                {
                    "structure_id": structure_id,
                    "structure_config": structure_config,
                }
            )
        return cls(entries)

    def metadata(self):
        return {
            "entries": deepcopy(self.entries),
            "sha256": self.digest,
            "token_mapping": {
                str(i): f"G{entry['structure_id']:03d}"
                for i, entry in enumerate(self.entries)
            },
        }

    def excluding(self, structure_ids):
        excluded = {int(value) for value in structure_ids}
        available = {int(entry["structure_id"]) for entry in self.entries}
        unknown = excluded - available
        if unknown:
            names = ", ".join(f"G{value:03d}" for value in sorted(unknown))
            raise ValueError(f"Unknown excluded structures: {names}.")
        remaining = [
            entry
            for entry in self.entries
            if int(entry["structure_id"]) not in excluded
        ]
        if not remaining:
            raise ValueError("Cannot exclude every structure from the library.")
        return StructureLibrary(remaining)

    @classmethod
    def from_metadata(cls, payload):
        library = cls(payload["entries"])
        if library.digest != payload["sha256"]:
            raise ValueError("Structure library hash mismatch.")
        return library

    def names(self, sequence, max_structures, min_structures=1):
        return [
            f"G{self.entries[i]['structure_id']:03d}"
            for i in StructureGrammar(
                len(self.entries), max_structures, min_structures
            ).decode(sequence)
        ]


def build_graph(config, rng):
    # Constructor randomness is separate from the actual optimization stream.
    construction_rng = NumpyTorchRNG(0)
    blocks = []
    for spec in config["blocks"]:
        block = BLOCK_TYPES[spec["type"]](
            rng=construction_rng, **spec.get("kwargs", {})
        )
        values = np.asarray(spec["parameters"], dtype=float)
        if values.size != block.parameter.numel() or not np.isfinite(values).all():
            raise ValueError("Exported parameter shape/value mismatch.")
        if values.size:
            tensor = torch.as_tensor(values, dtype=torch.float64)
            if torch.any(tensor < block.lower) or torch.any(tensor > block.upper):
                raise ValueError(
                    "Exported parameters exceed the supplied block's bounds."
                )
            block.set_parameters(tensor)
        block.rng = rng
        blocks.append(block)
    graph = np.zeros((len(blocks), len(blocks)))
    seen = set()
    for source, target, ratio in config["edges"]:
        if (
            not (0 <= source < len(blocks) and 0 <= target < len(blocks))
            or (source, target) in seen
            or not np.isfinite(ratio)
            or ratio <= 0
        ):
            raise ValueError("Invalid or repeated graph edge.")
        seen.add((source, target))
        graph[source, target] = ratio
    Block.Validity(blocks, graph)
    return blocks, graph


class BudgetExhausted(Exception):
    pass


class GraphProblem:
    def __init__(self, fid, dimension, instance, budget, seed):
        self.objective = ioh.get_problem(
            fid, instance, dimension, ioh.ProblemClass.BBOB
        )
        self.D, self.M, self.m = dimension, 1, 1
        self.device, self.dtype = torch.device("cpu"), torch.float64
        self.lower = torch.full((dimension,), -5.0, dtype=self.dtype)
        self.upper = torch.full((dimension,), 5.0, dtype=self.dtype)
        self.budget = budget
        self.rng = NumpyTorchRNG(seed)
        self.initializations = 0

    @property
    def evaluations(self):
        return int(self.objective.state.evaluations)

    def evaluation(self, decisions):
        decisions = torch.as_tensor(decisions, dtype=self.dtype).reshape(-1, self.D)
        decisions = decisions.clamp(-5, 5)
        if not torch.isfinite(decisions).all():
            raise FloatingPointError("Structure produced non-finite decisions.")
        values = []
        for decision in decisions:
            if self.evaluations >= self.budget:
                raise BudgetExhausted()
            value = float(self.objective(decision.numpy()))
            if not np.isfinite(value):
                raise FloatingPointError("Non-finite BBOB objective.")
            values.append(value)
        return Population(
            decisions,
            torch.tensor(values, dtype=self.dtype).reshape(-1, 1),
            torch.zeros((len(values), 1), dtype=self.dtype),
        )

    def initialize(self):
        self.initializations += 1
        return self.evaluation(self.rng.uniform(-5.0, 5.0, size=(100, self.D)))


class GraphOperator:
    def __init__(self, config, rng):
        self.blocks, self.graph = build_graph(config, rng)

    def step(self, population, problem):
        if not isinstance(population, Population) or len(population) != 100:
            raise ValueError(
                "Each graph must receive the previous graph's evaluated 100-member population."
            )
        for block in self.blocks:
            block.output, block.next_out = None, 0
        self.blocks[0].initialization(population)
        activated = np.zeros(len(self.blocks), dtype=bool)
        while not activated[0]:
            progressed = False
            for i, block in enumerate(self.blocks):
                if activated[i]:
                    continue
                predecessors = self.graph[:, i] > 0
                if np.all(
                    activated[predecessors]
                    | (np.arange(len(self.blocks))[predecessors] == 0)
                ):
                    block.Main(
                        problem,
                        [self.blocks[j] for j in np.where(predecessors)[0]],
                        self.graph[predecessors, i],
                    )
                    activated[i], progressed = True, True
            if not progressed:
                raise RuntimeError("Structure cannot complete a population update.")
        result = self.blocks[0].output
        if not isinstance(result, Population) or len(result) != 100:
            raise RuntimeError("Structure did not return 100 evaluated solutions.")
        return result


def run_sequence(
    library,
    tokens,
    fid,
    dimension,
    instance,
    budget,
    seed,
    max_structures=6,
    min_structures=1,
    *,
    observer=None,
):
    indices = StructureGrammar(
        len(library.entries), max_structures, min_structures
    ).decode(tokens)
    if budget < 100:
        raise ValueError(
            "Graph runs require at least 100 evaluations for initialization."
        )
    problem = GraphProblem(fid, dimension, instance, budget, seed)
    population = problem.initialize()
    operators = [
        GraphOperator(library.entries[i]["structure_config"], problem.rng)
        for i in indices
    ]
    completed = 0
    try:
        while problem.evaluations < budget:
            before_cycle = problem.evaluations
            for index, operator in zip(indices, operators):
                before = population
                population = operator.step(population, problem)
                completed += 1
                if observer is not None:
                    observer(index, before, population, problem)
                if problem.evaluations >= budget:
                    break
            if problem.evaluations == before_cycle:
                raise RuntimeError("Structure schedule consumed no evaluations.")
    except BudgetExhausted:
        # Keep IOH's best-so-far even when the final graph stops mid-batch.
        pass
    return SimpleNamespace(
        best=float(problem.objective.state.current_best.y),
        optimum=float(problem.objective.optimum.y),
        evaluations=problem.evaluations,
        completed_steps=completed,
        initializations=problem.initializations,
    )


class StructureBBOBObjective:
    def __init__(self, protocol, seed, library, max_structures=6, min_structures=1):
        if protocol.population_size != 100:
            raise ValueError("Exported graph operators require population_size=100.")
        self.protocol, self.seed, self.library = protocol, seed, library
        self.max_structures, self.min_structures = max_structures, min_structures
        self.round = 0

    def evaluate(self, actions, function_id, *, test=False):
        if hasattr(actions, "detach"):
            actions = actions.detach().cpu().numpy()
        protocol = self.protocol
        dimensions = protocol.test_dimensions if test else protocol.train_dimensions
        ids = protocol.test_instances if test else protocol.train_instances
        instances = [(d, i) for d in dimensions for i in ids]
        runs = protocol.test_runs if test else protocol.train_runs
        budget = protocol.test_evaluations if test else protocol.train_evaluations
        counter = 0 if test else self.round
        if not test:
            self.round += 1
        costs, records = [], []
        for sequence in actions:
            raw, optima, counts = [], [], []
            for case, (dimension, instance) in enumerate(instances):
                rows, case_counts = [], []
                for run in range(runs):
                    seed = int(
                        np.random.SeedSequence(
                            [self.seed, int(test), counter, case, run]
                        ).generate_state(1)[0]
                    )
                    result = run_sequence(
                        self.library,
                        sequence,
                        function_id,
                        dimension,
                        instance,
                        budget,
                        seed,
                        self.max_structures,
                        self.min_structures,
                    )
                    rows.append(result.best)
                    case_counts.append(result.evaluations)
                raw.append(rows)
                counts.append(case_counts)
                optima.append(result.optimum)
            raw, optima = np.asarray(raw), np.asarray(optima)[:, None]
            gaps = np.maximum(raw - optima, 0.0)
            costs.append(float(np.log1p(gaps).mean()))
            records.append(
                SimpleNamespace(
                    raw=raw,
                    gaps=gaps,
                    optimum=optima,
                    instances=instances,
                    evaluations=counts,
                )
            )
        return costs, records
