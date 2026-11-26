"""PFLACCO ELA features from exactly 100 objective evaluations per problem.

Use the original ALDes 32-column order. Its three runtime slots are zero;
all other columns are actual PFLACCO outputs, not custom approximations.
"""

from __future__ import annotations

from importlib.metadata import version

import ioh
import numpy as np
from pflacco.classical_ela_features import (
    calculate_dispersion,
    calculate_ela_meta,
    calculate_information_content,
    calculate_nbc,
)
from pflacco.sampling import create_initial_sample

from pflacco_feature import PAPER_FEATURE_NAMES

FEATURE_NAMES = PAPER_FEATURE_NAMES
FEATURE_SAMPLE_COUNT = 100
FEATURE_SCHEMA = "pflacco_aldes32_n100_v1"
QUADRATIC_CONDITION_CAP = 1e12


def extract_features(function_id, dimension, instance_id, *, sample_count=100, seed=0):
    if sample_count != FEATURE_SAMPLE_COUNT:
        raise ValueError("BBOB feature extraction requires exactly 100 sample points.")
    sample_seed = int(
        np.random.SeedSequence([seed, dimension, instance_id]).generate_state(1)[0]
    )
    objective = ioh.get_problem(
        function_id, instance_id, dimension, ioh.ProblemClass.BBOB
    )
    # PFLACCO sampling and IC/NBC use NumPy's global RNG. Keep feature extraction
    # deterministic without disturbing policy sampling or other caller state.
    state = np.random.get_state()
    try:
        x = create_initial_sample(
            dimension,
            n=FEATURE_SAMPLE_COUNT,
            lower_bound=-5.0,
            upper_bound=5.0,
            sample_type="lhs",
            seed=sample_seed,
        )
        y = np.asarray([objective(row) for row in x.to_numpy()])
        if not np.isfinite(y).all():
            raise FloatingPointError("Non-finite BBOB landscape sample.")
        values = {}
        values.update(calculate_ela_meta(x, y))
        values.update(
            calculate_dispersion(x, y, dist_method="euclidean", minimize=True)
        )
        values.update(calculate_information_content(x, y, seed=sample_seed))
        np.random.seed(sample_seed)
        values.update(calculate_nbc(x, y, minimize=True))
    finally:
        np.random.set_state(state)
    encoded = []
    for name in FEATURE_NAMES:
        value = 0.0 if name.endswith("costs_runtime") else float(values[name])
        if name == "ela_meta.quad_simple.cond" and np.isposinf(value):
            value = QUADRATIC_CONDITION_CAP
        encoded.append(value)
    vector = np.asarray(encoded)
    if not np.isfinite(vector).all():
        bad = [n for n, v in zip(FEATURE_NAMES, vector) if not np.isfinite(v)]
        raise FloatingPointError(
            f"Undefined PFLACCO features for F{function_id}/{dimension}D: {bad}"
        )
    if objective.state.evaluations != FEATURE_SAMPLE_COUNT:
        raise RuntimeError("Feature sampling exceeded its 100-evaluation budget.")
    return vector


class FeatureBank:
    def __init__(self, *, sample_count=100, seed=0, mean=None, scale=None):
        if sample_count != FEATURE_SAMPLE_COUNT:
            raise ValueError("BBOB feature sampling is fixed to 100 points.")
        self.sample_count, self.seed = sample_count, seed
        self.mean = None if mean is None else np.asarray(mean)
        self.scale = None if scale is None else np.asarray(scale)
        self.raw = {}

    def sample(self, key):
        key = tuple(key)
        if len(key) not in (3, 4):
            raise ValueError(
                "Feature keys must be (function, dimension, instance) or "
                "(function, dimension, instance, feature_seed)."
            )
        if key not in self.raw:
            problem = key[:3]
            seed = self.seed if len(key) == 3 else key[3]
            self.raw[key] = extract_features(
                *problem, sample_count=self.sample_count, seed=seed
            )
        return self.raw[key]

    def fit(self, training_keys):
        matrix = np.asarray([self.sample(key) for key in training_keys])
        self.mean = matrix.mean(axis=0)
        scale = matrix.std(axis=0)
        self.scale = np.where(scale > 1e-8, scale, 1.0)

    def transform(self, key):
        if self.mean is None or self.scale is None:
            raise RuntimeError("Fit training-only feature scaling first.")
        return np.clip((self.sample(key) - self.mean) / self.scale, -10, 10).astype(
            np.float32
        )

    def metadata(self):
        return {
            "schema": FEATURE_SCHEMA,
            "names": list(FEATURE_NAMES),
            "library": "pflacco",
            "library_version": version("pflacco"),
            "sample_count": self.sample_count,
            "sampling": "pflacco LHS in [-5,5]",
            "runtime_slots": "fixed to zero, following existing ALDes code",
            "quadratic_condition_cap": QUADRATIC_CONDITION_CAP,
            "seed": self.seed,
            "mean": self.mean.tolist(),
            "scale": self.scale.tolist(),
            "evaluations_per_problem": FEATURE_SAMPLE_COUNT,
        }

    @classmethod
    def from_metadata(cls, metadata):
        if metadata["schema"] != FEATURE_SCHEMA or metadata["names"] != list(
            FEATURE_NAMES
        ):
            raise ValueError(
                "Expected the PFLACCO ALDes-32, 100-sample feature schema; retrain older weights."
            )
        if metadata["library_version"] != version("pflacco"):
            raise ValueError(
                "PFLACCO version differs from the checkpoint's feature extractor."
            )
        return cls(
            **{k: metadata[k] for k in ("sample_count", "seed", "mean", "scale")}
        )
