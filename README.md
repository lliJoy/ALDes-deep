# ALDes

Pure-Python implementation of **Automated Metaheuristic Algorithm Design with
Autoregressive Learning** (ALDes).

ALDes treats algorithm design as constrained autoregressive sequence generation.
A Transformer policy generates a variable-length metaheuristic program, PPO
learns from the program's performance, and AutoOptLib executes the generated
algorithm.

## Release scope

This release supports the paper's 23 pseudo-Boolean optimization (PBO) tasks:

- independent design from scratch for one target problem;
- feature-conditioned continual design with EWC;
- the paper's training and test budgets;
- automatic neural-network acceleration on CUDA, ROCm, or Apple MPS;
- parallel CPU evaluation of generated algorithms.

The paper's RIS beamforming and power-system restoration experiments are not
part of this release. 

## Installation

Python 3.9--3.11 is supported. Install the released package from PyPI:

```bash
python -m pip install aldes
```

For an editable source installation, clone this repository and install it from
the repository root:

```bash
git clone --depth 1 https://github.com/auto4opt/ALDes.git
cd ALDes
python -m pip install -e .
```

The installation automatically downloads the compatible
`autooptlib[aldes]` dependency from PyPI.
Users do not need to clone AutoOptLib separately. The Python import name is
lowercase:

```python
import autooptlib
```

A Conda environment can be created instead:

```bash
conda env create -f aldes.yaml
conda activate aldes
```

For development and tests, install the test extra:

```bash
python -m pip install -e ".[test]"
```

## Quick start

The default command performs one independent design trial on PBO F1 with
seed 1:

```bash
aldes-train
```

The equivalent source command is:

```bash
python train.py
```

The inferred algorithm sequence is printed and written to the run log. A
single-problem run does not save a model checkpoint because the trained policy
is not reused after its final algorithm has been inferred.

Choose another problem or multiple explicit trials with command-line options:

```bash
aldes-train --problems 14 --seeds 1
aldes-train --problems 1,14,15 --seeds 1,2
aldes-train --problems 14 --seeds 1 --evaluate-test
```

`--evaluate-test` applies the paper's full 30-run test protocol after training.
Without it, only training and final algorithm inference are performed. In
continual mode, the flag evaluates all problems seen so far after every stage,
which produces the forgetting matrix used by the paper.

## Local extension: IOH BBOB (continuous problems)

### Sampled, feature-conditioned training with one shared checkpoint

The default action space uses the four independently trained graph structures
in `trained_structures`: G001, G002, G003 and G004. They use their actual block
implementations under `Algorithms/NEUROEA`. The structures are mapped to four
contiguous tokens, followed by BOS/EOS. Repeated EOS pads
shorter batch rows and has zero extra log probability. The old primitive
search space is available only with `--operator-space primitive`.

Each graph token performs one complete population-update cycle including
its own Population, parent/variation operations and Selection. The first
graph receives the single initialized 100-member evaluated population; each
following graph receives **the previous graph's returned population**. A
repeated graph receives the newest population as well. The generated sequence
is cycled until the global function-evaluation budget is exhausted. Temporary
block outputs/gather cursors reset per invocation, while population, fitness
values and the RNG stream carry forward. Trained graph parameters and edge
ratios stay fixed. No extra outer selection is added.

The grammar is BOS, 1--6 structure tokens, EOS (limit configurable via
`--min-structures` and `--max-structures`). Use `--prediction-length` for an
exact prediction length. A structure cannot be selected in two consecutive
positions, but it may appear again after a different structure. The original PBO
choose/search/update and global-search restrictions
do not apply to this mode. Exported selections retain 100 individuals, so
`--population-size` must be 100 (the new default). Initial population evaluation
counts toward the common run budget; structure intermediates consume that
same budget. The final graph may stop mid-batch at the exact FE limit, and
the best objective among all evaluated points is retained. PFLACCO's separate
100-point feature budget is unchanged.

```powershell
conda activate aldes
python train_bbob_joint.py --split-seed 0 --seeds 1 --evaluate-test --output-dir experiments/bbob_structures_10d
```

All four trained structures are available by default. Use `--exclude-structures`
to remove any of them. For example, this removes G002 and allows predicted
sequences of length 2 through 5:

```powershell
python train_bbob_joint.py --exclude-structures G002 --min-structures 2 --max-structures 5 --output-dir experiments/bbob_filtered
```

For an exact length of 4, use `--prediction-length 4` (equivalent to setting
both `--min-structures` and `--max-structures` to 4). Use
Omit `--exclude-structures` to use all four structures.

The evaluation budgets and population size are command-line options, not a
separate mutable configuration file. For example:

```powershell
python train_bbob_joint.py --population-size 100 --train-evaluations 5000 --test-evaluations 50000 --train-runs 5 --test-runs 30 --output-dir experiments/bbob_structures_10d
```

The resolved settings are saved in `config_seed1.json` and in the final result
JSON under `protocol`. To monitor a running job, open another terminal and run:

```powershell
python plot_bbob_training.py logs/<run-folder>/log.txt --watch-seconds 15
```

This refreshes `training_convergence.png` and `per_function_cost_live.png` in
the log directory every 15 seconds. Stop the monitor with Ctrl+C; training is
unaffected.

Joint BBOB training uses an entropy bonus by default (`--entropy-coef 0.01`):
`loss = PPO policy loss - entropy_coef * entropy`. Entropy is computed from
the full legal token distribution, averaged over positions with more than
one legal choice; forced EOS and padding are excluded. Logs include `entropy`
in nats and checkpoint metadata records the coefficient. Use `--entropy-coef 0`
to disable the bonus. This encourages exploration but does not guarantee
distinct structures or prevent consecutive repeats. Existing checkpoints
remain loadable; train a new run to measure the effect of the bonus.

Structure checkpoints use `aldes-bbob-structures-v4`, embedding all 10 graph
configs, their fixed trained parameters, token mapping and SHA-256 digest.
Evaluation-only restores this saved library instead of reading mutable JSON
files from the source directory. Old primitive checkpoints cannot supply the
new embeddings/output layer; train a new structure checkpoint.

```powershell
python train_bbob_joint.py --epochs 2 --ppo-epochs 1 --batch-size 4 --train-evaluations 1000 --train-runs 1 --test-evaluations 500 --test-runs 1 --evaluate-test --output-dir experiments/bbob_structure_smoke
```

Use `train_bbob_joint.py` to train **one model and one optimizer per training
seed**, jointly over 16 functions. The fixed split uses
`numpy.random.default_rng(0).choice(1..24, 16, replace=False)` (sorted):

- Training: F1, F2, F3, F4, F5, F6, F7, F8, F11, F13, F14, F15, F16, F18, F19, F23.
- Held out: F9, F10, F12, F17, F20, F21, F22, F24.

The split seed (`--split-seed 0`) is independent of model training seeds
(`--seeds 1`). Training and testing both default to **10 dimensions**, with
IOH instance ID 1. Their function IDs are disjoint, so the same dimension
and instance number are valid. Each epoch samples **one training problem**,
feeds that problem's 32-dimensional landscape descriptor to the shared
Transformer, generates a batch of algorithms, evaluates them on that sampled
problem only, and performs PPO updates with the same conditioning features.
Sampling uses shuffled balanced cycles: each of the 16 functions appears
once before the next shuffle. If multiple training dimensions or instance IDs
are provided, the cycle covers their Cartesian product with function IDs.
The optimizer and learning-rate schedule persist across problems; moving
baselines are separate per problem configuration. No function-ID encoding or
per-function neural weights are used.

The default 100 epochs are **100 sampled-problem updates in total**, not 100
epochs per function. Use `--epochs 1600` for 100 visits per function at the
default single dimension/instance. The policy is conditioned on measured
landscape features and can produce different algorithms for different problems;
different features do not guarantee different greedy token sequences.

Features are extracted directly by PFLACCO in `bbob_features.py`, using
`calculate_ela_meta`, `calculate_dispersion`, `calculate_information_content`,
and `calculate_nbc`. Columns follow the existing ALDes `PAPER_FEATURE_NAMES`
32-column order. Its three runtime slots remain fixed to zero; the other
29 columns are the library's outputs. BBOB uses minimization and Euclidean
geometry, not the PBO maximization/Hamming implementations. See the
[PFLACCO feature example](https://pflacco.readthedocs.io/en/latest/getting_started.html).

Feature sampling is **exactly 100 points / 100 objective calls per function,
dimension and instance**, including 10D. PFLACCO generates a seeded Latin
hypercube sample in `[-5,5]` via `create_initial_sample(n=100)`. All four
feature groups reuse these same `(X,y)` observations; none receives an
objective callable or requests further evaluations. `--feature-samples`
only accepts 100; the previous sample-factor option has been removed.
Features are cached within a run, so later sampled epochs do not resample.
Undefined/non-finite PFLACCO results fail explicitly, without substituting
custom statistics or silently adding observations. These samples are not
reused as search populations. Scaling is fitted only
on the 16 training functions, saved in the checkpoint and frozen for tests.
Held-out input features are extracted only at test time using their own
separately reported sample budget; their values never fit the scaler or update
weights. Optimum locations/values and function IDs are not feature inputs.

```powershell
conda activate aldes
python train_bbob_joint.py --split-seed 0 --seeds 1 --evaluate-test --output-dir experiments/bbob_structures_10d
```

The installed command is `aldes-bbob-joint`. Each seed produces one shared
`bbob_joint_seed<SEED>.pt`, a JSON summary, and a configuration JSON. The
checkpoint contains a portable model state dict, model architecture, split,
feature schema/scaling and evaluation protocol. It is saved **before** optional
test evaluation. With `--evaluate-test`, only the 8 held-out functions are
tested, at 10D by default. For each test problem, its measured features are
passed through the shared model to infer its algorithm; no PPO update occurs.
Raw objectives, gaps, input features and inferred programs are recorded.
Older unconditioned/custom-feature checkpoints require retraining; the primitive
checkpoint format remains `aldes-bbob-pflacco-v3`. Checkpoints record the exact
PFLACCO version, column names, 100-point sampling protocol and frozen scaler.

Load the shared weights and test without retraining:

```powershell
python train_bbob_joint.py --evaluate-only experiments/bbob_structures_10d/bbob_joint_seed1.pt --output-dir experiments/bbob_structure_eval
```

Evaluation-only uses the checkpoint's saved architecture, split, training seed
and test protocol; it does not use CLI dimension/budget overrides. For a short
test of sampled training and held-out feature-conditioned evaluation:

```powershell
python train_bbob_joint.py --operator-space primitive --epochs 2 --ppo-epochs 1 --batch-size 8 --population-size 4 --train-evaluations 80 --train-runs 1 --test-evaluations 40 --test-runs 1 --feature-samples 100 --evaluate-test --output-dir experiments/bbob_pflacco_smoke
```

Serial evaluation avoids Windows process-pool teardown problems. Smoke checkpoints
are only for checking the pipeline; they are not fully trained models.

### Independent per-function training

This checkout also provides `train_bbob.py` for the 24 continuous BBOB
functions in IOH. It trains an independent policy for each function and seed.
The defaults use **5, 10, and 20 dimensions for training and 40 dimensions for
testing**, with IOH instance ID 1 in both sets. The dimension-instance pairs
must be disjoint. These are new experimental settings, not a reproduction of
the paper's PBO results.

```powershell
conda activate aldes
python train_bbob.py --problems 1 --seeds 1 --evaluate-test
python train_bbob.py --problems 1,3,8,24 --seeds 1 --evaluate-test
```

All BBOB IDs F1--F24 are accepted. The installed entry point is `aldes-bbob`.
Dimensions, IOH instance IDs, and evaluation budgets are configurable:

```powershell
python train_bbob.py --train-dimensions 5,10,20 --test-dimensions 40 --train-instances 1,2 --test-instances 3,4 --evaluate-test
```

A short end-to-end smoke run (reduced budgets, not a performance experiment):

```powershell
python train_bbob.py --problems 1 --epochs 1 --ppo-epochs 1 --batch-size 4 --population-size 4 --train-evaluations 80 --train-runs 1 --test-evaluations 80 --test-runs 1 --evaluate-test --output-dir experiments/bbob_smoke
```

The BBOB backend preserves the existing 32-token grammar and its structural
restrictions, but assigns continuous operator meanings. Tokens 4/5 retain
coordinate crossovers on real values; 6/7 become arithmetic/SBX crossovers;
8/9/10 become Gaussian/polynomial/uniform mutation; and 11 becomes continuous
reinitialization. Choose and update operators are retained. Program tokens
must be interpreted with this BBOB mapping; PBO token names/checkpoints are
not interchangeable with it. The decoder uses AutoOptLib 1.3's grammar
helpers and execution engine.

Search uses real-valued coordinates, uniform initialization in `[-5,5]`, and
clipping to that box after search operations. This explicitly bounded protocol
differs from BBOB protocols that allow search outside the initialization box.
Evaluation is serial, including on Windows; `ALDES_EVAL_WORKERS` does not
change this backend. PyTorch still selects the configured compute device.

BBOB is minimized. The PPO cost is the average
`log1p(max(f_best - f_opt, 0))` across all training dimensions, instance IDs,
and runs. The optimum value is only an offset for the score; its coordinates
are never used in search. Lower costs and lower objective gaps are better.
Without `--evaluate-test`, no held-out evaluation is performed. With it, the
inferred algorithm is evaluated on 40-dimensional test problems using fresh
populations and an independent random stream. Default budgets remain 100 PPO
epochs, batch size 16, 5 PPO updates, population size 50, 5 training runs at
5,000 FEs and 30 test runs at 50,000 FEs, per dimension-instance pair.

Each completed trial writes `bbob_f<ID>_seed<SEED>.json` under the run's
`logs/.../bbob` directory (or `--output-dir`). JSON includes the protocol,
continuous component names, inferred tokens, and, when requested, raw test
objectives, optimum values and gaps for every dimension-instance pair and run.
Training histories are retained in the usual logs. The original `train.py`
and `run_paper_subset.py` commands remain PBO-only.

## Continual design

Continual mode extracts PBO landscape features, conditions one policy on those
features, and applies EWC between tasks:

```bash
aldes-train --mode continual --problems 1,2,11 --seeds 1
```

The default continual sequence is defined in `conf.py`. Checkpoint output is
optional and is only intended for a policy that must be reused across continual
stages:

```bash
aldes-train --mode continual --checkpoint-dir logs/continual
```

No checkpoint binary is distributed with this repository.

The two default continual problem sets in `conf.py` are independent paper
tasks. They are trained from separate initial models rather than concatenated
into one sequence. When `--problems` is supplied, that explicit list is one
continual task. Landscape vectors are standardized once over the complete
declared task so a previously seen problem keeps the same model input at every
stage. The sampled 100/225/400-dimensional populations are reused only during
training; the 625-dimensional test always starts from fresh populations.
PBO nearest-better and dispersion features explicitly use maximization
semantics and Hamming distance. The 32-column schema follows Table A3 of the
paper appendix; its three machine-dependent runtime slots are fixed to zero so
seeded artifacts remain reproducible. Pairwise statistics are computed in
blocks instead of allocating a square distance matrix, and the costly feature
regressions are calculated on the 100-dimensional training instance only;
sampled populations for all three training dimensions are still retained.

Automatic artifact generation uses the paper appendix's random-walk length of
`100 * dimension`. Feature extraction is consequently a long preprocessing
job. A factor of 10 can be requested explicitly for development smoke tests,
but artifacts generated that way are not paper-protocol inputs:

```bash
python pflacco_feature.py --sample-factor 10
```

## Compute devices

PyTorch devices are selected automatically in this order:

1. NVIDIA CUDA or AMD ROCm;
2. Apple Metal Performance Shaders (MPS);
3. CPU.

PyTorch must have been built for the user's accelerator. In ROCm builds, AMD
devices are exposed through PyTorch's `cuda` API. Override automatic selection
when needed:

```bash
ALDES_DEVICE=cpu aldes-train
ALDES_DEVICE=cuda aldes-train
ALDES_DEVICE=mps aldes-train
```

Only neural-network training and inference use the selected accelerator.
Generated algorithms and objective functions run on CPUs. Candidate algorithms
are evaluated in parallel using up to the available logical CPU cores:

```bash
ALDES_EVAL_WORKERS=8 aldes-train
ALDES_EVAL_WORKERS=1 aldes-train  # disable multiprocessing
```

## Paper protocol

The default configuration for each PBO design trial is:

- training instances: dimensions 100, 225, and 400;
- 100 PPO epochs and 16 generated algorithms per epoch;
- 5 PPO updates per epoch;
- 5 runs per training instance and 5,000 function evaluations per run;
- population size 50;
- test instance: dimension 625;
- 30 test runs and 50,000 function evaluations per run.

The single-problem mode does not extract or input problem features. Feature
conditioning is enabled only in continual mode.

## Time-bounded paper subset

The repository includes a runner that completes whole paper-protocol trials
until the next problem is predicted to exceed a wall-clock budget:

```bash
aldes-paper-subset --problems 1,14,15 --time-budget-minutes 60
```

The budget is checked only between problems. A started problem always retains
all 100 epochs and the complete 30-run test. Structured JSON results are written
under `experiments/`; this local output directory and all model checkpoints are
ignored by Git.

Historical comparison data used by the plotting notebooks are stored under
`draw/datas/reference_results` and can be read with SciPy.

The notebooks are source artifacts and should be opened from the repository
root after installing the plotting extra:

```bash
python -m pip install -e ".[plot]"
jupyter lab draw
```

Their historical `.pkl` inputs are trusted research artifacts committed to
this repository. Do not pass untrusted pickle files to the plotting helpers.

## Tests

Run the test suite and static checks with:

```bash
python -m pytest -q
python -m ruff check .
```

GitHub Actions repeats installation, linting, and tests on Linux with every
supported Python minor version (3.9--3.11). Tests cover device selection,
single and continual feature modes, grammar-valid generation, PPO likelihood
replay, EWC accumulation, and a pure-Python PPO update.

## Citation

If you use ALDes, cite:

```bash
@article{zhao2025automated,
title   = {Automated Metaheuristic Algorithm Design With Autoregressive Learning},
author  = {Zhao, Qi and Liu, Tengfei and Yan, Bai and Duan, Qiqi and Yang, Jian and Shi, Yuhui},
journal = {IEEE Transactions on Evolutionary Computation},
year    = {2025},
volume  = {29},
number  = {5},
pages   = {2004--2018},
doi     = {10.1109/TEVC.2024.3464677}}
```

## License

ALDes is released under the Apache License 2.0. See `LICENSE`.
