"""Plot convergence diagnostics from an ALDes BBOB training log."""

from __future__ import annotations

import argparse
import ast
import json
import os
import re
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

NUMBER = r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?"
METRIC_PATTERN = re.compile(
    rf"step\s*:\s*['\"]?\s*,?\s*(?P<progress>{NUMBER}).*?"
    rf"loss\s*:\s*['\"]?\s*,?\s*(?P<loss>{NUMBER}).*?"
    rf"cost_mean\s*:\s*['\"]?\s*,?\s*(?P<cost>{NUMBER}).*?"
    rf"baseline\s*:\s*['\"]?\s*,?\s*(?P<baseline>{NUMBER})"
)


def parse_log(path: Path) -> tuple[dict, np.ndarray]:
    metadata = {}
    rows = []
    with path.open("r", encoding="utf-8") as stream:
        for line in stream:
            if not metadata and "{'suite': 'BBOB'" in line:
                metadata = ast.literal_eval(line[line.index("{") :].strip())
            match = METRIC_PATTERN.search(line)
            if match:
                rows.append([float(match[name]) for name in match.groupdict()])
    if not rows:
        raise ValueError(f"No BBOB training metrics found in {path}")
    return metadata, np.asarray(rows, dtype=float)


def sampled_functions(metadata: dict, epochs: int) -> np.ndarray | None:
    function_ids = metadata.get("train_functions")
    protocol = metadata.get("protocol", {})
    dimensions = protocol.get("train_dimensions")
    instances = protocol.get("train_instances")
    feature_seeds = metadata.get("train_feature_seeds")
    if feature_seeds is None:
        feature_seeds = (metadata.get("features", {}).get("seed", 0),)
    seed = metadata.get("seed")
    if not function_ids or not dimensions or not instances or seed is None:
        return None

    keys = [
        (function_id, dimension, instance, feature_seed)
        for function_id in function_ids
        for dimension in dimensions
        for instance in instances
        for feature_seed in feature_seeds
    ]
    rng = np.random.default_rng(np.random.SeedSequence([seed, 731]))
    order = []
    sampled = []
    for _ in range(epochs):
        if not order:
            order = rng.permutation(len(keys)).tolist()
        sampled.append(keys[order.pop()][0])
    return np.asarray(sampled, dtype=int)


def rolling_mean(values: np.ndarray, window: int) -> np.ndarray:
    if window <= 1:
        return values.copy()
    result = np.full(values.shape, np.nan, dtype=float)
    cumulative = np.cumsum(np.insert(values, 0, 0.0))
    result[window - 1 :] = (cumulative[window:] - cumulative[:-window]) / window
    return result


def plot_log(log_path: Path, output: Path, window: int) -> None:
    metadata, metrics = parse_log(log_path)
    progress, loss, cost, baseline = metrics.T
    epochs = np.arange(1, len(metrics) + 1)
    functions = sampled_functions(metadata, len(metrics))

    figure, axes = plt.subplots(3, 1, figsize=(11, 11), constrained_layout=True)

    axes[0].plot(epochs, loss, color="#9ca3af", alpha=0.35, linewidth=0.7)
    axes[0].plot(
        epochs,
        rolling_mean(loss, window),
        color="#166534",
        linewidth=1.8,
        label=f"rolling mean ({window} epochs)",
    )
    axes[0].axhline(0, color="#111827", linewidth=0.7)
    axes[0].set(title="PPO loss", ylabel="Loss")
    axes[0].legend()

    axes[1].plot(epochs, cost, color="#2563eb", alpha=0.35, linewidth=0.7)
    axes[1].plot(
        epochs,
        rolling_mean(cost, window),
        color="#1d4ed8",
        linewidth=1.8,
        label=f"cost rolling mean ({window} epochs)",
    )
    axes[1].plot(
        epochs,
        rolling_mean(baseline, window),
        color="#dc2626",
        linewidth=1.2,
        label="baseline rolling mean",
    )
    axes[1].set(title="Mixed-function training cost", ylabel="log1p optimality gap")
    axes[1].legend()

    if functions is None:
        axes[2].plot(epochs, cost - baseline, color="#7c3aed", linewidth=0.8)
        axes[2].axhline(0, color="#111827", linewidth=0.7)
        axes[2].set(title="Cost relative to baseline", ylabel="Cost - baseline")
    else:
        colors = plt.get_cmap("tab20").resampled(len(np.unique(functions)))
        for index, function_id in enumerate(np.unique(functions)):
            selected = functions == function_id
            function_epochs = epochs[selected]
            function_costs = cost[selected]
            axes[2].plot(
                function_epochs,
                rolling_mean(function_costs, min(5, len(function_costs))),
                color=colors(index),
                linewidth=1.2,
                label=f"F{function_id}",
            )
        axes[2].set(
            title="Per-function cost (5 observations rolling mean)",
            ylabel="log1p optimality gap",
        )
        axes[2].legend(ncol=4, fontsize=8)

    axes[2].set_xlabel("Epoch")
    title = f"BBOB training convergence: {len(metrics)} epochs ({progress[-1]:.2f}%)"
    figure.suptitle(title, fontsize=14)
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=180)
    plt.close(figure)


def plot_per_function(log_path: Path, output: Path, window: int) -> None:
    metadata, metrics = parse_log(log_path)
    functions = sampled_functions(metadata, len(metrics))
    if functions is None:
        raise ValueError("The log metadata does not contain the BBOB sampling schedule")

    function_ids = np.unique(functions)
    columns = 4
    rows = (len(function_ids) + columns - 1) // columns
    figure, axes = plt.subplots(
        rows,
        columns,
        figsize=(16, 3.6 * rows),
        constrained_layout=True,
        squeeze=False,
    )
    cost, baseline = metrics[:, 2], metrics[:, 3]

    for axis, function_id in zip(axes.flat, function_ids):
        selected = functions == function_id
        function_cost = cost[selected]
        function_baseline = baseline[selected]
        observations = np.arange(1, len(function_cost) + 1)
        local_window = min(window, len(function_cost))

        axis.plot(
            observations,
            function_cost,
            color="#94a3b8",
            marker="o",
            markersize=2,
            alpha=0.45,
            linewidth=0.7,
            label="cost",
        )
        axis.plot(
            observations,
            rolling_mean(function_cost, local_window),
            color="#1d4ed8",
            linewidth=1.8,
            label=f"cost rolling mean ({local_window})",
        )
        axis.plot(
            observations,
            function_baseline,
            color="#dc2626",
            alpha=0.8,
            linewidth=1.1,
            label="baseline",
        )
        axis.set_title(f"BBOB F{function_id}")
        axis.grid(alpha=0.2)

    for axis in axes.flat[len(function_ids) :]:
        axis.set_visible(False)

    axes.flat[0].legend(loc="upper right", fontsize=8)
    figure.supxlabel("Sampling observation")
    figure.supylabel("log1p optimality gap")
    figure.suptitle(
        f"Per-function BBOB training cost ({len(metrics)} total epochs)", fontsize=15
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=180)
    plt.close(figure)


def write_dashboard(
    path: Path, overall: Path, per_function: Path | None, refresh_seconds: float
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    images = [overall] + ([per_function] if per_function is not None else [])
    relative = [
        os.path.relpath(image, path.parent).replace("\\", "/") for image in images
    ]
    sources = json.dumps(relative)
    interval = max(1, int(refresh_seconds * 1000))
    path.write_text(
        f"""<!doctype html>
<html><head><meta charset="utf-8"><title>ALDes BBOB Live Training</title>
<style>body{{font-family:Arial,sans-serif;margin:20px;background:#f8fafc}}
h1{{margin-bottom:4px}} p{{color:#475569}} img{{display:block;max-width:100%;margin:20px auto;
background:white;border:1px solid #cbd5e1;box-shadow:0 1px 4px #cbd5e1}}</style></head>
<body><h1>ALDes BBOB Live Training</h1>
<p>Images refresh every {refresh_seconds:g} seconds. The monitor only reads the log.</p>
<div id="plots"></div>
<script>const sources={sources}; const root=document.getElementById('plots');
const images=sources.map(src=>{{const img=document.createElement('img'); root.appendChild(img); return [img,src];}});
function refresh(){{const stamp=Date.now(); for(const [img,src] of images) img.src=src+'?t='+stamp;}}
refresh(); setInterval(refresh,{interval});</script></body></html>
""",
        encoding="utf-8",
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("log", type=Path, help="path to the BBOB log.txt")
    parser.add_argument("--output", type=Path, default=None, help="output PNG path")
    parser.add_argument("--window", type=int, default=32, help="overall rolling window")
    parser.add_argument(
        "--per-function-output",
        type=Path,
        default=None,
        help="optional output PNG with one subplot per training function",
    )
    parser.add_argument(
        "--per-function-window",
        type=int,
        default=5,
        help="rolling window for each function",
    )
    parser.add_argument(
        "--watch-seconds",
        type=float,
        default=0,
        help="continuously refresh plots at this interval; 0 disables watching",
    )
    parser.add_argument(
        "--dashboard",
        type=Path,
        default=None,
        help="auto-refreshing HTML path; defaults beside the log in watch mode",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.window < 1 or args.per_function_window < 1:
        raise SystemExit("rolling windows must be at least 1")
    if args.watch_seconds < 0:
        raise SystemExit("--watch-seconds cannot be negative")
    output = args.output or args.log.with_name("training_convergence.png")
    per_function_output = args.per_function_output
    if args.watch_seconds and per_function_output is None:
        per_function_output = args.log.with_name("per_function_cost_live.png")
    dashboard = args.dashboard
    if args.watch_seconds and dashboard is None:
        dashboard = args.log.with_name("training_dashboard.html")
    if dashboard is not None:
        write_dashboard(
            dashboard, output, per_function_output, args.watch_seconds or 15
        )
        print(f"Open the live dashboard: {dashboard.resolve()}", flush=True)

    while True:
        try:
            _, metrics = parse_log(args.log)
            plot_log(args.log, output, args.window)
            if per_function_output is not None:
                plot_per_function(
                    args.log, per_function_output, args.per_function_window
                )
            print(
                f"Refreshed {len(metrics)} epochs: {output}"
                + (f"; {per_function_output}" if per_function_output else ""),
                flush=True,
            )
        except (OSError, SyntaxError, ValueError) as exc:
            if not args.watch_seconds:
                raise
            print(f"Waiting for a complete log record: {exc}", flush=True)
        if not args.watch_seconds:
            break
        try:
            time.sleep(args.watch_seconds)
        except KeyboardInterrupt:
            print("Stopped live plot monitoring.")
            break


if __name__ == "__main__":
    main()
