"""Evaluate each trained structure independently on IOH BBOB and export Excel."""

from __future__ import annotations

import argparse
import math
import os
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import torch
from openpyxl import Workbook
from openpyxl.styles import Alignment, Font, PatternFill
from openpyxl.utils import get_column_letter

from structure_backend import StructureLibrary, run_sequence

_LIBRARY = None


def _initialize_worker(library_path):
    global _LIBRARY
    torch.set_num_threads(1)
    _LIBRARY = StructureLibrary.load(library_path)


def _evaluate(task):
    function_id, structure_index, seed, dimension, instance, budget = task
    count = len(_LIBRARY.entries)
    result = run_sequence(
        _LIBRARY,
        [count, structure_index, count + 1],
        function_id,
        dimension,
        instance,
        budget,
        seed,
        max_structures=1,
        min_structures=1,
    )
    gap = max(result.best - result.optimum, 0.0)
    return {
        "function": function_id,
        "structure": f"G{_LIBRARY.entries[structure_index]['structure_id']:03d}",
        "seed": seed,
        "best": result.best,
        "optimum": result.optimum,
        "gap": gap,
        "log_gap": math.log1p(gap),
        "evaluations": result.evaluations,
    }


def _style_summary(sheet, structures, rows):
    sheet.freeze_panes = "B3"
    sheet.merge_cells(start_row=1, start_column=1, end_row=2, end_column=1)
    sheet["A1"] = "BBOB Function"
    sheet["A1"].alignment = Alignment(horizontal="center", vertical="center")
    for index, structure in enumerate(structures):
        column = 2 + index * 2
        sheet.merge_cells(
            start_row=1, start_column=column, end_row=1, end_column=column + 1
        )
        sheet.cell(1, column, structure)
        sheet.cell(2, column, "Mean")
        sheet.cell(2, column + 1, "Std")
    header_fill = PatternFill("solid", fgColor="1F4E78")
    for row in sheet.iter_rows(min_row=1, max_row=2):
        for cell in row:
            cell.fill = header_fill
            cell.font = Font(color="FFFFFF", bold=True)
            cell.alignment = Alignment(horizontal="center", vertical="center")
    best_fill = PatternFill("solid", fgColor="C6EFCE")
    for row_number in range(3, 3 + rows):
        mean_columns = [2 + index * 2 for index in range(len(structures))]
        values = [sheet.cell(row_number, column).value for column in mean_columns]
        best = min(values)
        for column, value in zip(mean_columns, values):
            if value == best:
                sheet.cell(row_number, column).fill = best_fill
                sheet.cell(row_number, column).font = Font(bold=True)
    sheet.column_dimensions["A"].width = 16
    for column in range(2, 2 + len(structures) * 2):
        sheet.column_dimensions[get_column_letter(column)].width = 16
        for cell in sheet[get_column_letter(column)][2:]:
            cell.number_format = "0.000000"
    sheet.auto_filter.ref = f"A2:{get_column_letter(1 + len(structures) * 2)}{2 + rows}"


def _write_workbook(path, records, protocol):
    structures = sorted({record["structure"] for record in records})
    functions = sorted({record["function"] for record in records})
    workbook = Workbook()
    workbook.remove(workbook.active)
    metrics = (
        ("Log-gap", "log_gap"),
        ("Gap", "gap"),
        ("Final objective", "best"),
    )
    for title, metric in metrics:
        sheet = workbook.create_sheet(title)
        for row_number, function_id in enumerate(functions, start=3):
            sheet.cell(row_number, 1, f"F{function_id}")
            for structure_index, structure in enumerate(structures):
                values = np.asarray(
                    [
                        record[metric]
                        for record in records
                        if record["function"] == function_id
                        and record["structure"] == structure
                    ],
                    dtype=float,
                )
                column = 2 + structure_index * 2
                sheet.cell(row_number, column, float(values.mean()))
                sheet.cell(row_number, column + 1, float(values.std(ddof=1)))
        _style_summary(sheet, structures, len(functions))

    raw = workbook.create_sheet("Raw results")
    headers = [
        "Function",
        "Structure",
        "Seed",
        "Final objective",
        "Optimum",
        "Gap",
        "Log-gap",
        "Evaluations",
    ]
    raw.append(headers)
    for record in sorted(
        records, key=lambda item: (item["function"], item["structure"], item["seed"])
    ):
        raw.append(
            [
                f"F{record['function']}",
                record["structure"],
                record["seed"],
                record["best"],
                record["optimum"],
                record["gap"],
                record["log_gap"],
                record["evaluations"],
            ]
        )
    raw.freeze_panes = "A2"
    raw.auto_filter.ref = raw.dimensions
    for cell in raw[1]:
        cell.fill = PatternFill("solid", fgColor="1F4E78")
        cell.font = Font(color="FFFFFF", bold=True)
    for column, width in enumerate((12, 12, 10, 20, 14, 18, 18, 14), start=1):
        raw.column_dimensions[get_column_letter(column)].width = width
    for row in raw.iter_rows(min_row=2, min_col=4, max_col=7):
        for cell in row:
            cell.number_format = "0.0000000000"

    settings = workbook.create_sheet("Protocol", 0)
    settings.append(["Setting", "Value"])
    for key, value in protocol.items():
        settings.append([key, value])
    settings.column_dimensions["A"].width = 28
    settings.column_dimensions["B"].width = 70
    for cell in settings[1]:
        cell.fill = PatternFill("solid", fgColor="1F4E78")
        cell.font = Font(color="FFFFFF", bold=True)
    path.parent.mkdir(parents=True, exist_ok=True)
    workbook.save(path)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--library", type=Path, default=Path("trained_structures"))
    parser.add_argument("--dimension", type=int, default=10)
    parser.add_argument("--instance", type=int, default=1)
    parser.add_argument("--budget", type=int, default=10_000)
    parser.add_argument("--runs", type=int, default=30)
    parser.add_argument("--workers", type=int, default=min(4, os.cpu_count() or 1))
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("experiments/bbob_four_structures/bbob_f1_f24_instance1.xlsx"),
    )
    args = parser.parse_args(argv)
    library = StructureLibrary.load(args.library)
    tasks = [
        (function_id, structure_index, seed, args.dimension, args.instance, args.budget)
        for function_id in range(1, 25)
        for structure_index in range(len(library.entries))
        for seed in range(1, args.runs + 1)
    ]
    print(f"Running {len(tasks)} evaluations with {args.workers} workers...")
    with ProcessPoolExecutor(
        max_workers=args.workers,
        initializer=_initialize_worker,
        initargs=(args.library,),
    ) as executor:
        records = list(executor.map(_evaluate, tasks, chunksize=4))
    protocol = {
        "Suite": "IOH BBOB F1-F24",
        "Dimension": args.dimension,
        "Instance": args.instance,
        "MaxFE per run": args.budget,
        "Structures": ", ".join(
            f"G{entry['structure_id']:03d}" for entry in library.entries
        ),
        "Seeds": f"1-{args.runs}",
        "Runs per function and structure": args.runs,
        "Standard deviation": "Sample standard deviation (ddof=1)",
        "Gap": "max(final best objective - known optimum, 0)",
        "Log-gap": "log1p(gap)",
        "Boundary handling": "Clip to [-5, 5]",
        "Population size": 100,
    }
    _write_workbook(args.output, records, protocol)
    print(f"Saved {args.output}")


if __name__ == "__main__":
    main()
