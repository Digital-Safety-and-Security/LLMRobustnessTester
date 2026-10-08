"""Run black-box pipelines against prepared mutation datasets."""

from __future__ import annotations

import csv
from dataclasses import dataclass
import math
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping
import warnings

from .core import Distance, Output
from .mutator import MUTATION_METADATA_COLUMNS


@dataclass(frozen=True)
class TextOutput(Output[str]):
    text: str

    @property
    def value(self) -> str:
        return self.text


class Tester:
    """Test a callable pipeline(text, row_data) using an injected output distance.

    Factories wrap generated/reference values as Output objects. An optional
    batch_pipeline(rows) avoids per-input model initialization. Mutation
    generation and result persistence remain injectable, domain-specific calls.
    """

    def __init__(
        self, pipeline: Callable, output_distance: Distance, *,
        output_factory: Callable = TextOutput, reference_factory: Callable | None = None,
        input_column: str = "input", expected_output_column: str = "expected_output",
        original_input_column: str = "Original_Input", batch_pipeline: Callable | None = None,
    ):
        if not callable(pipeline) or not isinstance(output_distance, Distance):
            raise TypeError("pipeline must be callable and output_distance must be a Distance")
        if not callable(output_factory) or (reference_factory is not None and not callable(reference_factory)):
            raise TypeError("Output factories must be callable")
        if batch_pipeline is not None and not callable(batch_pipeline):
            raise TypeError("batch_pipeline must be callable")
        self.pipeline = pipeline
        self.batch_pipeline = batch_pipeline
        self.output_distance = output_distance
        self.output_factory = output_factory
        self.reference_factory = reference_factory if reference_factory is not None else output_factory
        self.input_column = input_column
        self.expected_output_column = expected_output_column
        self.original_input_column = original_input_column
        names = (input_column, expected_output_column, original_input_column)
        if any(not isinstance(name, str) or not name for name in names) or len(set(names)) != 3:
            raise ValueError("Input, expected-output, and original-input columns must have distinct non-empty names")
        self._required_columns = {*names, *MUTATION_METADATA_COLUMNS}

    def _validate_inputs(self, inputs: Iterable[Mapping[str, Any]]) -> list[dict]:
        rows, references, measures = [], {}, set()
        for source in inputs:
            row = dict(source)
            if missing := self._required_columns - row.keys():
                raise ValueError(f"Input row is missing columns: {', '.join(sorted(missing))}")
            if not isinstance(row[self.input_column], str) or not isinstance(row[self.original_input_column], str):
                raise TypeError("Input and original-input texts must be strings")
            for column in ("Input_Distance_To_Original_Input", "Max_Input_Distance_To_Original_Input"):
                if isinstance(row[column], bool):
                    raise TypeError(f"{column} must be a finite non-negative distance")
                row[column] = Distance.validate_distance_value(float(row[column]))
            if row["Input_Distance_To_Original_Input"] > row["Max_Input_Distance_To_Original_Input"]:
                raise ValueError("An input distance exceeds its recorded generation limit")
            original = row[self.original_input_column]
            if original in references and references[original] != row[self.expected_output_column]:
                raise ValueError("Each original input must have a single expected output")
            references[original] = row[self.expected_output_column]
            measures.add(row["Input_Distance_Measure"])
            rows.append(row)
        if not rows:
            raise ValueError("Evaluation datasets are empty")
        if len(measures) != 1:
            raise ValueError("All evaluation datasets must use the same input distance measure")
        return rows

    def load_inputs(self, filenames: Iterable[str | Path]) -> list[dict]:
        """Load CSVs produced by Mutator, retaining reference/context columns."""
        inputs = []
        for filename in filenames:
            with Path(filename).open(newline="", encoding="utf-8-sig") as stream:
                reader = csv.DictReader(stream)
                columns = reader.fieldnames or []
                if not columns or len(set(columns)) != len(columns):
                    raise ValueError(f"{filename}: missing or duplicate CSV column names")
                if missing := self._required_columns - set(columns):
                    raise ValueError(f"{filename} is missing columns: {', '.join(sorted(missing))}")
                rows = list(reader)
                if any(None in row or any(value is None for value in row.values()) for row in rows):
                    raise ValueError(f"{filename}: CSV rows must match the header width")
                inputs.extend(rows)
        return self._validate_inputs(inputs)

    def _evaluate(self, rows: list[dict], original_outputs: dict) -> list[dict]:
        outputs = list(self.batch_pipeline([dict(row) for row in rows])) if self.batch_pipeline is not None else [
            self.pipeline(row[self.input_column], dict(row)) for row in rows
        ]
        if len(outputs) != len(rows):
            raise RuntimeError(f"Pipeline returned {len(outputs)} outputs for {len(rows)} inputs")
        original_outputs.update({
            row[self.original_input_column]: output
            for row, output in zip(rows, outputs)
            if row[self.input_column] == row[self.original_input_column]
        })
        results = []
        for row, output in zip(rows, outputs):
            result = {
                **row, "Expected_Output": row[self.expected_output_column], "Generated_Output": output,
                "Output_Distance_Measure": type(self.output_distance).__name__,
            }
            for column, reference, factory in (
                ("Output_Distance_To_Expected_Output", row[self.expected_output_column], self.reference_factory),
                ("Output_Distance_To_Original_Output", original_outputs[row[self.original_input_column]], self.output_factory),
            ):
                try:
                    result[column] = self.output_distance(self.output_factory(output), factory(reference))
                except ValueError as exc:
                    result[column] = None
                    warnings.warn(f"{column} unavailable for input {row[self.input_column]!r}: {exc}", RuntimeWarning, stacklevel=2)
            results.append(result)
        return results

    @staticmethod
    def summarize(results, epsilon, threshold_check, threshold, *, input_column="input", original_input_column="Original_Input"):
        if threshold_check not in ("individual", "average"):
            raise ValueError("threshold_check must be individual or average")
        mutations = [row for row in results if row[input_column] != row[original_input_column]]
        distances = []
        for row in mutations:
            value = row["Output_Distance_To_Original_Output"]
            value = None if value is None else float(value)
            distances.append(None if value is None or math.isnan(value) else Distance.validate_distance_value(value))
        missing = sorted({row[original_input_column] for row in results} - {row[original_input_column] for row in mutations})
        complete = bool(distances) and all(value is not None for value in distances) and not missing
        maximum = max(distances) if complete else None
        average = math.fsum(value / len(distances) for value in distances) if complete else None
        checked = maximum if threshold_check == "individual" else average
        return {
            "Epsilon": epsilon, "Max_Delta": maximum, "Average_Delta": average,
            "Threshold_Check": threshold_check, "Checked_Delta": checked,
            "Output_Distance_Threshold": threshold,
            "Threshold_Holds": checked <= threshold if checked is not None and threshold is not None else None,
            "Mutated_Rows": len(mutations), "Unavailable_Output_Distances": distances.count(None),
            "Benchmarks_Without_Mutations": missing,
        }

    def run(
        self, inputs, *, mode="max-delta", epsilon=None, output_distance_threshold=None,
        threshold_check="individual", min_epsilon=1, epsilon_step="double", max_iterations=5,
        generate_mutations=None, on_iteration=None, clear_cache_each_iteration=False,
    ) -> dict:
        """Return final Results and a robustness summary over sampled mutations.

        generate_mutations(epsilon, iteration) returns fresh prepared rows;
        on_iteration(iteration, epsilon, results, observation) can persist them
        and add paths to the observation. Prior samples and baselines are reused.
        A run has independent caches, including when this Tester is reused.
        """
        if mode not in ("max-delta", "search-epsilon"):
            raise ValueError("mode must be max-delta or search-epsilon")
        if threshold_check not in ("individual", "average") or epsilon_step not in ("double", "add-one"):
            raise ValueError("Invalid threshold_check or epsilon_step")
        if isinstance(max_iterations, bool) or not isinstance(max_iterations, int) or max_iterations < 1:
            raise ValueError("max_iterations must be a positive integer")
        min_epsilon = Distance.validate_distance_value(min_epsilon)
        if min_epsilon == 0:
            raise ValueError("min_epsilon must be positive")
        if epsilon is not None:
            epsilon = Distance.validate_distance_value(epsilon)
        if output_distance_threshold is not None:
            output_distance_threshold = Distance.validate_distance_value(output_distance_threshold)
        search = mode == "search-epsilon"
        if search and (output_distance_threshold is None or epsilon is not None):
            raise ValueError("Search requires an output threshold and uses min_epsilon, not epsilon")
        if generate_mutations is not None and (not search or not callable(generate_mutations)):
            raise ValueError("generate_mutations must be callable and requires search-epsilon mode")
        if clear_cache_each_iteration and not search:
            raise ValueError("clear_cache_each_iteration requires search-epsilon mode")
        if on_iteration is not None and not callable(on_iteration):
            raise TypeError("on_iteration must be callable")
        rows = self._validate_inputs(inputs)
        originals = {}
        limits = {}
        for row in rows:
            original = row[self.original_input_column]
            if original not in originals:
                originals[original] = {**row, self.input_column: original, "Input_Distance_To_Original_Input": 0.0, "Mutation_Number": 0}
            limits[original] = max(limits.get(original, 0), row["Max_Input_Distance_To_Original_Input"])
        coverage = None if generate_mutations is not None else min(limits.values())
        epsilon = min_epsilon if search else (max(limits.values()) if epsilon is None else epsilon)
        suite = list(originals.values())
        if generate_mutations is None:
            suite.extend(row for row in rows if row[self.input_column] != row[self.original_input_column])
        cached, original_outputs, history, results = {}, {}, [], []
        passing = None
        reason = "iteration_limit" if search else "completed"
        for iteration in range(1, (max_iterations if search else 1) + 1):
            if coverage is not None and search and epsilon > coverage:
                reason = "dataset_limit"
                break
            if clear_cache_each_iteration:
                cached.clear()
                original_outputs.clear()
            if generate_mutations is not None:
                fresh = self._validate_inputs(generate_mutations(epsilon, iteration))
                for row in fresh:
                    original = row[self.original_input_column]
                    if original not in originals or row[self.expected_output_column] != originals[original][self.expected_output_column]:
                        raise ValueError("Generated inputs must preserve the original inputs and reference outputs")
                    if row["Input_Distance_Measure"] != rows[0]["Input_Distance_Measure"] or row["Input_Distance_To_Original_Input"] > epsilon:
                        raise ValueError("Generated input distances must use the configured measure and stay within epsilon")
                suite.extend(row for row in fresh if row[self.input_column] != row[self.original_input_column])
            selected = [index for index, row in enumerate(suite) if row["Input_Distance_To_Original_Input"] <= epsilon]
            pending = [index for index in selected if index not in cached]
            if pending:
                cached.update(zip(pending, self._evaluate([suite[index] for index in pending], original_outputs)))
            results = [cached[index] for index in selected]
            observation = self.summarize(results, epsilon, threshold_check, output_distance_threshold,
                                         input_column=self.input_column, original_input_column=self.original_input_column)
            if on_iteration is not None:
                on_iteration(iteration, epsilon, results, observation)
            history.append(observation)
            if observation["Checked_Delta"] is None:
                reason = "incomplete_distances_or_benchmarks"
                break
            if search:
                if not observation["Threshold_Holds"]:
                    reason = "threshold_exceeded"
                    break
                passing = epsilon
                if iteration < max_iterations:
                    epsilon = Distance.validate_distance_value(epsilon * 2 if epsilon_step == "double" else epsilon + 1)
        return {
            "Mode": mode, "Generate_Mutations": generate_mutations is not None,
            "Clear_Cache_Each_Iteration": clear_cache_each_iteration,
            "Input_Distance_Measure": rows[0]["Input_Distance_Measure"],
            "Output_Distance_Measure": type(self.output_distance).__name__,
            "Prepared_Input_Coverage_Limit": coverage, "Max_Tested_Passing_Epsilon": passing,
            "Stop_Reason": reason, "Iterations": history, "Results": results,
        }
