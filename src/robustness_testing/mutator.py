"""Generate reusable, annotated datasets of mutated natural-language inputs."""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Generic, Iterable, Mapping
import warnings

from .core import Input, InputT, Mutation


MUTATION_METADATA_COLUMNS = (
    "Modification", "Transformation", "Input_Distance_Measure", "Max_Input_Distance_To_Original_Input",
    "Num_Mutations", "Mutation_Number", "Input_Distance_To_Original_Input",
)


@dataclass(frozen=True)
class TextInput(Input[str]):
    text: str

    @property
    def value(self) -> str:
        return self.text


class Mutator(Generic[InputT]):
    """Mutate dataset rows while preserving reference outputs and other columns.

    Inject any Mutation implementation and, if needed, its concrete Input
    wrapper. Each source row targets num_mutations changed samples within a
    bounded number of attempts. Repeated changed samples are retained.
    """

    def __init__(
        self,
        mutation: Mutation[InputT],
        input_factory: Callable[[str], InputT] = TextInput,
        *,
        input_column: str = "input",
        original_input_column: str = "Original_Input",
        metadata: Mapping[str, Any] | None = None,
    ) -> None:
        if not isinstance(mutation, Mutation):
            raise TypeError("mutation must be a Mutation instance")
        self.mutation = mutation
        self.input_factory = input_factory
        self.input_column = input_column
        self.original_input_column = original_input_column
        labels = dict(metadata or {})
        if unknown := set(labels) - {"Modification", "Transformation", "Input_Distance_Measure"}:
            raise ValueError(f"Unknown mutation metadata labels: {', '.join(sorted(unknown))}")
        self.metadata = {
            "Modification": f"{type(mutation).__module__}.{type(mutation).__qualname__}",
            "Transformation": "",
            "Input_Distance_Measure": f"{type(mutation.distance).__module__}.{type(mutation.distance).__qualname__}",
            **labels,
        }
        self.added_columns = (
            original_input_column, "Input_Distance_To_Original_Input", "Mutation_Number",
            *self.metadata, "Max_Input_Distance_To_Original_Input", "Num_Mutations",
        )
        if len(set(self.added_columns)) != len(self.added_columns) or input_column in self.added_columns:
            raise ValueError("Input and original-input column names must be distinct from mutation metadata")

    def _validate_columns(self, columns: Iterable[str]) -> None:
        columns = set(columns)
        if self.input_column not in columns:
            raise ValueError(f"Source dataset must contain an {self.input_column!r} column")
        if columns.intersection(self.added_columns):
            raise ValueError("Source dataset already contains mutation metadata")

    def mutate(
        self, inputs: Iterable[Mapping[str, Any]], max_distance: float, num_mutations: int = 1,
        max_attempts: int = 100,
    ) -> list[dict[str, Any]]:
        """Collect changed samples, using at most max_attempts calls per input.

        Warn and return fewer samples if the attempt budget is exhausted.
        Num_Mutations records the requested count; Mutation_Number counts only
        accepted samples. All attempts mutate the original, not prior samples.
        """
        max_distance = self.mutation.validate_epsilon(max_distance)
        if isinstance(num_mutations, bool) or not isinstance(num_mutations, int) or num_mutations < 1:
            raise ValueError("num_mutations must be a positive integer")
        if isinstance(max_attempts, bool) or not isinstance(max_attempts, int) or max_attempts < 1:
            raise ValueError("max_attempts must be a positive integer")
        # Validate all rows before invoking a potentially expensive mutation.
        inputs = list(inputs)
        for row in inputs:
            self._validate_columns(row)
            if not isinstance(row[self.input_column], str):
                raise TypeError("Input text must be a string")
        rows = []
        for row in inputs:
            source = self.input_factory(row[self.input_column])
            collected = 0
            for _ in range(max_attempts):
                mutated = self.mutation(source, max_distance)
                if mutated.value == source.value:
                    continue
                collected += 1
                rows.append({
                    **row,
                    self.input_column: mutated.value,
                    self.original_input_column: source.value,
                    "Input_Distance_To_Original_Input": self.mutation.distance(source, mutated),
                    "Mutation_Number": collected,
                    **self.metadata,
                    "Max_Input_Distance_To_Original_Input": max_distance,
                    "Num_Mutations": num_mutations,
                })
                if collected == num_mutations:
                    break
            if collected < num_mutations:
                warnings.warn(
                    f"Collected {collected}/{num_mutations} changed samples for input {source.value!r} "
                    f"after {max_attempts} mutation attempts", stacklevel=2,
                )
        return rows

    def mutate_csv(
        self, input_file: str | Path, output_file: str | Path,
        max_distance: float, num_mutations: int = 1, max_attempts: int = 100,
    ) -> list[dict[str, Any]]:
        """Read source rows and save a prepared CSV only after successful mutation."""
        output = Path(output_file)
        if output.exists() and output.samefile(input_file):
            raise ValueError("Output CSV must not overwrite the source CSV")
        with Path(input_file).open(newline="", encoding="utf-8") as source:
            reader = csv.DictReader(source)
            columns = reader.fieldnames or []
            self._validate_columns(columns)
            if len(set(columns)) != len(columns):
                raise ValueError("Source CSV has duplicate column names")
            inputs = list(reader)
        if any(None in row or any(value is None for value in row.values()) for row in inputs):
            raise ValueError("Source CSV rows must match the header width")
        rows = self.mutate(inputs, max_distance, num_mutations, max_attempts)
        output.parent.mkdir(parents=True, exist_ok=True)
        with output.open("w", newline="", encoding="utf-8") as target:
            writer = csv.DictWriter(target, fieldnames=[*columns, *self.added_columns])
            writer.writeheader()
            writer.writerows(rows)
        return rows
