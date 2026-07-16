"""Core abstract contracts for robustness testing.

The classes in this module intentionally do not encode any concrete data type,
model, or distance. Domain-specific projects subclass them for text, images,
temporal logic formulas, trajectories, or other artifacts.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from math import isfinite
from numbers import Real
from typing import Any, Generic, TypeVar


ValueT = TypeVar("ValueT")
ArtifactT = TypeVar("ArtifactT", bound="RobustnessObject[Any]")
InputT = TypeVar("InputT", bound="Input[Any]")
OutputT = TypeVar("OutputT", bound="Output[Any]")


class RobustnessObject(ABC, Generic[ValueT]):
    """Abstract base for values used in robustness tests."""

    @property
    @abstractmethod
    def value(self) -> ValueT:
        """Return the wrapped concrete value."""

    def is_same_type_as(self, other: object) -> bool:
        """Return whether two robustness objects have the same concrete class."""

        return isinstance(other, RobustnessObject) and type(self) is type(other)

    @classmethod
    def are_same_type(
        cls,
        left: "RobustnessObject[Any]",
        right: "RobustnessObject[Any]",
    ) -> bool:
        """Return whether two robustness objects have the same concrete class."""

        return type(left) is type(right)

    def require_same_type_as(self, other: "RobustnessObject[Any]") -> None:
        """Raise ``TypeError`` unless ``other`` has the same concrete class."""

        if not self.is_same_type_as(other):
            raise TypeError(
                "Robustness objects must have the same concrete type: "
                f"{type(self).__name__} != {type(other).__name__}"
            )


class Input(RobustnessObject[ValueT], ABC):
    """Abstract input object for a system under test."""


class Output(RobustnessObject[ValueT], ABC):
    """Abstract output object produced by a system under test."""


class Distance(ABC, Generic[ArtifactT]):
    """Abstract distance over two inputs or two outputs of the same class."""

    def __call__(self, left: ArtifactT, right: ArtifactT) -> float:
        """Compute and validate a finite, non-negative real-valued distance."""

        self.validate_operands(left, right)
        raw_distance = self.distance(left, right)
        return self.validate_distance_value(raw_distance)

    def validate_operands(self, left: ArtifactT, right: ArtifactT) -> None:
        """Raise ``TypeError`` unless operands are same-type robustness objects."""

        if not isinstance(left, RobustnessObject):
            raise TypeError(f"left must be a robustness object, got {type(left).__name__}")
        if not isinstance(right, RobustnessObject):
            raise TypeError(f"right must be a robustness object, got {type(right).__name__}")
        left.require_same_type_as(right)

    @staticmethod
    def validate_distance_value(value: Real) -> float:
        """Return ``value`` as ``float`` after validating the distance contract."""

        if isinstance(value, bool) or not isinstance(value, Real):
            raise TypeError(f"distance must be a real scalar, got {type(value).__name__}")

        distance = float(value)
        if not isfinite(distance):
            raise ValueError("distance must be finite")
        if distance < 0.0:
            raise ValueError("distance must be non-negative")
        return distance

    @abstractmethod
    def distance(self, left: ArtifactT, right: ArtifactT) -> Real:
        """Return the concrete distance between two same-type objects."""


class Mutation(ABC, Generic[InputT]):
    """Abstract input mutation constrained by a distance threshold."""

    def __init__(self, distance: Distance[InputT]) -> None:
        self.distance = distance

    def __call__(self, source: InputT, epsilon: float) -> InputT:
        """Return a mutated input whose distance from ``source`` is at most epsilon."""

        self.validate_source(source)
        epsilon = self.validate_epsilon(epsilon)

        mutated = self.mutate(source, epsilon)
        self._validate_mutation(source, mutated, epsilon)
        return mutated

    @staticmethod
    def validate_source(source: InputT) -> None:
        """Raise ``TypeError`` unless ``source`` is an ``Input`` object."""

        if not isinstance(source, Input):
            raise TypeError(f"source must be an Input, got {type(source).__name__}")

    @staticmethod
    def validate_epsilon(epsilon: float) -> float:
        """Return ``epsilon`` as ``float`` after validating it is finite and non-negative."""

        if isinstance(epsilon, bool) or not isinstance(epsilon, Real):
            raise TypeError(f"epsilon must be a real scalar, got {type(epsilon).__name__}")

        epsilon_float = float(epsilon)
        if not isfinite(epsilon_float):
            raise ValueError("epsilon must be finite")
        if epsilon_float < 0.0:
            raise ValueError("epsilon must be non-negative")
        return epsilon_float
    
    def mutate_multiple_check_distance(self, source: InputT, epsilon: float) -> list[InputT]:
        """Return a list of mutated inputs whose distance from ``source`` is at most epsilon."""

        self.validate_source(source)
        epsilon = self.validate_epsilon(epsilon)

        mutated_list = self.mutate_multiple(source, epsilon)
        for mutated in mutated_list:
            self._validate_mutation(source, mutated, epsilon)
        return mutated_list

    def mutate(self, source: InputT, epsilon: float) -> InputT:
        """Return a concrete mutation of ``source``."""
        mutations = self.mutate_multiple(source, epsilon)
        if not mutations:
            raise ValueError("No mutations were possible.")
        return mutations[0]

    @abstractmethod
    def mutate_multiple(self, source: InputT, epsilon: float) -> list[InputT]:
        """Return a list of concrete mutations of ``source``."""

        raise NotImplementedError(
            f"{type(self).__name__} does not implement mutate_multiple()"
        )
    
    def _validate_mutation(self, source: InputT, mutated: InputT, epsilon: float) -> None:
        """Validate that a mutation is valid and within the distance threshold."""

        self.validate_source(mutated)
        source.require_same_type_as(mutated)
        
        actual_distance = self.distance(source, mutated)
        if actual_distance > epsilon:
            raise ValueError(
                "mutation exceeded epsilon: "
                f"distance={actual_distance}, epsilon={epsilon}"
            )


class SUT(ABC, Generic[InputT, OutputT]):
    """Abstract system under test mapping an input object to an output object."""

    def __call__(self, input_object: InputT) -> OutputT:
        """Run the system under test and validate that it returns an output object."""

        if not isinstance(input_object, Input):
            raise TypeError(
                f"input_object must be an Input, got {type(input_object).__name__}"
            )

        output = self.run(input_object)
        if not isinstance(output, Output):
            raise TypeError(f"SUT must return an Output, got {type(output).__name__}")
        return output

    @abstractmethod
    def run(self, input_object: InputT) -> OutputT:
        """Return the output produced by the concrete system under test."""
