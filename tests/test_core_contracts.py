from __future__ import annotations

import unittest
from dataclasses import dataclass
from numbers import Real

from robustness_testing import Distance, Input, Mutation, Output, SUT


@dataclass(frozen=True)
class TextInput(Input[str]):
    text: str

    @property
    def value(self) -> str:
        return self.text


@dataclass(frozen=True)
class OtherInput(Input[str]):
    text: str

    @property
    def value(self) -> str:
        return self.text


@dataclass(frozen=True)
class TextOutput(Output[str]):
    text: str

    @property
    def value(self) -> str:
        return self.text


class LengthDistance(Distance[TextInput]):
    def distance(self, left: TextInput, right: TextInput) -> Real:
        return abs(len(left.value) - len(right.value))


class NonRealDistance(Distance[TextInput]):
    def distance(self, left: TextInput, right: TextInput) -> Real:
        return "not-a-distance"  # type: ignore[return-value]


class NegativeDistance(Distance[TextInput]):
    def distance(self, left: TextInput, right: TextInput) -> Real:
        return -1.0


class IdentityMutation(Mutation[TextInput]):
    def mutate_multiple(self, source: TextInput, epsilon: float) -> list[TextInput]:
        return [source]


class TooFarMutation(Mutation[TextInput]):
    def mutate_multiple(self, source: TextInput, epsilon: float) -> list[TextInput]:
        return [TextInput(source.value + "xx")]


class WrongTypeMutation(Mutation[TextInput]):
    def mutate_multiple(self, source: TextInput, epsilon: float) -> list[TextInput]:
        return [OtherInput(source.value)]  # type: ignore[list-item]


class EchoSUT(SUT[TextInput, TextOutput]):
    def run(self, input_object: TextInput) -> TextOutput:
        return TextOutput(input_object.value)


class BadSUT(SUT[TextInput, TextOutput]):
    def run(self, input_object: TextInput) -> TextOutput:
        return input_object.value  # type: ignore[return-value]


class CoreContractTests(unittest.TestCase):
    def test_inputs_and_outputs_report_same_concrete_type(self) -> None:
        left = TextInput("hello")
        right = TextInput("world")
        other = OtherInput("hello")
        output = TextOutput("hello")

        self.assertTrue(left.is_same_type_as(right))
        self.assertTrue(Input.are_same_type(left, right))
        self.assertFalse(left.is_same_type_as(other))
        self.assertFalse(left.is_same_type_as(output))

    def test_distance_requires_same_concrete_type(self) -> None:
        distance = LengthDistance()

        self.assertEqual(distance(TextInput("a"), TextInput("abcd")), 3.0)
        with self.assertRaises(TypeError):
            distance(TextInput("a"), OtherInput("abcd"))  # type: ignore[arg-type]

    def test_distance_must_be_real_finite_and_non_negative(self) -> None:
        with self.assertRaises(TypeError):
            NonRealDistance()(TextInput("a"), TextInput("b"))
        with self.assertRaises(ValueError):
            NegativeDistance()(TextInput("a"), TextInput("b"))

    def test_mutation_returns_same_type_within_epsilon(self) -> None:
        mutation = IdentityMutation(LengthDistance())

        source = TextInput("hello")
        self.assertEqual(mutation(source, epsilon=0.0), source)

    def test_mutation_rejects_negative_epsilon(self) -> None:
        mutation = IdentityMutation(LengthDistance())

        with self.assertRaises(ValueError):
            mutation(TextInput("hello"), epsilon=-0.1)

    def test_mutation_rejects_wrong_type_and_excess_distance(self) -> None:
        with self.assertRaises(TypeError):
            WrongTypeMutation(LengthDistance())(TextInput("hello"), epsilon=10.0)

        with self.assertRaises(ValueError):
            TooFarMutation(LengthDistance())(TextInput("hello"), epsilon=1.0)

    def test_sut_maps_input_to_output(self) -> None:
        self.assertEqual(EchoSUT()(TextInput("hello")), TextOutput("hello"))

        with self.assertRaises(TypeError):
            EchoSUT()("hello")  # type: ignore[arg-type]
        with self.assertRaises(TypeError):
            BadSUT()(TextInput("hello"))

    def test_core_classes_are_abstract(self) -> None:
        with self.assertRaises(TypeError):
            Input()  # type: ignore[abstract]
        with self.assertRaises(TypeError):
            Output()  # type: ignore[abstract]
        with self.assertRaises(TypeError):
            Distance()  # type: ignore[abstract]
        with self.assertRaises(TypeError):
            Mutation(LengthDistance())  # type: ignore[abstract]
        with self.assertRaises(TypeError):
            SUT()  # type: ignore[abstract]


if __name__ == "__main__":
    unittest.main()
