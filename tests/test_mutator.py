import csv
import importlib.util
from dataclasses import dataclass
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from robustness_testing import Distance, Input, Mutation, Mutator, TextInput


class WordCountDistance(Distance):
    def distance(self, left, right):
        return abs(len(left.value.split()) - len(right.value.split()))


class AppendMutation(Mutation):
    def __init__(self):
        super().__init__(WordCountDistance())
        self.calls = 0

    def mutate_multiple(self, source, epsilon):
        self.calls += 1
        return [type(source)(f"{source.value} sample-{self.calls}")]


class MutatorTests(unittest.TestCase):
    def test_reference_script_uses_mutator_and_saves_inputs(self):
        path = Path(__file__).parents[1] / "examples" / "mutate_batch.py"
        spec = importlib.util.spec_from_file_location("mutator_example", path)
        example = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(example)
        with TemporaryDirectory() as directory:
            source, output = Path(directory) / "inputs.csv", Path(directory) / "mutated.csv"
            source.write_text("input,expected_output\nhello,reference\n", encoding="utf-8")
            with patch("sys.argv", [str(path), str(source), str(output), "--epsilon", "1", "-n", "2", "--max-attempts", "1"]), \
                 patch.object(example, "RandomWordsMutation", return_value=AppendMutation()), \
                 patch.object(example, "WordEditDistance", return_value=WordCountDistance()), patch("builtins.print"), \
                 self.assertWarnsRegex(UserWarning, "Collected 1/2"):
                example.main()
            with output.open(newline="", encoding="utf-8") as stream:
                rows = list(csv.DictReader(stream))
            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0]["Original_Input"], "hello")
            self.assertEqual(rows[0]["expected_output"], "reference")

    def test_retries_unchanged_results_per_input_and_keeps_repeated_changes(self):
        mutation = AppendMutation()
        attempts = []

        def transform(source, epsilon):
            attempts.append(source.value)
            return TextInput(source.value if len(attempts) % 2 else source.value + " changed")

        with patch.object(mutation, "mutate", side_effect=transform):
            rows = Mutator(mutation).mutate([{"input": "first"}, {"input": "second"}], 1, 2, max_attempts=10)
        self.assertEqual(attempts, ["first"] * 4 + ["second"] * 4)
        self.assertEqual([row["input"] for row in rows], ["first changed"] * 2 + ["second changed"] * 2)
        self.assertEqual([row["Mutation_Number"] for row in rows], [1, 2, 1, 2])
        self.assertTrue(all(row["Num_Mutations"] == 2 for row in rows))

    def test_attempt_limit_returns_partial_or_no_samples_and_warns(self):
        mutation = AppendMutation()
        with patch.object(mutation, "mutate", side_effect=lambda source, epsilon: source) as transform:
            with self.assertWarnsRegex(UserWarning, "Collected 0/3.*after 100 mutation attempts"):
                self.assertEqual(Mutator(mutation).mutate([{"input": "unchangeable"}], 1, 3), [])
            self.assertEqual(transform.call_count, 100)
        with self.assertWarnsRegex(UserWarning, "Collected 2/3.*after 2 mutation attempts"):
            rows = Mutator(mutation).mutate([{"input": "hello"}], 1, 3, max_attempts=2)
        self.assertEqual(len(rows), 2)
        self.assertEqual([row["Mutation_Number"] for row in rows], [1, 2])
        self.assertTrue(all(row["Num_Mutations"] == 3 for row in rows))
        with patch.object(mutation, "mutate", side_effect=ValueError("mutation failure")) as transform:
            with self.assertRaisesRegex(ValueError, "mutation failure"):
                Mutator(mutation).mutate([{"input": "hello"}], 1, 3)
            self.assertEqual(transform.call_count, 1)

    def test_changed_text_is_accepted_even_when_distance_is_zero(self):
        mutation = AppendMutation()
        with patch.object(mutation, "mutate", return_value=TextInput("different")):
            rows = Mutator(mutation).mutate([{"input": "original"}], 0)
        self.assertEqual(rows[0]["input"], "different")
        self.assertEqual(rows[0]["Input_Distance_To_Original_Input"], 0)

    def test_preserves_references_and_metadata_and_does_not_change_source_rows(self):
        source = [{"input": "always a", "expected_output": "G a", "id": "one"}]
        mutation = AppendMutation()
        mutator = Mutator(mutation, metadata={"Modification": "AppendMutation"})
        rows = mutator.mutate(iter(source), 1, 3)
        self.assertEqual([row["input"] for row in rows], [f"always a sample-{number}" for number in (1, 2, 3)])
        self.assertEqual([row["Mutation_Number"] for row in rows], [1, 2, 3])
        for row in rows:
            self.assertEqual(row["Original_Input"], "always a")
            self.assertEqual(row["expected_output"], "G a")
            self.assertEqual(row["id"], "one")
            self.assertEqual(row["Input_Distance_To_Original_Input"], 1)
            self.assertEqual(row["Max_Input_Distance_To_Original_Input"], 1)
            self.assertEqual(row["Num_Mutations"], 3)
            self.assertEqual(row["Modification"], "AppendMutation")
            self.assertTrue(row["Input_Distance_Measure"].endswith(".WordCountDistance"))
        self.assertEqual(source, [{"input": "always a", "expected_output": "G a", "id": "one"}])

    def test_accepts_custom_wrapper_and_column_names(self):
        @dataclass(frozen=True)
        class CustomInput(Input[str]):
            text: str

            @property
            def value(self):
                return self.text

        factory_inputs = []

        def factory(text):
            factory_inputs.append(text)
            return CustomInput(text)

        mutator = Mutator(AppendMutation(), factory, input_column="text", original_input_column="source_text")
        rows = mutator.mutate([{"text": "some text", "reference": 42}], 1)
        self.assertEqual(factory_inputs, ["some text"])
        self.assertEqual(rows[0]["source_text"], "some text")
        self.assertEqual(rows[0]["reference"], 42)
        self.assertNotIn("Original_Input", rows[0])

    def test_validates_parameters_schema_and_distance_bounds(self):
        mutation = AppendMutation()
        mutator = Mutator(mutation)
        for epsilon in (-1, float("inf"), float("nan"), True):
            with self.subTest(epsilon=epsilon), self.assertRaises((ValueError, TypeError)):
                mutator.mutate([{"input": "hello"}], epsilon)
        for count in (0, -1, 1.5, True):
            with self.subTest(count=count), self.assertRaises(ValueError):
                mutator.mutate([{"input": "hello"}], 1, count)
        for attempts in (0, -1, 1.5, True, "2", None):
            with self.subTest(attempts=attempts), self.assertRaises(ValueError):
                mutator.mutate([{"input": "hello"}], 1, max_attempts=attempts)
        for row in ({"other": "hello"}, {"input": "hello", "Original_Input": "old"}, {"input": 5}):
            with self.subTest(row=row), self.assertRaises((ValueError, TypeError)):
                mutator.mutate([{"input": "valid"}, row], 1)
        self.assertEqual(mutation.calls, 0)
        with self.assertRaises(ValueError):
            mutator.mutate([{"input": "hello"}], 0)
        with self.assertRaises(ValueError):
            Mutator(mutation, input_column="Original_Input")
        with self.assertRaises(ValueError):
            Mutator(mutation, metadata={"Num_Mutations": 5})

    def test_csv_round_trip_and_empty_dataset(self):
        with TemporaryDirectory() as directory:
            source = Path(directory) / "source.csv"
            output = Path(directory) / "nested" / "mutated.csv"
            source.write_text('input,expected_output\n"hello, world",reference\n', encoding="utf-8")
            mutator = Mutator(AppendMutation())
            rows = mutator.mutate_csv(source, output, 1, 2)
            with output.open(newline="", encoding="utf-8") as stream:
                reader = csv.DictReader(stream)
                self.assertEqual(reader.fieldnames, ["input", "expected_output", *mutator.added_columns])
                stored = list(reader)
            self.assertEqual(len(stored), 2)
            self.assertEqual(stored[0]["input"], rows[0]["input"])
            self.assertEqual(stored[0]["expected_output"], "reference")
            with self.assertWarnsRegex(UserWarning, "Collected 1/2"):
                rows = mutator.mutate_csv(source, output, 1, 2, max_attempts=1)
            self.assertEqual(len(rows), 1)
            with patch.object(mutator.mutation, "mutate", side_effect=lambda source, epsilon: source), \
                 self.assertWarnsRegex(UserWarning, "Collected 0/2"):
                self.assertEqual(mutator.mutate_csv(source, output, 1, 2, max_attempts=3), [])
            self.assertEqual(len(output.read_text().splitlines()), 1)
            source.write_text("input,expected_output\n", encoding="utf-8")
            self.assertEqual(mutator.mutate_csv(source, output, 1), [])
            self.assertEqual(len(output.read_text().splitlines()), 1)

    def test_invalid_csv_does_not_replace_existing_output(self):
        with TemporaryDirectory() as directory:
            source = Path(directory) / "source.csv"
            output = Path(directory) / "mutated.csv"
            output.write_text("previous results", encoding="utf-8")
            mutator = Mutator(AppendMutation())
            for content in ("input,expected\nhello\n", "input,expected\nhello,reference,extra\n", "input,input\na,b\n"):
                source.write_text(content, encoding="utf-8")
                with self.subTest(content=content), self.assertRaises(ValueError):
                    mutator.mutate_csv(source, output, 1)
                self.assertEqual(output.read_text(), "previous results")
            source.write_text("input\nhello\n", encoding="utf-8")
            with self.assertRaises(ValueError):
                mutator.mutate_csv(source, output, 0)
            self.assertEqual(output.read_text(), "previous results")
            with self.assertRaises(ValueError):
                mutator.mutate_csv(source, source, 1)
            self.assertEqual(source.read_text(), "input\nhello\n")


if __name__ == "__main__":
    unittest.main()
