import csv
import importlib.util
import io
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import Mock, patch

from robustness_testing import Distance, Mutator, Mutation, Tester, TextInput


class LengthDistance(Distance):
    def distance(self, left, right):
        return abs(len(left.value) - len(right.value))


class AppendMutation(Mutation):
    def __init__(self):
        super().__init__(LengthDistance())

    def mutate_multiple(self, source, epsilon):
        return [TextInput(source.value + "x" * int(epsilon))]


def prepared(distances=(1, 2, 4), original="a", limit=4):
    mutator = Mutator(AppendMutation())
    return [
        {**mutator.mutate([{"input": original, "expected_output": "reference", "prefix": "context:"}], epsilon)[0],
         "Max_Input_Distance_To_Original_Input": limit}
        for epsilon in distances
    ]


class TesterTests(unittest.TestCase):
    def test_reference_runner_loads_inputs_and_prints_results(self):
        path = Path(__file__).parents[1] / "robustness.py"
        spec = importlib.util.spec_from_file_location("tester_example", path)
        example = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(example)
        with TemporaryDirectory() as directory:
            source = Path(directory) / "inputs.csv"
            rows = prepared()
            with source.open("w", newline="") as stream:
                writer = csv.DictWriter(stream, fieldnames=rows[0].keys())
                writer.writeheader()
                writer.writerows(rows)
            with patch("sys.argv", [str(path), "--eval-files", str(source), "--epsilon", "2"]), \
                 patch("sys.stdout", new_callable=io.StringIO) as output:
                example.main()
            result = json.loads(output.getvalue())
            self.assertEqual(result["Results"][0]["Generated_Output"], "A")
            self.assertEqual(len(result["Results"]), 3)
            self.assertEqual(result["Iterations"][0]["Max_Delta"], 1)

    def test_callable_receives_text_and_context_and_both_distances_are_computed(self):
        calls = []

        def pipeline(text, data):
            calls.append((text, data["prefix"]))
            return data["prefix"] + text

        tester = Tester(pipeline, LengthDistance())
        inputs = prepared()
        result = tester.run(inputs, epsilon=2)
        self.assertEqual(calls, [("a", "context:"), ("ax", "context:"), ("axx", "context:")])
        self.assertEqual([row["Output_Distance_To_Original_Output"] for row in result["Results"]], [0, 1, 2])
        self.assertEqual([row["Output_Distance_To_Expected_Output"] for row in result["Results"]], [0, 1, 2])
        self.assertEqual(result["Iterations"][0]["Max_Delta"], 2)
        self.assertNotIn("Generated_Output", inputs[0])
        calls.clear()
        tester.run(inputs, epsilon=1)
        self.assertEqual([text for text, _ in calls], ["a", "ax"])

    def test_prepared_search_grows_stops_and_reuses_baselines(self):
        cases = [
            ({"output_distance_threshold": 2}, [1, 2, 4], 2, "threshold_exceeded"),
            ({"output_distance_threshold": 2, "threshold_check": "average"}, [1, 2, 4], 2, "threshold_exceeded"),
            ({"output_distance_threshold": 2, "threshold_check": "average", "epsilon_step": "add-one"}, [1, 2, 3, 4], 3, "threshold_exceeded"),
            ({"output_distance_threshold": 100, "max_iterations": 2}, [1, 2], 2, "iteration_limit"),
            ({"output_distance_threshold": 100}, [1, 2, 4], 4, "dataset_limit"),
            ({"output_distance_threshold": 0}, [1], None, "threshold_exceeded"),
            ({"output_distance_threshold": 100, "min_epsilon": 5}, [], None, "dataset_limit"),
        ]
        for options, epsilons, passing, reason in cases:
            with self.subTest(options=options):
                pipeline = Mock(side_effect=lambda text, data: text)
                history = []
                tester = Tester(pipeline, LengthDistance())
                result = tester.run(prepared(), mode="search-epsilon", **options,
                                    on_iteration=lambda iteration, epsilon, rows, summary: history.append((iteration, epsilon, len(rows))))
                self.assertEqual([row["Epsilon"] for row in result["Iterations"]], epsilons)
                self.assertEqual(result["Max_Tested_Passing_Epsilon"], passing)
                self.assertEqual(result["Stop_Reason"], reason)
                self.assertEqual([epsilon for _, epsilon, _ in history], epsilons)
                self.assertEqual(sum(call.args[0] == "a" for call in pipeline.call_args_list), bool(epsilons))
                if epsilons:
                    self.assertEqual(result["Results"][0]["Output_Distance_To_Original_Output"], 0)

    def test_fresh_search_uses_mutator_and_continues_beyond_prepared_coverage(self):
        mutator = Mutator(AppendMutation())
        generate = Mock(side_effect=lambda epsilon, iteration: mutator.mutate([
            {"input": "a", "expected_output": "a"}
        ], epsilon, 2))
        inputs = mutator.mutate([{"input": "a", "expected_output": "a"}], 1)
        pipeline = Mock(side_effect=lambda text, data: text)
        result = Tester(pipeline, LengthDistance()).run(
            inputs, mode="search-epsilon", output_distance_threshold=2, generate_mutations=generate,
        )
        self.assertEqual([call.args for call in generate.call_args_list], [(1, 1), (2, 2), (4, 3)])
        self.assertIsNone(result["Prepared_Input_Coverage_Limit"])
        self.assertEqual(result["Max_Tested_Passing_Epsilon"], 2)
        self.assertEqual([row["Mutated_Rows"] for row in result["Iterations"]], [2, 4, 6])
        self.assertEqual(sum(call.args[0] == "a" for call in pipeline.call_args_list), 1)

    def test_invalid_output_and_missing_benchmarks_do_not_pass(self):
        class RejectInvalid(LengthDistance):
            def distance(self, left, right):
                if "invalid" in (left.value, right.value):
                    raise ValueError("invalid output")
                return super().distance(left, right)

        with self.assertWarns(RuntimeWarning):
            result = Tester(lambda text, data: "a" if text == "a" else "invalid", RejectInvalid()).run(
                prepared(), mode="search-epsilon", output_distance_threshold=100,
            )
        self.assertEqual(result["Stop_Reason"], "incomplete_distances_or_benchmarks")
        self.assertIsNone(result["Iterations"][0]["Threshold_Holds"])
        self.assertEqual(result["Iterations"][0]["Unavailable_Output_Distances"], 1)
        result = Tester(lambda text, data: text, LengthDistance()).run(
            [*prepared(), *prepared((2,), original="b")], epsilon=1,
        )
        self.assertEqual(result["Iterations"][0]["Benchmarks_Without_Mutations"], ["b"])

    def test_custom_output_factories_and_batch_callable(self):
        batch = Mock(side_effect=lambda rows: [{"value": row["input"]} for row in rows])
        tester = Tester(
            Mock(side_effect=AssertionError("single-input callable should not run")), LengthDistance(),
            batch_pipeline=batch, output_factory=lambda output: TextInput(output["value"]), reference_factory=TextInput,
        )
        result = tester.run(prepared(), mode="search-epsilon", output_distance_threshold=2)
        self.assertEqual(batch.call_count, 3)
        self.assertEqual(result["Iterations"][-1]["Max_Delta"], 4)
        batch.return_value = []
        batch.side_effect = None
        with self.assertRaises(RuntimeError):
            tester.run(prepared())

    def test_loading_custom_column_names(self):
        rows = prepared()
        aliases = {"input": "text", "expected_output": "reference", "Original_Input": "source_text"}
        rows = [{aliases.get(key, key): value for key, value in row.items()} for row in rows]
        with TemporaryDirectory() as directory:
            source = Path(directory) / "inputs.csv"
            with source.open("w", newline="") as stream:
                writer = csv.DictWriter(stream, fieldnames=rows[0].keys())
                writer.writeheader()
                writer.writerows(rows)
            tester = Tester(lambda text, data: text, LengthDistance(), input_column="text",
                            expected_output_column="reference", original_input_column="source_text")
            inputs = tester.load_inputs([source])
            self.assertEqual(tester.run(inputs)["Results"][0]["source_text"], "a")
            empty_bad_source = Path(directory) / "bad.csv"
            empty_bad_source.write_text("text\n")
            with self.assertRaises(ValueError):
                tester.load_inputs([source, empty_bad_source])
            for content in ("text,text\na,b\n", "text,reference\na\n"):
                source.write_text(content)
                with self.assertRaises(ValueError):
                    tester.load_inputs([source])

    def test_legacy_metadata_columns_are_rejected(self):
        pipeline = Mock()
        tester = Tester(pipeline, LengthDistance())
        for old, current in (
            ("Actual_Distance", "Input_Distance_To_Original_Input"),
            ("Distance_Measure", "Input_Distance_Measure"),
            ("Max_Distance", "Max_Input_Distance_To_Original_Input"),
        ):
            with self.subTest(column=old), TemporaryDirectory() as directory:
                rows = prepared()
                for row in rows:
                    row[old] = row.pop(current)
                with self.assertRaisesRegex(ValueError, current):
                    tester.run(rows)
                source = Path(directory) / "inputs.csv"
                with source.open("w", newline="") as stream:
                    writer = csv.DictWriter(stream, fieldnames=rows[0].keys())
                    writer.writeheader()
                    writer.writerows(rows)
                with self.assertRaisesRegex(ValueError, current):
                    tester.load_inputs([source])
        pipeline.assert_not_called()

    def test_validation_happens_before_pipeline_runs(self):
        pipeline = Mock()
        tester = Tester(pipeline, LengthDistance())
        cases = [{"mode": "bad"}, {"mode": "search-epsilon"}, {"max_iterations": 0}, {"min_epsilon": 0},
                 {"threshold_check": "bad"}, {"epsilon_step": "bad"}, {"epsilon": -1}, {"output_distance_threshold": float("inf")},
                 {"generate_mutations": lambda epsilon, iteration: []}]
        for options in cases:
            with self.subTest(options=options), self.assertRaises(ValueError):
                tester.run(prepared(), **options)
        for column, value in (("Input_Distance_To_Original_Input", -1), ("Max_Input_Distance_To_Original_Input", 0),
                              ("Input_Distance_Measure", "another measure"), ("expected_output", "different reference")):
            inputs = prepared()
            inputs[0][column] = value
            with self.subTest(column=column), self.assertRaises(ValueError):
                tester.run(inputs)
        pipeline.assert_not_called()

    def test_invalid_fresh_inputs_are_rejected(self):
        for column, value in (("Original_Input", "unknown"), ("expected_output", "wrong reference"),
                              ("Input_Distance_To_Original_Input", 2), ("Input_Distance_Measure", "another measure")):
            with self.subTest(column=column):
                fresh = prepared((1,))
                fresh[0][column] = value
                pipeline = Mock()
                with self.assertRaises(ValueError):
                    Tester(pipeline, LengthDistance()).run(
                        prepared(), mode="search-epsilon", output_distance_threshold=100,
                        generate_mutations=lambda epsilon, iteration: fresh,
                    )
                pipeline.assert_not_called()


if __name__ == "__main__":
    unittest.main()
