import csv
import importlib.util
import io
import json
import signal
import time
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
    def test_search_continues_after_empty_generated_rounds(self):
        for later_mutation in (False, True):
            with self.subTest(later_mutation=later_mutation):
                pipeline = Mock(side_effect=lambda text, row: text)
                result = Tester(pipeline, LengthDistance()).run(
                    prepared(), mode="search-epsilon", output_distance_threshold=100, max_iterations=2,
                    generate_mutations=lambda epsilon, iteration: (
                        prepared((2,)) if later_mutation and iteration == 2 else []
                    ),
                )
                self.assertEqual(result["Stop_Reason"], "iteration_limit")
                self.assertEqual([row["Mutated_Rows"] for row in result["Iterations"]], [0, 1 if later_mutation else 0])
                self.assertEqual(result["Max_Tested_Passing_Epsilon"], 2 if later_mutation else None)
                self.assertEqual(pipeline.call_count, 2 if later_mutation else 1)

    def test_max_delta_reports_all_estimates_without_repeating_pipeline_calls(self):
        template = prepared((1,))[0]
        inputs = [{**template, "input": f"mutation-{index}", "test_output": "x" * distance}
                  for index, distance in enumerate([2] * 19 + [9])]
        for check in ("individual", "average", "percentile"):
            for percentage, expected in ((0, 9), (5, 2), (50, 2), (100, 0)):
                with self.subTest(check=check, percentage=percentage):
                    pipeline = Mock(side_effect=lambda text, row: "" if text == "a" else row["test_output"])
                    result = Tester(pipeline, LengthDistance()).run(
                        inputs, threshold_check=check, max_exceedance_percentage=percentage,
                    )
                    result = json.loads(json.dumps(result, allow_nan=False))
                    self.assertEqual(result["Mode"], "max-delta")
                    self.assertEqual(result["Stop_Reason"], "completed")
                    self.assertEqual(len(result["Iterations"]), 1)
                    observation = result["Iterations"][0]
                    self.assertEqual(observation["Max_Delta"], 9)
                    self.assertAlmostEqual(observation["Average_Delta"], 2.35)
                    self.assertEqual(observation["Percentile_Delta"], expected)
                    self.assertEqual(observation["Max_Exceedance_Percentage"], percentage)
                    self.assertEqual(observation["Checked_Delta"],
                                     observation[{"individual": "Max_Delta", "average": "Average_Delta",
                                                  "percentile": "Percentile_Delta"}[check]])
                    self.assertIsNone(observation["Threshold_Holds"])
                    self.assertEqual(pipeline.call_count, 21)

    def test_wall_time_limit_aborts_pipeline_and_preserves_completed_iterations(self):
        for clear_cache in (False, True):
            with self.subTest(clear_cache=clear_cache):
                calls, saved = [], []

                def pipeline(rows):
                    calls.append([row["input"] for row in rows])
                    if len(calls) == 2:
                        try:
                            time.sleep(2)
                        except Exception:
                            self.fail("Cancellation must bypass pipeline Exception handlers")
                    return [row["input"] for row in rows]

                handler = signal.getsignal(signal.SIGALRM)
                started = time.monotonic()
                result = Tester(lambda text, row: text, LengthDistance(), batch_pipeline=pipeline).run(
                    prepared(), mode="search-epsilon", output_distance_threshold=100,
                    clear_cache_each_iteration=clear_cache, wall_time_limit_hours=0.2 / 3600,
                    on_iteration=lambda iteration, epsilon, rows, observation: saved.append(iteration),
                )
                self.assertLess(time.monotonic() - started, 1.5)
                self.assertEqual(result["Stop_Reason"], "wall-time limit exceeded")
                self.assertEqual(result["Aborted_Iteration"], 2)
                self.assertEqual(result["Aborted_Epsilon"], 2)
                self.assertEqual(result["Max_Tested_Passing_Epsilon"], 1)
                self.assertEqual([row["Epsilon"] for row in result["Iterations"]], [1])
                self.assertEqual([row["input"] for row in result["Results"]], ["a", "ax"])
                self.assertEqual(saved, [1])
                self.assertEqual(signal.getsignal(signal.SIGALRM), handler)
                self.assertEqual(signal.getitimer(signal.ITIMER_REAL), (0, 0))

    def test_wall_time_limit_during_generation_and_before_any_result(self):
        def slow_generation(epsilon, iteration):
            time.sleep(2)
            return prepared((1,))

        pipeline = Mock()
        result = Tester(pipeline, LengthDistance()).run(
            prepared(), mode="search-epsilon", output_distance_threshold=100,
            generate_mutations=slow_generation, wall_time_limit_hours=0.05 / 3600,
        )
        self.assertEqual(result["Stop_Reason"], "wall-time limit exceeded")
        self.assertEqual(result["Aborted_Iteration"], 1)
        self.assertEqual(result["Results"], [])
        self.assertEqual(result["Iterations"], [])
        self.assertIsNone(result["Max_Tested_Passing_Epsilon"])
        pipeline.assert_not_called()

    def test_wall_time_limit_does_not_interrupt_persistence_and_restores_timer_on_errors(self):
        saved = []

        def persist(iteration, epsilon, rows, observation):
            self.assertEqual(signal.getitimer(signal.ITIMER_REAL), (0, 0))
            time.sleep(0.05)
            saved.append(iteration)

        result = Tester(lambda text, data: text, LengthDistance()).run(
            prepared(), mode="search-epsilon", output_distance_threshold=100,
            wall_time_limit_hours=0.02 / 3600, on_iteration=persist,
        )
        self.assertEqual(saved, [1])
        self.assertEqual(result["Stop_Reason"], "wall-time limit exceeded")
        self.assertEqual(len(result["Iterations"]), 1)
        self.assertIsNone(result["Aborted_Iteration"])
        self.assertEqual(result["Max_Tested_Passing_Epsilon"], 1)
        handler = signal.getsignal(signal.SIGALRM)
        with self.assertRaisesRegex(RuntimeError, "pipeline failure"):
            Tester(Mock(side_effect=RuntimeError("pipeline failure")), LengthDistance()).run(
                prepared(), mode="search-epsilon", output_distance_threshold=100, wall_time_limit_hours=1,
            )
        self.assertEqual(signal.getsignal(signal.SIGALRM), handler)
        self.assertEqual(signal.getitimer(signal.ITIMER_REAL), (0, 0))

    def test_percentile_search_allows_outliers_and_stops_on_exceeded_percentage(self):
        template = prepared((1,))[0]
        inputs = []
        for epsilon, count in ((1, 20), (2, 1), (4, 1)):
            for index in range(count):
                inputs.append({**template, "input": f"changed-{epsilon}-{index}",
                               "Input_Distance_To_Original_Input": epsilon,
                               "test_output": "x" * (9 if epsilon > 1 or index == 19 else 2)})
        for percentage, epsilons, passing in [(5, [1, 2], 1), (10, [1, 2, 4], 2), (0, [1], None)]:
            with self.subTest(percentage=percentage):
                result = Tester(lambda text, data: "" if text == "a" else data["test_output"], LengthDistance()).run(
                    inputs, mode="search-epsilon", output_distance_threshold=2,
                    threshold_check="percentile", max_exceedance_percentage=percentage,
                )
                self.assertEqual([row["Epsilon"] for row in result["Iterations"]], epsilons)
                self.assertEqual(result["Max_Tested_Passing_Epsilon"], passing)
                self.assertEqual(result["Stop_Reason"], "threshold_exceeded")
                self.assertEqual(result["Max_Exceedance_Percentage"], percentage)
                first = result["Iterations"][0]
                self.assertEqual(first["Outputs_Above_Threshold"], 1)
                self.assertEqual(first["Outputs_Above_Threshold_Percentage"], 5)
                self.assertEqual(first["Percentile_Delta"], 9 if percentage == 0 else 2)
                self.assertEqual(first["Max_Delta"], 9)

    def test_percentile_summary_nan_infinity_and_boundaries(self):
        def summarize(values, **kwargs):
            results = [{"input": "a", "Original_Input": "a", "Output_Distance_To_Original_Output": 999}]
            results.extend({"input": "changed", "Original_Input": "a", "Output_Distance_To_Original_Output": value} for value in values)
            return Tester.summarize(results, 1, "percentile", 2, **kwargs)

        summary = summarize([2] * 19 + [float("inf")])
        self.assertTrue(summary["Threshold_Holds"])
        self.assertEqual(summary["Percentile_Delta"], 2)
        self.assertFalse(summarize([2] * 18 + [float("inf")] * 2)["Threshold_Holds"])
        self.assertTrue(summarize([float("inf")], max_exceedance_percentage=100)["Threshold_Holds"])
        self.assertTrue(summarize([2] * 18 + [9, None])["Threshold_Holds"])
        self.assertIsNone(summarize([2] * 18 + [9, None], stop_on_nan=True)["Threshold_Holds"])
        self.assertIsNone(summarize([None, float("nan")])["Threshold_Holds"])
        self.assertIsNone(summarize([])["Threshold_Holds"])
        self.assertEqual(summarize([2] * 20, max_exceedance_percentage=0)["Percentile_Delta"], 2)
        self.assertFalse(summarize([2] * 19 + [9], max_exceedance_percentage=4.99)["Threshold_Holds"])
        for check in ("individual", "average", "percentile"):
            for values, expected in (([2] * 19 + [float("inf")], 2), ([2] * 18 + [9, None], 2),
                                     ([None, float("nan")], None), ([], None)):
                rows = [{"input": "a", "Original_Input": "a", "Output_Distance_To_Original_Output": 999}]
                rows.extend({"input": "changed", "Original_Input": "a", "Output_Distance_To_Original_Output": value}
                            for value in values)
                with self.subTest(check=check, values=values):
                    self.assertEqual(Tester.summarize(rows, 1, check, None)["Percentile_Delta"], expected)
                    if None in values:
                        self.assertIsNone(Tester.summarize(rows, 1, check, None, stop_on_nan=True)["Percentile_Delta"])

    def test_reference_runner_loads_inputs_and_prints_results(self):
        path = Path(__file__).parents[1] / "examples" / "robustness.py"
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
            with patch("sys.argv", [str(path), "--eval-files", str(source), "--epsilon", "2", "--stop-on-nan",
                                    "--threshold-check", "percentile", "--max-exceedance-percentage", "7.5"]), \
                 patch("sys.stdout", new_callable=io.StringIO) as output:
                example.main()
            result = json.loads(output.getvalue())
            self.assertEqual(result["Results"][0]["Generated_Output"], "A")
            self.assertEqual(len(result["Results"]), 3)
            self.assertEqual(result["Iterations"][0]["Max_Delta"], 1)
            self.assertEqual(result["Iterations"][0]["Average_Delta"], 1)
            self.assertEqual(result["Iterations"][0]["Percentile_Delta"], 1)
            self.assertTrue(result["Stop_On_NaN"])
            self.assertEqual(result["Max_Exceedance_Percentage"], 7.5)
            self.assertEqual(result["Iterations"][0]["Threshold_Check"], "percentile")
            self.assertIsNone(result["Wall_Time_Limit_Hours"])
            with patch("sys.argv", [str(path), "--eval-files", str(source), "--mode", "search-epsilon",
                                    "--delta", "100", "--wall-time-limit-hours", "1"]), \
                 patch("sys.stdout", new_callable=io.StringIO) as output:
                example.main()
            self.assertEqual(json.loads(output.getvalue())["Wall_Time_Limit_Hours"], 1)

    def test_nan_policy_controls_search_and_summary_statistics(self):
        class NanDistance(LengthDistance):
            def __call__(self, left, right):
                return float("nan") if left.value == "ax" else super().__call__(left, right)

        for threshold_check in ("individual", "average"):
            for stop_on_nan in (False, True):
                with self.subTest(threshold_check=threshold_check, stop_on_nan=stop_on_nan):
                    result = Tester(lambda text, data: text, NanDistance()).run(
                        prepared(), mode="search-epsilon", output_distance_threshold=2,
                        threshold_check=threshold_check, stop_on_nan=stop_on_nan,
                    )
                    self.assertEqual(result["Stop_On_NaN"], stop_on_nan)
                    self.assertEqual(result["Stop_Reason"],
                                     "incomplete_distances_or_benchmarks" if stop_on_nan else "threshold_exceeded")
                    self.assertEqual([row["Epsilon"] for row in result["Iterations"]],
                                     [1] if stop_on_nan else [1, 2, 4])
                    if not stop_on_nan:
                        self.assertEqual(result["Iterations"][1]["Average_Delta"], 2)
                        self.assertEqual(result["Iterations"][-1]["Average_Delta"], 3)
                        self.assertEqual(result["Iterations"][-1]["Unavailable_Output_Distances"], 1)
                        self.assertEqual(result["Max_Tested_Passing_Epsilon"], 2)

        result = Tester(lambda text, data: text, NanDistance()).run(
            prepared((1,)), mode="search-epsilon", output_distance_threshold=2,
            generate_mutations=lambda epsilon, iteration: prepared((1,)), max_iterations=3,
        )
        self.assertEqual(len(result["Iterations"]), 3)
        self.assertEqual(result["Stop_Reason"], "iteration_limit")
        self.assertIsNone(result["Max_Tested_Passing_Epsilon"])
        self.assertTrue(all(row["Checked_Delta"] is None for row in result["Iterations"]))

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

    def test_invalid_outputs_stop_search_and_missing_benchmarks_are_reported(self):
        class RejectInvalid(LengthDistance):
            def distance(self, left, right):
                if "invalid" in (left.value, right.value):
                    raise ValueError("invalid output")
                return super().distance(left, right)

        with self.assertWarns(RuntimeWarning):
            result = Tester(lambda text, data: "a" if text == "a" else "invalid", RejectInvalid()).run(
                prepared(), mode="search-epsilon", output_distance_threshold=100, stop_on_nan=True,
            )
        self.assertEqual(result["Stop_Reason"], "incomplete_distances_or_benchmarks")
        self.assertIsNone(result["Iterations"][0]["Threshold_Holds"])
        self.assertEqual(result["Iterations"][0]["Unavailable_Output_Distances"], 1)
        result = Tester(lambda text, data: text, LengthDistance()).run(
            [*prepared(), *prepared((2,), original="b")], epsilon=1,
        )
        self.assertEqual(result["Iterations"][0]["Benchmarks_Without_Mutations"], ["b"])

    def test_search_passes_with_missing_benchmarks_and_continues(self):
        for threshold_check in ("individual", "average"):
            for fresh in (False, True):
                with self.subTest(threshold_check=threshold_check, fresh=fresh):
                    inputs = [*prepared((1,), original="a"), *prepared((2,), original="b")]
                    def generate(epsilon, iteration):
                        second = prepared((2,), original="b")[0]
                        if epsilon < 2:
                            second.update(input="b", Input_Distance_To_Original_Input=0)
                        return [*prepared((1,)), second]
                    result = Tester(lambda text, data: text, LengthDistance()).run(
                        inputs, mode="search-epsilon", output_distance_threshold=10,
                        threshold_check=threshold_check, max_iterations=2,
                        generate_mutations=generate if fresh else None,
                    )
                    self.assertEqual([row["Epsilon"] for row in result["Iterations"]], [1, 2])
                    self.assertEqual(result["Iterations"][0]["Benchmarks_Without_Mutations"], ["b"])
                    self.assertEqual(result["Iterations"][1]["Benchmarks_Without_Mutations"], [])
                    self.assertEqual(result["Max_Tested_Passing_Epsilon"], 2)
                    self.assertEqual(result["Stop_Reason"], "iteration_limit")
                    partial = Tester(lambda text, data: text, LengthDistance()).run(
                        inputs, mode="search-epsilon", output_distance_threshold=10,
                        threshold_check=threshold_check, max_iterations=1,
                        generate_mutations=generate if fresh else None,
                    )
                    self.assertEqual(partial["Iterations"][0]["Benchmarks_Without_Mutations"], ["b"])
                    self.assertTrue(partial["Iterations"][0]["Threshold_Holds"])
                    self.assertEqual(partial["Max_Tested_Passing_Epsilon"], 1)

    def test_search_with_no_changed_samples_does_not_claim_a_passing_bound(self):
        unchanged = prepared((1,))[0]
        unchanged.update(input="a", Input_Distance_To_Original_Input=0)
        for later_mutation in (False, True):
            with self.subTest(later_mutation=later_mutation):
                result = Tester(lambda text, data: text, LengthDistance()).run(
                    [unchanged], mode="search-epsilon", output_distance_threshold=10, max_iterations=2,
                    generate_mutations=lambda epsilon, iteration: (
                        prepared((2,)) if later_mutation and iteration == 2 else [unchanged]
                    ),
                )
                self.assertEqual([row["Epsilon"] for row in result["Iterations"]], [1, 2])
                self.assertIsNone(result["Iterations"][0]["Checked_Delta"])
                self.assertIsNone(result["Iterations"][0]["Threshold_Holds"])
                self.assertEqual(result["Stop_Reason"], "iteration_limit")
                self.assertEqual(result["Max_Tested_Passing_Epsilon"], 2 if later_mutation else None)

    def test_missing_benchmarks_do_not_hide_threshold_or_distance_failures(self):
        inputs = [*prepared((1,)), *prepared((2,), original="b")]
        for threshold_check in ("individual", "average"):
            with self.subTest(threshold_check=threshold_check):
                result = Tester(lambda text, data: text, LengthDistance()).run(
                    inputs, mode="search-epsilon", output_distance_threshold=0,
                    threshold_check=threshold_check,
                )
                self.assertEqual(result["Stop_Reason"], "threshold_exceeded")
                self.assertEqual(len(result["Iterations"]), 1)
                self.assertIsNone(result["Max_Tested_Passing_Epsilon"])
        class RejectInvalid(LengthDistance):
            def distance(self, left, right):
                if "invalid" in (left.value, right.value):
                    raise ValueError("invalid output")
                return super().distance(left, right)
        with self.assertWarns(RuntimeWarning):
            result = Tester(lambda text, data: "invalid" if text == "ax" else text, RejectInvalid()).run(
                inputs, mode="search-epsilon", output_distance_threshold=10, stop_on_nan=True,
            )
        self.assertEqual(result["Stop_Reason"], "incomplete_distances_or_benchmarks")
        self.assertEqual(len(result["Iterations"]), 1)

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
        cases += [{"max_exceedance_percentage": value} for value in (-1, 101, float("nan"), float("inf"))]
        cases += [{"mode": "search-epsilon", "output_distance_threshold": 100, "wall_time_limit_hours": value}
                  for value in (0, -1, float("nan"), float("inf"), 1e308)]
        cases.append({"wall_time_limit_hours": 1})
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
