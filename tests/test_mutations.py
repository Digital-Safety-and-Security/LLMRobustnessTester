import importlib.util
from pathlib import Path
from types import ModuleType
import unittest
from unittest.mock import Mock, patch


class MutationPresetTests(unittest.TestCase):
    def setUp(self):
        path = Path(__file__).parents[1] / "src/robustness_testing/mutations.py"
        spec = importlib.util.spec_from_file_location("mutation_presets_under_test", path)
        self.module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(self.module)

    def test_importing_presets_does_not_import_textattack(self):
        self.assertIsInstance(self.module.random_word_swap_qwerty_mutation, self.module.RandomWordsMutation)
        self.module.import_module = Mock(side_effect=AssertionError("TextAttack imported eagerly"))
        self.assertEqual(self.module.random_word_swap_qwerty_mutation._transformation, "WordSwapQWERTY")

    def test_mutation_runs_without_annotation_only_types_at_runtime(self):
        from robustness_testing import TextInput

        class AttackedText:
            def __init__(self, text):
                self.text = text

        shared = ModuleType("textattack.shared")
        shared.AttackedText = AttackedText
        swap = Mock(return_value=[AttackedText("two")])
        mutation = self.module.RandomWordsMutation(swap)
        with patch.object(self.module, "import_module", return_value=shared):
            result = mutation(TextInput("one"), epsilon=1)
        self.assertEqual(result.value, "two")
        self.assertEqual(swap.call_args.kwargs, {"indices_to_modify": [0]})

    def test_transformation_class_is_initialized_once_and_instances_still_work(self):
        constructed = Mock()

        class Swap:
            def __init__(self):
                constructed()

        mutation = self.module.RandomWordsMutation(Swap)
        constructed.assert_not_called()
        transformations = ModuleType("textattack.transformations")
        transformations.Swap = Swap
        with patch.object(self.module, "import_module", return_value=transformations):
            self.assertIsInstance(mutation.transformation, Swap)
            self.assertIs(mutation.transformation, mutation.transformation)
        constructed.assert_called_once_with()
        instance = Swap()
        self.assertIs(self.module.RandomWordsMutation(instance).transformation, instance)

    def test_gradient_factory_passes_the_callers_model(self):
        model = object()
        transformations = ModuleType("textattack.transformations")
        transformations.WordSwapGradientBased = Mock()
        with patch.object(self.module, "import_module", return_value=transformations):
            mutation = self.module.random_word_swap_gradient_based_mutation(model, top_n=3)
        transformations.WordSwapGradientBased.assert_called_once_with(model, top_n=3)
        self.assertIs(mutation.transformation, transformations.WordSwapGradientBased.return_value)

    def test_package_exports_all_mutations_and_distances(self):
        import robustness_testing.distances as distances

        path = Path(__file__).parents[1] / "src/robustness_testing/__init__.py"
        spec = importlib.util.spec_from_file_location("robustness_testing", path)
        package = importlib.util.module_from_spec(spec)
        with patch.dict("sys.modules", {
            "robustness_testing": package,
            "robustness_testing.mutations": self.module,
            "robustness_testing.distances": distances,
        }):
            spec.loader.exec_module(package)
            expected = {"EditDistance", "WordEditDistance", "RandomWordsMutation"}
            expected.update(name for name in vars(self.module) if name.startswith("random_"))
            self.assertTrue(expected <= set(package.__all__))
            namespace = {}
            exec("from robustness_testing import *", namespace)
            for name in expected:
                module = distances if name.endswith("EditDistance") else self.module
                self.assertIs(namespace[name], getattr(module, name))
            with self.assertRaises(AttributeError):
                getattr(package, "unknown_mutation")


if __name__ == "__main__":
    unittest.main()
