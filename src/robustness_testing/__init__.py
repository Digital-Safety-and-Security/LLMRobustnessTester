"""Contracts, distances, and mutations for robustness testing."""

from importlib import import_module
from typing import TYPE_CHECKING, Any

from robustness_testing.core import Distance, Input, Mutation, Output, SUT
from robustness_testing.mutator import Mutator, TextInput
from robustness_testing.tester import Tester, TextOutput

if TYPE_CHECKING:
    from robustness_testing.distances import EditDistance, WordEditDistance
    from robustness_testing.mutations import (
        RandomWordsMutation,
        random_chinese_homophone_character_swap_mutation,
        random_chinese_morphonym_character_swap_mutation,
        random_chinese_word_swap_how_net_mutation,
        random_chinese_word_swap_masked_lm_mutation,
        random_word_swap_change_location_mutation,
        random_word_swap_change_name_mutation,
        random_word_swap_change_number_mutation,
        random_word_swap_contract_mutation,
        random_word_swap_embedding_mutation,
        random_word_swap_extend_mutation,
        random_word_swap_gradient_based_mutation,
        random_word_swap_homoglyph_swap_mutation,
        random_word_swap_how_net_mutation,
        random_word_swap_inflections_mutation,
        random_word_swap_masked_lm_mutation,
        random_word_swap_neighboring_character_swap_mutation,
        random_word_swap_qwerty_mutation,
        random_word_swap_random_character_deletion_mutation,
        random_word_swap_random_character_insertion_mutation,
        random_word_swap_random_character_substitution_mutation,
        random_word_swap_word_net_mutation,
    )

__all__ = [
    "Distance",
    "Input",
    "Mutation",
    "Mutator",
    "TextInput",
    "Tester",
    "TextOutput",
    "Output",
    "SUT",
    "EditDistance",
    "WordEditDistance",
    "RandomWordsMutation",
    "random_word_swap_how_net_mutation",
    "random_word_swap_qwerty_mutation",
    "random_word_swap_embedding_mutation",
    "random_word_swap_homoglyph_swap_mutation",
    "random_word_swap_inflections_mutation",
    "random_word_swap_neighboring_character_swap_mutation",
    "random_word_swap_random_character_deletion_mutation",
    "random_word_swap_random_character_insertion_mutation",
    "random_word_swap_random_character_substitution_mutation",
    "random_word_swap_word_net_mutation",
    "random_word_swap_masked_lm_mutation",
    "random_word_swap_contract_mutation",
    "random_word_swap_extend_mutation",
    "random_word_swap_change_number_mutation",
    "random_word_swap_change_location_mutation",
    "random_word_swap_change_name_mutation",
    "random_chinese_homophone_character_swap_mutation",
    "random_chinese_morphonym_character_swap_mutation",
    "random_chinese_word_swap_how_net_mutation",
    "random_chinese_word_swap_masked_lm_mutation",
    "random_word_swap_gradient_based_mutation",
]


def __getattr__(name: str) -> Any:
    # Keep core imports independent of TextAttack's model and resource setup.
    if name not in __all__:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module = "distances" if name in ("EditDistance", "WordEditDistance") else "mutations"
    value = getattr(import_module(f".{module}", __name__), name)
    globals()[name] = value
    return value
