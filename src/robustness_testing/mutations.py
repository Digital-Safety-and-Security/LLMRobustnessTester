from __future__ import annotations

import random
from functools import cached_property
from importlib import import_module
from typing import TYPE_CHECKING, Callable, cast

if TYPE_CHECKING:
    from textattack.shared import AttackedText
    from textattack.transformations import Transformation

from robustness_testing import Mutation, Distance, Input
from robustness_testing.distances import WordEditDistance


class RandomWordsMutation(Mutation):
    def __init__(
            self,
            transformation: Transformation | type[Transformation] | str,
            distance: Distance = WordEditDistance()
        ) -> None:
        super().__init__(distance)
        self._transformation = transformation

    @cached_property
    def transformation(self) -> Transformation:
        # Load model weights and dictionaries only when this swap is used.
        if isinstance(self._transformation, str):
            self._transformation = getattr(
                import_module("textattack.transformations"), self._transformation
            )
        if isinstance(self._transformation, type):
            return self._transformation()
        return self._transformation

    def mutate_multiple(self, source: Input[str], epsilon: float) -> list[Input[str]]:
        """Return a mutated string in which at most ``epsilon`` words have been changed from ``source``."""
        attacked_text = import_module("textattack.shared").AttackedText
        num_swaps = int(epsilon)
        assert float(num_swaps) - epsilon == 0.0, f"epsilon must be an integer, got {epsilon}"
        if num_swaps <= 0:
            return [source]

        words = source.value.split()
        indices = random.sample(range(len(words)), min(num_swaps, len(words)))
        print(f"Mutating words at indices: {indices}")
        current_sentences = [source.value]
        # We need a loop because TextAttack transformations only change one index at a time
        for i in indices:
            print(f"Word to mutate: {words[i]} (index {i})")
            new_sentences = []
            for current_sentence in current_sentences:
                at = attacked_text(current_sentence)
                transformed_texts = self.transformation(at, indices_to_modify=[i])
                if len(transformed_texts) > 0:
                    new_sentences.extend([t.text for t in transformed_texts])
                else:
                    new_sentences.append(current_sentence)
            print("Obtained mutations:", new_sentences)
            current_sentences = new_sentences

        make_input = cast(Callable[[str], Input[str]], type(source))
        return [make_input(sentence) for sentence in current_sentences]


random_word_swap_how_net_mutation = RandomWordsMutation(transformation="WordSwapHowNet", distance=WordEditDistance())
random_word_swap_qwerty_mutation = RandomWordsMutation(transformation="WordSwapQWERTY", distance=WordEditDistance())
random_word_swap_embedding_mutation = RandomWordsMutation(transformation="WordSwapEmbedding", distance=WordEditDistance())
random_word_swap_homoglyph_swap_mutation = RandomWordsMutation(transformation="WordSwapHomoglyphSwap", distance=WordEditDistance())
random_word_swap_inflections_mutation = RandomWordsMutation(transformation="WordSwapInflections", distance=WordEditDistance())
random_word_swap_neighboring_character_swap_mutation = RandomWordsMutation(transformation="WordSwapNeighboringCharacterSwap", distance=WordEditDistance())
random_word_swap_random_character_deletion_mutation = RandomWordsMutation(transformation="WordSwapRandomCharacterDeletion", distance=WordEditDistance())
random_word_swap_random_character_insertion_mutation = RandomWordsMutation(transformation="WordSwapRandomCharacterInsertion", distance=WordEditDistance())
random_word_swap_random_character_substitution_mutation = RandomWordsMutation(transformation="WordSwapRandomCharacterSubstitution", distance=WordEditDistance())
random_word_swap_word_net_mutation = RandomWordsMutation(transformation="WordSwapWordNet", distance=WordEditDistance())
random_word_swap_masked_lm_mutation = RandomWordsMutation(transformation="WordSwapMaskedLM", distance=WordEditDistance())
random_word_swap_contract_mutation = RandomWordsMutation(transformation="WordSwapContract", distance=WordEditDistance())
random_word_swap_extend_mutation = RandomWordsMutation(transformation="WordSwapExtend", distance=WordEditDistance())
random_word_swap_change_number_mutation = RandomWordsMutation(transformation="WordSwapChangeNumber", distance=WordEditDistance())
random_word_swap_change_location_mutation = RandomWordsMutation(transformation="WordSwapChangeLocation", distance=WordEditDistance())
random_word_swap_change_name_mutation = RandomWordsMutation(transformation="WordSwapChangeName", distance=WordEditDistance())
random_chinese_homophone_character_swap_mutation = RandomWordsMutation(transformation="ChineseHomophoneCharacterSwap", distance=WordEditDistance())
random_chinese_morphonym_character_swap_mutation = RandomWordsMutation(transformation="ChineseMorphonymCharacterSwap", distance=WordEditDistance())
random_chinese_word_swap_how_net_mutation = RandomWordsMutation(transformation="ChineseWordSwapHowNet", distance=WordEditDistance())
random_chinese_word_swap_masked_lm_mutation = RandomWordsMutation(transformation="ChineseWordSwapMaskedLM", distance=WordEditDistance())


def random_word_swap_gradient_based_mutation(model_wrapper, top_n: int = 1) -> RandomWordsMutation:
    """Build a gradient swap using the caller's TextAttack model wrapper."""
    transformation = getattr(import_module("textattack.transformations"), "WordSwapGradientBased")
    return RandomWordsMutation(transformation=transformation(model_wrapper, top_n=top_n), distance=WordEditDistance())
