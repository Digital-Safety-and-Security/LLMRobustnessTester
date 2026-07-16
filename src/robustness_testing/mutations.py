import random

from textattack.transformations import Transformation, WordSwapHowNet, WordSwapQWERTY
from textattack.shared import AttackedText

from robustness_testing import Mutation, Distance, Input
from robustness_testing.distances import WordEditDistance


class RandomWordsMutation(Mutation):
    def __init__(
            self,
            transformation: Transformation,
            distance: Distance = WordEditDistance()
        ) -> None:
        super().__init__(distance)
        self.transformation = transformation

    def mutate_multiple(self, source: Input[str], epsilon: float) -> list[Input[str]]:
        """Return a mutated string in which at most ``epsilon`` words have been changed from ``source``."""
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
                at = AttackedText(current_sentence)
                transformed_texts = self.transformation(at, indices_to_modify=[i])
                if len(transformed_texts) > 0:
                    new_sentences.extend([t.text for t in transformed_texts])
                else:
                    new_sentences.append(current_sentence)
            print("Obtained mutations:", new_sentences)
            current_sentences = new_sentences

        return [type(source)(sentence) for sentence in current_sentences]


random_word_swap_how_net_mutation = RandomWordsMutation(transformation=WordSwapHowNet(), distance=WordEditDistance())
random_word_swap_qwerty_mutation = RandomWordsMutation(transformation=WordSwapQWERTY(), distance=WordEditDistance())
