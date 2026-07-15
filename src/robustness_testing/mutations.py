import random

from textattack.transformations import Transformation
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

    def mutate(self, source: Input[str], epsilon: float) -> Input[str]:
        """Return a mutated string in which at most ``epsilon`` words have been changed from ``source``."""
        num_swaps = int(epsilon)
        assert float(num_swaps) - epsilon == 0.0, f"epsilon must be an integer, got {epsilon}"
        if num_swaps <= 0:
            return source

        words = source.value.split()
        indices = random.sample(range(len(words)), min(num_swaps, len(words)))
        print(f"Mutating words at indices: {indices}")
        current_sentence = source.value
        # We need a loop because TextAttack transformations only change one index at a time
        for i in indices:
            at = AttackedText(current_sentence)
            print(f"Word to mutate: {at.words[i]} (index {i})")
            transformed_texts = self.transformation(at, indices_to_modify=[i])
            print("Obtained mutations (we take the first, if any):", [t.text for t in transformed_texts])
            current_sentence = transformed_texts[0].text if transformed_texts else current_sentence

        return type(source)(current_sentence)
