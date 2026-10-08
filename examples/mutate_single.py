from dataclasses import dataclass

from textattack.transformations import WordSwapHowNet, WordSwapQWERTY

from robustness_testing import Distance, Input, Mutation, Output, SUT
from robustness_testing.distances import WordEditDistance
from robustness_testing.mutations import RandomWordsMutation

from flair.embeddings import FlairEmbeddings
from flair.models import SequenceTagger
from textattack.shared.utils import strings as textattack_strings


def configure_textattack_pos_tagger() -> None:
    tagger = SequenceTagger.load("upos-fast")

    for embedding in tagger.embeddings.embeddings:
        if isinstance(embedding, FlairEmbeddings):
            embedding.with_whitespace = False

    # AttackedText.pos_of_word_index() uses this global cached tagger.
    textattack_strings._flair_pos_tagger = tagger
configure_textattack_pos_tagger()


@dataclass(frozen=True)
class TextInput(Input[str]):
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


distance = WordEditDistance()
mutation = RandomWordsMutation(transformation=WordSwapHowNet(), distance=distance)

tinput = TextInput("The quick brown fox jumps over the lazy dog.")

mutated = mutation(tinput, epsilon=3)

print (f"Original text: {tinput.value}")
print (f"Mutated text: {mutated.value}")
