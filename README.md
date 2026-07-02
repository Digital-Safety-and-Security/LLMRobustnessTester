# Robustness Testing Architecture

This starter project defines abstract Python contracts for robustness testing of
generative AI models and other systems under test.

The core abstractions are:

- `Input`: an abstract typed input object, such as natural language, an image,
  or a temporal formula.
- `Output`: an abstract typed output object.
- `Distance`: an abstract distance over two inputs or two outputs of the same
  concrete class.
- `Mutation`: an abstract perturbation of an input that must remain within an
  epsilon ball under a configured distance.
- `SUT`: an abstract system under test mapping an input object to an output
  object.

Concrete domains should subclass these contracts and implement only the
domain-specific pieces. For example, a natural-language robustness benchmark
could define `TextInput`, `TextOutput`, `EmbeddingDistance`, `ParaphraseMutation`,
and `ChatModelSUT`.

## Quick Example

```python
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
class TextOutput(Output[str]):
    text: str

    @property
    def value(self) -> str:
        return self.text


class LengthDistance(Distance[TextInput]):
    def distance(self, left: TextInput, right: TextInput) -> Real:
        return abs(len(left.value) - len(right.value))


class IdentityMutation(Mutation[TextInput]):
    def mutate(self, source: TextInput, epsilon: float) -> TextInput:
        return source


class EchoSUT(SUT[TextInput, TextOutput]):
    def run(self, input_object: TextInput) -> TextOutput:
        return TextOutput(input_object.value)


distance = LengthDistance()
mutation = IdentityMutation(distance)
sut = EchoSUT()

mutated = mutation(TextInput("hello"), epsilon=0.0)
output = sut(mutated)
```

Run the tests with:

```powershell
$env:PYTHONPATH = "src"
python -m unittest discover -s tests
```
