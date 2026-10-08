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
- `Mutator`: generates annotated mutation datasets from text inputs while
  preserving reference outputs and arbitrary source columns.

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

## Preparing mutation datasets

`robustness_testing.mutator.Mutator` works with an injected `Mutation` instance,
including project-specific mutations and distances. It does not run a model or
interpret expected outputs. The default text wrapper is `TextInput`; pass an
`input_factory` callable to use your own `Input` subclass.

```python
from robustness_testing import Mutator, RandomWordsMutation, WordEditDistance

mutation = RandomWordsMutation("WordSwapQWERTY", distance=WordEditDistance())
mutator = Mutator(mutation, metadata={"Transformation": "WordSwapQWERTY"})
rows = mutator.mutate(
    [{"input": "The light must stay on", "expected_output": "some reference"}],
    max_distance=1, num_mutations=5,
)
mutator.mutate_csv("inputs.csv", "mutated.csv", max_distance=1, num_mutations=5)
```

The CSV method expects an `input` column by default and preserves every other
column, including any reference outputs. It adds `Original_Input`, per-row
`Input_Distance_To_Original_Input`, the maximum input distance, sample count and
number, and mutation/distance labels. Column names can be adapted with
`input_column` and `original_input_column`. Labels default to fully qualified
class names; optional `metadata` labels record CLI aliases and transformations.
Repeated and unchanged samples are retained; they are not guaranteed unique.

The reference script uses the same class and standard-library CSV handling:

```bash
python -m pip install -e .
python mutate.py inputs.csv mutated.csv --max-distance 1 -n 5 --transformation WordSwapQWERTY
```

Use `--input-column text` for a differently named source column. TextAttack
resources are loaded only when its mutation is invoked. `Mutator` itself uses
only the standard library and does not import TextAttack or pandas.
