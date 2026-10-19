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
- `Tester`: loads prepared inputs, invokes a black-box pipeline, compares its
  outputs to references and original-input outputs, and searches epsilon bounds.

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

The batch example uses the same class and standard-library CSV handling:

```bash
python -m pip install -e .
python examples/mutate_batch.py inputs.csv mutated.csv --max-distance 1 -n 5 --transformation WordSwapQWERTY
```

Use `--input-column text` for a differently named source column. TextAttack
resources are loaded only when its mutation is invoked. `Mutator` itself uses
only the standard library and does not import TextAttack or pandas.

`examples/mutate_single.py` demonstrates mutating one `TextInput` directly. It
creates a `WordEditDistance` and `RandomWordsMutation`, applies a TextAttack
transformation within epsilon 3, and prints the original and mutated text. This
small example uses TextAttack/Flair directly; unlike `mutate_batch.py`, it does
not read or write CSV files.

## Testing a black-box pipeline

```python
from robustness_testing import Tester, WordEditDistance

def pipeline(text, row_data):
    # Pass only the context your model needs; references are for evaluation.
    return text.upper()

tester = Tester(pipeline, WordEditDistance())
inputs = tester.load_inputs(["mutated.csv"])
result = tester.run(inputs, epsilon=2)
search = tester.run(
    inputs, mode="search-epsilon", output_distance_threshold=3,
    min_epsilon=1, epsilon_step="double", max_iterations=5,
    threshold_check="individual", clear_cache_each_iteration=True,
)
```

The callable receives an input text and its dataset row, including any context
columns. Default columns are `input`, `Original_Input`, and `expected_output`;
all three names are configurable. Prepared datasets use `Mutator`'s mutation
metadata. Generated and expected values are wrapped in `TextOutput` by default;
inject `output_factory` and optionally `reference_factory` for structured values
or project-specific `Output` subclasses. Inject any library or project-specific
`Distance`. `batch_pipeline(rows)` is optional and useful for model reuse; it
must return one generated value per row in the same order.

`run()` returns final `Results`, an `Iterations` history, the maximum tested
passing epsilon, and a stop reason. Results preserve source columns and add
`Expected_Output`, `Generated_Output`, the output-distance measure, and distances
to both expected and original outputs. Each original runs once per call to
`run()` by default, with independent caches between runs. In epsilon search,
`clear_cache_each_iteration=True` reruns all inputs selected at that epsilon,
including each original baseline, which is useful for nondeterministic pipelines.
Baselines are excluded from the
maximum and global mutation-row average; thresholds are inclusive. Invalid
output distances remain unavailable and cannot establish a passing bound.
Search continues when originals have no changed samples, even if there are no
changed samples at all yet. Distances are computed over changed samples only;
`Benchmarks_Without_Mutations` records missing coverage but does not prevent an
epsilon from passing. Passing requires at least one changed sample and the
checked distance to stay within the threshold. NaN and None output distances are
ignored in statistics by default, but counted in `Unavailable_Output_Distances`.
If none remain, statistics stay unset and search continues without declaring a
passing bound. Set `run(stop_on_nan=True)` (or `--stop-on-nan` in the example)
to restore stopping on unavailable distances. Infinity still exceeds any finite
threshold under individual/average checks. Summaries record the policy in `Stop_On_NaN`.
Use `threshold_check="percentile", max_exceedance_percentage=5` to continue
epsilon search while at most 5% of individual mutated result rows have distance
to the original output strictly above the configured output threshold. Search
stops at the first iteration exceeding this percentage. Exactly 5% passes, and
distances equal to the output threshold are within it. The percentage accepts
values from 0 to 100 and defaults to 5. The example runner exposes these as
`--threshold-check percentile --max-exceedance-percentage 5`.
Repeated mutation rows count separately; baselines are excluded. Unavailable
distances do not count as exceedances but stay in the mutated-row denominator,
matching the reporting criterion. Thus with missing distances this is not a
guarantee that 95% have known within-threshold outputs. No usable distances
still means no passing bound, and `stop_on_nan=True` retains its stopping rule.
Infinite distances count as exceedances and can be tolerated within the budget.
Percentile observations include `Percentile_Delta` (also `Checked_Delta`),
`Max_Exceedance_Percentage`, `Outputs_Above_Threshold`, and
`Outputs_Above_Threshold_Percentage`. The percentile delta is the smallest
non-negative observed-distance cutoff meeting the percentage budget, without
interpolation; it is compared to the configured output threshold.
Bounds are empirical over the sampled inputs, not universal guarantees.

For fresh sampling at every search step, supply
`generate_mutations(epsilon, iteration)`, returning prepared rows using a
`Mutator` or another generator. Previous observations are retained. Without a
generator, search is limited by each original's recorded generation coverage.
An optional `on_iteration(iteration, epsilon, results, observation)` callback
can save result CSVs and add file paths to the observation; no storage format is
imposed on model-specific results.

The example script demonstrates the uppercase callable without LTL or model
dependencies (its inputs need an `expected_output` column):

```bash
python examples/robustness.py --eval-files mutated.csv --epsilon 2
python examples/robustness.py --eval-files eps1.csv eps2.csv eps4.csv --mode search-epsilon --delta 3
```
