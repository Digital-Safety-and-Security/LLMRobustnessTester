"""Prepare mutated text inputs for any black-box pipeline."""

import argparse
from pathlib import Path

from robustness_testing import Mutator, RandomWordsMutation, WordEditDistance


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input_file", type=Path, help="CSV with input text and optional reference-output columns")
    parser.add_argument("output_file", type=Path)
    parser.add_argument("--input-column", default="input")
    parser.add_argument("--max-distance", "--epsilon", type=float, required=True)
    parser.add_argument("--num-mutations", "-n", type=int, default=1)
    parser.add_argument("--max-attempts", type=int, default=100,
                        help="Maximum mutation calls per original input (default: 100).")
    parser.add_argument("--transformation", default="WordSwapQWERTY")
    args = parser.parse_args()

    mutation = RandomWordsMutation(args.transformation, distance=WordEditDistance())
    mutator = Mutator(mutation, input_column=args.input_column, metadata={"Transformation": args.transformation})
    rows = mutator.mutate_csv(args.input_file, args.output_file, args.max_distance, args.num_mutations, args.max_attempts)
    print(f"Saved {len(rows)} mutated inputs to {args.output_file}")


if __name__ == "__main__":
    main()
