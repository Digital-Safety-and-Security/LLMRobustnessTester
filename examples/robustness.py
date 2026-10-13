"""Reference Tester script: replace the uppercase pipeline with your own model."""

import argparse
import json

from robustness_testing import Tester, WordEditDistance


def pipeline(text, data):
    return text.upper()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--eval-files", nargs="+", required=True)
    parser.add_argument("--mode", choices=("max-delta", "search-epsilon"), default="max-delta")
    parser.add_argument("--epsilon", type=float)
    parser.add_argument("--delta", type=float)
    parser.add_argument("--threshold-check", choices=("individual", "average"), default="individual")
    parser.add_argument("--min-epsilon", type=float, default=1)
    parser.add_argument("--epsilon-step", choices=("double", "add-one"), default="double")
    parser.add_argument("--max-iterations", type=int, default=5)
    parser.add_argument("--stop-on-nan", action="store_true", help="Stop on unavailable output distances instead of ignoring them.")
    args = parser.parse_args()
    tester = Tester(pipeline, WordEditDistance())
    result = tester.run(
        tester.load_inputs(args.eval_files), mode=args.mode, epsilon=args.epsilon,
        output_distance_threshold=args.delta, threshold_check=args.threshold_check,
        min_epsilon=args.min_epsilon, epsilon_step=args.epsilon_step, max_iterations=args.max_iterations,
        stop_on_nan=args.stop_on_nan,
    )
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
