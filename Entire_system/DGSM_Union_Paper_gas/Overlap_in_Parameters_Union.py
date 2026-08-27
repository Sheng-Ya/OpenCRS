# code for comparing between two DGSM runs for the overlap in sensitive parameters
import re
from collections import defaultdict, Counter

import torch


def parse_sensitivity_file(filename):
    """
    Parse sensitivity output file.
    Returns:
    - output_params: dict {output_name: set(parameters)}
    - param_counts: Counter of parameter appearances across outputs
    """
    output_params = defaultdict(set)
    current_output = None

    output_header = re.compile(r"^Output:\s*(.+)$")
    param_line = re.compile(r"^\s*([A-Za-z0-9_]+)\s*:\s*[-+0-9\.eE]+\s*\(")

    with open(filename, "r") as f:
        for line in f:
            header_match = output_header.search(line)
            if header_match:
                current_output = header_match.group(1).strip()
                continue

            param_match = param_line.match(line)
            if param_match and current_output is not None:
                param = param_match.group(1)
                output_params[current_output].add(param)

    param_counts = Counter()
    for params in output_params.values():
        param_counts.update(params)

    return output_params, param_counts


def get_all_params(output_params):
    """Flatten all parameters from all outputs into a single set."""
    all_params = set()
    for params in output_params.values():
        all_params.update(params)
    return all_params


def format_python_set(name, params):
    """Return a Python assignment for a sorted set of parameter names."""
    sorted_params = sorted(params, key=str.lower)
    if not sorted_params:
        return f"{name} = set()"

    values = ", ".join(repr(param) for param in sorted_params)
    return f"{name} = {{{values}}}"


if __name__ == "__main__":

    fileRest = r"DGSM_Union_Rest.txt"
    fileExercise = r"DGSM_Union_Exercise.txt"

    output_params_Exercise, param_counts_Exercise = parse_sensitivity_file(fileExercise)
    output_params_Rest, param_counts_Rest = parse_sensitivity_file(fileRest)

    params_Exercise = get_all_params(output_params_Exercise)
    params_Rest = get_all_params(output_params_Rest)

    # ── Overlap analysis ──────────────────────────────────────────────────────
    overlap       = params_Exercise & params_Rest
    rest_only     = params_Rest - params_Exercise
    exercise_only = params_Exercise - params_Rest
    only_in_Exercise    = exercise_only
    only_in_Rest    = rest_only

    print("=" * 80)
    print(f"DGSM_Exercise  — unique parameters : {len(params_Exercise)}")
    print(f"DGSM_Rest  — unique parameters : {len(params_Rest)}")
    print(f"Overlap (in both)            : {len(overlap)}")
    print(f"Only in DGSM_Exercise              : {len(only_in_Exercise)}")
    print(f"Only in DGSM_Rest              : {len(only_in_Rest)}")
    print("=" * 80)

    print(f"\n{'-'*40}")
    print(f"Parameters in BOTH files ({len(overlap)}):")
    print(f"{'-'*40}")
    for p in sorted(overlap, key=str.lower):
        print(f"  {p}")

    if only_in_Exercise:
        print(f"\n{'-'*40}")
        print(f"Only in DGSM_Exercise ({len(only_in_Exercise)}):")
        print(f"{'-'*40}")
        for p in sorted(only_in_Exercise, key=str.lower):
            print(f"  {p}")

    if only_in_Rest:
        print(f"\n{'-'*40}")
        print(f"Only in DGSM_Rest ({len(only_in_Rest)}):")
        print(f"{'-'*40}")
        for p in sorted(only_in_Rest, key=str.lower):
            print(f"  {p}")

    # ── Per-output overlap ────────────────────────────────────────────────────
    all_outputs = sorted(set(output_params_Exercise) | set(output_params_Rest))

    print(f"\n{'='*100}")
    print("Per-output overlap")
    print(f"{'='*100}")
    print(f"{'Output':<30} {'Exercise-only':>8} {'overlap':>8} {'Rest-only':>8}  overlapping parameters")
    print(f"{'-'*100}")

    for out in all_outputs:
        pExercise = output_params_Exercise.get(out, set())
        pRest = output_params_Rest.get(out, set())
        ov  = pExercise & pRest
        print(
            f"{out:<30} {len(pExercise-pRest):>8} {len(ov):>8} {len(pRest-pExercise):>8}"
            f"  {', '.join(sorted(ov, key=str.lower)) if ov else '-'}"
        )

    print()
    print(format_python_set("Overlap", overlap))
    print()
    print(format_python_set("Rest_only", rest_only))
    print()
    print(format_python_set("Exercise_only", exercise_only))
