"""
convert_mbpp_to_rlvr.py

Converts the MBPP dataset from Hugging Face into the VERL format
expected by SageMaker's prime_code preset reward function.

Usage:
    pip install datasets
    python convert_mbpp_to_rlvr.py

Output:
    mbpp_rlvr_train.jsonl  (ready to upload to S3 for SageMaker RLVR training)
"""

import json
import re

from datasets import load_dataset


def parse_assert(assertion: str, fn_name: str):
    """
    Parse an MBPP assertion like:
        assert similar_elements((3, 4, 5, 6),(5, 7, 4, 10)) == (4, 5)

    Returns (input_str, output_str) in the format prime_code expects:
        - input_str: JSON-encoded arguments separated by newlines
        - output_str: JSON-encoded expected return value
    """
    # Remove 'assert ' prefix
    assertion = assertion.strip()
    if assertion.startswith("assert "):
        assertion = assertion[7:]

    # Split on ' == ' to get the call and expected result
    parts = assertion.split(" == ", 1)
    if len(parts) != 2:
        return None, None

    call_str = parts[0].strip()
    expected_str = parts[1].strip()

    # Extract arguments from the function call
    pattern = rf"^{re.escape(fn_name)}\((.*)\)$"
    match = re.match(pattern, call_str, re.DOTALL)
    if not match:
        return None, None

    args_str = match.group(1)

    # Evaluate the full call to extract individual arguments
    try:
        captured_args = []

        def _capture(*args, **kwargs):
            captured_args.extend(args)

        exec(f"_capture({args_str})", {"_capture": _capture})

        # Convert each arg to JSON string, join with newlines
        input_str = "\n".join(json.dumps(arg) for arg in captured_args)

        # Convert expected output to JSON
        # prime_code converts tuples to lists for comparison
        expected_val = eval(expected_str)
        if isinstance(expected_val, tuple):
            expected_val = list(expected_val)
        output_str = json.dumps(expected_val)

        return input_str, output_str
    except Exception:
        return None, None


def extract_fn_name(test_list: list) -> str:
    """Extract the function name from the first assertion."""
    if not test_list:
        return None

    first_test = test_list[0].strip()
    if first_test.startswith("assert "):
        first_test = first_test[7:]

    match = re.match(r"^([a-zA-Z_][a-zA-Z0-9_]*)\(", first_test)
    if match:
        return match.group(1)
    return None


def convert_mbpp_entry(row):
    """Convert one MBPP row to VERL prime_code format."""
    fn_name = extract_fn_name(row["test_list"])
    if not fn_name:
        return None

    inputs = []
    outputs = []

    for assertion in row["test_list"]:
        input_str, output_str = parse_assert(assertion, fn_name)
        if input_str is not None and output_str is not None:
            inputs.append(input_str)
            outputs.append(output_str)

    if not inputs:
        return None

    # Build the ground_truth dict that prime_code's testing_util.py expects
    ground_truth = {
        "fn_name": fn_name,
        "inputs": inputs,
        "outputs": outputs,
    }

    # Build the full training entry in VERL format
    entry = {
        "data_source": "mbpp",
        "prompt": [
            {
                "role": "user",
                "content": (
                    f"{row['prompt']}\n\n"
                    f"The function should be named `{fn_name}`. "
                    f"Write only the Python function."
                ),
            }
        ],
        "ability": "code",
        "reward_model": {
            "style": "rule",
            "ground_truth": json.dumps(ground_truth),
        },
    }

    return entry


def main():
    # Load MBPP from Hugging Face (combine train + test for more data)
    print("Loading MBPP dataset...")
    dataset = load_dataset(
        "google-research-datasets/mbpp", "sanitized", split="train+test"
    )

    print(f"Processing {len(dataset)} problems...")

    converted = []
    skipped = 0

    for row in dataset:
        entry = convert_mbpp_entry(row)
        if entry:
            converted.append(entry)
        else:
            skipped += 1

    # Write output
    output_file = "mbpp_rlvr_train.jsonl"
    with open(output_file, "w") as f:
        for entry in converted:
            f.write(json.dumps(entry) + "\n")

    print(f"\nDone!")
    print(f"  Converted: {len(converted)} problems")
    print(f"  Skipped:   {skipped} (could not parse assertions)")
    print(f"  Output:    {output_file}")

    print(f"\nNext steps:")
    print(f"  aws s3 cp {output_file} s3://<YOUR_BUCKET>/rlvr-code/{output_file}")

    # Print a sample entry for verification
    print(f"\nSample entry:")
    print(json.dumps(converted[0], indent=2))


if __name__ == "__main__":
    main()
