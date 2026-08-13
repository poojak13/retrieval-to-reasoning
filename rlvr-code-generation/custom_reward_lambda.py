"""
Custom Lambda reward function for RLVR code generation.

Scores model-generated code on:
    1. Test case correctness (80% of score)
    2. Code conciseness (20% of score) - penalizes unnecessarily verbose solutions

Deploy as an AWS Lambda function and pass the ARN to RLVRTrainer's
custom_reward_function parameter.

Input format (verl, sent by SageMaker as a list with one item):
[
  {
    "data_source": "mbpp",
    "prompt": [...],
    "response": "<model generated code>",
    "reward_model": {"style": "rule", "ground_truth": "<json string>"},
    "extra_info": {"reference_answer": {"text": "..."}}
  }
]

Output format (required by SageMaker):
[
  {
    "aggregate_reward_score": 0.85,
    "metrics_list": [
      {"name": "correctness", "value": 0.9, "type": "Reward"},
      {"name": "conciseness", "value": 0.8, "type": "Metric"}
    ]
  }
]

Reference: https://docs.aws.amazon.com/sagemaker/latest/dg/model-customize-evaluation-preset-custom-scorers.html
"""

import json
import subprocess


def extract_code(response: str) -> str:
    """Extract Python code from model response, handling markdown fences."""
    if "```python" in response:
        code = response.split("```python")[1].split("```")[0]
    elif "```" in response:
        code = response.split("```")[1].split("```")[0]
    else:
        code = response
    return code.strip()


def run_tests(code: str, test_list: list) -> tuple:
    """
    Execute code against test assertions.

    Returns (passed_count, total_count)
    """
    passed = 0
    total = len(test_list)

    for test_case in test_list:
        full_code = f"{code}\n{test_case}"
        try:
            result = subprocess.run(
                ["python", "-c", full_code],
                capture_output=True,
                timeout=5,
            )
            if result.returncode == 0:
                passed += 1
        except (subprocess.TimeoutExpired, Exception):
            pass

    return passed, total


def score_conciseness(code: str, reference_length: int = 10) -> float:
    """
    Score code conciseness. Returns a value between 0.0 and 1.0.

    Scoring logic:
        - If code is at or below reference length: full score (1.0)
        - If code is up to 3x reference length: partial score (linear decay)
        - If code is more than 3x reference length: minimum score (0.2)
    """
    lines = [line for line in code.split("\n") if line.strip()]
    actual_length = len(lines)

    if actual_length <= reference_length:
        return 1.0

    max_before_floor = reference_length * 3
    if actual_length >= max_before_floor:
        return 0.2

    ratio = (actual_length - reference_length) / (max_before_floor - reference_length)
    return 1.0 - (0.8 * ratio)


def build_test_list(ground_truth_raw: str) -> list:
    """
    Parse ground_truth into a list of assertion strings.

    Supports:
        - Format A: JSON list of assertion strings
        - Format B: JSON dict with fn_name, inputs, outputs
    """
    try:
        ground_truth = json.loads(ground_truth_raw)
    except (json.JSONDecodeError, TypeError):
        return []

    if isinstance(ground_truth, list):
        return ground_truth

    if isinstance(ground_truth, dict):
        fn_name = ground_truth.get("fn_name", "")
        inputs = ground_truth.get("inputs", [])
        outputs = ground_truth.get("outputs", [])
        if not fn_name or not inputs or not outputs:
            return []
        test_list = []
        for inp, out in zip(inputs, outputs):
            try:
                args = [json.loads(a) for a in inp.split("\n")]
                expected = json.loads(out)
                args_str = ", ".join(repr(a) for a in args)
                test_list.append(f"assert {fn_name}({args_str}) == {repr(expected)}")
            except (json.JSONDecodeError, Exception):
                continue
        return test_list

    return []


def lambda_handler(event, context):
    """
    AWS Lambda handler for RLVR reward scoring.

    SageMaker sends a list with one sample per invocation.
    Must return a list with one result per input sample.
    """
    # event is always a list of samples
    if not isinstance(event, list):
        event = [event]

    results = []
    for sample in event:
        result = _score_sample(sample)
        results.append(result)

    return results


def _score_sample(sample: dict) -> dict:
    """
    Score a single sample.

    Returns dict with aggregate_reward_score and optional metrics_list.
    """
    # Build result skeleton with id if present
    result = {}
    if "id" in sample:
        result["id"] = sample["id"]

    # Get model response (verl format: "response" field)
    model_response = sample.get("response", "")

    # Get ground truth from reward_model.ground_truth (verl format)
    reward_model = sample.get("reward_model", {})
    ground_truth_raw = ""
    if isinstance(reward_model, dict):
        ground_truth_raw = reward_model.get("ground_truth", "")

    # Fallback: try extra_info.reference_answer.text
    if not ground_truth_raw:
        extra_info = sample.get("extra_info", {})
        ref_answer = extra_info.get("reference_answer", {}) if extra_info else {}
        ground_truth_raw = ref_answer.get("text", "") if ref_answer else ""

    # Parse ground truth into test assertions
    test_list = build_test_list(ground_truth_raw)
    if not test_list:
        result["aggregate_reward_score"] = 0.0
        return result

    # Extract code from model response
    code = extract_code(model_response)
    if not code:
        result["aggregate_reward_score"] = 0.0
        return result

    # Score 1: Test correctness (80%)
    passed, total = run_tests(code, test_list)
    correctness_score = passed / total if total > 0 else 0.0

    # Score 2: Conciseness (20%)
    conciseness_score = score_conciseness(code, reference_length=10)

    # Combine: correctness dominates, conciseness is a tiebreaker
    total_score = (0.8 * correctness_score) + (0.2 * conciseness_score)

    result["aggregate_reward_score"] = round(total_score, 4)
    result["metrics_list"] = [
        {"name": "correctness", "value": correctness_score, "type": "Reward"},
        {"name": "conciseness", "value": conciseness_score, "type": "Metric"},
        {"name": "tests_passed", "value": passed, "type": "Metric"},
        {
            "name": "code_lines",
            "value": len([l for l in code.split("\n") if l.strip()]),
            "type": "Metric",
        },
    ]

    return result
