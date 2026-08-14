"""
Custom Lambda reward function for RLAIF news summarization.

Scores model-generated summaries on 5 dimensions using an LLM judge (Bedrock):
    1. Factual Faithfulness (does it only state facts from the article?)
    2. Key Event Coverage (does it capture the main news story?)
    3. No Redundancy (does it avoid repeating the same point?)
    4. Appropriate Length (is it 2-4 sentences?)
    5. Coherence (does it read as a natural, standalone paragraph?)

Each dimension is scored on a 1-5 scale by the judge.
Final score = normalized aggregate of all dimensions (0.0 to 1.0).


=== Why use a custom Lambda instead of just setting reward_prompt? ===

The built-in reward_prompt approach sends one judge call per sample.
A custom Lambda lets you:
    - Add deterministic checks BEFORE the expensive judge call
      (e.g., reject empty responses, check length constraints)
    - Combine judge scores with rule-based scores
    - Add retry logic and error handling for judge API calls
    - Log detailed metrics per dimension for debugging

Deploy as an AWS Lambda function. The function name MUST contain "SageMaker"
(e.g., "SummarizationRewardSageMaker").

Input format (sent by SageMaker as a list):
[
  {
    "data_source": "cnn_dailymail",
    "prompt": [{"role": "user", "content": "Summarize..."}],
    "response": "<model generated summary>",
    "reward_model": {"style": "llmj", "ground_truth": "<reference summary>"},
    "extra_info": {...}
  }
]

Output format (required by SageMaker):
[
  {
    "aggregate_reward_score": 0.8,
    "metrics_list": [
      {"name": "faithfulness", "value": 4, "type": "Reward"},
      {"name": "coverage", "value": 3, "type": "Reward"},
      {"name": "no_redundancy", "value": 5, "type": "Reward"},
      {"name": "appropriate_length", "value": 5, "type": "Reward"},
      {"name": "coherence", "value": 4, "type": "Reward"}
    ]
  }
]

Reference: https://docs.aws.amazon.com/sagemaker/latest/dg/model-customize-evaluation-preset-custom-scorers.html
"""

import json
import os
from concurrent.futures import ThreadPoolExecutor

import boto3


# --- Configuration ---
# The Bedrock model used as the AI judge.
# Set this via environment variable when deploying the Lambda,
# so it matches the reward_model_id in train_rlaif.py.
JUDGE_MODEL_ID = os.environ.get("JUDGE_MODEL_ID", "<YOUR_BEDROCK_JUDGE_MODEL_ID>")

# AWS region for Bedrock API calls
REGION = os.environ.get("AWS_REGION", "us-east-1")

# Bedrock client (reused across invocations for Lambda warm starts)
bedrock_client = boto3.client("bedrock-runtime", region_name=REGION)


# --- The Judge Prompt ---
# This is the core of RLAIF: a natural language description of what
# "good" looks like. The judge reads this, then evaluates the summary.

JUDGE_PROMPT_TEMPLATE = """You are a senior news editor evaluating article summaries.

Your task is to assess a summary written by a junior reporter. Score each dimension
from 1 to 5. Be strict but fair.

## Evaluation Criteria

1. **Factual Faithfulness** (1-5): Does the summary only contain facts from the article?
   5 = perfectly faithful, no invented details
   3 = mostly faithful, one minor inaccuracy
   1 = contains hallucinated facts or fabricated details

2. **Key Event Coverage** (1-5): Does the summary capture the main news story?
   5 = reader fully understands what happened
   3 = main event mentioned but key details missing
   1 = main event not mentioned or buried

3. **No Redundancy** (1-5): Does the summary avoid repeating information?
   5 = every sentence adds new information
   3 = some overlap between sentences
   1 = multiple sentences say the same thing

4. **Appropriate Length** (1-5): Is the summary 2-4 sentences?
   5 = exactly 2-4 concise sentences
   3 = slightly too long (5 sentences) or too short (1 sentence)
   1 = way too long (6+) or empty

5. **Coherence** (1-5): Does the summary read as a natural paragraph?
   5 = flows perfectly, logical ordering
   3 = understandable but awkward transitions
   1 = disconnected sentences, no logical flow

## Input

Original Article:
{article}

Model's Summary:
{summary}

Reference Summary (for context on what's important, not an exact target):
{reference}

## Required Output Format

Respond with ONLY a JSON object, no other text:
{{
  "faithfulness": 1-5,
  "coverage": 1-5,
  "no_redundancy": 1-5,
  "appropriate_length": 1-5,
  "coherence": 1-5
}}"""


def extract_article_from_prompt(prompt: list) -> str:
    """
    Extract the original article text from the chat-format prompt.

    The prompt looks like:
    [{"role": "user", "content": "Summarize the following...\n\nArticle:\n<text>"}]

    We need to pull out just the article portion for the judge.
    """
    if not prompt or not isinstance(prompt, list):
        return ""

    # Get the user message content
    for msg in prompt:
        if msg.get("role") == "user":
            content = msg.get("content", "")
            # The article comes after "Article:\n"
            if "Article:\n" in content:
                return content.split("Article:\n", 1)[1].strip()
            # Fallback: return the whole content
            return content

    return ""


def check_length_rule(summary: str) -> int:
    """
    Deterministic length check. No need to waste a judge call on this.

    Returns 1 (pass) if summary is 2-4 sentences, 0 (fail) otherwise.
    Sentences are approximated by splitting on period + space or period + end.
    """
    if not summary.strip():
        return 0

    # Simple sentence counting: split on '. ' or final '.'
    # This isn't perfect but catches obvious violations
    sentences = [s.strip() for s in summary.replace(".\n", ". ").split(". ") if s.strip()]

    # Account for last sentence that might not end with '. '
    if sentences and not sentences[-1].endswith("."):
        pass  # Already counted
    elif summary.strip().endswith(".") and sentences:
        pass  # Already counted

    count = len(sentences)
    return 1 if 2 <= count <= 4 else 0


def call_judge(article: str, summary: str, reference: str) -> dict:
    """
    Call the Bedrock judge model to evaluate the summary.

    Sends the judge prompt with the article, summary, and reference filled in.
    Parses the JSON response to get per-dimension pass/fail scores.

    Returns a dict like:
        {"faithfulness": 1, "coverage": 1, "no_redundancy": 0, ...}

    On any failure (API error, parse error), returns neutral scores (0.5)
    to avoid crashing the training job.
    """
    # Fill in the template
    judge_prompt = JUDGE_PROMPT_TEMPLATE.format(
        article=article[:3000],  # Truncate to stay within judge context
        summary=summary,
        reference=reference[:500],
    )

    try:
        # Call Bedrock with the judge prompt
        # Format depends on the model provider. We use the Converse API
        # which works across all Bedrock models with a unified format.
        response = bedrock_client.converse(
            modelId=JUDGE_MODEL_ID,
            messages=[
                {"role": "user", "content": [{"text": judge_prompt}]}
            ],
            inferenceConfig={
                "maxTokens": 200,
                "temperature": 0.0,
            },
        )

        # Parse the response (Converse API has a unified output format)
        judge_text = response["output"]["message"]["content"][0]["text"].strip()

        # Extract JSON from judge response
        # Sometimes the judge wraps it in markdown code blocks
        if "```json" in judge_text:
            judge_text = judge_text.split("```json")[1].split("```")[0].strip()
        elif "```" in judge_text:
            judge_text = judge_text.split("```")[1].split("```")[0].strip()

        scores = json.loads(judge_text)

        # Validate that all expected keys are present
        expected_keys = ["faithfulness", "coverage", "no_redundancy",
                         "appropriate_length", "coherence"]
        for key in expected_keys:
            if key not in scores:
                scores[key] = 3  # Neutral if missing

        return scores

    except Exception as e:
        # On any failure, return neutral scores rather than crashing.
        # This is important: a training job processes thousands of samples,
        # and one bad API call shouldn't kill the whole job.
        print(f"Judge call failed: {e}")
        return {
            "faithfulness": 3,
            "coverage": 3,
            "no_redundancy": 3,
            "appropriate_length": 3,
            "coherence": 3,
        }


def _score_sample(sample: dict) -> dict:
    """
    Score a single sample by combining deterministic checks + AI judge.

    The flow:
    1. Extract the article, summary, and reference from the sample
    2. Run cheap deterministic checks first (length)
    3. Call the AI judge for subjective dimensions
    4. Combine all scores into a single aggregate reward

    Returns the result dict in SageMaker's required format.
    """
    result = {}
    if "id" in sample:
        result["id"] = sample["id"]

    # --- Extract fields from the sample ---

    # The model's generated summary
    model_response = sample.get("response", "").strip()

    # The original prompt (contains the article)
    prompt = sample.get("prompt", [])
    article = extract_article_from_prompt(prompt)

    # The reference summary from ground_truth
    reward_model = sample.get("reward_model", {})
    reference = ""
    if isinstance(reward_model, dict):
        reference = reward_model.get("ground_truth", "")

    # --- Edge case: empty response ---
    if not model_response:
        result["aggregate_reward_score"] = 0.0
        result["metrics_list"] = [
            {"name": "faithfulness", "value": 0.0, "type": "Reward"},
            {"name": "coverage", "value": 0.0, "type": "Reward"},
            {"name": "no_redundancy", "value": 0.0, "type": "Reward"},
            {"name": "appropriate_length", "value": 0.0, "type": "Reward"},
            {"name": "coherence", "value": 0.0, "type": "Reward"},
        ]
        return result

    # --- Step 1: Deterministic check (free, instant) ---
    length_score = check_length_rule(model_response)

    # --- Step 2: AI judge for subjective dimensions ---
    judge_scores = call_judge(article, model_response, reference)

    # Override the length score with our deterministic check
    # (more reliable than the judge for counting sentences)
    # Convert to 1-5 scale: pass=5, fail=1
    judge_scores["appropriate_length"] = 5 if length_score == 1 else 1

    # --- Step 3: Compute aggregate score ---
    # Normalize from 1-5 scale to 0-1 scale: (score - 1) / 4
    dimensions = ["faithfulness", "coverage", "no_redundancy",
                  "appropriate_length", "coherence"]

    raw_total = sum(judge_scores.get(d, 1) for d in dimensions)
    # Convert sum of 1-5 scores to 0-1 range
    # Min possible = 5 (all 1s), Max possible = 25 (all 5s)
    aggregate = (raw_total - 5) / 20

    result["aggregate_reward_score"] = round(aggregate, 4)
    result["metrics_list"] = [
        {"name": "faithfulness", "value": judge_scores.get("faithfulness", 1), "type": "Reward"},
        {"name": "coverage", "value": judge_scores.get("coverage", 1), "type": "Reward"},
        {"name": "no_redundancy", "value": judge_scores.get("no_redundancy", 1), "type": "Reward"},
        {"name": "appropriate_length", "value": judge_scores.get("appropriate_length", 1), "type": "Reward"},
        {"name": "coherence", "value": judge_scores.get("coherence", 1), "type": "Reward"},
        {"name": "summary_sentences", "value": len([s for s in model_response.split(". ") if s.strip()]), "type": "Metric"},
    ]

    return result


def lambda_handler(event, context):
    """
    AWS Lambda handler for RLAIF reward scoring.

    SageMaker sends a list of samples. We score them in parallel
    using threads to reduce latency (each sample needs a Bedrock API call).

    Must return a list with one result per input sample.
    """
    if not isinstance(event, list):
        event = [event]

    # Process samples in parallel (each one makes a Bedrock API call)
    # ThreadPoolExecutor is ideal here since the work is I/O-bound (API calls)
    max_workers = min(len(event), 10)  # Cap at 10 concurrent judge calls

    print(f"Evaluating {len(event)} samples with {max_workers} threads...")

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(_score_sample, sample) for sample in event]
        results = [future.result() for future in futures]

    print(f"Completed {len(results)} evaluations")
    return results
