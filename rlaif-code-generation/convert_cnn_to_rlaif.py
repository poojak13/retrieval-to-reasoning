"""
convert_cnn_to_rlaif.py

Converts the CNN/DailyMail dataset from Hugging Face into the JSONL format
expected by SageMaker's RLAIFTrainer.

=== What this script does ===

1. Downloads the CNN/DailyMail dataset from HuggingFace (300k news articles
   with human-written summary highlights)

2. For each article, creates a training record with:
   - A "prompt" that asks the model to summarize the article
   - A "reward_model" section that tells SageMaker to use an LLM judge
     (not test cases) to score the model's summaries
   - The human-written summary as "ground_truth" so the judge has a
     reference point when scoring

3. Writes everything to a JSONL file (one JSON object per line) that
   SageMaker can directly consume

=== Why this format? ===

SageMaker's RLAIF trainer expects:
- reward_model.style = "llmj" (meaning "LLM as judge", not "rule-based")
- reward_model.ground_truth = a reference the judge can compare against
- prompt = the conversation that the model will complete during training

The judge doesn't require an exact match with ground_truth. It uses the
reference to understand what a good summary should cover, then scores the
model's attempt on faithfulness, coverage, and conciseness.

Usage:
    pip install datasets
    python convert_cnn_to_rlaif.py

Output:
    cnn_rlaif_train.jsonl  (ready to upload to S3 for SageMaker RLAIF training)
"""

import json

from datasets import load_dataset


# How many articles to include. CNN/DailyMail has ~287k training examples.
# For a first run, 5000 is plenty. Scale up once you've validated the pipeline.
MAX_SAMPLES = 5000

# Maximum article length in characters. Articles longer than this get truncated
# to avoid context window issues during training.
MAX_ARTICLE_LENGTH = 4000


def build_prompt(article: str) -> list:
    """
    Build the chat-format prompt that the model will see during training.

    The model's job is to complete this conversation by writing a summary.
    During training, it generates candidate summaries, and the AI judge
    scores each one.

    Args:
        article: The news article text to summarize

    Returns:
        A list of message dicts in OpenAI chat format
    """
    return [
        {
            "role": "user",
            "content": (
                "Summarize the following news article in 2-4 sentences. "
                "Capture the most important facts and events. "
                "Be concise and faithful to the source.\n\n"
                f"Article:\n{article}"
            ),
        }
    ]


def convert_entry(row: dict) -> dict | None:
    """
    Convert one CNN/DailyMail row into SageMaker RLAIF format.

    Args:
        row: A dataset row with 'article' and 'highlights' fields

    Returns:
        A dict in RLAIF format, or None if the article is too short
    """
    article = row["article"].strip()
    highlights = row["highlights"].strip()

    # Skip very short articles (likely malformed)
    if len(article) < 100 or len(highlights) < 20:
        return None

    # Truncate very long articles to stay within context limits
    if len(article) > MAX_ARTICLE_LENGTH:
        # Cut at the last sentence boundary before the limit
        truncated = article[:MAX_ARTICLE_LENGTH]
        last_period = truncated.rfind(".")
        if last_period > MAX_ARTICLE_LENGTH // 2:
            article = truncated[: last_period + 1]
        else:
            article = truncated

    # Build the training entry
    # - "prompt": what the model sees and must respond to
    # - "reward_model.style": "llmj" tells SageMaker to use an LLM judge
    # - "reward_model.ground_truth": the reference summary the judge uses
    #   for comparison (it doesn't require an exact match, just uses it
    #   to understand what good looks like)
    entry = {
        "data_source": "cnn_dailymail",
        "prompt": build_prompt(article),
        "ability": "summarization",
        "reward_model": {
            "style": "llmj",
            "ground_truth": highlights,
        },
    }

    return entry


def main():
    # Step 1: Load the dataset from HuggingFace
    # Version 3.0.0 is the standard one with article + highlights fields
    print("Loading CNN/DailyMail dataset from HuggingFace...")
    dataset = load_dataset("abisee/cnn_dailymail", "3.0.0", split="train")
    print(f"  Total articles available: {len(dataset)}")

    # Step 2: Convert each article into RLAIF format
    # We take a subset to keep training time and cost manageable
    print(f"\nConverting up to {MAX_SAMPLES} articles...")

    converted = []
    skipped = 0

    for i, row in enumerate(dataset):
        if len(converted) >= MAX_SAMPLES:
            break

        entry = convert_entry(row)
        if entry:
            converted.append(entry)
        else:
            skipped += 1

    # Step 3: Write the JSONL output file
    output_file = "cnn_rlaif_train.jsonl"
    with open(output_file, "w") as f:
        for entry in converted:
            f.write(json.dumps(entry) + "\n")

    # Print summary
    print(f"\nDone!")
    print(f"  Converted: {len(converted)} articles")
    print(f"  Skipped:   {skipped} (too short or malformed)")
    print(f"  Output:    {output_file}")

    print(f"\nNext step:")
    print(
        f"  aws s3 cp {output_file} "
        f"s3://<YOUR_BUCKET>/rlaif-summarization/{output_file}"
    )

    # Print a sample so you can verify the format looks right
    print(f"\n{'='*60}")
    print("Sample entry (first record):")
    print("=" * 60)
    sample = converted[0]
    print(json.dumps(sample, indent=2)[:2000])


if __name__ == "__main__":
    main()
