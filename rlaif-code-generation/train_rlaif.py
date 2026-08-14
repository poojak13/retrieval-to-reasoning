"""
train_rlaif.py

Trains a summarization model using SageMaker's RLAIFTrainer.

=== What this script does (plain English) ===

This script kicks off a training job on SageMaker that works like this:

1. SageMaker loads your base model (e.g., Llama or Qwen)
2. For each news article in your dataset:
   a. The model generates several candidate summaries (controlled by rollout_n)
   b. An AI judge scores each summary
   c. The scores tell the model which summaries were good and which were bad
   d. The model's weights get updated to produce more "good" summaries
3. This repeats for all articles, for multiple epochs
4. The result is a fine-tuned model that writes better summaries

=== Two reward options ===

Option A: Built-in judge template (simpler, no Lambda needed)
    - Uses a preset like "summarize.jinja" which is pre-written inside the
      training container
    - The judge evaluates faithfulness, coverage, conciseness
    - NOTE: This may not work via the Python SDK in all regions. If you get
      a HubContent validation error, use Option B instead.

Option B: Custom Lambda judge (more control, what we used)
    - Your Lambda calls Bedrock with a custom prompt
    - 1-5 scale scoring on 5 specific dimensions
    - Adds deterministic length checks before the judge call
    - Better for production where you need specific criteria

Usage:
    1. First run: python convert_cnn_to_rlaif.py
    2. Upload the output to S3
    3. Deploy the Lambda (see README)
    4. Update the placeholder values below
    5. Set USE_PRESET = True or False
    6. Run: python train_rlaif.py
"""

import sagemaker
from sagemaker.train import RLAIFTrainer
from sagemaker.train.common import TrainingType
from sagemaker.ai_registry.evaluator import Evaluator


# ============================================================
# CONFIGURATION - Update these with your own values
# ============================================================

# The base model to fine-tune
MODEL = "huggingface-reasoning-qwen3-8b"

# Where your converted dataset lives in S3
TRAINING_DATA = "s3://<YOUR_BUCKET>/rlaif-summarization/cnn_rlaif_train.jsonl"

# SageMaker stores the fine-tuned model in this group
MODEL_PACKAGE_GROUP = "summarization-rlaif"

# Your SageMaker execution role
ROLE_ARN = "arn:aws:iam::<YOUR_ACCOUNT_ID>:role/service-role/<YOUR_SAGEMAKER_EXECUTION_ROLE>"

# The Bedrock model that acts as the AI judge
REWARD_MODEL_ID = "<YOUR_BEDROCK_JUDGE_MODEL_ID>"

# Custom Lambda ARN (only needed if USE_PRESET = False)
LAMBDA_ARN = "arn:aws:lambda:<REGION>:<YOUR_ACCOUNT_ID>:function:SummarizationRewardSageMaker"

# ============================================================
# CHOOSE YOUR REWARD APPROACH
# ============================================================
# True  = Use built-in preset (no Lambda needed, may not work via SDK)
# False = Use custom Lambda with 5-dimension scoring (recommended)
USE_PRESET = False


# ============================================================
# TRAINING FUNCTIONS
# ============================================================

def train_with_preset():
    """
    Option A: Use a built-in judge template.

    SageMaker has pre-written evaluation templates inside the training
    container (e.g., "summarize.jinja" for summarization quality).

    You just select the template name and SageMaker handles the rest:
    it fills in the article/summary/reference, sends it to the judge
    model, and returns a score.

    NOTE: This path may fail with a HubContent validation error when
    using the Python SDK. If so, use the custom Lambda approach instead.
    The preset templates may only be selectable via the SageMaker console UI.
    """
    trainer = RLAIFTrainer(
        model=MODEL,
        training_type=TrainingType.LORA,
        model_package_group=MODEL_PACKAGE_GROUP,
        reward_model_id=REWARD_MODEL_ID,
        reward_prompt="summarize",  # Built-in template name
        training_dataset=TRAINING_DATA,
        role=ROLE_ARN,
    )

    trainer.hyperparameters.rollout_n = 4
    trainer.hyperparameters.rollout_temperature = 0.7
    trainer.hyperparameters.learning_rate = 2e-5
    trainer.hyperparameters.max_epochs = 2
    trainer.hyperparameters.global_batch_size = 128
    trainer.hyperparameters.clip_ratio = 0.2
    trainer.hyperparameters.lora_rank = 16
    trainer.hyperparameters.lora_alpha = 32

    print("Using PRESET judge template: summarize")
    print("(Judge evaluates faithfulness, coverage, conciseness)")
    return trainer.train(wait=True)


def train_with_custom_lambda():
    """
    Option B: Use a custom Lambda function as the reward.

    The Lambda (custom_reward_lambda.py) does everything itself:
    - Deterministic length checks (free, instant)
    - Calls Bedrock judge for subjective dimensions (1-5 scale)
    - Combines scores into a normalized 0-1 aggregate reward
    - Logs per-dimension metrics for debugging

    Advantages over preset:
    - Mix deterministic + AI scoring (hybrid approach)
    - Full control over retry logic and error handling
    - Parallel processing with ThreadPoolExecutor
    - Detailed per-dimension metrics in CloudWatch
    - Actually works via the Python SDK

    Requires deploying the Lambda first (see README).
    """
    evaluator = Evaluator.create(
        name="summarization-reward-evaluator",
        type="RewardFunction",
        source=LAMBDA_ARN,
        role=ROLE_ARN,
    )

    trainer = RLAIFTrainer(
        model=MODEL,
        training_type=TrainingType.LORA,
        model_package_group=MODEL_PACKAGE_GROUP,
        reward_model_id=REWARD_MODEL_ID,
        reward_prompt=evaluator,
        training_dataset=TRAINING_DATA,
        role=ROLE_ARN,
    )

    trainer.hyperparameters.rollout_n = 4
    trainer.hyperparameters.rollout_temperature = 0.7
    trainer.hyperparameters.learning_rate = 2e-5
    trainer.hyperparameters.max_epochs = 2
    trainer.hyperparameters.global_batch_size = 128
    trainer.hyperparameters.clip_ratio = 0.2
    trainer.hyperparameters.lora_rank = 16
    trainer.hyperparameters.lora_alpha = 32

    print("Using CUSTOM LAMBDA reward function")
    print(f"Lambda: {LAMBDA_ARN}")
    print("(Deterministic length check + AI judge scoring 1-5 on 4 dimensions)")
    return trainer.train(wait=True)


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":
    print("=" * 60)
    print("RLAIF Summarization Training")
    print("=" * 60)
    print(f"  Base model:    {MODEL}")
    print(f"  Training data: {TRAINING_DATA}")
    print(f"  Judge model:   {REWARD_MODEL_ID}")
    print()

    if USE_PRESET:
        training_job = train_with_preset()
    else:
        training_job = train_with_custom_lambda()

    print()
    print("=" * 60)
    print("Training complete!")
    print("=" * 60)
    print(f"  Job name:         {training_job.training_job_name}")
    print(f"  Fine-tuned model: {training_job.output_model_package_arn}")
    print()
    print("The model has learned to write summaries that the AI judge")
    print("rates highly. Deploy it and send new articles for summarization.")
