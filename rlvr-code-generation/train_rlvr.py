import sagemaker
from sagemaker.train import RLVRTrainer
from sagemaker.train.common import TrainingType

# Configuration (update these with your own values)
MODEL = "huggingface-reasoning-qwen3-8b"
TRAINING_DATA = "s3://<YOUR_BUCKET>/rlvr-code/mbpp_rlvr_train.jsonl"
MODEL_PACKAGE_GROUP = "code-gen-rlvr"
ROLE_ARN = "arn:aws:iam::<YOUR_ACCOUNT_ID>:role/service-role/<YOUR_SAGEMAKER_EXECUTION_ROLE>"

# Choose one:
USE_PRESET = False  # True = no reward function (built-in), False = custom Lambda

# Custom Lambda ARN (only needed if USE_PRESET = False)
LAMBDA_ARN = "arn:aws:lambda:<REGION>:<YOUR_ACCOUNT_ID>:function:mbpp-reward"


def train_with_preset():
    """Option A: Use built-in reward (no custom Lambda). Scores based on data format."""
    trainer = RLVRTrainer(
        model=MODEL,
        training_type=TrainingType.LORA,
        model_package_group=MODEL_PACKAGE_GROUP,
        training_dataset=TRAINING_DATA,
        role=ROLE_ARN,
    )

    trainer.hyperparameters.rollout_n = 8
    trainer.hyperparameters.rollout_temperature = 0.7
    trainer.hyperparameters.learning_rate = 5e-6
    trainer.hyperparameters.max_epochs = 3
    trainer.hyperparameters.global_batch_size = 128
    trainer.hyperparameters.clip_ratio = 0.2
    trainer.hyperparameters.lora_rank = 16
    trainer.hyperparameters.lora_alpha = 32

    print("Training with built-in reward (correctness only)")
    return trainer.train(wait=True)


def train_with_custom_lambda():
    """Option B: Use custom Lambda. Scores correctness + conciseness."""
    from sagemaker.ai_registry.evaluator import Evaluator

    # Pre-create the Evaluator with explicit role to avoid SDK role resolution bug
    evaluator = Evaluator.create(
        name="mbpp-reward-evaluator",
        type="RewardFunction",
        source=LAMBDA_ARN,
        role=ROLE_ARN,
    )

    trainer = RLVRTrainer(
        model=MODEL,
        training_type=TrainingType.LORA,
        model_package_group=MODEL_PACKAGE_GROUP,
        custom_reward_function=evaluator,
        training_dataset=TRAINING_DATA,
        role=ROLE_ARN,
        skip_reward_validation=True,
    )

    trainer.hyperparameters.rollout_n = 8
    trainer.hyperparameters.rollout_temperature = 0.7
    trainer.hyperparameters.learning_rate = 5e-6
    trainer.hyperparameters.max_epochs = 3
    trainer.hyperparameters.global_batch_size = 128
    trainer.hyperparameters.clip_ratio = 0.2
    trainer.hyperparameters.lora_rank = 16
    trainer.hyperparameters.lora_alpha = 32

    print("Training with custom Lambda (correctness + conciseness)")
    print(f"Lambda: {LAMBDA_ARN}")

    return trainer.train(wait=True)


if __name__ == "__main__":
    print(f"Model: {MODEL}")
    print(f"Data: {TRAINING_DATA}")
    print()

    if USE_PRESET:
        training_job = train_with_preset()
    else:
        training_job = train_with_custom_lambda()

    print(f"\nTraining complete!")
    print(f"Job name: {training_job.training_job_name}")
    print(f"Fine-tuned model: {training_job.output_model_package_arn}")
