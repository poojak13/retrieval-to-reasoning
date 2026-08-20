"""
train_mtrl.py

Trains a text-to-SQL agent using SageMaker's MultiTurnRLTrainer.

=== What this script does (plain English) ===

This script kicks off a multi-turn RL training job. Unlike single-turn RL
(RLVR/RLAIF) where the model generates one response and gets scored,
multi-turn RL works like this:

1. SageMaker loads the base model and connects it to an agent environment
   (the "agentic application" that has the SQL execution tool)

2. For each question in the dataset:
   a. The agent receives the question + schema
   b. The agent thinks, writes SQL, and executes it (turn 1)
   c. The environment returns the result or error
   d. The agent can iterate: fix errors, try different queries (turns 2-5)
   e. The agent submits its final answer with <solution> tags

3. The reward function evaluates the FULL trajectory:
   - Did the final SQL produce the correct result? (1.0 or 0.0)
   - Did the agent use proper format? (small shaping bonus)

4. The policy update considers the ENTIRE sequence of decisions:
   - Which search strategies worked?
   - Which debugging approaches led to correct answers?
   - When should the agent stop and submit vs. keep iterating?

=== Key Differences from Single-Turn RL ===

Single-turn (RLVR/RLAIF):
- Model generates ONE response
- Reward is per-response
- No interaction with environment between generation and scoring

Multi-turn (MTRL):
- Model generates MULTIPLE responses across turns
- Reward is per-EPISODE (the whole conversation)
- Model interacts with tools/environment between turns
- Model can observe results and adapt its strategy

=== The Agent Environment ===

The "agent_env" parameter points to a Bedrock AgentCore runtime.
This runtime is a deployed application that:
- Hosts the model for inference
- Provides tools (in our case, a SQL execution tool)
- Manages the conversation loop (sending observations back to the model)
- Enforces turn limits

During training, SageMaker sends prompts to this agent, lets it run
through multiple turns with the SQL tool, then collects the full trajectory
and rewards it.

Usage:
    1. First run: python convert_spider_to_mtrl.py
    2. Upload the output to S3
    3. Deploy the agent environment (Bedrock AgentCore)
    4. Update the placeholder values below
    5. Run: python train_mtrl.py
"""

from sagemaker.train.multi_turn_rl_trainer import MultiTurnRLTrainer


# ============================================================
# CONFIGURATION - Update these with your own values
# ============================================================

# The base model to fine-tune.
# Must be one of the supported multi-turn RL models:
# - "openai-reasoning-gpt-oss-20b" (us-east-1, us-west-2)
# - Nova Lite 2.0 (us-east-1, us-west-2)
# - Gemma-4-31B-it (us-west-2)
# - Qwen 3.6 27B (us-west-2)
MODEL = "openai-reasoning-gpt-oss-20b"

# The Bedrock AgentCore runtime that serves as the agent environment.
# This runtime has the SQL execution tool registered and manages the
# multi-turn conversation loop.
AGENT_ENV_ARN = "arn:aws:bedrock-agentcore:<REGION>:<YOUR_ACCOUNT_ID>:runtime/<YOUR_AGENTCORE_RUNTIME_ID>"

# Where your converted dataset lives in S3
TRAINING_DATA = "s3://<YOUR_BUCKET>/mtrl-text2sql/spider_mtrl_train.parquet"

# S3 path for training outputs (model checkpoints, logs)
OUTPUT_PATH = "s3://<YOUR_BUCKET>/mtrl-text2sql/output/"

# Accept the model's end-user license agreement
ACCEPT_EULA = True

# How long the LOCAL script waits for the job before giving up (seconds). This
# is only the client-side wait; the job runs on SageMaker independent of this.
# The SDK default is 3000s (50 min), which is too short for a real MTRL run.
WAIT_TIMEOUT_S = 4 * 60 * 60  # 4 hours

# Your SageMaker execution role (needed when running from local machine)
ROLE_ARN = "arn:aws:iam::<YOUR_ACCOUNT_ID>:role/service-role/<YOUR_SAGEMAKER_EXECUTION_ROLE>"


# ============================================================
# TRAINING
# ============================================================

def train():
    """
    Launch the multi-turn RL training job.

    Key parameters explained:

    - model: The base LLM to fine-tune. It learns to be a better SQL agent
      through RL over the course of many episodes.

    - agent_env: The deployed agent application. During training, SageMaker
      sends prompts to this agent and lets it interact with the SQL tool
      across multiple turns. The agent_env defines WHAT tools are available
      and HOW the conversation loop works.

    - training_dataset: Just prompts (questions + schemas). Unlike SFT,
      there are no target responses. The model generates its own trajectories
      during training and learns from the rewards.

    - global_batch_size: Number of EPISODES per optimizer step. This is
      smaller than single-turn (10 vs 128) because each episode involves
      multiple turns of inference + tool execution, which is expensive.

    - max_epochs: How many times to go through the full dataset. Each pass
      generates new trajectories (the model's behavior changes, so it
      produces different SQL strategies each epoch).
    """
    trainer = MultiTurnRLTrainer(
        model=MODEL,
        agent_env=AGENT_ENV_ARN,
        training_dataset=TRAINING_DATA,
        s3_output_path=OUTPUT_PATH,
        accept_eula=ACCEPT_EULA,
        role=ROLE_ARN,
    )

    # --- Hyperparameters ---

    # max_epochs: Number of full passes through the dataset.
    # Each epoch generates fresh trajectories with the current policy.
    trainer.hyperparameters.max_epochs = 1

    # global_batch_size: Episodes per optimizer step.
    # Smaller than single-turn because each episode is expensive
    # (multiple turns of model inference + tool calls).
    # 10 means: run 10 complete episodes, collect all rewards,
    # then do one weight update.
    trainer.hyperparameters.global_batch_size = 32

    # learning_rate: Step size for weight updates.
    # Similar to single-turn RL, kept conservative for stability.
    trainer.hyperparameters.learning_rate = 1e-5

    # --- Launch Training ---
    print("=" * 60)
    print("Multi-Turn RL: Text-to-SQL Agent Training")
    print("=" * 60)
    print(f"  Model:       {MODEL}")
    print(f"  Agent Env:   {AGENT_ENV_ARN}")
    print(f"  Dataset:     {TRAINING_DATA}")
    print(f"  Batch size:  {trainer.hyperparameters.global_batch_size} episodes per step")
    print(f"  Epochs:      {trainer.hyperparameters.max_epochs}")
    print()
    print("Each episode = agent receives question, writes SQL,")
    print("executes it, sees results/errors, iterates, submits answer.")
    print()
    print("Starting training job...")

    # Submit WITHOUT blocking. train(wait=True) internally calls job.wait()
    # with a default timeout of 3000s (50 min) and RAISES if the job is still
    # running past that, even though the job keeps running on SageMaker. A real
    # multi-turn RL run (256 agent invocations per step) easily exceeds 50 min,
    # so we submit non-blocking and then wait with a generous timeout below.
    job = trainer.train(wait=False)

    print(f"Submitted job: {job.job_name}")
    print(f"Waiting for completion (timeout {WAIT_TIMEOUT_S}s)...")
    print("Note: the job runs on SageMaker regardless of this local wait. If")
    print("this script times out or you Ctrl-C, the job keeps going; check the")
    print("SageMaker console or rerun with WAIT to re-attach.")

    try:
        job.wait(timeout=WAIT_TIMEOUT_S)
    except Exception as e:  # noqa: BLE001
        # A client-side wait timeout is NOT a training failure. Report status
        # and exit gracefully instead of implying the job failed.
        job.refresh()
        if str(job.job_status or "").lower() in ("inprogress", "in progress"):
            print()
            print("=" * 60)
            print("Local wait timed out, but the JOB IS STILL RUNNING.")
            print("=" * 60)
            print(f"  Job name:   {job.job_name}")
            print(f"  Job status: {job.job_status}")
            print(f"  Reattach:   check the SageMaker console, or query")
            print(f"              job.job_status until it is Completed.")
            return job
        # Some other error while waiting; surface it.
        print(f"Wait raised: {e}")

    # AgentRFTJob (returned by MultiTurnRLTrainer.train) exposes job_name,
    # job_arn, job_status, output_model_package_arn, and failure_reason as
    # properties. Note: there is no `training_job_name` (that is a single-turn
    # estimator attribute).
    status = str(job.job_status or "").lower()
    succeeded = status == "completed"

    print()
    print("=" * 60)
    print("Training complete!" if succeeded else "Training did NOT succeed")
    print("=" * 60)
    print(f"  Job name:    {job.job_name}")
    print(f"  Job status:  {job.job_status}")

    if succeeded:
        print(f"  Fine-tuned model: {job.output_model_package_arn}")
        print(f"  Output path:      {job.s3_output_path}")
        print()
        print("The agent has learned to write SQL iteratively:")
        print("  - When to explore the schema")
        print("  - How to debug SQL errors")
        print("  - When to submit vs. keep iterating")
    else:
        # Only surface the "learned" narrative on success. On failure, show
        # what went wrong so it can be debugged.
        print(f"  Failure reason:   {job.failure_reason}")
        print(f"  Output path:      {job.s3_output_path}")
        print()
        print("Check the agent's CloudWatch logs and the MLflow traces for")
        print("this job to diagnose the failure.")

    return job


if __name__ == "__main__":
    train()
