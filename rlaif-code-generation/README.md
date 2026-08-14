# RLAIF News Summarization with Amazon SageMaker

This example demonstrates Reinforcement Learning from AI Feedback (RLAIF) for news article summarization using Amazon SageMaker. It uses the CNN/DailyMail dataset, where the model writes summaries of news articles and gets scored by an AI judge model (not by running tests or checking a single correct answer).

## What is RLAIF?

RLAIF stands for **Reinforcement Learning from AI Feedback**. Here's how it works at a high level:

1. **You have a model** that you want to make better at a task (in our case, summarizing news articles)
2. **You give it prompts** (news articles) and it generates candidate summaries
3. **An AI judge** (a separate, stronger LLM) reads each summary and scores it on quality criteria like faithfulness, conciseness, and coverage
4. **The scores become rewards** that guide the model to produce better summaries over time

This is different from RLVR (which we used for code generation) where rewards come from running code against test cases. Summarization has no "correct answer" you can verify programmatically, so an AI judge is the natural fit.

## Why CNN/DailyMail?

- 300k news articles with human-written summary highlights
- No single "correct" summary exists for any article, making it inherently subjective
- The AI judge can evaluate quality dimensions that no unit test could check:
  - Is the summary faithful to the article (no hallucinated facts)?
  - Does it cover the most important points?
  - Is it concise and well-written?
- Well-studied benchmark in the summarization research community

## Files

| File | Description |
|------|-------------|
| `convert_cnn_to_rlaif.py` | Converts the CNN/DailyMail dataset from HuggingFace into SageMaker's expected JSONL format for RLAIF |
| `train_rlaif.py` | Training script using SageMaker's RLAIFTrainer with an LLM judge |

## Prerequisites

- AWS CLI configured with appropriate credentials
- Python 3.12+
- An IAM role with SageMaker execution permissions (trust policy must include `sagemaker.amazonaws.com`)
- An S3 bucket for training data

```bash
pip install datasets sagemaker
```

## Steps

### 1. Convert the dataset

Downloads CNN/DailyMail from HuggingFace and converts it to the format SageMaker's RLAIF trainer expects.

```bash
python convert_cnn_to_rlaif.py
```

Output: `cnn_rlaif_train.jsonl`

Each record contains:
- A prompt asking the model to summarize an article
- The `reward_model.style` set to `"llmj"` (LLM-as-judge)
- A `reward_model.ground_truth` containing the reference summary for the judge to compare against

### 2. Upload dataset to S3

```bash
aws s3 cp cnn_rlaif_train.jsonl s3://<YOUR_BUCKET>/rlaif-summarization/cnn_rlaif_train.jsonl
```

### 3. Create the Model Package Group

SageMaker needs a Model Package Group to store the fine-tuned model output.

```bash
aws sagemaker create-model-package-group \
    --model-package-group-name summarization-rlaif \
    --model-package-group-description "RLAIF fine-tuned summarization models"
```

### 4. Configure and run training

Edit `train_rlaif.py` and replace the placeholder values:
- `<YOUR_BUCKET>` with your S3 bucket name
- `<YOUR_ACCOUNT_ID>` with your AWS account ID
- `<REGION>` with your AWS region (e.g., `us-east-1`)
- `<YOUR_SAGEMAKER_EXECUTION_ROLE>` with your SageMaker execution role name
- `<YOUR_BEDROCK_JUDGE_MODEL_ID>` with your Bedrock judge model (e.g., `anthropic.claude-sonnet-4-20250514`)

```bash
python train_rlaif.py
```

### 5. Monitor training

Track these metrics in CloudWatch or MLflow:
- **Mean reward**: should climb and plateau (judge is giving higher scores)
- **Policy entropy**: should decrease gradually (model becomes more confident)
- **Mean advantage**: should hover near zero
- **Gradient norm**: should stabilize

## How the Training Loop Works (Step by Step)

```
┌─────────────────────────────────────────────────────────────────┐
│ For each batch of articles:                                      │
│                                                                  │
│  1. ROLLOUT: Model generates N candidate summaries per article   │
│                                                                  │
│  2. JUDGE: An AI judge (e.g., Claude on Bedrock) reads each      │
│     summary and scores it using the "summarize.jinja" template:  │
│     - Is it faithful to the source article?                      │
│     - Does it cover key information?                             │
│     - Is it concise and well-structured?                         │
│                                                                  │
│  3. REWARD: Judge scores become reward signals                   │
│                                                                  │
│  4. UPDATE: Model weights are adjusted (via GRPO) to increase    │
│     probability of generating high-scoring summaries             │
│                                                                  │
│  Repeat until convergence                                        │
└─────────────────────────────────────────────────────────────────┘
```

## Key Hyperparameters

| Parameter | Value | Purpose |
|-----------|-------|---------|
| `judge_prompt_template` | `summarize.jinja` | Tells the judge how to evaluate summaries |
| `rollout_n` | 4 | Candidate summaries generated per article |
| `learning_rate` | 5e-6 | Step size for weight updates |
| `max_epochs` | 2 | Passes through the dataset |
| `global_batch_size` | 128 | Articles per optimizer step |
| `clip_ratio` | 0.2 | Limits policy change per update (GRPO) |
| `lora_rank` | 16 | LoRA adapter dimensionality |
| `lora_alpha` | 32 | LoRA scaling factor |

## RLAIF vs RLVR: When to Use Which

| Scenario | Use RLVR | Use RLAIF |
|----------|----------|-----------|
| Code generation (tests exist) | ✅ | |
| Math problems (answer verifiable) | ✅ | |
| Summarization | | ✅ |
| Creative writing | | ✅ |
| Instruction following | | ✅ |
| Tone/style alignment | | ✅ |
| Safety/harmlessness | | ✅ |

The rule of thumb: if you can write a program to check correctness, use RLVR. If quality is subjective and best described in natural language, use RLAIF.

## References

- [RLAIF: Scaling Reinforcement Learning from Human Feedback with AI Feedback](https://arxiv.org/abs/2309.00267)
- [SageMaker RLAIFTrainer SDK Documentation](https://sagemaker.readthedocs.io/en/stable/api/generated/sagemaker.train.rlaif_trainer.html)
- [CNN/DailyMail Dataset](https://huggingface.co/datasets/abisee/cnn_dailymail)
