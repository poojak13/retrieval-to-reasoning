# RLVR Code Generation with Amazon SageMaker

This example demonstrates Reinforcement Learning with Verifiable Rewards (RLVR) for code generation using Amazon SageMaker. It uses the MBPP (Mostly Basic Python Programs) dataset, where the model writes Python functions and gets scored by running test cases.

## Files

| File | Description |
|------|-------------|
| `convert_mbpp_to_rlvr.py` | Converts the MBPP dataset from HuggingFace into SageMaker's expected JSONL format |
| `custom_reward_lambda.py` | Custom AWS Lambda reward function that scores on correctness (80%) + conciseness (20%) |
| `train_rlvr.py` | Training script with two options: built-in preset reward or custom Lambda |

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

Downloads MBPP from HuggingFace and converts it to the VERL format SageMaker expects.

```bash
python convert_mbpp_to_rlvr.py
```

Output: `mbpp_rlvr_train.jsonl`

### 2. Upload dataset to S3

```bash
aws s3 cp mbpp_rlvr_train.jsonl s3://<YOUR_BUCKET>/rlvr-code/mbpp_rlvr_train.jsonl
```

### 3. Create the Model Package Group

SageMaker needs a Model Package Group to store the fine-tuned model output.

```bash
aws sagemaker create-model-package-group \
    --model-package-group-name code-gen-rlvr \
    --model-package-group-description "RLVR fine-tuned code generation models"
```

### 4. (Optional) Deploy the custom reward Lambda

Skip this step if using the built-in `prime_code` preset.

#### 4a. Create an IAM role for Lambda

```bash
echo '{"Version":"2012-10-17","Statement":[{"Effect":"Allow","Principal":{"Service":"lambda.amazonaws.com"},"Action":"sts:AssumeRole"}]}' > trust-policy.json

aws iam create-role \
    --role-name mbpp-reward-lambda-role \
    --assume-role-policy-document file://trust-policy.json

aws iam attach-role-policy \
    --role-name mbpp-reward-lambda-role \
    --policy-arn arn:aws:iam::aws:policy/service-role/AWSLambdaBasicExecutionRole
```

#### 4b. Package and deploy the Lambda

```bash
# On Linux/Mac:
zip reward_function.zip custom_reward_lambda.py

# On Windows (PowerShell):
Compress-Archive -Path custom_reward_lambda.py -DestinationPath reward_function.zip -Force
```

```bash
aws lambda create-function \
    --function-name mbpp-reward \
    --runtime python3.12 \
    --handler custom_reward_lambda.lambda_handler \
    --role arn:aws:iam::<YOUR_ACCOUNT_ID>:role/mbpp-reward-lambda-role \
    --zip-file fileb://reward_function.zip \
    --timeout 60 \
    --memory-size 256
```

#### 4c. Grant SageMaker permission to invoke the Lambda

Ensure your SageMaker execution role has `lambda:InvokeFunction` permission on this Lambda.

### 5. Configure and run training

Edit `train_rlvr.py` and replace the placeholder values:
- `<YOUR_BUCKET>` with your S3 bucket name
- `<YOUR_ACCOUNT_ID>` with your AWS account ID
- `<REGION>` with your AWS region (e.g., `us-east-1`)
- `<YOUR_SAGEMAKER_EXECUTION_ROLE>` with your SageMaker execution role name

Set `USE_PRESET = True` for the built-in reward, or `False` for the custom Lambda.

```bash
python train_rlvr.py
```

### 6. Monitor training

Track these metrics in CloudWatch or MLflow:
- **Mean reward**: should climb and plateau
- **Policy entropy**: should decrease gradually (not crash to zero)
- **Mean advantage**: should hover near zero
- **Gradient norm**: should stabilize

## Reward Function Options

### Option A: Built-in `prime_code` (no Lambda needed)

Runs inside the training container. Executes the model's code against test cases and returns the pass ratio. Zero network overhead.

### Option B: Custom Lambda

Adds custom scoring logic (e.g., conciseness penalty). The training container invokes the Lambda once per sample via network call. More flexible but adds latency at scale.

## Key Hyperparameters

| Parameter | Value | Purpose |
|-----------|-------|---------|
| `rollout_n` | 8 | Candidate responses per prompt |
| `learning_rate` | 5e-6 | Step size for weight updates |
| `max_epochs` | 3 | Passes through the dataset |
| `global_batch_size` | 128 | Prompts per optimizer step |
| `clip_ratio` | 0.2 | Limits policy change per update (GRPO) |
| `lora_rank` | 16 | LoRA adapter dimensionality |
| `lora_alpha` | 32 | LoRA scaling factor |

## Notes

- The `preset_reward_function` hyperparameter is documented but may not be settable via the Python SDK's hyperparameters object. The built-in preset is used automatically when no `custom_reward_function` is provided.
- When running from a local machine (not a SageMaker notebook), you must pass `role=ROLE_ARN` explicitly to the trainer.
