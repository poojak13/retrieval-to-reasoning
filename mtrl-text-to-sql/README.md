# Multi-Turn RL: Text-to-SQL Agent with Amazon SageMaker

This example demonstrates Multi-Turn Reinforcement Learning (MTRL) for training a text-to-SQL agent using Amazon SageMaker. The agent receives a natural language question and a database schema, writes SQL queries, executes them against a live database, sees errors or results, and iterates until it produces the correct query.

## What is Multi-Turn RL?

In single-turn RL (like RLVR or RLAIF), the model gets one shot: it generates one response and gets a reward. Multi-turn RL is different. The model operates as an **agent** that takes multiple actions over a sequence of steps:

1. The agent **observes** the current state (question + schema + any prior results)
2. It **takes an action** (writes and executes a SQL query)
3. It **receives feedback** from the environment (query results or an error message)
4. It **decides what to do next** (fix the query, try a different approach, or submit the final answer)
5. At the end of the episode, the **full trajectory** gets a reward

The model learns which sequences of decisions lead to correct answers, not just which individual outputs score well.

## Why Text-to-SQL?

- **Tool setup is simple**: the environment is a SQLite database file (bundled with the agent)
- **Multi-turn is natural**: write query → execute → see error → fix → re-execute
- **Reward is verifiable**: does the final query return the correct result set?
- **Public dataset**: Spider (10k questions across 200 databases) on HuggingFace
- **Clear progression from single-turn**: compare one-shot SQL generation vs. iterative debugging

## How a Single Episode Works

```
Question: "How many departments have a budget over 10 billion?"
Schema: CREATE TABLE department (Department_ID int, Name text, Budget_in_Billions real, ...)

┌─── Turn 1 ────────────────────────────────────────────────┐
│ Agent thinks:                                              │
│   <think>                                                  │
│   I need to count departments where budget > 10.          │
│   Let me write the SQL.                                    │
│   </think>                                                 │
│                                                            │
│ Agent executes:                                            │
│   <sql>                                                    │
│   SELECT COUNT(*) FROM department                          │
│   WHERE Budget_in_Billions > 10                            │
│   </sql>                                                   │
│                                                            │
│ Environment returns:                                       │
│   <observation>                                            │
│   Result: [(3,)]                                           │
│   </observation>                                           │
└────────────────────────────────────────────────────────────┘

┌─── Turn 2 ────────────────────────────────────────────────┐
│ Agent thinks:                                              │
│   <think>                                                  │
│   The query returned 3. That seems reasonable.            │
│   I'm confident in this answer.                            │
│   </think>                                                 │
│                                                            │
│ Agent submits final answer:                                │
│   <solution>                                               │
│   SELECT COUNT(*) FROM department                          │
│   WHERE Budget_in_Billions > 10                            │
│   </solution>                                              │
└────────────────────────────────────────────────────────────┘

Reward: 1.0 (query result matches gold answer)
```

### When the Agent Needs to Iterate (Error Recovery)

```
Question: "List names of departments with above-average budgets"

┌─── Turn 1 ────────────────────────────────────────────────┐
│ Agent executes:                                            │
│   <sql>                                                    │
│   SELECT Name FROM department                              │
│   WHERE Budget_in_Billions > AVG(Budget_in_Billions)       │
│   </sql>                                                   │
│                                                            │
│ Environment returns:                                       │
│   <observation>                                            │
│   Error: misuse of aggregate function AVG()               │
│   </observation>                                           │
└────────────────────────────────────────────────────────────┘

┌─── Turn 2 ────────────────────────────────────────────────┐
│ Agent thinks:                                              │
│   <think>                                                  │
│   I can't use AVG() in WHERE directly.                    │
│   I need a subquery.                                       │
│   </think>                                                 │
│                                                            │
│ Agent executes:                                            │
│   <sql>                                                    │
│   SELECT Name FROM department                              │
│   WHERE Budget_in_Billions > (                             │
│       SELECT AVG(Budget_in_Billions) FROM department       │
│   )                                                        │
│   </sql>                                                   │
│                                                            │
│ Environment returns:                                       │
│   <observation>                                            │
│   Result: [("Treasury",), ("Defense",)]                   │
│   </observation>                                           │
└────────────────────────────────────────────────────────────┘

┌─── Turn 3 ────────────────────────────────────────────────┐
│ Agent submits:                                             │
│   <solution>                                               │
│   SELECT Name FROM department                              │
│   WHERE Budget_in_Billions > (                             │
│       SELECT AVG(Budget_in_Billions) FROM department       │
│   )                                                        │
│   </solution>                                              │
└────────────────────────────────────────────────────────────┘

Reward: 1.0 (correct result after self-correction)
```

## What the Agent Learns

Before training:
- Writes SQL in one shot, often with syntax errors or logic bugs
- Doesn't recover from mistakes (single-turn mindset)
- Doesn't verify results before submitting

After training:
- **Explores** the database when unsure (SELECT * FROM table LIMIT 5)
- **Debugs** errors by reading the error message and fixing the query
- **Verifies** results before submitting (does this make sense?)
- **Knows when to stop** (don't keep querying once you have the answer)

## Files

| File | Description |
|------|-------------|
| `convert_spider_to_mtrl.py` | Converts Spider dataset from HuggingFace into SageMaker's multi-turn RL prompt format |
| `train_mtrl.py` | Training script using SageMaker's MultiTurnRLTrainer |
| `permissions.json` | IAM policy templates for all three roles (copy-paste ready) |
| `sql-agent/` | AgentCore project: agent code + bundled SQLite databases |

## Prerequisites

- AWS CLI configured with appropriate credentials
- Python 3.12+
- Node.js (for AgentCore CLI)
- An S3 bucket for training data
- Spider dataset download (for the `database/` folder of SQLite files): https://yale-lily.github.io/spider

```bash
pip install datasets sagemaker pyarrow
npm install -g @aws/agentcore
```

## IAM Setup

Multi-turn RL requires **three roles** with specific permissions. See [`permissions.json`](./permissions.json) for the full policy templates.

| Role | What it does | Key permissions |
|------|-------------|-----------------|
| Caller (your IAM user or Studio role) | Submits the training job | `sagemaker:CreateJob`, `iam:PassRole` |
| SageMaker Execution Role | Runs the training job | `AmazonSageMakerJobFullAccess`, must trust `job.sagemaker.amazonaws.com` |
| AgentCore Runtime Role | Runs your deployed agent | `AmazonSageMakerJobRuntimeAccess`, must trust `bedrock-agentcore.amazonaws.com` |

Setup commands (replace placeholders):

```bash
aws iam attach-role-policy --role-name <EXECUTION_ROLE_NAME> --policy-arn arn:aws:iam::aws:policy/AmazonSageMakerJobFullAccess
```

```bash
aws iam attach-role-policy --role-name <AGENTCORE_ROLE_NAME> --policy-arn arn:aws:iam::aws:policy/AmazonSageMakerJobRuntimeAccess
```

Update the execution role trust policy (IAM Console > Roles > your role > Trust relationships > Edit):
- Add `job.sagemaker.amazonaws.com` with actions `sts:AssumeRole` and `sts:TagSession`

## Steps

### 1. Download Spider data

Download from https://yale-lily.github.io/spider and extract. You need the
`database/` folder, placed inside `sql-agent/sqlagent/app/sqlagentmtrl/`. Each
database ships as a `<db_id>/<db_id>.sqlite` file alongside a `schema.sql`; the
agent executes queries against the `.sqlite` file, and the conversion script
reads CREATE TABLE statements from the `schema.sql`.

### 2. Convert the dataset

```bash
python convert_spider_to_mtrl.py
```

This reads the CREATE TABLE statements from each bundled `schema.sql` and creates
`spider_mtrl_train.parquet` with the real schema in each prompt. Only questions
whose database is bundled under the agent's `database/` folder are included, so
the dataset stays in sync with what the deployed agent can actually query.

### 3. Upload dataset to S3

```bash
aws s3 cp spider_mtrl_train.parquet s3://<YOUR_BUCKET>/mtrl-text2sql/spider_mtrl_train.parquet
```

### 4. Deploy the Agent

From inside `sql-agent/sqlagent/`:

```bash
agentcore deploy
```

Note the Runtime ARN from the output. Update `AGENT_ENV_ARN` in `train_mtrl.py`.

### 5. Configure and run training

Edit `train_mtrl.py` and replace the placeholder values:
- `<YOUR_BUCKET>` with your S3 bucket name
- `AGENT_ENV_ARN` with the ARN from step 4
- `ROLE_ARN` with your SageMaker execution role ARN

```bash
python train_mtrl.py
```

### 6. Monitor training

Track these metrics:
- **Mean reward**: should climb (agent is solving more questions correctly)
- **Episode length (turns)**: should stabilize or decrease (agent learns to be efficient)
- **Policy entropy**: should decrease gradually
- **KL divergence**: should climb alongside reward

## Key Hyperparameters

| Parameter | Value | Purpose |
|-----------|-------|---------|
| `max_epochs` | 1 | Passes through the dataset |
| `global_batch_size` | 32 | Episodes per optimizer step |
| `learning_rate` | 1e-5 | Step size for weight updates |
| `max_turns` | 5 | Maximum turns per episode before timeout |

## MTRL vs Single-Turn RL: When to Use Which

| Scenario | Single-Turn (RLVR/RLAIF) | Multi-Turn RL |
|----------|--------------------------|---------------|
| One-shot code generation | ✅ | |
| SQL with no iteration | ✅ | |
| SQL with debugging/iteration | | ✅ |
| Agent with tool use | | ✅ |
| Summarization | ✅ | |
| Multi-step reasoning | | ✅ |
| Tasks requiring error recovery | | ✅ |

The rule: if the task benefits from the model seeing the result of its action and adapting, use multi-turn RL. If one shot is enough, single-turn is simpler and cheaper.

## References

- [SageMaker Multi-Turn RL Documentation](https://docs.aws.amazon.com/sagemaker/latest/dg/model-customize-mtrl.html)
- [Spider Dataset](https://huggingface.co/datasets/xlangai/spider)
- [Best Practices for Multi-Turn RL](https://aws.amazon.com/blogs/machine-learning/best-practices-for-multi-turn-reinforcement-learning-in-amazon-sagemaker-ai/)
