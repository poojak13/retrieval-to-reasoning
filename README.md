# From Retrieval to Reasoning

Companion code for the [From Retrieval to Reasoning](https://poojak13.substack.com/) series on Substack.

This series explores how to move beyond basic LLM usage into production-grade model customization on AWS, starting with retrieval-augmented generation (RAG) and progressing through reinforcement learning techniques that teach models to reason.

## Examples

| Folder | Topic | Substack Post |
|--------|-------|---------------|
| [`rlvr-code-generation/`](./rlvr-code-generation/) | RLVR for code generation using MBPP dataset on SageMaker | [From Retrieval to Reasoning - Part 2](https://poojak13.substack.com/p/from-retrieval-to-reasoning-part) |
| [`rlaif-code-generation/`](./rlaif-code-generation/) | RLAIF for news summarization using CNN/DailyMail on SageMaker | Coming soon |
| [`mtrl-text-to-sql/`](./mtrl-text-to-sql/) | Multi-Turn RL for text-to-SQL agent using Spider dataset on SageMaker | Coming soon |

## What's Covered

- **Part 1 (RAG)**: Giving models knowledge without modifying weights
- **Part 2 (RLVR)**: Teaching models to write code that passes tests using reinforcement learning with verifiable rewards
- **Part 3 (RLAIF)**: Using LLM-as-a-judge for subjective quality improvements (news summarization)
- **Part 4 (MTRL)**: Multi-turn agent training with tool use (text-to-SQL with iterative debugging)

## The RL Progression

Each example builds on the previous one, showing when and why to use different RL techniques:

| Technique | Reward Source | When to Use | Example |
|-----------|--------------|-------------|---------|
| RLVR | Programmatic verification (test cases) | Task has objectively correct answers | Code generation |
| RLAIF | AI judge (LLM evaluates output) | Quality is subjective, no verifier exists | Summarization |
| MTRL | Outcome-based (end of multi-step episode) | Task requires iteration, tool use, error recovery | Text-to-SQL agent |

## Getting Started

Each folder is self-contained with its own README and setup instructions. Pick the example that matches the post you're reading.

### Prerequisites (all examples)

- AWS CLI configured with appropriate credentials
- Python 3.12+
- An IAM role with SageMaker execution permissions
- An S3 bucket for training data

### Additional for MTRL

- Node.js (for AgentCore CLI)
- AgentCore CLI: `npm install -g @aws/agentcore`
- Spider dataset download (for database schemas and SQLite files)
- See [`mtrl-text-to-sql/permissions.json`](./mtrl-text-to-sql/permissions.json) for required IAM policies
