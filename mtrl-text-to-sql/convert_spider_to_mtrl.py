"""
convert_spider_to_mtrl.py

Converts the Spider text-to-SQL dataset from HuggingFace into the format
expected by SageMaker's MultiTurnRLTrainer.

=== About the Spider Dataset ===

Spider is a text-to-SQL benchmark with:
- ~7000 training examples and ~1000 dev examples
- 200 different database schemas (cross-domain)
- Each example has: a natural language question, a database schema, and
  the gold (correct) SQL query

Fields in Spider:
- db_id: which database this question is about (e.g., "concert_singer")
- question: the natural language question (e.g., "How many singers are there?")
- query: the correct SQL answer (e.g., "SELECT COUNT(*) FROM singer")
- question_toks: tokenized question (we don't need this)
- query_toks: tokenized query (we don't need this)

Spider does NOT include the actual database schemas in the HuggingFace version.
We need to download those separately from the Spider GitHub repo to include
CREATE TABLE statements in our prompts.

=== What this script produces ===

A Parquet file where each row has a single "prompt" column containing a
JSON string. This JSON string is what SageMaker passes to the agent at the
start of each episode.

The prompt includes:
- The system instructions (telling the agent how to behave)
- The user message (question + schema)
- The reward spec (gold SQL for evaluation)
- The env_class (tells SageMaker which tool environment to use)

=== Why Parquet? ===

SageMaker's MultiTurnRLTrainer recommends Parquet for efficient storage
and fast loading. JSONL also works, but Parquet is better for larger datasets.

Usage:
    pip install datasets pyarrow
    python convert_spider_to_mtrl.py

Output:
    spider_mtrl_train.parquet
"""

import json
import os
import re

import pyarrow as pa
import pyarrow.parquet as pq
from datasets import load_dataset


# The agent can only execute SQL against databases whose .sqlite file is
# bundled into the deployed AgentCore runtime. Prompts for databases that
# aren't bundled would always error at SQL execution time. We also read the
# per-database schema.sql from here, so this one directory is the single
# source of truth for which databases exist and what their schemas are.
AGENT_DB_DIR = os.path.join(
    "sql-agent", "sqlagent", "app", "sqlagentmtrl", "database"
)


# How many examples to include. Spider has ~7000 training examples.
# Multi-turn RL is expensive (each episode runs multiple turns with tool calls),
# so start small. You can scale up once you've validated the pipeline.
MAX_SAMPLES = 500

# The system prompt that tells the agent how to behave during training.
# This is critical: it defines the agent's action format, when to think,
# when to execute SQL, and when to submit the final answer.
SYSTEM_PROMPT = """Task Overview:
You are a data science expert. Your task is to understand the database schema and
generate a valid SQL query to answer the given question.

Instructions:
- Think through the problem step by step before writing SQL.
- You can execute SQL queries to explore the database or verify your logic.
- If your query returns an error, read the error message and fix your approach.
- You have a maximum of 5 turns to produce the correct query.

Format:
- Conduct your reasoning inside <think>...</think> blocks.
- Execute exploratory or verification queries using <sql>your query</sql>.
- The execution result will appear inside <observation>...</observation>.
- When you are confident in your final answer, submit it inside <solution>...</solution>.

Important:
- Only use tables and columns that exist in the provided schema.
- If you get an error, do NOT repeat the same query. Fix it.
- Submit your final answer as soon as you are confident. Do not waste turns."""


def build_prompt_json(question: str, db_id: str, gold_query: str, schema: str) -> str:
    """
    Build the full prompt JSON that SageMaker passes to the agent.

    This JSON string contains everything the agent and environment need:
    - prompt: the conversation messages (system instructions + user question)
    - env_class: tells SageMaker which tool environment to activate
    - reward_spec: the gold SQL for the reward function to compare against
    - extra_info: metadata for logging and debugging

    Args:
        question: The natural language question
        db_id: The database identifier
        gold_query: The correct SQL query (used by reward function)
        schema: The CREATE TABLE statements for this database

    Returns:
        A JSON string (this whole string becomes one cell in the Parquet file)
    """
    prompt_data = {
        # Conversation messages the agent sees at episode start
        "prompt": [
            {
                "role": "system",
                "content": SYSTEM_PROMPT,
            },
            {
                "role": "user",
                "content": (
                    f"Database Schema:\n{schema}\n\n"
                    f"Question: {question}"
                ),
            },
        ],
        # Tells SageMaker which environment/tool to provide
        # "text2sql" activates the SQL execution tool
        "env_class": "text2sql",
        # The reward function uses this to evaluate the agent's final answer
        "reward_spec": {
            "ground_truth": gold_query,
            "style": "rule",
        },
        # Metadata for logging (not used in training)
        "extra_info": {
            "db_id": db_id,
            "question": question,
            "split": "train",
        },
    }

    return json.dumps(prompt_data)


def extract_create_statements(schema_sql: str) -> str:
    """Pull just the CREATE TABLE statements out of a full schema.sql file.

    The bundled schema.sql files contain PRAGMA lines, CREATE TABLE blocks,
    and INSERT rows. The model only needs the CREATE TABLE statements to write
    correct SQL, so we strip the rest to keep the prompt focused.
    """
    statements = []
    # Match "CREATE TABLE ... ( ... );" including newlines inside the parens.
    for match in re.finditer(
        r"CREATE TABLE.*?\(.*?\);", schema_sql, re.DOTALL | re.IGNORECASE
    ):
        statements.append(match.group(0).strip())
    return "\n\n".join(statements)


def load_schemas() -> dict:
    """
    Load real CREATE TABLE schemas from the agent's bundled schema.sql files.

    Each bundled database ships a schema.sql (full DDL + data) next to its
    .sqlite file, at:
        AGENT_DB_DIR/<db_id>/schema.sql
    We read the CREATE TABLE statements straight from those, which guarantees
    the schema in the prompt matches the database the agent actually queries.

    Returns:
        A dict mapping db_id -> schema string (CREATE TABLE statements only)
    """
    schemas = {}

    if not os.path.isdir(AGENT_DB_DIR):
        print(f"  WARNING: agent database dir not found at {AGENT_DB_DIR}")
        return schemas

    for db_id in os.listdir(AGENT_DB_DIR):
        schema_path = os.path.join(AGENT_DB_DIR, db_id, "schema.sql")
        if not os.path.isfile(schema_path):
            continue
        with open(schema_path, "r", encoding="utf-8") as f:
            schema_sql = f.read()
        create_only = extract_create_statements(schema_sql)
        if create_only:
            schemas[db_id] = create_only

    print(f"  Loaded real schemas for {len(schemas)} databases from schema.sql files")
    return schemas


def build_schema_from_query(query: str, db_id: str) -> str:
    """
    Fallback schema when no schema.sql is available for a database.

    This should not trigger in normal use, because we only keep examples for
    databases that are bundled (and bundled databases have schema.sql). It's a
    clearly-labeled last resort so a missing schema is obvious rather than silent.
    """
    return f"-- Database: {db_id}\n-- WARNING: no schema.sql found; schema unknown"


def get_available_db_ids() -> set:
    """Return the set of db_ids that have a bundled .sqlite file in the agent.

    Each database lives at AGENT_DB_DIR/<db_id>/<db_id>.sqlite. We only keep
    a db_id if that file actually exists, so the dataset stays in sync with
    what the deployed agent can open.
    """
    available = set()
    if not os.path.isdir(AGENT_DB_DIR):
        print(f"  WARNING: agent database dir not found at {AGENT_DB_DIR}")
        return available

    for db_id in os.listdir(AGENT_DB_DIR):
        sqlite_path = os.path.join(AGENT_DB_DIR, db_id, f"{db_id}.sqlite")
        if os.path.isfile(sqlite_path):
            available.add(db_id)

    print(f"  Found {len(available)} bundled databases: {sorted(available)}")
    return available


def main():
    # Step 1: Load Spider dataset from HuggingFace
    print("Loading Spider dataset from HuggingFace...")
    dataset = load_dataset("xlangai/spider", split="train")
    print(f"  Total examples available: {len(dataset)}")

    # Step 2: Load database schemas
    # In production, this would load actual CREATE TABLE statements
    print("Loading database schemas...")
    schemas = load_schemas()

    # Step 2b: Discover which databases the agent can actually query.
    print("Discovering bundled agent databases...")
    available_db_ids = get_available_db_ids()
    if not available_db_ids:
        print(
            "  ERROR: no bundled databases found. The agent cannot execute SQL "
            "for any prompt. Add .sqlite files under the agent's database/ dir "
            "before generating the dataset."
        )

    # Step 3: Convert each example to MTRL format
    print(f"\nConverting up to {MAX_SAMPLES} examples (bundled databases only)...")

    prompts = []
    skipped = 0
    skipped_missing_db = 0

    for i, row in enumerate(dataset):
        if len(prompts) >= MAX_SAMPLES:
            break

        question = row["question"]
        db_id = row["db_id"]
        gold_query = row["query"]

        # Skip examples with empty questions or queries
        if not question.strip() or not gold_query.strip():
            skipped += 1
            continue

        # Skip examples whose database is not bundled in the agent. Without the
        # .sqlite file, the agent's SQL execution always errors and the episode
        # can never succeed.
        if db_id not in available_db_ids:
            skipped_missing_db += 1
            continue

        # Get schema for this database
        schema = schemas.get(db_id, build_schema_from_query(gold_query, db_id))

        # Build the full prompt JSON
        prompt_json = build_prompt_json(question, db_id, gold_query, schema)
        prompts.append(prompt_json)

    # Step 4: Write to Parquet
    # SageMaker reads the "prompt" column and passes each value to the agent
    output_file = "spider_mtrl_train.parquet"

    table = pa.table({"prompt": prompts})
    pq.write_table(table, output_file)

    print(f"\nDone!")
    print(f"  Converted:            {len(prompts)} examples")
    print(f"  Skipped (empty):      {skipped}")
    print(f"  Skipped (no local DB):{skipped_missing_db}")
    print(f"  Output:               {output_file}")

    if not prompts:
        print(
            "\n  WARNING: 0 examples written. None of the first "
            f"{MAX_SAMPLES} scanned Spider rows matched a bundled database. "
            "Either bundle more .sqlite files or raise MAX_SAMPLES so the scan "
            "reaches rows for your bundled databases."
        )

    print(f"\nNext step:")
    print(f"  aws s3 cp {output_file} s3://<YOUR_BUCKET>/mtrl-text2sql/{output_file}")

    # Print a sample to verify format
    if prompts:
        print(f"\n{'='*60}")
        print("Sample prompt (first record, parsed):")
        print("=" * 60)
        sample = json.loads(prompts[0])
        print(json.dumps(sample, indent=2)[:2000])


if __name__ == "__main__":
    main()
