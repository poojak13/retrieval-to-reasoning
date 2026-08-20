"""
main.py

Text-to-SQL agent for multi-turn RL training on SageMaker.

This agent:
1. Receives a question + database schema from SageMaker
2. Connects to the policy model (the model being trained) via RFT Runtime
3. Runs a multi-turn loop: model generates SQL, we execute it, model sees results
4. Returns a reward based on whether the final SQL produces correct results

Deployed to AgentCore via: agentcore deploy
"""

import json
import os
import re
import sqlite3
import time

import botocore.session
from bedrock_agentcore.runtime import BedrockAgentCoreApp
import openai
from sagemaker.core.token_generator import generate_token, CredentialProvider
from sagemaker.train.rft import sagemaker_rft_handler
from sagemaker.train.rft.headers import make_inference_headers


app = BedrockAgentCoreApp()
log = app.logger

# ============================================================
# CONFIGURATION
# ============================================================

MAX_TURNS = 5
DB_BASE_PATH = os.environ.get("DB_BASE_PATH", os.path.join(os.path.dirname(__file__), "database"))

# Strict submission by default: only an explicit <solution> earns reward. This
# is what we want during training so the model learns to submit. Set
# ALLOW_LAST_SQL_FALLBACK=1 in the environment (e.g. for offline eval) to
# leniently score the last successful query when the model never submits.
ALLOW_LAST_SQL_FALLBACK = os.environ.get("ALLOW_LAST_SQL_FALLBACK", "0") == "1"

AWS_REGION = os.environ.get("AWS_REGION", "us-east-1")

# Default RFT Runtime endpoint. During training SageMaker passes the endpoint in
# the payload metadata; this is the fallback the docs recommend.
DEFAULT_RFT_ENDPOINT = f"https://job-runtime.sagemaker.{AWS_REGION}.api.aws"

# Reuse a single botocore Session across invocations in this container. The
# AgentCore runtime supplies credentials via the container credential provider,
# but that provider can transiently return None on a cold start. Creating the
# Session once lets the credential chain initialize and cache. We use botocore
# directly (guaranteed dependency) rather than boto3.
_BOTO_SESSION = botocore.session.get_session()

# One-time startup diagnostic: does this container have AWS credentials at boot?
# If this logs False consistently, the problem is credential DELIVERY to the
# AgentCore runtime (role/identity/metadata), not a cold-start race, and the
# retry in generate_rft_token() is only a band-aid. If it logs True, creds are
# resolving and any later "No AWS credentials" is a transient timing issue.
try:
    _startup_creds = _BOTO_SESSION.get_credentials()
    log.info(
        "STARTUP credential check: credentials_present=%s region=%s",
        _startup_creds is not None,
        AWS_REGION,
    )
except Exception as _e:  # noqa: BLE001
    log.warning("STARTUP credential check raised: %s", _e)


class _CachedSessionCredentialProvider(CredentialProvider):
    """Feeds generate_token the credentials from our cached botocore session.

    IMPORTANT: generate_token(region=...) with no provider creates its OWN
    fresh botocore Session() internally and calls get_credentials() on it.
    That fresh session must re-resolve credentials from scratch every call,
    and under concurrency it intermittently returns None -> "No AWS credentials
    found". Our module-level _BOTO_SESSION resolved and cached credentials at
    startup (confirmed: credentials_present=True), so we pass those cached
    credentials in explicitly instead of letting generate_token do a fresh,
    flaky lookup.
    """

    def load(self):
        return _BOTO_SESSION.get_credentials()


_CACHED_CRED_PROVIDER = _CachedSessionCredentialProvider()


def generate_rft_token(max_attempts: int = 5, base_delay: float = 0.5) -> str:
    """Generate an RFT bearer token using cached credentials, with retries.

    Two-part fix for the intermittent "No AWS credentials found":
    1. Pass a provider backed by the cached _BOTO_SESSION so generate_token
       does NOT spin up a fresh (flaky) Session per call.
    2. Retry with exponential backoff as a safety net if credentials are
       briefly unavailable anyway.
    """
    last_err: Exception | None = None
    for attempt in range(max_attempts):
        try:
            token = generate_token(
                region=AWS_REGION,
                aws_credentials_provider=_CACHED_CRED_PROVIDER,
            )
            if attempt > 0:
                log.info("Credentials resolved after %d retrie(s)", attempt)
            return token
        except RuntimeError as e:
            last_err = e
            sleep_s = base_delay * (2 ** attempt)
            log.warning(
                "generate_token failed (attempt %d/%d): %s; retrying in %.1fs",
                attempt + 1,
                max_attempts,
                e,
                sleep_s,
            )
            time.sleep(sleep_s)
    raise RuntimeError(
        f"No AWS credentials found after {max_attempts} attempts. "
        f"Last error: {last_err}"
    )


# ============================================================
# SQL EXECUTION TOOL
# ============================================================

def execute_sql(query: str, db_id: str) -> str:
    """
    Execute SQL against a Spider SQLite database.
    Returns results as a string, or an error message.
    """
    db_path = os.path.join(DB_BASE_PATH, db_id, f"{db_id}.sqlite")

    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        cursor.execute(query)
        results = cursor.fetchall()
        conn.close()

        if not results:
            return "Query executed successfully. No rows returned."

        output = f"Result ({len(results)} rows):\n"
        for row in results[:20]:
            output += f"  {row}\n"
        if len(results) > 20:
            output += f"  ... ({len(results) - 20} more rows)\n"
        return output

    except Exception as e:
        return f"Error: {str(e)}"


# ============================================================
# PARSING HELPERS
# ============================================================

def extract_sql(response: str | None) -> str | None:
    """Extract SQL from <sql>...</sql> tags."""
    if not response:
        return None
    match = re.search(r"<sql>(.*?)</sql>", response, re.DOTALL)
    return match.group(1).strip() if match else None


def extract_solution(response: str | None) -> str | None:
    """Extract final answer from <solution>...</solution> tags."""
    if not response:
        return None
    match = re.search(r"<solution>(.*?)</solution>", response, re.DOTALL)
    return match.group(1).strip() if match else None


def tag_trace(**attrs) -> None:
    """Attach per-episode attributes to the current trace span.

    These attributes show up on the MLflow/OTel trace for this rollout, so you
    can filter and group episodes by reward, verdict, db_id, etc. in the trace
    UI, rather than digging them out of the returned dict or CloudWatch.

    Guarded: if the OpenTelemetry API isn't available or there's no active
    span, this is a no-op and never breaks the rollout.
    """
    try:
        from opentelemetry import trace as _otel_trace

        span = _otel_trace.get_current_span()
        if span is None:
            return
        for key, value in attrs.items():
            if value is None:
                value = ""
            # OTel attributes must be primitives; stringify anything else.
            if not isinstance(value, (str, bool, int, float)):
                value = str(value)
            span.set_attribute(f"episode.{key}", value)
    except Exception:  # noqa: BLE001
        # Tracing is best-effort telemetry; never fail the episode over it.
        pass


def extract_tool_call_sql(message) -> str | None:
    """Extract SQL from a native OpenAI-style tool call, if present.

    The policy model often calls SQL as a structured tool call instead of
    emitting <sql>...</sql> text. In that case message.content is empty and the
    query lives in message.tool_calls. We must read it from there, otherwise
    the query is silently ignored and the episode runs 0 SQL queries.

    Handles both argument shapes we've observed:
      - "arguments": "SELECT ..."                 (raw string)
      - "arguments": "{\"sql\": \"SELECT ...\"}"  (JSON object with a sql field)
    """
    tool_calls = getattr(message, "tool_calls", None)
    if not tool_calls:
        return None

    for tc in tool_calls:
        fn = getattr(tc, "function", None)
        if fn is None:
            continue
        raw_args = getattr(fn, "arguments", None)
        if not raw_args:
            continue
        # arguments is usually a JSON string; sometimes already a dict.
        if isinstance(raw_args, str):
            try:
                parsed = json.loads(raw_args)
            except json.JSONDecodeError:
                # Not JSON: treat the whole string as the SQL query.
                return raw_args.strip()
        else:
            parsed = raw_args

        if isinstance(parsed, dict):
            # Look for a sql/query field; fall back to the first string value.
            for key in ("sql", "query", "statement"):
                if isinstance(parsed.get(key), str):
                    return parsed[key].strip()
            for v in parsed.values():
                if isinstance(v, str):
                    return v.strip()
        elif isinstance(parsed, str):
            return parsed.strip()

    return None


# ============================================================
# REWARD COMPUTATION
# ============================================================

def compute_reward(
    final_sql: str | None, gold_query: str, db_id: str
) -> tuple[float, str]:
    """
    Score the episode and explain why.

    Returns (reward, reason):
      1.0  = correct   -> "correct: result matches gold"
      0.0  = wrong     -> "incorrect: result does not match gold"
      0.0  = SQL error -> "final SQL failed to execute: <error>"
     -0.1  = no submit -> "no <solution> submitted within turn limit"
    """
    if final_sql is None:
        return -0.1, "no <solution> submitted within turn limit"

    db_path = os.path.join(DB_BASE_PATH, db_id, f"{db_id}.sqlite")

    if not os.path.isfile(db_path):
        return 0.0, f"database file not found for db_id '{db_id}' ({db_path})"

    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()

        cursor.execute(final_sql)
        predicted = set(map(tuple, cursor.fetchall()))

        cursor.execute(gold_query)
        gold = set(map(tuple, cursor.fetchall()))

        conn.close()

        if predicted == gold:
            return 1.0, "correct: result matches gold"
        return 0.0, "incorrect: result does not match gold"

    except Exception as e:
        return 0.0, f"final SQL failed to execute: {e}"


# ============================================================
# AGENT ENTRY POINT
# ============================================================

@app.entrypoint
@sagemaker_rft_handler
async def invoke(payload):
    """
    Called by SageMaker during MTRL training, once per episode.

    Runs the full multi-turn interaction:
    - Parse prompt (question + schema)
    - Connect to the policy model via RFT Runtime
    - Loop: model generates -> execute SQL -> observe -> repeat
    - Compute and return reward

    Decorator order matters:
    - @app.entrypoint (outer) registers the HTTP route AgentCore invokes.
    - @sagemaker_rft_handler (inner) wires this rollout into the SageMaker RFT
      Runtime: it establishes rollout tracking and, on return, issues
      CompleteRollout + UpdateReward using the "reward" field we return.

    Without @sagemaker_rft_handler, the Sample calls below are never correlated
    to the rollout and the trainer fails with:
    "No sampling requests were received for this rollout".

    Signature note: @sagemaker_rft_handler wraps this in a single-argument
    handler (it calls func(payload)). The entrypoint must therefore accept
    ONLY `payload`. Adding a second `context` parameter causes AgentCore to
    fail every invocation with:
    "TypeError: invoke() takes 1 positional argument but 2 were given".
    """
    log.info("Starting MTRL episode...")

    # --- Parse the incoming payload ---
    metadata = payload.get("metadata", {})
    prompt_raw = payload.get("prompt", "")
    inference_params = payload.get("inferenceParams", {})

    # NOTE: The @sagemaker_rft_handler decorator already calls
    # set_rollout_context(metadata, inference_params) before this function
    # runs, which is what correlates every Sample call to this rollout.
    # Do NOT call set_rollout_context(metadata) here: passing metadata alone
    # would overwrite the decorator's context and drop the inference params.

    # The prompt is a JSON string from our Parquet dataset
    try:
        prompt_data = json.loads(prompt_raw)
    except (json.JSONDecodeError, TypeError):
        prompt_data = {"prompt": [{"role": "user", "content": prompt_raw}]}

    messages = prompt_data.get("prompt", [])
    reward_spec = prompt_data.get("reward_spec", {})
    extra_info = prompt_data.get("extra_info", {})

    gold_query = reward_spec.get("ground_truth", "")
    db_id = extra_info.get("db_id", "")

    # --- Connect to the policy model (model being trained) ---
    # Prefer the endpoint SageMaker injects in metadata; fall back to the env
    # var, then to the documented default. Never let this be empty, otherwise
    # base_url becomes "/v1" and every inference call fails silently.
    endpoint = (
        metadata.get("endpoint")
        or os.environ.get("RFT_RUNTIME_ENDPOINT")
        or DEFAULT_RFT_ENDPOINT
    )
    token = generate_rft_token()
    headers = make_inference_headers(metadata)

    log.info(f"Using RFT Runtime endpoint: {endpoint}")

    client = openai.OpenAI(
        base_url=endpoint.rstrip("/") + "/v1",
        api_key=token,
        default_headers=headers,
    )

    # --- Multi-turn loop ---
    conversation = list(messages)
    final_sql = None
    num_turns = 0
    sql_attempts = 0          # how many <sql> the agent ran
    sql_errors = 0            # how many of those errored
    empty_responses = 0       # turns where the model returned no text
    last_observation = ""     # most recent tool result/error the agent saw
    last_successful_sql = None  # last <sql> that executed without error
    submitted = False         # did the model explicitly submit a <solution>?

    try:
        for turn in range(MAX_TURNS):
            num_turns += 1

            # Call the policy model
            response = client.chat.completions.create(
                model="default",
                messages=conversation,
                max_tokens=inference_params.get("maxTokens", 1024),
                temperature=inference_params.get("temperature", 1.0),
                top_p=inference_params.get("topP", 1.0),
            )

            message = response.choices[0].message

            # message.content can be None (empty generation, or the SQL was
            # emitted as a native tool_call rather than text). Coerce to "" so
            # the regex extractors and conversation history never receive None.
            assistant_msg = message.content or ""
            if not assistant_msg:
                empty_responses += 1
            log.info(
                "Turn %d: content_len=%d chars, tool_calls=%s",
                num_turns,
                len(assistant_msg),
                bool(getattr(message, "tool_calls", None)),
            )
            conversation.append({"role": "assistant", "content": assistant_msg})

            # Check if model submitted final answer (text <solution>)
            solution = extract_solution(assistant_msg)
            if solution:
                final_sql = solution
                submitted = True
                log.info("Agent submitted solution at turn %d", num_turns)
                break

            # Find SQL from either a <sql> text tag OR a native tool_call.
            # The model uses both forms, so we must handle both or its queries
            # get silently dropped (leading to "Ran 0 SQL queries").
            tag_sql = extract_sql(assistant_msg)
            sql = tag_sql or extract_tool_call_sql(message)

            # If the SQL came from a tool_call (empty text content), record it
            # in the transcript so the model sees its own prior query on the
            # next turn. Otherwise the assistant turn is blank and the model
            # loses context.
            if sql and not tag_sql and not assistant_msg:
                conversation[-1]["content"] = f"<sql>\n{sql}\n</sql>"

            if sql:
                sql_attempts += 1
                result = execute_sql(sql, db_id)
                errored = result.startswith("Error:")
                if errored:
                    sql_errors += 1
                else:
                    last_successful_sql = sql
                last_observation = result

                turns_left = MAX_TURNS - num_turns
                # After every query, explicitly remind the model that it must
                # wrap its final answer in <solution>...</solution>. Without
                # this, the model tends to keep running <sql> forever and never
                # submits, so the episode ends with no answer (reward -0.1).
                if errored:
                    guidance = (
                        "Fix the query based on the error above. When your query "
                        "is correct, submit the final SQL wrapped in "
                        "<solution>...</solution>."
                    )
                else:
                    guidance = (
                        "If this result answers the question, submit the final "
                        "SQL now as <solution>your query</solution>. "
                        "Do not keep exploring once you have the answer."
                    )
                conversation.append({
                    "role": "user",
                    "content": (
                        f"<observation>\n{result}\n</observation>\n"
                        f"{guidance} You have {turns_left} turn(s) left."
                    ),
                })
            else:
                # Nudge the model to take action
                conversation.append({
                    "role": "user",
                    "content": (
                        "Use <sql>...</sql> to execute a query or "
                        "<solution>...</solution> to submit your final answer."
                    ),
                })
    except Exception as e:
        # Any unexpected error: surface it clearly in the trace output instead
        # of a bare "trajectory failed". Still return a (negative) reward so
        # the trainer gets a usable signal rather than an errored trajectory.
        log.exception("Episode raised an unexpected error")
        tag_trace(
            reward=-1.0,
            verdict="error",
            db_id=db_id,
            num_turns=num_turns,
            sql_attempts=sql_attempts,
            sql_errors=sql_errors,
            error=f"{type(e).__name__}: {e}",
        )
        return {
            "reward": -1.0,
            "status": "error",
            "outcome": f"agent raised {type(e).__name__}: {e}",
            "metrics": {
                "num_turns": num_turns,
                "sql_attempts": sql_attempts,
                "sql_errors": sql_errors,
            },
        }

    # During TRAINING we deliberately do NOT fall back to the last successful
    # query. The model must explicitly submit with <solution> to earn reward;
    # otherwise it never learns to signal a final answer and just loops on
    # <sql>. Only an explicit <solution> counts. (For offline EVAL you may want
    # to be lenient and score the last successful query instead; flip
    # ALLOW_LAST_SQL_FALLBACK via the env var below for that case.)
    used_fallback = False
    if (
        final_sql is None
        and last_successful_sql is not None
        and ALLOW_LAST_SQL_FALLBACK
    ):
        final_sql = last_successful_sql
        used_fallback = True
        log.info("No explicit <solution>; falling back to last successful SQL")

    # --- Compute reward and a human-readable explanation ---
    reward, reason = compute_reward(final_sql, gold_query, db_id)

    # Build a clear outcome string that always explains the result, or why
    # there wasn't one. This shows up directly in the trace "Outputs".
    if final_sql is None:
        outcome = (
            f"No answer produced. Ran {sql_attempts} SQL quer(ies), "
            f"{sql_errors} errored, over {num_turns} turn(s) before the limit. "
            f"Reason: {reason}."
        )
        if sql_errors and sql_errors == sql_attempts and sql_attempts > 0:
            outcome += " Every SQL query errored: last error -> " + last_observation
    else:
        verdict = "CORRECT" if reward == 1.0 else "WRONG"
        how = (
            "no explicit <solution>; used last successful query"
            if used_fallback
            else "submitted via <solution>"
        )
        outcome = (
            f"Answer {verdict} ({how}). {reason}. "
            f"Ran {sql_attempts} SQL quer(ies) ({sql_errors} errored) "
            f"over {num_turns} turn(s)."
        )

    log.info("Episode complete: reward=%s | %s", reward, outcome)

    # Normalized verdict for easy filtering in the trace UI.
    if final_sql is None:
        verdict_tag = "no_submission"
    elif reward == 1.0:
        verdict_tag = "correct"
    else:
        verdict_tag = "incorrect"

    # Attach per-episode attributes to the MLflow/OTel trace so you can filter
    # and group episodes by reward, verdict, db_id, etc. in the trace UI.
    tag_trace(
        reward=reward,
        verdict=verdict_tag,
        db_id=db_id,
        num_turns=num_turns,
        sql_attempts=sql_attempts,
        sql_errors=sql_errors,
        empty_responses=empty_responses,
        submitted_solution=submitted,
        used_last_sql_fallback=used_fallback,
        final_sql=final_sql,
    )

    return {
        "reward": reward,
        "status": "ok",
        "outcome": outcome,
        "final_sql": final_sql,
        "metrics": {
            "num_turns": num_turns,
            "sql_attempts": sql_attempts,
            "sql_errors": sql_errors,
            "empty_responses": empty_responses,
            "submitted_solution": submitted,
            "used_last_sql_fallback": used_fallback,
        },
    }


if __name__ == "__main__":
    app.run()
