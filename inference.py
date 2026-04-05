"""
Inference script for the Customer Service Queue Environment.

Mandatory environment variables:
    API_BASE_URL       LLM API endpoint  (default: HuggingFace router)
    MODEL_NAME         Model identifier  (default: Qwen2.5-72B-Instruct)
    HF_TOKEN / API_KEY HuggingFace or API key

Optional:
    CUSTOMER_SERVICE_TASK  Task ID to run (default: easy_billing_refund)
                           One of: easy_billing_refund | medium_support_queue |
                                   hard_escalation_judgment | all

Stdout format (one episode per task):
    [START] task=<task_id> env=customer_service_queue model=<model>
    [STEP]  step=<n> action=<action_str> reward=<0.00> done=<true|false> error=<msg|null>
    [END]   success=<true|false> steps=<n> score=<0.000> rewards=<r1,r2,...>
"""

from __future__ import annotations

import asyncio
import json
import os
import textwrap
from typing import Any, Dict, List, Optional

from openai import OpenAI

from my_env.graders import grade_task
from my_env.models import ActionType, CustomerServiceAction, CustomerServiceObservation
from my_env.server.my_env_environment import CustomerServiceEnvironment
from my_env.tasks import TASK_REGISTRY, get_task

# ── Configuration ─────────────────────────────────────────────────────────────
API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME   = os.getenv("MODEL_NAME", "Qwen/Qwen2.5-72B-Instruct")
API_KEY      = os.getenv("HF_TOKEN") or os.getenv("API_KEY", "")
TASK_NAME    = os.getenv("CUSTOMER_SERVICE_TASK", "easy_billing_refund")
BENCHMARK    = "customer_service_queue"

SUCCESS_SCORE_THRESHOLD = 0.8
TEMPERATURE  = 0.0
MAX_TOKENS   = 300

# ── System prompt ─────────────────────────────────────────────────────────────
SYSTEM_PROMPT = textwrap.dedent("""
    You are a Tier-1 customer support agent. Reply ONLY with a JSON object.

    Required fields:
      "action_type": one of  respond | categorize | request_info | transfer | escalate | resolve
      "message":     your customer-facing reply or internal note (non-empty string)

    Optional fields — include only when the action requires them:
      "category":           billing | technical | account | product | shipping | general
      "priority":           low | medium | high | urgent
      "escalation_reason":  string  (required when action_type=escalate)
      "transfer_department": string  (required when action_type=transfer)

    Workflow rules:
    - First action on a NEW ticket must be  categorize  (set category + priority).
    - Use empathy words (sorry, apologize, understand) for negative/angry sentiment.
    - If critical info is missing, use  request_info  once, then proceed.
    - Escalate ONLY when the ticket clearly requires Tier-2 attention.
    - resolve  closes the ticket — use it only when the issue is fully addressed.
""").strip()


# ── Logging ───────────────────────────────────────────────────────────────────
def log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)


def log_step(step: int, action: str, reward: float, done: bool, error: Optional[str]) -> None:
    print(
        f"[STEP] step={step} action={action} reward={reward:.2f} "
        f"done={str(done).lower()} error={error if error else 'null'}",
        flush=True,
    )


def log_end(success: bool, steps: int, score: float, rewards: List[float]) -> None:
    print(
        f"[END] success={str(success).lower()} steps={steps} score={score:.3f} "
        f"rewards={','.join(f'{r:.2f}' for r in rewards)}",
        flush=True,
    )


# ── Prompt builder ────────────────────────────────────────────────────────────
def build_user_prompt(obs: CustomerServiceObservation, step: int) -> str:
    history_lines = [
        f"  {m.sender}: {m.content}"
        for m in obs.conversation_history[-6:]
    ]
    return textwrap.dedent(f"""
        Task : {obs.task_title} ({obs.task_level.value})
        Ticket: {obs.ticket_id}  Customer: {obs.customer_name}
        Sentiment: {obs.customer_sentiment or 'unknown'}
        Status: {obs.status.value}
        Category assigned: {obs.category.value if obs.category else 'none'}
        Priority assigned: {obs.priority.value}
        Tickets remaining after this one: {obs.remaining_tickets}
        Hint: {obs.resolution_hint or 'none'}
        Expected next action: {obs.info.expected_outcome}

        Conversation history:
{chr(10).join(history_lines) if history_lines else '  (none)'}

        Step {step}: choose the next action as a JSON object.
    """).strip()


# ── LLM call ──────────────────────────────────────────────────────────────────
def get_action(
    client: OpenAI,
    obs: CustomerServiceObservation,
    step: int,
) -> CustomerServiceAction:
    prompt = build_user_prompt(obs, step)
    payload: Dict[str, Any] = {}
    try:
        completion = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
            temperature=TEMPERATURE,
            max_tokens=MAX_TOKENS,
            response_format={"type": "json_object"},
        )
        payload = json.loads(completion.choices[0].message.content or "{}")
    except Exception as exc:
        print(f"[DEBUG] LLM call failed: {exc}", flush=True)

    return CustomerServiceAction(
        action_type=ActionType(payload.get("action_type", ActionType.RESPOND.value)),
        message=str(payload.get("message", "I am reviewing your ticket now.")),
        category=payload.get("category"),
        priority=payload.get("priority"),
        escalation_reason=payload.get("escalation_reason"),
        transfer_department=payload.get("transfer_department"),
    )


# ── Episode runner ─────────────────────────────────────────────────────────────
def run_episode(task_id: str) -> None:
    """Run one full episode for task_id and emit START / STEP* / END to stdout."""
    task    = get_task(task_id)
    env     = CustomerServiceEnvironment(task_id=task_id)
    client  = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)
    budget  = task.max_steps + 2          # small overrun buffer

    log_start(task=task_id, env=BENCHMARK, model=MODEL_NAME)

    rewards:     List[float] = []
    steps_taken: int         = 0
    score:       float       = 0.0
    success:     bool        = False

    try:
        obs = env.reset(episode_id=task_id)

        for step in range(1, budget + 1):
            if obs.done:
                break

            action     = get_action(client, obs, step)
            action_str = (
                f"{action.action_type.value}:"
                f"{action.message[:40].replace(' ', '_')}"
            )

            obs        = env.step(action)
            reward     = float(obs.reward)
            done       = bool(obs.done)
            error      = obs.info.policy_violations[-1] if obs.info.policy_violations else None

            rewards.append(reward)
            steps_taken = step
            log_step(step=step, action=action_str, reward=reward, done=done, error=error)

            if done:
                break

        grade   = env.state.final_grade or grade_task(task, env.state)
        score   = float(grade.score)
        success = score >= SUCCESS_SCORE_THRESHOLD

    except Exception as exc:
        print(f"[DEBUG] Episode error: {exc}", flush=True)

    finally:
        log_end(success=success, steps=steps_taken, score=score, rewards=rewards)


# ── Entry point ───────────────────────────────────────────────────────────────
async def main() -> None:
    if TASK_NAME == "all":
        for task_id in sorted(TASK_REGISTRY):
            run_episode(task_id)
    else:
        run_episode(TASK_NAME)


if __name__ == "__main__":
    asyncio.run(main())
