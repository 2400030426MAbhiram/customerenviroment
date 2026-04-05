"""Baseline inference runner for the customer service environment."""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass
from typing import Any

from openai import OpenAI

from .graders import grade_task
from .models import ActionType, CustomerServiceAction, CustomerServiceObservation
from .server.my_env_environment import CustomerServiceEnvironment
from .tasks import TASK_REGISTRY, get_task


SYSTEM_PROMPT = """
You are a Tier-1 customer support agent operating inside a deterministic simulator.
Return exactly one JSON object with keys:
- action_type: one of respond, escalate, resolve, request_info, categorize, transfer
- message: short string
- category: optional one of billing, technical, account, product, shipping, general
- priority: optional one of low, medium, high, urgent
- escalation_reason: optional string
- transfer_department: optional string

Rules:
- First classify each new ticket before resolving or escalating it.
- Escalate only when the ticket clearly requires Tier-2.
- Use empathy for angry or frustrated customers.
- If information is missing, request it once and continue.
- Keep messages concise and professional.
""".strip()


@dataclass
class BaselineResult:
    task_id: str
    score: float
    steps: int
    cumulative_reward: float
    summary: str


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a reproducible OpenAI baseline on all tasks.")
    parser.add_argument("--model", default="gpt-4.1-mini")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--max-steps-buffer", type=int, default=2)
    args = parser.parse_args()

    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise SystemExit("OPENAI_API_KEY is required to run the baseline.")

    client = OpenAI(api_key=api_key)
    results: list[BaselineResult] = []

    for task_id in sorted(TASK_REGISTRY):
        result = run_task(client, task_id=task_id, model=args.model, seed=args.seed, max_steps_buffer=args.max_steps_buffer)
        results.append(result)

    mean_score = sum(item.score for item in results) / len(results)
    output = {
        "model": args.model,
        "seed": args.seed,
        "mean_score": round(mean_score, 4),
        "tasks": [item.__dict__ for item in results],
    }
    print(json.dumps(output, indent=2))


def run_task(
    client: OpenAI,
    *,
    task_id: str,
    model: str,
    seed: int,
    max_steps_buffer: int,
) -> BaselineResult:
    env = CustomerServiceEnvironment(task_id=task_id)
    observation = env.reset(seed=seed, episode_id=task_id)
    task = get_task(task_id)

    total_budget = task.max_steps + max_steps_buffer
    while not observation.done and env.state.step_count < total_budget:
        action = choose_action(client, observation, model=model, seed=seed)
        observation = env.step(action)

    grade = env.state.final_grade or grade_task(task, env.state)
    return BaselineResult(
        task_id=task_id,
        score=grade.score,
        steps=env.state.step_count,
        cumulative_reward=env.state.cumulative_reward,
        summary=grade.summary,
    )


def choose_action(
    client: OpenAI,
    observation: CustomerServiceObservation,
    *,
    model: str,
    seed: int,
) -> CustomerServiceAction:
    prompt = build_user_prompt(observation)
    response = client.chat.completions.create(
        model=model,
        temperature=0,
        seed=seed,
        response_format={"type": "json_object"},
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
    )
    content = response.choices[0].message.content or "{}"
    payload = json.loads(content)
    return parse_action_payload(payload)


def build_user_prompt(observation: CustomerServiceObservation) -> str:
    history_lines = [
        f"{message.sender}: {message.content}" for message in observation.conversation_history[-6:]
    ]
    return "\n".join(
        [
            f"Task: {observation.task_title} ({observation.task_level.value})",
            f"Ticket id: {observation.ticket_id}",
            f"Customer: {observation.customer_name}",
            f"Sentiment: {observation.customer_sentiment}",
            f"Status: {observation.status.value}",
            f"Current category: {observation.category.value if observation.category else 'unknown'}",
            f"Current priority: {observation.priority.value}",
            f"Hint: {observation.resolution_hint or 'none'}",
            f"Expected outcome: {observation.info.expected_outcome}",
            "Conversation:",
            *history_lines,
            "Choose the next best action.",
        ]
    )


def parse_action_payload(payload: dict[str, Any]) -> CustomerServiceAction:
    action_type = ActionType(payload.get("action_type", ActionType.RESPOND.value))
    return CustomerServiceAction(
        action_type=action_type,
        message=str(payload.get("message", "I am reviewing the ticket now.")),
        category=payload.get("category"),
        priority=payload.get("priority"),
        escalation_reason=payload.get("escalation_reason"),
        transfer_department=payload.get("transfer_department"),
    )


if __name__ == "__main__":
    main()
