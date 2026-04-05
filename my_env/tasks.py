"""Deterministic task definitions for the customer service environment."""

from __future__ import annotations

from typing import Dict, List

from pydantic import BaseModel, Field

from .models import TaskLevel, TicketCategory, TicketPriority


class TicketSpec(BaseModel):
    """Ground-truth requirements for a single support ticket."""

    scenario_id: str
    customer_name: str
    opening_message: str
    true_category: TicketCategory
    true_priority: TicketPriority
    sentiment: str
    resolution_hint: str
    requires_empathy: bool = False
    needs_request_info: bool = False
    requires_escalation: bool = False
    allowed_to_resolve: bool = True
    expected_transfer_department: str | None = None
    required_response_keywords: List[str] = Field(default_factory=list)
    customer_follow_up: str | None = None


class TaskSpec(BaseModel):
    """A full OpenEnv task with deterministic grading inputs."""

    task_id: str
    title: str
    level: TaskLevel
    objective: str
    max_steps: int = Field(default=12, ge=1)
    success_criteria: List[str]
    tickets: List[TicketSpec]


TASK_REGISTRY: Dict[str, TaskSpec] = {
    "easy_billing_refund": TaskSpec(
        task_id="easy_billing_refund",
        title="Duplicate Charge Refund Triage",
        level=TaskLevel.EASY,
        objective=(
            "Handle a straightforward billing ticket by categorizing it correctly, "
            "acknowledging the duplicate charge, and resolving it without escalation."
        ),
        max_steps=4,
        success_criteria=[
            "Classify the ticket as billing with high priority.",
            "Send an empathetic response that mentions the refund or duplicate charge.",
            "Resolve the ticket without escalating it.",
        ],
        tickets=[
            TicketSpec(
                scenario_id="billing_refund",
                customer_name="Alice Johnson",
                opening_message=(
                    "I was charged twice for my subscription this month. Please refund the extra charge."
                ),
                true_category=TicketCategory.BILLING,
                true_priority=TicketPriority.HIGH,
                sentiment="negative",
                resolution_hint=(
                    "Confirm the duplicate transaction, apologize, and process a refund under the billing policy."
                ),
                requires_empathy=True,
                required_response_keywords=["refund", "duplicate", "charge"],
            )
        ],
    ),
    "medium_support_queue": TaskSpec(
        task_id="medium_support_queue",
        title="Multi-Ticket Tier-1 Resolution Queue",
        level=TaskLevel.MEDIUM,
        objective=(
            "Work through a realistic Tier-1 queue by triaging each ticket, collecting "
            "needed information, and closing the issues without unnecessary escalation."
        ),
        max_steps=10,
        success_criteria=[
            "Correctly categorize and prioritize all three tickets.",
            "Use request-info only on the account lockout ticket.",
            "Resolve all tickets without escalation.",
        ],
        tickets=[
            TicketSpec(
                scenario_id="account_lockout",
                customer_name="Bob Smith",
                opening_message=(
                    "I can't log into my account. I've tried resetting my password three times."
                ),
                true_category=TicketCategory.ACCOUNT,
                true_priority=TicketPriority.HIGH,
                sentiment="negative",
                resolution_hint=(
                    "Gather the username or account email, confirm the account lock, then unlock it."
                ),
                requires_empathy=True,
                needs_request_info=True,
                required_response_keywords=["unlock", "account", "email"],
                customer_follow_up=(
                    "My login email is bob@example.com. Please unlock the account so I can sign in."
                ),
            ),
            TicketSpec(
                scenario_id="csv_export",
                customer_name="Carol Davis",
                opening_message=(
                    "How do I export my data to CSV? I can't find the option anywhere."
                ),
                true_category=TicketCategory.TECHNICAL,
                true_priority=TicketPriority.LOW,
                sentiment="neutral",
                resolution_hint=(
                    "Point the customer to Settings > Account > Export and mention CSV export."
                ),
                required_response_keywords=["export", "csv", "settings"],
            ),
            TicketSpec(
                scenario_id="checkout_error",
                customer_name="Frank Wilson",
                opening_message=(
                    "I need to upgrade my plan but the checkout page keeps giving me an error."
                ),
                true_category=TicketCategory.BILLING,
                true_priority=TicketPriority.MEDIUM,
                sentiment="neutral",
                resolution_hint=(
                    "Acknowledge the checkout issue, offer a manual upgrade path, and close the loop."
                ),
                required_response_keywords=["upgrade", "checkout", "manual"],
            ),
        ],
    ),
    "hard_escalation_judgment": TaskSpec(
        task_id="hard_escalation_judgment",
        title="Escalation Judgment Under Customer Pressure",
        level=TaskLevel.HARD,
        objective=(
            "Handle a five-ticket queue where the agent must distinguish between tickets "
            "that need escalation and tickets that should stay in Tier-1 support."
        ),
        max_steps=16,
        success_criteria=[
            "Escalate the repeat-contact urgent ticket.",
            "Do not escalate tickets that Tier-1 can resolve.",
            "Keep customer satisfaction high by using empathetic, issue-specific responses.",
        ],
        tickets=[
            TicketSpec(
                scenario_id="repeat_contact",
                customer_name="Grace Kim",
                opening_message=(
                    "This is the THIRD time I'm contacting support about this issue. Nobody is helping me!"
                ),
                true_category=TicketCategory.GENERAL,
                true_priority=TicketPriority.URGENT,
                sentiment="angry",
                resolution_hint=(
                    "Acknowledge the frustration, summarize the repeat-contact risk, and escalate to Tier-2."
                ),
                requires_empathy=True,
                requires_escalation=True,
                allowed_to_resolve=False,
                required_response_keywords=["sorry", "escalate", "priority"],
            ),
            TicketSpec(
                scenario_id="late_shipment",
                customer_name="David Lee",
                opening_message=(
                    "My order #4521 still hasn't arrived. It's been 2 weeks past the estimated delivery."
                ),
                true_category=TicketCategory.SHIPPING,
                true_priority=TicketPriority.HIGH,
                sentiment="angry",
                resolution_hint=(
                    "Apologize, confirm the delayed shipment, and arrange a replacement or refund without escalation."
                ),
                requires_empathy=True,
                required_response_keywords=["replacement", "refund", "order"],
            ),
            TicketSpec(
                scenario_id="dashboard_bug",
                customer_name="Eva Martinez",
                opening_message=(
                    "The new dashboard update is great! But I noticed a small bug in the charts."
                ),
                true_category=TicketCategory.PRODUCT,
                true_priority=TicketPriority.LOW,
                sentiment="positive",
                resolution_hint=(
                    "Thank the customer, capture the bug, and transfer it to product feedback."
                ),
                expected_transfer_department="product_feedback",
                required_response_keywords=["thanks", "bug", "charts"],
            ),
            TicketSpec(
                scenario_id="enterprise_features",
                customer_name="Henry Patel",
                opening_message="Can you tell me more about the enterprise plan features?",
                true_category=TicketCategory.PRODUCT,
                true_priority=TicketPriority.LOW,
                sentiment="neutral",
                resolution_hint=(
                    "Provide a brief feature summary and transfer to sales for follow-up."
                ),
                expected_transfer_department="sales",
                required_response_keywords=["enterprise", "features", "sales"],
            ),
            TicketSpec(
                scenario_id="billing_refund_repeatable",
                customer_name="Alice Johnson",
                opening_message=(
                    "I was charged twice for my subscription this month. Please refund the extra charge."
                ),
                true_category=TicketCategory.BILLING,
                true_priority=TicketPriority.HIGH,
                sentiment="negative",
                resolution_hint=(
                    "Acknowledge the billing error, confirm the refund, and resolve without escalation."
                ),
                requires_empathy=True,
                required_response_keywords=["refund", "duplicate", "charge"],
            ),
        ],
    ),
}


def get_task(task_id: str) -> TaskSpec:
    """Return a task spec or raise a clear error for unknown ids."""

    try:
        return TASK_REGISTRY[task_id]
    except KeyError as exc:  # pragma: no cover - defensive guard
        known = ", ".join(sorted(TASK_REGISTRY))
        raise ValueError(f"Unknown task_id '{task_id}'. Known tasks: {known}") from exc


def task_from_seed(seed: int | None) -> TaskSpec:
    """Select a task deterministically from a seed."""

    ordered = [TASK_REGISTRY[key] for key in sorted(TASK_REGISTRY)]
    if seed is None:
        return ordered[0]
    return ordered[seed % len(ordered)]
