"""Deterministic task graders for the customer service environment."""

from __future__ import annotations

from .models import CustomerServiceState, TaskGrade, TicketOutcome
from .tasks import TaskSpec, TicketSpec

# Violations that immediately zero-score the entire task — no partial credit.
CATASTROPHIC_VIOLATIONS: frozenset[str] = frozenset(
    {
        "resolved_instead_of_escalating",  # agent resolved a ticket that required Tier-2
    }
)


def grade_task(task: TaskSpec, state: CustomerServiceState) -> TaskGrade:
    """Grade a completed episode: 4-stage pipeline returning a 0.0–1.0 score.

    Stage 1 — Violation gate (binary).
    Stage 2 — Per-ticket outcome scoring with feature-flag rubric.
    Stage 3 — Normalize each ticket by its theoretical maximum.
    Stage 4 — Task-level weighted aggregation.
    """
    outcomes_by_id = {o.scenario_id: o for o in state.completed_tickets}

    # ── Stage 1: Violation gate ──────────────────────────────────────────────
    for outcome in state.completed_tickets:
        if CATASTROPHIC_VIOLATIONS & set(outcome.policy_violations):
            return TaskGrade(
                task_id=task.task_id,
                task_title=task.title,
                score=0.0,
                passed=False,
                per_ticket_scores={t.scenario_id: 0.0 for t in task.tickets},
                metrics={"violation_gate": 1.0},
                summary=(
                    f"FAILED (violation gate): catastrophic policy violation on '{task.title}'."
                ),
            )

    # ── Stage 2 + 3: Per-ticket scoring and normalization ────────────────────
    per_ticket_scores: dict[str, float] = {}
    for ticket in task.tickets:
        outcome = outcomes_by_id.get(
            ticket.scenario_id, TicketOutcome.placeholder(ticket.scenario_id)
        )
        raw = _score_ticket(ticket, outcome)          # uncapped raw score
        ceiling = _max_ticket_score(ticket)            # theoretical max for this ticket
        per_ticket_scores[ticket.scenario_id] = round(min(1.0, raw / ceiling), 4)

    mean_ticket = sum(per_ticket_scores.values()) / len(per_ticket_scores)

    # ── Stage 4: Task-level aggregation ─────────────────────────────────────
    # Escalation accuracy — binary: all required escalations hit, nothing more.
    required_escalations = sum(1 for t in task.tickets if t.requires_escalation)
    correct_escalations = sum(
        1
        for t in task.tickets
        if t.requires_escalation
        and outcomes_by_id.get(
            t.scenario_id, TicketOutcome.placeholder(t.scenario_id)
        ).escalated
    )
    escalation_acc = 1.0 if correct_escalations == required_escalations else 0.0

    # Efficiency — 1.0 within budget, linear decay only when over budget.
    overrun = max(0, state.step_count - task.max_steps)
    efficiency = max(0.0, 1.0 - overrun / max(1, task.max_steps))

    final_score = max(
        0.0,
        min(
            1.0,
            round(mean_ticket * 0.80 + escalation_acc * 0.15 + efficiency * 0.05, 4),
        ),
    )

    return TaskGrade(
        task_id=task.task_id,
        task_title=task.title,
        score=final_score,
        passed=final_score >= 0.8,
        per_ticket_scores=per_ticket_scores,
        metrics={
            "mean_ticket_score": round(mean_ticket, 4),
            "escalation_accuracy": escalation_acc,
            "efficiency_score": round(efficiency, 4),
            "tickets_completed": float(len(state.completed_tickets)),
        },
        summary=_grade_summary(task, final_score, state),
    )


def _max_ticket_score(ticket: TicketSpec) -> float:
    """Theoretical maximum raw score for a ticket given its active feature flags.

    Used by Stage 3 to normalise raw scores so a perfect agent always returns 1.0
    regardless of which features a ticket exercises.
    """
    score = 0.25 + 0.10                                     # category + priority
    score += 0.15 if ticket.requires_empathy else 0.10      # response tone
    if ticket.required_response_keywords:
        score += 0.10                                        # keyword extraction
    if ticket.needs_request_info:
        score += 0.15                                        # info-request handling
    score += 0.20 + 0.05                                    # resolution quality + clean-finish bonus
    if ticket.expected_transfer_department:
        score += 0.05                                        # transfer accuracy
    score += 0.05 + 0.05                                    # verbosity check + no-violation bonus
    return score


def _score_ticket(ticket: TicketSpec, outcome: TicketOutcome) -> float:
    """Raw (unnormalized) rubric score for a single ticket outcome.

    Returns a non-negative float; the caller is responsible for capping at 1.0
    after normalization.
    """
    score = 0.0

    # Category and priority classification
    score += 0.25 if outcome.category_correct else 0.0
    score += 0.10 if outcome.priority_correct else 0.0

    # Response tone: empathy rewarded more when the ticket demands it
    if ticket.requires_empathy:
        score += 0.15 if outcome.empathetic_response else 0.0
    elif outcome.responded or outcome.transferred:
        score += 0.10

    # Keyword extraction — issue-specific content
    if ticket.required_response_keywords:
        score += 0.10 if outcome.issue_specific_response else 0.0

    # Information-request handling
    if ticket.needs_request_info:
        score += 0.15 if outcome.requested_info else 0.0

    # Resolution quality
    if ticket.requires_escalation:
        score += 0.20 if outcome.escalated else 0.0
        score += 0.05 if not outcome.resolved else 0.0        # didn't incorrectly resolve
    else:
        score += 0.20 if outcome.resolved else 0.0
        score += 0.05 if not outcome.unnecessary_escalation else 0.0

    # Transfer accuracy
    if ticket.expected_transfer_department:
        score += 0.05 if outcome.transferred else 0.0

    # Verbosity check + no-violation bonus
    score += 0.05 if not outcome.excessive_steps else 0.0
    score += 0.05 if not outcome.policy_violations else 0.0

    return max(0.0, score)


def _grade_summary(task: TaskSpec, score: float, state: CustomerServiceState) -> str:
    if score >= 0.9:
        verdict = "Strong execution"
    elif score >= 0.75:
        verdict = "Usable but inconsistent"
    else:
        verdict = "Needs improvement"

    return (
        f"{verdict} on '{task.title}': completed {len(state.completed_tickets)}/"
        f"{len(task.tickets)} tickets in {state.step_count} steps with "
        f"{state.total_tickets_escalated} escalation(s)."
    )
