"""Customer service environment with deterministic task grading."""

from __future__ import annotations

from datetime import datetime, timezone
from uuid import uuid4

from openenv.core.env_server.interfaces import Environment

try:
    from ..graders import grade_task
    from ..models import (
        ActionType,
        CustomerMessage,
        CustomerServiceAction,
        CustomerServiceInfo,
        CustomerServiceObservation,
        CustomerServiceReward,
        CustomerServiceState,
        TaskLevel,
        TicketOutcome,
        TicketPriority,
        TicketStatus,
    )
    from ..tasks import TASK_REGISTRY, TaskSpec, TicketSpec, get_task, task_from_seed
except ImportError:  # pragma: no cover
    from graders import grade_task
    from models import (
        ActionType,
        CustomerMessage,
        CustomerServiceAction,
        CustomerServiceInfo,
        CustomerServiceObservation,
        CustomerServiceReward,
        CustomerServiceState,
        TaskLevel,
        TicketOutcome,
        TicketPriority,
        TicketStatus,
    )
    from tasks import TASK_REGISTRY, TaskSpec, TicketSpec, get_task, task_from_seed


MAX_IDLE_REPEAT = 2


class CustomerServiceEnvironment(Environment):
    """OpenEnv environment for a realistic customer-support workflow."""

    SUPPORTS_CONCURRENT_SESSIONS: bool = True

    def __init__(self, task_id: str | None = None):
        self._configured_task_id = task_id
        self._task: TaskSpec = task_from_seed(0)
        self._state = CustomerServiceState(episode_id=str(uuid4()))
        self._conversation: list[CustomerMessage] = []
        self._ticket_cursor = -1
        self._current_ticket: TicketSpec | None = None
        self._current_outcome = TicketOutcome()
        self._current_reward_total = 0.0
        self._steps_on_ticket = 0
        self._awaiting_customer_reply = False

    def reset(
        self, seed: int | None = None, episode_id: str | None = None
    ) -> CustomerServiceObservation:
        if self._configured_task_id is not None:
            self._task = get_task(self._configured_task_id)
        elif episode_id in TASK_REGISTRY:
            self._task = get_task(episode_id)
        else:
            self._task = task_from_seed(seed)

        self._state = CustomerServiceState(
            episode_id=episode_id or str(uuid4()),
            task_id=self._task.task_id,
            task_title=self._task.title,
            task_level=self._task.level,
            total_tickets=len(self._task.tickets),
            remaining_tickets=len(self._task.tickets),
        )
        self._conversation = []
        self._ticket_cursor = -1
        self._current_ticket = None
        self._current_outcome = TicketOutcome()
        self._current_reward_total = 0.0
        self._steps_on_ticket = 0
        self._awaiting_customer_reply = False
        return self._advance_to_next_ticket(
            recent_event="task_reset",
            reward=CustomerServiceReward(),
            grader_score=None,
        )

    def step(self, action: CustomerServiceAction) -> CustomerServiceObservation:  # type: ignore[override]
        self._state.step_count += 1
        self._steps_on_ticket += 1

        if self._current_ticket is None:
            return self._terminal_observation(
                reward=CustomerServiceReward(),
                recent_event="task_complete",
            )

        reward = CustomerServiceReward()
        violations: list[str] = []
        self._record_agent_message(action)
        self._state.action_history.append(
            f"{self._state.ticket_id}:{action.action_type.value}:{action.message[:48]}"
        )

        if not action.message.strip():
            reward.safety_penalty -= 0.1
            violations.append("empty_message")

        if self._is_repeated_action(action.action_type):
            reward.efficiency_penalty -= 0.05
            violations.append("looping_action_pattern")

        if action.action_type == ActionType.CATEGORIZE:
            self._handle_categorize(action, reward, violations)
        elif action.action_type == ActionType.RESPOND:
            self._handle_respond(action, reward, violations)
        elif action.action_type == ActionType.REQUEST_INFO:
            self._handle_request_info(action, reward, violations)
        elif action.action_type == ActionType.TRANSFER:
            self._handle_transfer(action, reward, violations)
        elif action.action_type == ActionType.ESCALATE:
            return self._finish_ticket(action, reward, violations, resolved=False, escalated=True)
        elif action.action_type == ActionType.RESOLVE:
            return self._finish_ticket(action, reward, violations, resolved=True, escalated=False)

        if self._steps_on_ticket > self._task.max_steps:
            reward.efficiency_penalty -= 0.2
            self._current_outcome.excessive_steps = True
            violations.append("step_budget_exceeded")
            return self._finish_ticket(action, reward, violations, resolved=False, escalated=False)

        return self._build_observation(
            reward=reward,
            recent_event="step",
            violations=violations,
            grader_score=None,
        )

    @property
    def state(self) -> CustomerServiceState:
        return self._state

    def _handle_categorize(
        self,
        action: CustomerServiceAction,
        reward: CustomerServiceReward,
        violations: list[str],
    ) -> None:
        ticket = self._require_ticket()
        self._state.status = TicketStatus.IN_PROGRESS

        if action.category == ticket.true_category:
            reward.classification += 0.25
            self._current_outcome.category_correct = True
        else:
            reward.safety_penalty -= 0.15
            violations.append("wrong_category")
            self._current_outcome.category_correct = False

        if action.priority == ticket.true_priority:
            reward.classification += 0.1
            self._current_outcome.priority_correct = True
        else:
            reward.safety_penalty -= 0.05
            violations.append("wrong_priority")
            self._current_outcome.priority_correct = False

        if action.category is not None:
            self._state.category = action.category
        if action.priority is not None:
            self._state.priority = action.priority

    def _handle_respond(
        self,
        action: CustomerServiceAction,
        reward: CustomerServiceReward,
        violations: list[str],
    ) -> None:
        ticket = self._require_ticket()
        self._state.status = TicketStatus.IN_PROGRESS
        self._current_outcome.responded = True

        message = action.message.lower()
        empathy_keywords = ("sorry", "apologize", "understand", "frustrating", "help")
        response_keywords = {keyword.lower() for keyword in ticket.required_response_keywords}

        empathetic = any(word in message for word in empathy_keywords)
        issue_specific = all(keyword in message for keyword in response_keywords) if response_keywords else True

        if ticket.requires_empathy and empathetic:
            reward.communication += 0.15
            self._current_outcome.empathetic_response = True
            self._state.satisfaction_score = min(1.0, self._state.satisfaction_score + 0.1)
        elif ticket.requires_empathy:
            reward.safety_penalty -= 0.2
            violations.append("missing_empathy")
            self._state.satisfaction_score = max(0.0, self._state.satisfaction_score - 0.12)
        else:
            reward.communication += 0.05
            self._state.satisfaction_score = min(1.0, self._state.satisfaction_score + 0.04)

        if issue_specific:
            reward.communication += 0.1
            self._current_outcome.issue_specific_response = True
        elif response_keywords:
            reward.efficiency_penalty -= 0.05
            violations.append("generic_response")

        if self._awaiting_customer_reply:
            self._awaiting_customer_reply = False
            reward.task_progress += 0.05

    def _handle_request_info(
        self,
        action: CustomerServiceAction,
        reward: CustomerServiceReward,
        violations: list[str],
    ) -> None:
        ticket = self._require_ticket()
        self._state.status = TicketStatus.WAITING_CUSTOMER

        if ticket.needs_request_info:
            reward.task_progress += 0.15
            self._current_outcome.requested_info = True
            follow_up = ticket.customer_follow_up or "Here is the missing information you requested."
            self._conversation.append(
                CustomerMessage(
                    sender="customer",
                    content=follow_up,
                    timestamp=self._timestamp(),
                )
            )
            self._awaiting_customer_reply = True
        else:
            reward.efficiency_penalty -= 0.05
            violations.append("unnecessary_request_info")

    def _handle_transfer(
        self,
        action: CustomerServiceAction,
        reward: CustomerServiceReward,
        violations: list[str],
    ) -> None:
        ticket = self._require_ticket()
        self._state.status = TicketStatus.IN_PROGRESS
        self._current_outcome.transferred = True

        if ticket.expected_transfer_department:
            expected = ticket.expected_transfer_department.lower()
            actual = (action.transfer_department or "").lower()
            if actual == expected:
                reward.task_progress += 0.1
            else:
                reward.efficiency_penalty -= 0.05
                violations.append("transfer_to_wrong_queue")
        else:
            reward.efficiency_penalty -= 0.02

    def _finish_ticket(
        self,
        action: CustomerServiceAction,
        reward: CustomerServiceReward,
        violations: list[str],
        *,
        resolved: bool,
        escalated: bool,
    ) -> CustomerServiceObservation:
        ticket = self._require_ticket()

        if resolved:
            self._state.status = TicketStatus.RESOLVED
            self._state.resolved = True
            self._state.total_tickets_resolved += 1
            self._current_outcome.resolved = True

            if ticket.requires_escalation:
                reward.safety_penalty -= 0.35
                violations.append("resolved_instead_of_escalating")
            elif self._state.category is None:
                reward.safety_penalty -= 0.1
                violations.append("resolved_without_triage")
            else:
                reward.resolution += 0.35
                if self._current_outcome.responded:
                    reward.resolution += 0.1
                self._state.satisfaction_score = min(1.0, self._state.satisfaction_score + 0.08)

        if escalated:
            self._state.status = TicketStatus.ESCALATED
            self._state.escalated = True
            self._state.total_tickets_escalated += 1
            self._current_outcome.escalated = True

            if ticket.requires_escalation:
                reward.resolution += 0.35
            else:
                reward.safety_penalty -= 0.25
                violations.append("unnecessary_escalation")
                self._current_outcome.unnecessary_escalation = True

        if ticket.requires_escalation and action.action_type != ActionType.ESCALATE:
            self._state.satisfaction_score = max(0.0, self._state.satisfaction_score - 0.05)

        self._current_outcome.steps_taken = self._steps_on_ticket
        self._current_outcome.policy_violations = sorted(set(violations))
        self._current_outcome.final_satisfaction = self._state.satisfaction_score
        reward.normalized_progress = self._estimate_ticket_progress()
        reward.total = round(
            reward.classification
            + reward.communication
            + reward.task_progress
            + reward.resolution
            + reward.efficiency_penalty
            + reward.safety_penalty,
            4,
        )
        finished_reward_total = self._current_reward_total + reward.total
        self._current_outcome.reward_total = round(finished_reward_total, 4)
        self._state.cumulative_reward = round(self._state.cumulative_reward + reward.total, 4)
        self._state.last_reward = reward

        finished_outcome = self._current_outcome.model_copy(deep=True)
        self._state.completed_tickets.append(finished_outcome)
        done = self._ticket_cursor >= len(self._task.tickets) - 1

        if done:
            self._state.remaining_tickets = 0
            final_grade = grade_task(self._task, self._state)
            self._state.final_grade = final_grade
            return self._terminal_observation(
                reward=reward,
                recent_event="task_complete",
                violations=violations,
                grader_score=final_grade.score,
            )

        return self._advance_to_next_ticket(
            recent_event="ticket_complete",
            reward=reward,
            violations=violations,
            grader_score=None,
            apply_reward=False,
        )

    def _advance_to_next_ticket(
        self,
        *,
        recent_event: str,
        reward: CustomerServiceReward,
        violations: list[str] | None = None,
        grader_score: float | None,
        apply_reward: bool = True,
    ) -> CustomerServiceObservation:
        self._ticket_cursor += 1
        self._current_ticket = self._task.tickets[self._ticket_cursor]
        self._current_outcome = TicketOutcome(
            scenario_id=self._current_ticket.scenario_id,
            ticket_id=f"TKT-{uuid4().hex[:8].upper()}",
        )
        self._current_reward_total = 0.0
        self._steps_on_ticket = 0
        self._awaiting_customer_reply = False
        self._state.current_ticket_index = self._ticket_cursor + 1
        self._state.remaining_tickets = len(self._task.tickets) - self._ticket_cursor
        self._state.ticket_id = self._current_outcome.ticket_id
        self._state.status = TicketStatus.OPEN
        self._state.category = None
        self._state.priority = TicketPriority.MEDIUM
        self._state.satisfaction_score = 0.5
        self._state.resolved = False
        self._state.escalated = False
        self._conversation = [
            CustomerMessage(
                sender="customer",
                content=self._current_ticket.opening_message,
                timestamp=self._timestamp(),
            )
        ]

        return self._build_observation(
            reward=reward,
            recent_event=recent_event,
            violations=violations or [],
            grader_score=grader_score,
            apply_reward=apply_reward,
        )

    def _terminal_observation(
        self,
        *,
        reward: CustomerServiceReward,
        recent_event: str,
        violations: list[str] | None = None,
        grader_score: float | None = None,
    ) -> CustomerServiceObservation:
        reward.normalized_progress = 1.0 if self._state.final_grade else 0.0
        reward.total = round(
            reward.classification
            + reward.communication
            + reward.task_progress
            + reward.resolution
            + reward.efficiency_penalty
            + reward.safety_penalty,
            4,
        )
        self._state.last_reward = reward
        return CustomerServiceObservation(
            ticket_id=self._state.ticket_id,
            task_id=self._task.task_id,
            task_title=self._task.title,
            task_level=self._task.level,
            customer_name="",
            customer_message="All tickets for this task have been processed.",
            category=self._state.category,
            priority=self._state.priority,
            status=TicketStatus.CLOSED,
            conversation_history=list(self._conversation),
            customer_sentiment=None,
            resolution_hint=None,
            remaining_tickets=0,
            available_actions=[],
            reward=reward.total,
            done=True,
            reward_details=reward,
            info=self._build_info(recent_event, violations or [], grader_score),
            metadata={
                "task_id": self._task.task_id,
                "grade": grader_score,
                "completed_tickets": len(self._state.completed_tickets),
            },
        )

    def _build_observation(
        self,
        *,
        reward: CustomerServiceReward,
        recent_event: str,
        violations: list[str],
        grader_score: float | None,
        apply_reward: bool = True,
    ) -> CustomerServiceObservation:
        ticket = self._require_ticket()
        reward.normalized_progress = self._estimate_ticket_progress()
        reward.total = round(
            reward.classification
            + reward.communication
            + reward.task_progress
            + reward.resolution
            + reward.efficiency_penalty
            + reward.safety_penalty,
            4,
        )
        if apply_reward:
            self._current_reward_total += reward.total
            self._state.cumulative_reward = round(self._state.cumulative_reward + reward.total, 4)
        self._state.last_reward = reward
        self._state.last_info = self._build_info(recent_event, violations, grader_score)

        return CustomerServiceObservation(
            ticket_id=self._state.ticket_id,
            task_id=self._task.task_id,
            task_title=self._task.title,
            task_level=self._task.level,
            customer_name=ticket.customer_name,
            customer_message=self._latest_customer_message(),
            category=self._state.category,
            priority=self._state.priority,
            status=self._state.status,
            conversation_history=list(self._conversation),
            customer_sentiment=ticket.sentiment,
            resolution_hint=ticket.resolution_hint,
            remaining_tickets=max(0, len(self._task.tickets) - self._ticket_cursor - 1),
            available_actions=list(ActionType),
            reward=reward.total,
            done=False,
            reward_details=reward,
            info=self._state.last_info,
            metadata={
                "task_id": self._task.task_id,
                "step_count": self._state.step_count,
                "current_ticket_index": self._state.current_ticket_index,
            },
        )

    def _build_info(
        self, recent_event: str, violations: list[str], grader_score: float | None
    ) -> CustomerServiceInfo:
        ticket = self._current_ticket
        expected_outcome = ""
        if ticket is not None:
            expected_outcome = (
                "escalate"
                if ticket.requires_escalation
                else "request_info_then_resolve"
                if ticket.needs_request_info and not self._current_outcome.requested_info
                else "resolve_or_transfer"
                if ticket.expected_transfer_department
                else "resolve"
            )

        info = CustomerServiceInfo(
            task_id=self._task.task_id,
            task_level=self._task.level,
            task_title=self._task.title,
            current_ticket_index=self._state.current_ticket_index,
            total_tickets=len(self._task.tickets),
            remaining_tickets=max(0, len(self._task.tickets) - self._ticket_cursor - 1),
            expected_outcome=expected_outcome,
            recent_event=recent_event,
            policy_violations=sorted(set(violations)),
            grader_score=grader_score,
        )
        self._state.last_info = info
        return info

    def _estimate_ticket_progress(self) -> float:
        ticket = self._current_ticket
        if ticket is None:
            return 1.0

        score = 0.0
        if self._current_outcome.category_correct:
            score += 0.25
        if self._current_outcome.priority_correct:
            score += 0.1
        if self._current_outcome.responded:
            score += 0.15
        if ticket.needs_request_info and self._current_outcome.requested_info:
            score += 0.15
        if self._current_outcome.transferred:
            score += 0.1
        if self._current_outcome.resolved or self._current_outcome.escalated:
            score += 0.25
        return max(0.0, min(1.0, round(score, 4)))

    def _record_agent_message(self, action: CustomerServiceAction) -> None:
        self._conversation.append(
            CustomerMessage(
                sender="agent",
                content=action.message,
                timestamp=self._timestamp(),
            )
        )

    def _is_repeated_action(self, action_type: ActionType) -> bool:
        if len(self._state.action_history) < MAX_IDLE_REPEAT:
            return False
        recent_types = [
            entry.split(":")[1]
            for entry in self._state.action_history[-MAX_IDLE_REPEAT:]
            if ":" in entry
        ]
        return recent_types.count(action_type.value) == MAX_IDLE_REPEAT

    def _require_ticket(self) -> TicketSpec:
        if self._current_ticket is None:  # pragma: no cover - defensive guard
            raise RuntimeError("No active ticket is loaded.")
        return self._current_ticket

    def _latest_customer_message(self) -> str:
        for message in reversed(self._conversation):
            if message.sender == "customer":
                return message.content
        return ""

    @staticmethod
    def _timestamp() -> str:
        return datetime.now(timezone.utc).isoformat()
