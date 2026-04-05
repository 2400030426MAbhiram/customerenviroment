"""Customer Service Environment client."""

from __future__ import annotations

from typing import Dict, List

from openenv.core import EnvClient
from openenv.core.client_types import StepResult

from .models import (
    ActionType,
    CustomerMessage,
    CustomerServiceAction,
    CustomerServiceInfo,
    CustomerServiceObservation,
    CustomerServiceReward,
    CustomerServiceState,
    TaskGrade,
    TaskLevel,
    TicketCategory,
    TicketOutcome,
    TicketPriority,
    TicketStatus,
)


class CustomerServiceEnv(
    EnvClient[CustomerServiceAction, CustomerServiceObservation, CustomerServiceState]
):
    """WebSocket client for the customer-support OpenEnv environment."""

    def _step_payload(self, action: CustomerServiceAction) -> Dict:
        payload = {
            "action_type": action.action_type.value,
            "message": action.message,
        }
        if action.category is not None:
            payload["category"] = action.category.value
        if action.priority is not None:
            payload["priority"] = action.priority.value
        if action.escalation_reason is not None:
            payload["escalation_reason"] = action.escalation_reason
        if action.transfer_department is not None:
            payload["transfer_department"] = action.transfer_department
        return payload

    def _parse_result(self, payload: Dict) -> StepResult[CustomerServiceObservation]:
        obs_data = payload.get("observation", {})
        observation = self._parse_observation(obs_data, payload)
        return StepResult(
            observation=observation,
            reward=payload.get("reward", observation.reward),
            done=payload.get("done", observation.done),
        )

    def _parse_state(self, payload: Dict) -> CustomerServiceState:
        completed_tickets = [
            self._parse_ticket_outcome(item) for item in payload.get("completed_tickets", [])
        ]
        final_grade_raw = payload.get("final_grade")
        final_grade = TaskGrade(**final_grade_raw) if final_grade_raw else None

        return CustomerServiceState(
            episode_id=payload.get("episode_id", ""),
            task_id=payload.get("task_id", ""),
            task_title=payload.get("task_title", ""),
            task_level=TaskLevel(payload.get("task_level", TaskLevel.EASY.value)),
            step_count=payload.get("step_count", 0),
            current_ticket_index=payload.get("current_ticket_index", 0),
            total_tickets=payload.get("total_tickets", 0),
            remaining_tickets=payload.get("remaining_tickets", 0),
            ticket_id=payload.get("ticket_id", ""),
            status=TicketStatus(payload.get("status", TicketStatus.OPEN.value)),
            category=self._parse_category(payload.get("category")),
            priority=TicketPriority(payload.get("priority", TicketPriority.MEDIUM.value)),
            satisfaction_score=payload.get("satisfaction_score", 0.5),
            resolved=payload.get("resolved", False),
            escalated=payload.get("escalated", False),
            total_tickets_resolved=payload.get("total_tickets_resolved", 0),
            total_tickets_escalated=payload.get("total_tickets_escalated", 0),
            cumulative_reward=payload.get("cumulative_reward", 0.0),
            action_history=payload.get("action_history", []),
            completed_tickets=completed_tickets,
            last_reward=self._parse_reward(payload.get("last_reward", {})),
            last_info=self._parse_info(payload.get("last_info", {})),
            final_grade=final_grade,
        )

    def _parse_observation(
        self, obs_data: Dict, result_payload: Dict
    ) -> CustomerServiceObservation:
        history = [
            CustomerMessage(
                sender=item.get("sender", ""),
                content=item.get("content", ""),
                timestamp=item.get("timestamp", ""),
            )
            for item in obs_data.get("conversation_history", [])
        ]

        return CustomerServiceObservation(
            ticket_id=obs_data.get("ticket_id", ""),
            task_id=obs_data.get("task_id", ""),
            task_title=obs_data.get("task_title", ""),
            task_level=TaskLevel(obs_data.get("task_level", TaskLevel.EASY.value)),
            customer_name=obs_data.get("customer_name", ""),
            customer_message=obs_data.get("customer_message", ""),
            category=self._parse_category(obs_data.get("category")),
            priority=TicketPriority(obs_data.get("priority", TicketPriority.MEDIUM.value)),
            status=TicketStatus(obs_data.get("status", TicketStatus.OPEN.value)),
            conversation_history=history,
            customer_sentiment=obs_data.get("customer_sentiment"),
            resolution_hint=obs_data.get("resolution_hint"),
            remaining_tickets=obs_data.get("remaining_tickets", 0),
            available_actions=[
                ActionType(value)
                for value in obs_data.get("available_actions", [])
            ],
            reward=result_payload.get("reward", obs_data.get("reward", 0.0)),
            done=result_payload.get("done", obs_data.get("done", False)),
            metadata=obs_data.get("metadata", {}),
            reward_details=self._parse_reward(obs_data.get("reward_details", {})),
            info=self._parse_info(obs_data.get("info", {})),
        )

    @staticmethod
    def _parse_category(value: str | None) -> TicketCategory | None:
        return TicketCategory(value) if value else None

    @staticmethod
    def _parse_reward(payload: Dict) -> CustomerServiceReward:
        return CustomerServiceReward(**payload) if payload else CustomerServiceReward()

    @staticmethod
    def _parse_info(payload: Dict) -> CustomerServiceInfo:
        return CustomerServiceInfo(**payload) if payload else CustomerServiceInfo()

    @staticmethod
    def _parse_ticket_outcome(payload: Dict) -> TicketOutcome:
        return TicketOutcome(**payload)
