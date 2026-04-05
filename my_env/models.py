"""Typed models for the customer service OpenEnv environment."""

from __future__ import annotations

from enum import Enum
from typing import Dict, List, Optional

from openenv.core.env_server.types import Action, Observation
from pydantic import BaseModel, Field


class TicketPriority(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    URGENT = "urgent"


class TicketCategory(str, Enum):
    BILLING = "billing"
    TECHNICAL = "technical"
    ACCOUNT = "account"
    PRODUCT = "product"
    SHIPPING = "shipping"
    GENERAL = "general"


class TicketStatus(str, Enum):
    OPEN = "open"
    IN_PROGRESS = "in_progress"
    WAITING_CUSTOMER = "waiting_customer"
    ESCALATED = "escalated"
    RESOLVED = "resolved"
    CLOSED = "closed"


class ActionType(str, Enum):
    RESPOND = "respond"
    ESCALATE = "escalate"
    RESOLVE = "resolve"
    REQUEST_INFO = "request_info"
    CATEGORIZE = "categorize"
    TRANSFER = "transfer"


class TaskLevel(str, Enum):
    EASY = "easy"
    MEDIUM = "medium"
    HARD = "hard"


class CustomerServiceAction(Action):
    """Action available to the support agent."""

    action_type: ActionType = Field(..., description="Type of action to perform")
    message: str = Field(..., description="Customer-facing reply or internal note")
    category: Optional[TicketCategory] = Field(
        default=None, description="Category to assign for categorize actions"
    )
    priority: Optional[TicketPriority] = Field(
        default=None, description="Priority to assign for categorize actions"
    )
    escalation_reason: Optional[str] = Field(
        default=None, description="Reason for escalation when action_type=escalate"
    )
    transfer_department: Optional[str] = Field(
        default=None, description="Target queue when action_type=transfer"
    )


class CustomerMessage(BaseModel):
    """A single conversational event in the ticket thread."""

    sender: str = Field(..., description="'customer' or 'agent'")
    content: str = Field(..., description="Message body")
    timestamp: str = Field(..., description="ISO-8601 timestamp")


class CustomerServiceReward(BaseModel):
    """Typed reward breakdown for the latest transition."""

    classification: float = Field(default=0.0, description="Reward for correct routing")
    communication: float = Field(default=0.0, description="Reward for response quality")
    task_progress: float = Field(default=0.0, description="Reward for moving the ticket forward")
    resolution: float = Field(default=0.0, description="Reward for a correct close or escalation")
    efficiency_penalty: float = Field(default=0.0, description="Penalty for loops or excess steps")
    safety_penalty: float = Field(default=0.0, description="Penalty for destructive decisions")
    total: float = Field(default=0.0, description="Scalar reward returned by the environment")
    normalized_progress: float = Field(
        default=0.0, ge=0.0, le=1.0, description="Normalized progress estimate for the active ticket"
    )


class CustomerServiceInfo(BaseModel):
    """Structured info payload for the latest transition."""

    task_id: str = Field(default="", description="Current task id")
    task_level: TaskLevel = Field(default=TaskLevel.EASY, description="Difficulty level")
    task_title: str = Field(default="", description="Human-readable task title")
    current_ticket_index: int = Field(default=0, description="1-based active ticket index")
    total_tickets: int = Field(default=0, description="Number of tickets in the task")
    remaining_tickets: int = Field(default=0, description="How many tickets remain after this step")
    expected_outcome: str = Field(default="", description="What a good agent should do next")
    recent_event: str = Field(default="", description="Lifecycle event for the current transition")
    policy_violations: List[str] = Field(default_factory=list, description="Detected mistakes")
    grader_score: Optional[float] = Field(
        default=None, ge=0.0, le=1.0, description="Final task score when the episode is complete"
    )


class TicketOutcome(BaseModel):
    """Outcome summary for a single ticket after it leaves the active queue."""

    scenario_id: str = Field(default="", description="Stable scenario identifier")
    ticket_id: str = Field(default="", description="Runtime ticket id")
    category_correct: bool = Field(default=False)
    priority_correct: bool = Field(default=False)
    responded: bool = Field(default=False)
    empathetic_response: bool = Field(default=False)
    issue_specific_response: bool = Field(default=False)
    requested_info: bool = Field(default=False)
    transferred: bool = Field(default=False)
    escalated: bool = Field(default=False)
    resolved: bool = Field(default=False)
    unnecessary_escalation: bool = Field(default=False)
    excessive_steps: bool = Field(default=False)
    policy_violations: List[str] = Field(default_factory=list)
    steps_taken: int = Field(default=0)
    reward_total: float = Field(default=0.0)
    final_satisfaction: float = Field(default=0.5, ge=0.0, le=1.0)

    @classmethod
    def placeholder(cls, scenario_id: str) -> "TicketOutcome":
        return cls(scenario_id=scenario_id)


class TaskGrade(BaseModel):
    """Deterministic 0.0-1.0 task grade."""

    task_id: str
    task_title: str
    score: float = Field(..., ge=0.0, le=1.0)
    passed: bool
    per_ticket_scores: Dict[str, float] = Field(default_factory=dict)
    metrics: Dict[str, float] = Field(default_factory=dict)
    summary: str = Field(default="")


class CustomerServiceObservation(Observation):
    """Observation returned by reset() and step()."""

    ticket_id: str = Field(default="", description="Unique runtime ticket id")
    task_id: str = Field(default="", description="Current task identifier")
    task_title: str = Field(default="", description="Current task title")
    task_level: TaskLevel = Field(default=TaskLevel.EASY, description="Task difficulty")
    customer_name: str = Field(default="", description="Customer display name")
    customer_message: str = Field(default="", description="Latest customer message")
    category: Optional[TicketCategory] = Field(
        default=None, description="Current ticket category chosen by the agent"
    )
    priority: TicketPriority = Field(
        default=TicketPriority.MEDIUM, description="Current ticket priority chosen by the agent"
    )
    status: TicketStatus = Field(default=TicketStatus.OPEN, description="Current ticket status")
    conversation_history: List[CustomerMessage] = Field(
        default_factory=list, description="Conversation transcript so far"
    )
    customer_sentiment: Optional[str] = Field(
        default=None, description="Customer sentiment label"
    )
    resolution_hint: Optional[str] = Field(
        default=None, description="Internal hint about how the issue can be solved"
    )
    remaining_tickets: int = Field(default=0, description="How many tickets remain after this one")
    available_actions: List[ActionType] = Field(
        default_factory=list, description="Actions the agent may choose from"
    )
    reward_details: CustomerServiceReward = Field(
        default_factory=CustomerServiceReward, description="Typed reward breakdown"
    )
    info: CustomerServiceInfo = Field(
        default_factory=CustomerServiceInfo, description="Typed info payload"
    )


class CustomerServiceState(BaseModel):
    """Serializable environment state returned by the OpenEnv state endpoint."""

    episode_id: str = Field(default="", description="Current episode identifier")
    task_id: str = Field(default="", description="Active task id")
    task_title: str = Field(default="", description="Active task title")
    task_level: TaskLevel = Field(default=TaskLevel.EASY, description="Difficulty level")
    step_count: int = Field(default=0, description="Steps taken in the episode")
    current_ticket_index: int = Field(default=0, description="1-based ticket index")
    total_tickets: int = Field(default=0, description="Number of tickets in the task")
    remaining_tickets: int = Field(default=0, description="Tickets left in the task")
    ticket_id: str = Field(default="", description="Runtime ticket id")
    status: TicketStatus = Field(default=TicketStatus.OPEN, description="Current ticket status")
    category: Optional[TicketCategory] = Field(default=None, description="Current ticket category")
    priority: TicketPriority = Field(default=TicketPriority.MEDIUM, description="Current priority")
    satisfaction_score: float = Field(
        default=0.5, ge=0.0, le=1.0, description="Current simulated customer satisfaction"
    )
    resolved: bool = Field(default=False, description="Whether the active ticket is resolved")
    escalated: bool = Field(default=False, description="Whether the active ticket is escalated")
    total_tickets_resolved: int = Field(default=0)
    total_tickets_escalated: int = Field(default=0)
    cumulative_reward: float = Field(default=0.0, description="Total reward over the episode")
    action_history: List[str] = Field(default_factory=list, description="Compact action summaries")
    completed_tickets: List[TicketOutcome] = Field(
        default_factory=list, description="Finished ticket summaries"
    )
    last_reward: CustomerServiceReward = Field(default_factory=CustomerServiceReward)
    last_info: CustomerServiceInfo = Field(default_factory=CustomerServiceInfo)
    final_grade: Optional[TaskGrade] = Field(
        default=None, description="Task score once the episode is complete"
    )
