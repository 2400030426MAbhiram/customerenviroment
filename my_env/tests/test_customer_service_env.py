from my_env.graders import grade_task
from my_env.models import ActionType, CustomerServiceAction, TicketPriority
from my_env.server.my_env_environment import CustomerServiceEnvironment
from my_env.tasks import get_task


def test_easy_task_can_score_perfectly() -> None:
    env = CustomerServiceEnvironment(task_id="easy_billing_refund")
    env.reset(episode_id="easy_billing_refund")

    env.step(
        CustomerServiceAction(
            action_type=ActionType.CATEGORIZE,
            message="Classifying the duplicate charge.",
            category="billing",
            priority="high",
        )
    )
    env.step(
        CustomerServiceAction(
            action_type=ActionType.RESPOND,
            message="I am sorry about the duplicate charge. I will refund the extra payment.",
        )
    )
    final_obs = env.step(
        CustomerServiceAction(
            action_type=ActionType.RESOLVE,
            message="The refund has been submitted and your ticket is resolved.",
        )
    )

    assert final_obs.done is True
    assert env.state.final_grade is not None
    assert env.state.final_grade.score == 1.0


def test_medium_task_partial_progress_rewards_request_info() -> None:
    env = CustomerServiceEnvironment(task_id="medium_support_queue")
    env.reset(episode_id="medium_support_queue")

    env.step(
        CustomerServiceAction(
            action_type=ActionType.CATEGORIZE,
            message="Routing this login issue.",
            category="account",
            priority="high",
        )
    )
    observation = env.step(
        CustomerServiceAction(
            action_type=ActionType.REQUEST_INFO,
            message="Please confirm the email address tied to the locked account.",
        )
    )

    assert observation.reward > 0.0
    assert observation.reward_details.task_progress > 0.0
    assert observation.status.value in {"waiting_customer", "in_progress"}


def test_hard_task_penalizes_unnecessary_escalation() -> None:
    env = CustomerServiceEnvironment(task_id="hard_escalation_judgment")
    env.reset(episode_id="hard_escalation_judgment")

    env.step(
        CustomerServiceAction(
            action_type=ActionType.CATEGORIZE,
            message="Routing repeat-contact complaint.",
            category="general",
            priority=TicketPriority.URGENT,
        )
    )
    env.step(
        CustomerServiceAction(
            action_type=ActionType.RESPOND,
            message="I am sorry this has been so frustrating. I am escalating this with priority.",
        )
    )
    env.step(
        CustomerServiceAction(
            action_type=ActionType.ESCALATE,
            message="Escalating to Tier-2 for repeat-contact recovery.",
            escalation_reason="Repeat contact with unresolved issue",
        )
    )

    env.step(
        CustomerServiceAction(
            action_type=ActionType.CATEGORIZE,
            message="Routing the late shipment issue.",
            category="shipping",
            priority="high",
        )
    )
    observation = env.step(
        CustomerServiceAction(
            action_type=ActionType.ESCALATE,
            message="Escalating the shipping issue.",
            escalation_reason="Delayed order",
        )
    )

    assert observation.reward < 0.0
    assert "unnecessary_escalation" in observation.info.policy_violations


def test_medium_grader_returns_normalized_score() -> None:
    env = CustomerServiceEnvironment(task_id="medium_support_queue")
    env.reset(episode_id="medium_support_queue")
    task = get_task("medium_support_queue")

    grade = grade_task(task, env.state)

    assert 0.0 <= grade.score <= 1.0
