# Customer Service Queue Environment

`my_env` is an OpenEnv environment that simulates a real customer-support workflow. An agent works a queue of inbound support tickets and must behave like a Tier-1 operations teammate: triage the issue, choose the right priority, respond with useful guidance, request missing information when needed, transfer or escalate when appropriate, and close the ticket cleanly.

This is a real-world task simulation, not a toy benchmark. The ticket set models work humans actually do in customer support teams: duplicate-charge refunds, account lockouts, late shipments, product bug reporting, and enterprise-plan routing.

## OpenEnv Interface

The environment implements the standard OpenEnv server contract:

- `reset(seed=None, episode_id=None) -> CustomerServiceObservation`
- `step(action: CustomerServiceAction) -> CustomerServiceObservation`
- `state -> CustomerServiceState`

Typed models:

- Action: `CustomerServiceAction`
- Observation: `CustomerServiceObservation`
- Reward: `CustomerServiceReward`
- State: `CustomerServiceState`
- Grading result: `TaskGrade`

`step()` returns an observation whose inherited OpenEnv fields carry the scalar `reward`, `done`, and `metadata`, and whose typed fields include `reward_details` and `info` for structured inspection.

## Action Space

`CustomerServiceAction` supports six actions:

- `categorize`: assign `category` and `priority`
- `respond`: send a customer-facing reply
- `request_info`: ask for missing details
- `transfer`: route the ticket to another internal queue
- `escalate`: hand the ticket to Tier-2 or a specialist
- `resolve`: close the ticket

Optional structured fields:

- `category`: `billing | technical | account | product | shipping | general`
- `priority`: `low | medium | high | urgent`
- `escalation_reason`
- `transfer_department`

## Observation Space

Each `CustomerServiceObservation` includes:

- task context: `task_id`, `task_title`, `task_level`
- active ticket data: `ticket_id`, `customer_name`, `customer_message`, `customer_sentiment`
- agent state: `category`, `priority`, `status`
- transcript: `conversation_history`
- assistance: `resolution_hint`, `available_actions`
- progress: `remaining_tickets`, `reward_details`, `info`
- OpenEnv wire fields: `reward`, `done`, `metadata`

## Reward Function

The scalar reward is built from a typed `CustomerServiceReward` breakdown:

- `classification`: rewards correct category and priority assignment
- `communication`: rewards empathy and issue-specific responses
- `task_progress`: rewards useful moves like requesting missing account info or transferring to the correct queue
- `resolution`: rewards correct resolution or escalation decisions
- `efficiency_penalty`: penalizes loops, generic replies, and blown step budgets
- `safety_penalty`: penalizes harmful actions like unnecessary escalation, wrong triage, or resolving tickets that should be escalated

This creates partial progress signals over the whole trajectory instead of a binary end-of-episode reward.

## Tasks and Graders

The environment ships with three deterministic tasks, each scored from `0.0` to `1.0` by a programmatic grader:

1. `easy_billing_refund`
   Single-ticket duplicate-charge refund triage.
2. `medium_support_queue`
   Three-ticket Tier-1 queue with one request-for-info dependency.
3. `hard_escalation_judgment`
   Five-ticket queue where correct escalation judgment is critical.

The grader is deterministic and operates on the final `CustomerServiceState.completed_tickets` trail. It scores category/priority accuracy, response quality, request-for-info usage, escalation correctness, resolution correctness, efficiency, and customer satisfaction.

## Setup

From [`my_env`](/C:/Users/HP/customerenviroment/my_env):

```bash
uv sync
```

Run the server locally:

```bash
uv run server
```

Validate the environment:

```bash
uv run openenv validate
```

Run the tests:

```bash
uv run pytest tests
```

## Baseline Inference Script

The package includes a reproducible baseline runner that uses the OpenAI API client and reads credentials from `OPENAI_API_KEY`.

```bash
$env:OPENAI_API_KEY="..."
uv run baseline --model gpt-4.1-mini --seed 7
```

Baseline behavior is stabilized by:

- deterministic task IDs
- fixed model prompt
- `temperature=0`
- fixed OpenAI `seed`
- fixed per-task step budgets

The script prints JSON with per-task scores and a mean score across all three tasks.

## Project Layout

```text
my_env/
|-- __init__.py
|-- baseline.py
|-- client.py
|-- graders.py
|-- models.py
|-- openenv.yaml
|-- tasks.py
|-- tests/
|   `-- test_customer_service_env.py
`-- server/
    |-- app.py
    `-- my_env_environment.py
```
