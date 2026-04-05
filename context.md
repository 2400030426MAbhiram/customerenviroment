# Project Context: Autonomous Customer Care Service

## Mission
Building a scalable, AI-driven Customer Care backend for the OpenEnv hackathon. The service simulates a realistic customer support ticket queue where an agent must triage, respond to, escalate, and resolve Tier-1/Tier-2 support tickets autonomously.

## Critical Directives
You are operating inside a structured, organizer-managed OpenEnv environment. Before writing any code or proposing changes, read:
1. `.claude/docs/REPO_WALKTHROUGH.md` — component locations
2. `.claude/docs/INVARIANTS.md` — strict rules and boundaries

## Tech Stack
* **Language:** Python 3.10+
* **Package Management:** `uv` (adhere to `pyproject.toml` and `uv.lock`)
* **Framework:** FastAPI via OpenEnv's `create_app()` + WebSocket
* **Core Dependency:** `openenv-core[core]>=0.2.2`
* **Infrastructure:** Containerized via Docker (see `my_env/Dockerfile`)
* **Agentic Tools:** `.claude/skills/` and `.claude/agents/`

## Current Implementation Status

### Completed
- **models.py** — Typed Pydantic models: `CustomerServiceAction` (6 action types: respond, escalate, resolve, request_info, categorize, transfer), `CustomerServiceObservation` (ticket_id, customer_name, message, category, priority, status, conversation history, sentiment, resolution hint), `CustomerServiceState` (episode tracking, satisfaction score, resolution counters). Enums: `TicketPriority`, `TicketCategory`, `TicketStatus`, `ActionType`.
- **server/my_env_environment.py** — `CustomerServiceEnvironment` with full `reset()`/`step()`/`state()`. 8 sample tickets with varied categories, priorities, and sentiments. Reward signals: correct categorization (+0.2), correct priority (+0.1), empathetic response to angry customers (+0.2/-0.3), resolution (+0.5, +0.3 bonus if satisfaction >0.7), appropriate escalation (+0.3), step limit penalty (-0.2). 10-step limit per ticket, auto-advances queue.
- **client.py** — `CustomerServiceEnv(EnvClient[CustomerServiceAction, CustomerServiceObservation, CustomerServiceState])` with full payload serialization and enum-safe parsing.
- **server/app.py** — FastAPI app via `create_app()`, 5 concurrent sessions, standard endpoints (POST /reset, POST /step, GET /state, GET /schema, WS /ws).
- **__init__.py** — Exports all public types and client class.
- **openenv.yaml** — Basic metadata (needs task definitions).
- **pyproject.toml** — Dependencies configured.
- **Dockerfile** — Multi-stage build with openenv-base.
- **CLAUDE.md** — Repository guidance for Claude Code.

### In Progress — Next Sprint
- **3 Graded Tasks with Agent Graders** (easy/medium/hard, scoring 0.0-1.0):
  - **Task 1 (Easy): Single Ticket Triage** — Categorize + prioritize 1 straightforward billing ticket. Grader checks categorization accuracy and priority match.
  - **Task 2 (Medium): Multi-Ticket Resolution** — Handle 3 tickets end-to-end: categorize, respond empathetically, resolve. Graded on resolution rate + satisfaction score.
  - **Task 3 (Hard): Escalation Judgment** — Handle 5 tickets including angry repeat-contact customers, ambiguous priorities, escalation-vs-resolve decisions. Graded on correct escalation calls, no unnecessary escalations, overall satisfaction.
- **Meaningful Reward Function** — Already has partial progress signals; needs per-task normalization to 0.0-1.0 scale.
- **Baseline Inference Script** — OpenAI API client that runs a model against all 3 tasks, reads `OPENAI_API_KEY` from env, produces reproducible scores.
- **README.md** — Needs full rewrite: environment description, action/observation spaces, setup instructions, task descriptions, baseline results.
- **openenv.yaml** — Needs task metadata and grader definitions.

## Environment Architecture

```
my_env/
├── __init__.py              # Exports: CustomerServiceEnv, all models & enums
├── models.py                # Pydantic Action, Observation, State + enums
├── client.py                # EnvClient subclass (WebSocket)
├── openenv.yaml             # Environment manifest
├── pyproject.toml           # Dependencies
├── Dockerfile               # Multi-stage Docker build
├── README.md                # Environment docs
└── server/
    ├── __init__.py
    ├── app.py               # FastAPI create_app() entry point
    ├── my_env_environment.py # CustomerServiceEnvironment (reset/step/state)
    └── requirements.txt     # Docker dependencies
```

### Dual-API Model (OpenEnv Invariant)
| Boundary | API | Consumer |
|----------|-----|----------|
| Agent | MCP tools | AI agent interacting with environment |
| Infrastructure | Gym-like (reset/step/state) | Training orchestration via WebSocket |

Agents must NEVER access reset/step/state directly.

## State Management Protocol
1. Check `memory.md` when starting a new session for prior context.
2. Update `memory.md` with resolution and next steps before ending a session.
