# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Autonomous Customer Care Service built on the **OpenEnv** framework (Meta) for an OpenEnv hackathon. The service ingests customer inquiries, routes them intelligently, and uses agentic tools to resolve Tier-1/Tier-2 support tickets. Each environment follows the standard OpenEnv pattern with a client, server, and Pydantic models.

## Before Writing Code

Read these files first — they define the rules and structure:
- `.claude/docs/INVARIANTS.md` — strict rules that must never be violated
- `.claude/docs/REPO_WALKTHROUGH.md` — where components live
- `context.md` — current sprint focus and mission details

## Build & Run Commands

```bash
# Install dependencies (from an environment directory, e.g. my_env/)
cd my_env && uv sync

# Run server locally (from environment directory)
uvicorn server.app:app --reload --host 0.0.0.0 --port 8000
# Or via entry point:
uv run --project . server

# Run tests
PYTHONPATH=src:envs uv run pytest tests/ -v --tb=short

# Run a single test
PYTHONPATH=src:envs uv run pytest tests/path/test_file.py::test_name -v

# Run tests via hook (excludes environments needing special setup)
bash .claude/hooks/test.sh

# Lint
uv run ruff check .

# Docker build (from root)
docker build -t my_env-env .
```

## Architecture

### Dual-API Model (Critical)

OpenEnv exposes two separate APIs to two different consumers:

| Boundary | API | Consumer |
|----------|-----|----------|
| **Agent** | MCP (Model Context Protocol) | The AI agent interacting with the environment |
| **Infrastructure** | Gym-like (`reset`, `step`, `state`) | Training orchestration via WebSocket |

The agent **must never** access `reset()`, `step()`, or `state()` — these are infrastructure-only. Exposing simulation control to agents breaks the training paradigm.

### Environment Structure

Every environment (`my_env/`, `echo_env/`, `create_app/`) follows this pattern:

```
<env_name>/
├── __init__.py          # Exports Action, Observation, Client
├── models.py            # Pydantic Action & Observation (shared between client/server)
├── client.py            # EnvClient[ActT, ObsT, StateT] subclass
├── openenv.yaml         # Environment manifest
├── pyproject.toml       # Dependencies (uses openenv-core[core]>=0.2.2)
└── server/
    ├── app.py               # create_app() with HTTPEnvServer
    ├── <name>_environment.py # Environment[ActT, ObsT, StateT] subclass
    ├── requirements.txt     # Docker dependencies
    └── Dockerfile
```

Scaffold new environments with `openenv init <name>`.

### Key Separation Rules

- **Clients must never import from `server/`** — shared types go in `models.py`
- **Server must never import client code**
- **All wire types** (Action, Observation, State) must be Pydantic models
- **Rewards** are computed inside the environment, not externally

## Key Invariants

1. **Gymnasium API signatures** (`reset`, `step`, `state`) must not change without a major version bump
2. **Generic type safety** — environments use `Environment[ActT, ObsT, StateT]`, clients use `EnvClient[ActT, ObsT, StateT]`
3. **Agent isolation** — agents cannot access reset/simulation controls
4. **Container isolation** — environments run in isolated Docker containers

If a change would violate an invariant, **stop and flag for human review** with an ALIGNMENT FLAG.

## TDD Workflow

```bash
# Start TDD on a GitHub issue (creates worktree + enforces Red-Green-Refactor)
/work-on-issue #42

# Manual worktree without TDD
.claude/scripts/worktree-create.sh my-feature

# Parallel multi-issue batch
/sprint 67,68,69
```

Cycle: `/write-tests` (Red) → `/implement` (Green) → `/update-docs` → `/simplify` (Refactor) → `/pre-submit-pr`

## Package Management

Uses `uv` exclusively. Dependencies are declared in each environment's `pyproject.toml` and locked in `uv.lock`. The core dependency is `openenv-core[core]>=0.2.2`.
