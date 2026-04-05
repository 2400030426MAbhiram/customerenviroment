"""
FastAPI application for the Customer Service Environment.

This module creates an HTTP server that exposes the CustomerServiceEnvironment
over HTTP and WebSocket endpoints, compatible with EnvClient.

Endpoints:
    - POST /reset: Reset the environment (new ticket queue)
    - POST /step: Execute an action on the current ticket
    - GET /state: Get current environment state
    - GET /schema: Get action/observation schemas
    - WS /ws: WebSocket endpoint for persistent sessions

Usage:
    # Development (with auto-reload):
    uvicorn server.app:app --reload --host 0.0.0.0 --port 8000

    # Production:
    uvicorn server.app:app --host 0.0.0.0 --port 8000 --workers 4

    # Or run directly:
    python -m server.app
"""

try:
    from openenv.core.env_server.http_server import create_app
except Exception as e:  # pragma: no cover
    raise ImportError(
        "openenv is required for the web interface. Install dependencies with '\n    uv sync\n'"
    ) from e

try:
    from ..models import CustomerServiceAction, CustomerServiceObservation
    from .my_env_environment import CustomerServiceEnvironment
except ModuleNotFoundError:
    from models import CustomerServiceAction, CustomerServiceObservation
    from server.my_env_environment import CustomerServiceEnvironment


# Create the app with web interface and README integration
app = create_app(
    CustomerServiceEnvironment,
    CustomerServiceAction,
    CustomerServiceObservation,
    env_name="my_env",
    max_concurrent_envs=5,  # support multiple concurrent agent sessions
)


def main() -> None:
    """Entry point for direct execution via uv run or python -m."""
    import uvicorn
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
