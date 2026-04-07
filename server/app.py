# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
FastAPI application for the SniffTest Environment.

Exposes SniffTestEnvironment over HTTP and WebSocket endpoints.

Endpoints:
    - POST /reset: Reset the environment (accepts task_level in body)
    - POST /step: Execute an InvestigateAction
    - GET /state: Get current environment state (includes adversarial metadata)
    - GET /schema: Get action/observation schemas
    - WS /ws: WebSocket endpoint for persistent sessions

Usage:
    uvicorn server.app:app --host 0.0.0.0 --port 8000
"""

try:
    from openenv.core.env_server.http_server import create_app
except Exception as e:  # pragma: no cover
    raise ImportError(
        "openenv is required for the web interface. Install dependencies with '\n    uv sync\n'"
    ) from e

try:
    from ..models import InvestigateAction, SniffTestObservation
    from .snifftest_environment import SniffTestEnvironment
except ImportError:
    from models import InvestigateAction, SniffTestObservation
    from server.snifftest_environment import SniffTestEnvironment


app = create_app(
    SniffTestEnvironment,
    InvestigateAction,
    SniffTestObservation,
    env_name="snifftest_env",
    max_concurrent_envs=1,
)


def main(host: str = "0.0.0.0", port: int = 8000):
    """Entry point for uv run server or python -m snifftest_env.server.app."""
    import uvicorn

    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()
    if args.port == 8000 and args.host == "0.0.0.0":
        main()
    else:
        main(host=args.host, port=args.port)
