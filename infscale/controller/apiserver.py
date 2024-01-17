"""apiserver class."""
from __future__ import annotations

import asyncio
from enum import Enum
from typing import TYPE_CHECKING

from fastapi import FastAPI, Request
from infscale.constants import APISERVER_PORT
from pydantic import BaseModel
from uvicorn import Config, Server

if TYPE_CHECKING:
    from infscale.controller.controller import Controller


app = FastAPI()


class ApiServer:
    """ApiServer class."""

    def __init__(self, ctrl: Controller, port: int = APISERVER_PORT):
        """Initialize an instance."""
        self.api = API(ctrl)
        self.port = port

    async def run(self):
        """Run apiserver."""
        config = Config(
            app=app,
            host="0.0.0.0",
            port=self.port,
            loop=asyncio.get_event_loop(),
        )

        server = Server(config)
        await server.serve()


class ReqType(str, Enum):
    """Enum class for request type."""

    UNKNOWN = "unknown"
    SERVE = "serve"


class ServeSpec(BaseModel):
    """ServiceSpec model."""

    name: str
    model: str
    num_failures: int


class Response(BaseModel):
    """Response model."""

    message: str


class API:
    """Collection of API."""

    def __init__(self, ctrl: Controller):
        """Initialize instance."""
        self.ctrl = ctrl

    @app.post("/models", response_model=Response)
    def serve(self, spec: ServeSpec):
        """Serve a model."""
        self.ctrl.handle_fastapi_request(ReqType.SERVE, spec)

        res = {"message": "started serving"}
        return res
