"""serve subcommand."""
from __future__ import annotations

from typing import TYPE_CHECKING

import click
from infscale.constants import CONTROLLER_PORT, LOCALHOST

if TYPE_CHECKING:
    from io import BufferedReader


@click.command()
@click.option("--host", default=LOCALHOST, help="Controller's IP or hostname")
@click.option("--port", default=CONTROLLER_PORT, help="Controller's port number")
@click.argument("config", type=click.File("rb"))
def serve(host: str, port: int, config: BufferedReader):
    """Serve model based on config yaml file."""
    pass
