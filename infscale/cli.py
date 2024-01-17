"""command line tool."""
import click

from infscale.cmd.run import run
from infscale.cmd.serve import serve
from infscale.version import VERSION


@click.group()
@click.version_option(version=VERSION)
def cli():  # noqa: D103
    pass


cli.add_command(run)
cli.add_command(serve)


if __name__ == "__main__":
    cli()
