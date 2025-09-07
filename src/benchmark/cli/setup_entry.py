"""
Setup entry point for the CLI to be used with setuptools.

This module provides the main entry point that can be registered
as a console script in setup.py or pyproject.toml.
"""

from benchmark.cli.main import cli


def main() -> None:
    """Main entry point for the benchmark CLI."""
    cli()


if __name__ == "__main__":
    main()
