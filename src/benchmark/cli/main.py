"""
Main CLI entry point for the LLM Cybersecurity Benchmark system.

This module provides the primary command-line interface that aggregates
all available CLI commands and subcommands.
"""

import click
from rich.console import Console

from benchmark.cli.config_commands import config

console = Console()


@click.group()
@click.version_option(version="1.0.0", prog_name="benchmark")
@click.option("--verbose", "-v", is_flag=True, help="Enable verbose output")
def cli(verbose: bool) -> None:
    """
    🚀 LLM Cybersecurity Benchmark CLI

    A comprehensive toolkit for benchmarking large language models on
    cybersecurity tasks. This CLI provides tools for configuration
    management, experiment execution, and result analysis.
    """
    if verbose:
        console.print("[dim]Verbose mode enabled[/dim]")


# Add command groups
cli.add_command(config)


@cli.command()
def info() -> None:
    """Show system information and status."""
    console.print("\n[bold blue]🛡️  LLM Cybersecurity Benchmark System[/bold blue]")
    console.print("─" * 50)

    # System info
    import sys
    from pathlib import Path

    console.print(f"[dim]Python version:[/dim] {sys.version.split()[0]}")
    console.print(f"[dim]System platform:[/dim] {sys.platform}")
    console.print(f"[dim]Working directory:[/dim] {Path.cwd()}")

    # Available commands
    console.print("\n[bold]📋 Available Commands:[/bold]")
    console.print("  [blue]config[/blue]     - Configuration management")
    console.print("    ├── validate   - Validate configuration files")
    console.print("    ├── generate   - Generate sample configurations")
    console.print("    ├── show       - Display parsed configurations")
    console.print("    └── check-env  - Check environment variables")

    console.print("\n[bold green]✅ System Status:[/bold green] Ready")

    # Quick start guide
    console.print("\n[bold blue]🚀 Quick Start:[/bold blue]")
    console.print("1. Generate a config:  [bold]benchmark config generate[/bold]")
    console.print("2. Validate config:    [bold]benchmark config validate config.yaml[/bold]")
    console.print("3. Check environment:  [bold]benchmark config check-env config.yaml[/bold]")


@cli.command()
def version() -> None:
    """Show version information."""
    console.print("[bold blue]LLM Cybersecurity Benchmark[/bold blue] v1.0.0")
    console.print("[dim]A comprehensive benchmarking toolkit for LLM security evaluation[/dim]")


if __name__ == "__main__":
    cli()
