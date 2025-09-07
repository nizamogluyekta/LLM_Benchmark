"""
CLI commands for configuration management and validation.

This module provides comprehensive command-line tools for working with
benchmark configurations, including validation, generation, and inspection.
"""

import asyncio
import json
import os
import sys
from pathlib import Path
from typing import Any

import click
import yaml
from rich.console import Console
from rich.panel import Panel
from rich.prompt import Confirm, Prompt
from rich.syntax import Syntax
from rich.table import Table

from benchmark.services.configuration_service import ConfigurationService

console = Console()


@click.group()
def config() -> None:
    """Configuration management commands for benchmark experiments."""
    pass


@config.command()
@click.argument("config_file", type=click.Path(exists=True))
@click.option("--quiet", "-q", is_flag=True, help="Show only errors and critical warnings")
@click.option("--json-output", is_flag=True, help="Output validation results as JSON")
def validate(config_file: str, quiet: bool, json_output: bool) -> None:
    """Validate configuration file and show comprehensive analysis.

    This command performs extensive validation including:
    - Configuration structure and syntax
    - Model compatibility and API key format
    - Dataset availability and format validation
    - Resource requirement analysis
    - Performance optimization recommendations
    - Cross-field consistency checks
    """

    async def _validate() -> None:
        config_path = Path(config_file)

        try:
            # Initialize configuration service
            service = ConfigurationService()
            await service.initialize()

            if not json_output:
                console.print(
                    f"\n[bold blue]🔍 Validating configuration:[/bold blue] {config_path.name}"
                )
                console.print("─" * 60)

            # Load configuration
            try:
                experiment_config = await service.load_experiment_config(config_path)
                if not json_output:
                    console.print("[green]✓[/green] Configuration loaded successfully")
                    console.print(f"[dim]Experiment:[/dim] {experiment_config.name}")
            except Exception as e:
                if json_output:
                    result = {"success": False, "error": str(e), "warnings": []}
                    click.echo(json.dumps(result, indent=2))
                else:
                    console.print(f"[red]✗ Failed to load configuration:[/red] {str(e)}")
                await service.shutdown()
                sys.exit(1)

            # Run validation
            try:
                warnings = await service.validate_config(experiment_config)

                if json_output:
                    # JSON output format
                    result = {
                        "success": True,
                        "config_name": experiment_config.name,
                        "total_warnings": len(warnings),
                        "warnings": [
                            {
                                "message": w,
                                "level": "warning",  # Simplified for JSON
                            }
                            for w in warnings
                        ],
                    }
                    click.echo(json.dumps(result, indent=2))
                else:
                    # Rich console output
                    await _display_validation_results(warnings, quiet)

            except Exception as e:
                if json_output:
                    result = {
                        "success": False,
                        "error": f"Validation failed: {str(e)}",
                        "warnings": [],
                    }
                    click.echo(json.dumps(result, indent=2))
                else:
                    console.print(f"[red]✗ Validation failed:[/red] {str(e)}")
                await service.shutdown()
                sys.exit(1)

            await service.shutdown()

            # Exit with appropriate code
            if warnings and not json_output:
                sys.exit(
                    1
                    if any("error" in w.lower() or "critical" in w.lower() for w in warnings)
                    else 0
                )

        except Exception as e:
            if json_output:
                result = {"success": False, "error": f"Unexpected error: {str(e)}", "warnings": []}
                click.echo(json.dumps(result, indent=2))
            else:
                console.print(f"[red]✗ Unexpected error:[/red] {str(e)}")
            sys.exit(1)

    asyncio.run(_validate())


async def _display_validation_results(warnings: list[str], quiet: bool) -> None:
    """Display validation results with rich formatting."""
    if not warnings:
        console.print("\n[bold green]🎉 Configuration validation passed![/bold green]")
        console.print("[green]No issues found. Configuration is ready for use.[/green]")
        return

    # Categorize warnings by type
    error_warnings = []
    critical_warnings = []
    warning_warnings = []
    info_warnings = []

    for warning in warnings:
        lower_warning = warning.lower()
        if "error" in lower_warning or "missing" in lower_warning or "not found" in lower_warning:
            error_warnings.append(warning)
        elif "critical" in lower_warning:
            critical_warnings.append(warning)
        elif "warning" in lower_warning or "exceeds" in lower_warning or "invalid" in lower_warning:
            warning_warnings.append(warning)
        else:
            info_warnings.append(warning)

    # Display summary
    total_issues = len(error_warnings) + len(critical_warnings) + len(warning_warnings)
    if not quiet:
        total_info = len(info_warnings)
        console.print("\n[bold yellow]📋 Validation Summary:[/bold yellow]")
        console.print(f"  [red]● Errors: {len(error_warnings)}[/red]")
        console.print(f"  [magenta]● Critical: {len(critical_warnings)}[/magenta]")
        console.print(f"  [yellow]● Warnings: {len(warning_warnings)}[/yellow]")
        console.print(f"  [blue]● Info: {total_info}[/blue]")

    # Display detailed results
    if error_warnings:
        console.print(f"\n[bold red]🚨 Errors ({len(error_warnings)}):[/bold red]")
        for warning in error_warnings:
            console.print(f"  [red]✗[/red] {warning}")

    if critical_warnings:
        console.print(
            f"\n[bold magenta]⚠️  Critical Issues ({len(critical_warnings)}):[/bold magenta]"
        )
        for warning in critical_warnings:
            console.print(f"  [magenta]![/magenta] {warning}")

    if warning_warnings:
        console.print(f"\n[bold yellow]⚡ Warnings ({len(warning_warnings)}):[/bold yellow]")
        for warning in warning_warnings:
            console.print(f"  [yellow]▲[/yellow] {warning}")

    if info_warnings and not quiet:
        console.print(f"\n[bold blue]💡 Recommendations ({len(info_warnings)}):[/bold blue]")
        for warning in info_warnings:
            console.print(f"  [blue]i[/blue] {warning}")

    # Overall status
    if error_warnings or critical_warnings:
        console.print(
            f"\n[bold red]❌ Configuration has {total_issues} issue(s) that need attention.[/bold red]"
        )
    elif warning_warnings:
        console.print(
            f"\n[bold yellow]⚠️  Configuration is functional but has {len(warning_warnings)} warning(s).[/bold yellow]"
        )
    else:
        console.print(
            f"\n[bold green]✅ Configuration is valid with {len(info_warnings)} optimization suggestion(s).[/bold green]"
        )


@config.command()
@click.option("--output", "-o", default="config.yaml", help="Output file name")
@click.option("--interactive", "-i", is_flag=True, help="Interactive configuration generation")
@click.option("--format", type=click.Choice(["yaml", "json"]), default="yaml", help="Output format")
def generate(output: str, interactive: bool, format: str) -> None:
    """Generate sample configuration file with optional interactive mode.

    Creates a complete configuration template with example models, datasets,
    and evaluation settings. Interactive mode allows customization of the
    generated configuration.
    """

    async def _generate() -> None:
        service = ConfigurationService()
        await service.initialize()

        try:
            # Get default configuration
            default_config = await service.get_default_config()

            if interactive:
                console.print("\n[bold blue]🚀 Interactive Configuration Generator[/bold blue]")
                console.print("Let's create a customized configuration for your experiment.\n")

                # Customize configuration interactively
                default_config = await _interactive_config_generation(default_config)

            # Write configuration file
            output_path = Path(output)

            # Ensure proper extension
            if format == "json" and not output_path.suffix:
                output_path = output_path.with_suffix(".json")
            elif format == "yaml" and not output_path.suffix:
                output_path = output_path.with_suffix(".yaml")

            if format == "json":
                with open(output_path, "w", encoding="utf-8") as f:
                    json.dump(default_config, f, indent=2)
            else:
                with open(output_path, "w", encoding="utf-8") as f:
                    yaml.dump(default_config, f, indent=2, default_flow_style=False)

            console.print("\n[green]✅ Configuration generated successfully![/green]")
            console.print(f"[dim]Output file:[/dim] {output_path}")
            console.print(f"[dim]Format:[/dim] {format.upper()}")

            if interactive:
                console.print("\n[blue]💡 Next steps:[/blue]")
                console.print("1. Review the generated configuration")
                console.print("2. Update API keys and dataset paths")
                console.print(
                    f"3. Validate with: [bold]benchmark config validate {output_path}[/bold]"
                )

        except Exception as e:
            console.print(f"[red]✗ Failed to generate configuration:[/red] {str(e)}")
            sys.exit(1)
        finally:
            await service.shutdown()

    asyncio.run(_generate())


async def _interactive_config_generation(config: dict[str, Any]) -> dict[str, Any]:
    """Interactive configuration customization."""

    # Experiment basics
    console.print("[bold]📝 Experiment Information[/bold]")
    name = Prompt.ask("Experiment name", default=config["name"])
    description = Prompt.ask("Description", default=config["description"])

    config["name"] = name
    config["description"] = description

    # Models configuration
    console.print("\n[bold]🤖 Model Configuration[/bold]")

    # Ask about model types
    use_openai = Confirm.ask("Include OpenAI models?", default=True)
    use_anthropic = Confirm.ask("Include Anthropic models?", default=True)

    new_models = []
    if use_openai:
        console.print("  [blue]Configuring OpenAI model...[/blue]")
        openai_model = {
            "name": Prompt.ask("  Model name", default="gpt-3.5-turbo"),
            "type": "openai_api",
            "path": Prompt.ask("  Model path", default="gpt-3.5-turbo"),
            "config": {"api_key": "${OPENAI_API_KEY}"},
            "max_tokens": int(Prompt.ask("  Max tokens", default="1024")),
            "temperature": float(Prompt.ask("  Temperature", default="0.1")),
        }
        new_models.append(openai_model)

    if use_anthropic:
        console.print("  [blue]Configuring Anthropic model...[/blue]")
        anthropic_model = {
            "name": Prompt.ask("  Model name", default="claude-3-haiku"),
            "type": "anthropic_api",
            "path": Prompt.ask("  Model path", default="claude-3-haiku-20240307"),
            "config": {"api_key": "${ANTHROPIC_API_KEY}"},
            "max_tokens": int(Prompt.ask("  Max tokens", default="1024")),
            "temperature": float(Prompt.ask("  Temperature", default="0.1")),
        }
        new_models.append(anthropic_model)

    config["models"] = new_models

    # Dataset configuration
    console.print("\n[bold]📊 Dataset Configuration[/bold]")
    dataset_path = Prompt.ask("Dataset path", default="./data/samples.jsonl")
    max_samples = int(Prompt.ask("Maximum samples", default="1000"))

    config["datasets"][0]["path"] = dataset_path
    config["datasets"][0]["max_samples"] = max_samples

    # Evaluation settings
    console.print("\n[bold]⚙️  Evaluation Settings[/bold]")
    parallel_jobs = int(Prompt.ask("Parallel jobs", default="2"))
    batch_size = int(Prompt.ask("Batch size", default="16"))
    timeout = int(Prompt.ask("Timeout (minutes)", default="30"))

    config["evaluation"]["parallel_jobs"] = parallel_jobs
    config["evaluation"]["batch_size"] = batch_size
    config["evaluation"]["timeout_minutes"] = timeout

    return config


@config.command()
@click.argument("config_file", type=click.Path(exists=True))
@click.option(
    "--format",
    type=click.Choice(["yaml", "json"]),
    help="Output format (auto-detect if not specified)",
)
@click.option("--pretty", is_flag=True, default=True, help="Pretty print with syntax highlighting")
def show(config_file: str, format: str, pretty: bool) -> None:
    """Display parsed configuration with syntax highlighting.

    Shows the complete configuration after environment variable resolution
    and validation. Useful for debugging configuration issues.
    """

    async def _show() -> None:
        config_path = Path(config_file)

        try:
            # Initialize service
            service = ConfigurationService()
            await service.initialize()

            # Load and parse configuration
            experiment_config = await service.load_experiment_config(config_path)

            # Convert to dictionary for display
            config_dict = experiment_config.model_dump()

            # Determine format
            if not format:
                output_format = "json" if config_path.suffix.lower() == ".json" else "yaml"
            else:
                output_format = format

            # Generate output
            if output_format == "json":
                content = json.dumps(config_dict, indent=2)
                syntax_lang = "json"
            else:
                content = yaml.dump(config_dict, indent=2, default_flow_style=False)
                syntax_lang = "yaml"

            # Display with rich formatting
            if pretty:
                console.print(f"\n[bold blue]📋 Configuration:[/bold blue] {config_path.name}")
                console.print("─" * 60)

                # Show basic info
                table = Table(show_header=False, box=None)
                table.add_row("[dim]Name:[/dim]", experiment_config.name)
                table.add_row("[dim]Models:[/dim]", str(len(experiment_config.models)))
                table.add_row("[dim]Datasets:[/dim]", str(len(experiment_config.datasets)))
                table.add_row("[dim]Output:[/dim]", experiment_config.output_dir or "Not specified")
                console.print(table)

                console.print(f"\n[bold]📄 Configuration Content ({output_format.upper()}):[/bold]")
                syntax = Syntax(content, syntax_lang, theme="monokai", line_numbers=True)
                console.print(Panel(syntax, border_style="blue"))
            else:
                # Plain output
                click.echo(content)

            await service.shutdown()

        except Exception as e:
            console.print(f"[red]✗ Failed to display configuration:[/red] {str(e)}")
            sys.exit(1)

    asyncio.run(_show())


@config.command("check-env")
@click.argument("config_file", type=click.Path(exists=True))
@click.option("--set-missing", is_flag=True, help="Prompt to set missing environment variables")
def check_env(config_file: str, set_missing: bool) -> None:
    """Check environment variable requirements for configuration.

    Analyzes the configuration file to identify required environment variables
    and checks if they are properly set in the current environment.
    """

    async def _check_env() -> None:
        config_path = Path(config_file)

        try:
            # Initialize service
            service = ConfigurationService()
            await service.initialize()

            # Load configuration
            experiment_config = await service.load_experiment_config(config_path)

            # Get required environment variables
            required_vars = service.get_required_env_vars(experiment_config)

            console.print(
                f"\n[bold blue]🔍 Environment Variables Check:[/bold blue] {config_path.name}"
            )
            console.print("─" * 60)

            if not required_vars:
                console.print("[green]✅ No environment variables required.[/green]")
                await service.shutdown()
                return

            # Check each variable
            missing_vars = []
            set_vars = []

            for var in required_vars:
                value = os.getenv(var)
                if value:
                    set_vars.append((var, value))
                else:
                    missing_vars.append(var)

            # Display results
            if set_vars:
                console.print(f"[bold green]✅ Set Variables ({len(set_vars)}):[/bold green]")
                for var, value in set_vars:
                    # Mask sensitive values
                    if "key" in var.lower() or "secret" in var.lower() or "token" in var.lower():
                        display_value = (
                            value[:4] + "*" * (len(value) - 8) + value[-4:]
                            if len(value) > 8
                            else "*" * len(value)
                        )
                    else:
                        display_value = value
                    console.print(f"  [green]●[/green] {var} = {display_value}")

            if missing_vars:
                console.print(f"\n[bold red]❌ Missing Variables ({len(missing_vars)}):[/bold red]")
                for var in missing_vars:
                    console.print(f"  [red]●[/red] {var}")

                # Provide suggestions
                console.print("\n[bold blue]💡 Set missing variables:[/bold blue]")
                for var in missing_vars:
                    console.print(f"  export {var}=your_value_here")

                # Interactive setting
                if set_missing:
                    console.print("\n[bold yellow]🛠️  Interactive Setup:[/bold yellow]")
                    env_commands = []
                    for var in missing_vars:
                        if Confirm.ask(f"Set {var} now?"):
                            value = Prompt.ask(
                                f"Value for {var}",
                                password="key" in var.lower() or "secret" in var.lower(),
                            )
                            os.environ[var] = value
                            env_commands.append(f"export {var}={value}")
                            console.print(f"  [green]✓[/green] {var} set for this session")

                    if env_commands:
                        console.print(
                            "\n[bold blue]📝 To make permanent, add to your shell profile:[/bold blue]"
                        )
                        for cmd in env_commands:
                            console.print(f"  {cmd}")

                # Exit with error if variables are missing
                if not set_missing:
                    sys.exit(1)
            else:
                console.print(
                    "\n[bold green]🎉 All required environment variables are set![/bold green]"
                )

            await service.shutdown()

        except Exception as e:
            console.print(f"[red]✗ Failed to check environment variables:[/red] {str(e)}")
            sys.exit(1)

    asyncio.run(_check_env())


if __name__ == "__main__":
    config()
