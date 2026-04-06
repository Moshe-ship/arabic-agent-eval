"""CLI for Arabic Agent Eval."""

from __future__ import annotations

import json
import sys

import click
from rich.console import Console

from arabic_agent_eval.dataset import Dataset
from arabic_agent_eval.evaluator import Evaluator, BenchmarkResult
from arabic_agent_eval.providers import (
    PROVIDER_CONFIGS,
    get_available_providers,
    get_api_key,
    save_config,
    load_config,
    make_call_fn,
)
from arabic_agent_eval import display

console = Console()


@click.group()
@click.version_option(package_name="arabic-agent-eval")
def main() -> None:
    """Arabic Agent Eval - The first Arabic function-calling benchmark.

    Evaluate how well LLMs handle function/tool calling in Arabic.
    """
    pass


@main.command()
@click.option("--provider", "-p", help="Specific provider to benchmark")
@click.option("--model", "-m", help="Override model name")
@click.option("--json-output", is_flag=True, help="Output as JSON")
@click.option("--min-score", type=float, help="Fail if score below this threshold (0-1)")
def run(provider: str | None, model: str | None, json_output: bool, min_score: float | None) -> None:
    """Run full benchmark across providers."""
    dataset = Dataset()

    if provider:
        providers = [provider]
    else:
        providers = get_available_providers()
        if not providers:
            console.print("[red]No API keys configured. Run: aae config[/red]")
            sys.exit(1)

    results = []
    for prov in providers:
        console.print(f"[dim]Evaluating {prov}...[/dim]")
        try:
            call_fn = make_call_fn(prov, model)
            pconf = PROVIDER_CONFIGS.get(prov, {})
            evaluator = Evaluator(
                call_fn=call_fn,
                provider=prov,
                model=model or pconf.get("model", "unknown"),
            )
            result = evaluator.evaluate(dataset)
            results.append(result)
        except Exception as e:
            console.print(f"[red]{prov}: {e}[/red]")

    if not results:
        console.print("[red]No results.[/red]")
        sys.exit(1)

    if json_output:
        click.echo(json.dumps([r.to_dict() for r in results], indent=2, ensure_ascii=False))
    else:
        for r in results:
            display.print_benchmark_result(r)
            console.print()

    if min_score is not None:
        for r in results:
            if r.overall_score < min_score:
                console.print(f"[red]{r.provider} score {r.overall_score:.1%} below threshold {min_score:.1%}[/red]")
                sys.exit(1)


@main.command()
@click.argument("provider")
@click.option("--model", "-m", help="Override model name")
@click.option("--json-output", is_flag=True, help="Output as JSON")
def quick(provider: str, model: str | None, json_output: bool) -> None:
    """Quick single-provider evaluation (subset of items)."""
    dataset = Dataset()

    call_fn = make_call_fn(provider, model)
    pconf = PROVIDER_CONFIGS.get(provider, {})
    evaluator = Evaluator(
        call_fn=call_fn,
        provider=provider,
        model=model or pconf.get("model", "unknown"),
    )

    console.print(f"[dim]Quick eval: {provider} (12 items)...[/dim]")
    result = evaluator.evaluate_quick(dataset)

    if json_output:
        click.echo(json.dumps(result.to_dict(), indent=2, ensure_ascii=False))
    else:
        display.print_benchmark_result(result)


@main.command()
@click.argument("provider_a")
@click.argument("provider_b")
@click.option("--model-a", help="Model for provider A")
@click.option("--model-b", help="Model for provider B")
def compare(provider_a: str, provider_b: str, model_a: str | None, model_b: str | None) -> None:
    """Compare two providers side by side."""
    dataset = Dataset()

    results = []
    for prov, model in [(provider_a, model_a), (provider_b, model_b)]:
        console.print(f"[dim]Evaluating {prov}...[/dim]")
        call_fn = make_call_fn(prov, model)
        pconf = PROVIDER_CONFIGS.get(prov, {})
        evaluator = Evaluator(
            call_fn=call_fn,
            provider=prov,
            model=model or pconf.get("model", "unknown"),
        )
        results.append(evaluator.evaluate(dataset))

    display.print_comparison(results[0], results[1])


@main.command("dataset")
def show_dataset() -> None:
    """Show dataset statistics."""
    dataset = Dataset()
    display.print_dataset_stats(dataset)


@main.command()
def config() -> None:
    """Configure API keys."""
    cfg = load_config()
    keys = cfg.get("keys", {})

    console.print("[bold]Arabic Agent Eval - Configuration[/bold]")
    console.print()

    for provider, pconf in sorted(PROVIDER_CONFIGS.items()):
        env_key = pconf["env_key"]
        current = keys.get(provider) or ""
        mask = f"{current[:4]}...{current[-4:]}" if len(current) > 8 else ""

        prompt_text = f"{provider} ({env_key})"
        if mask:
            prompt_text += f" [{mask}]"

        value = click.prompt(prompt_text, default="", show_default=False)
        if value:
            keys[provider] = value

    cfg["keys"] = keys
    save_config(cfg)

    available = get_available_providers()
    console.print(f"\n[green]Saved. {len(available)} providers configured.[/green]")
    for p in available:
        console.print(f"  {p}")
