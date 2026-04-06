"""Rich terminal output for Arabic Agent Eval."""

from __future__ import annotations

from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.text import Text

from arabic_agent_eval.dataset import Dataset, CATEGORIES, DIALECTS
from arabic_agent_eval.scoring import CategoryScore, grade
from arabic_agent_eval.evaluator import BenchmarkResult, EvalResult

console = Console()

GRADE_COLORS = {
    "A": "green bold",
    "B": "blue",
    "C": "yellow",
    "D": "red",
    "F": "red bold",
}


def print_benchmark_result(result: BenchmarkResult) -> None:
    """Print full benchmark result."""
    g = result.overall_grade
    color = GRADE_COLORS.get(g, "white")

    header = Text()
    header.append(f"{result.provider}", style="bold")
    header.append(f" ({result.model})", style="dim")
    header.append(f"  Score: {result.overall_score:.1%}", style="bold")
    header.append(f"  Grade: ", style="dim")
    header.append(f"{g}", style=color)

    console.print(Panel(header, title="Arabic Agent Eval", border_style="bold"))
    console.print()

    # Category breakdown
    table = Table(title="Category Scores")
    table.add_column("Category", style="cyan")
    table.add_column("Arabic", style="dim")
    table.add_column("Items", justify="right")
    table.add_column("Func Select", justify="right")
    table.add_column("Arg Accuracy", justify="right")
    table.add_column("Arabic Pres.", justify="right")
    table.add_column("Total", justify="right")
    table.add_column("Grade", justify="center")

    for cs in result.category_scores:
        g = grade(cs.avg_total)
        gc = GRADE_COLORS.get(g, "white")
        table.add_row(
            cs.category,
            cs.name_ar,
            str(cs.count),
            f"{cs.avg_function_selection:.0%}",
            f"{cs.avg_argument_accuracy:.0%}",
            f"{cs.avg_arabic_preservation:.0%}",
            f"{cs.avg_total:.0%}",
            Text(g, style=gc),
        )

    console.print(table)

    # Errors
    if result.errors:
        console.print()
        console.print(f"[red]{len(result.errors)} errors encountered:[/red]")
        for r in result.errors:
            console.print(f"  [red]x[/red] {r.item.id}: {r.error}")


def print_comparison(a: BenchmarkResult, b: BenchmarkResult) -> None:
    """Print side-by-side comparison of two providers."""
    table = Table(title=f"{a.provider} vs {b.provider}")
    table.add_column("Category")
    table.add_column(a.provider, justify="right")
    table.add_column(b.provider, justify="right")
    table.add_column("Diff", justify="right")

    a_cats = {cs.category: cs for cs in a.category_scores}
    b_cats = {cs.category: cs for cs in b.category_scores}

    all_cats = set(list(a_cats.keys()) + list(b_cats.keys()))
    for cat in sorted(all_cats):
        a_score = a_cats.get(cat)
        b_score = b_cats.get(cat)
        a_val = a_score.avg_total if a_score else 0
        b_val = b_score.avg_total if b_score else 0
        diff = a_val - b_val
        diff_style = "green" if diff > 0 else "red" if diff < 0 else "dim"
        diff_str = f"+{diff:.0%}" if diff > 0 else f"{diff:.0%}"

        table.add_row(
            cat,
            f"{a_val:.0%}",
            f"{b_val:.0%}",
            Text(diff_str, style=diff_style),
        )

    # Overall
    diff = a.overall_score - b.overall_score
    diff_style = "green" if diff > 0 else "red" if diff < 0 else "dim"
    diff_str = f"+{diff:.0%}" if diff > 0 else f"{diff:.0%}"
    table.add_row(
        Text("OVERALL", style="bold"),
        Text(f"{a.overall_score:.0%}", style="bold"),
        Text(f"{b.overall_score:.0%}", style="bold"),
        Text(diff_str, style=f"{diff_style} bold"),
    )

    console.print(table)


def print_dataset_stats(dataset: Dataset) -> None:
    """Print dataset statistics."""
    console.print(f"[bold]Dataset: {len(dataset)} items[/bold]")
    console.print()

    # By category
    table = Table(title="By Category")
    table.add_column("Category", style="cyan")
    table.add_column("Arabic")
    table.add_column("Count", justify="right")
    table.add_column("Weight", justify="right")

    cats = dataset.categories()
    for cat, count in cats.items():
        info = CATEGORIES.get(cat, {})
        table.add_row(
            cat,
            info.get("name_ar", ""),
            str(count),
            f"{info.get('weight', 0):.0%}",
        )
    console.print(table)
    console.print()

    # By dialect
    table = Table(title="By Dialect")
    table.add_column("Dialect", style="cyan")
    table.add_column("Count", justify="right")

    for dialect, count in dataset.dialects().items():
        table.add_row(dialect, str(count))
    console.print(table)

    # By difficulty
    console.print()
    table = Table(title="By Difficulty")
    table.add_column("Difficulty", style="cyan")
    table.add_column("Count", justify="right")

    for diff in ["easy", "medium", "hard"]:
        items = dataset.by_difficulty(diff)
        table.add_row(diff, str(len(items)))
    console.print(table)


def print_leaderboard(results: list[BenchmarkResult]) -> None:
    """Print ranked leaderboard."""
    sorted_results = sorted(results, key=lambda r: r.overall_score, reverse=True)

    table = Table(title="Leaderboard")
    table.add_column("#", style="dim", justify="right")
    table.add_column("Provider", style="cyan")
    table.add_column("Model", style="dim")
    table.add_column("Score", justify="right")
    table.add_column("Grade", justify="center")
    table.add_column("Items", justify="right")

    for i, r in enumerate(sorted_results, 1):
        g = r.overall_grade
        gc = GRADE_COLORS.get(g, "white")
        table.add_row(
            str(i),
            r.provider,
            r.model,
            f"{r.overall_score:.1%}",
            Text(g, style=gc),
            str(r.total_items),
        )

    console.print(table)
