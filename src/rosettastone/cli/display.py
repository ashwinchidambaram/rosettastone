"""Rich-based CLI display components for RosettaStone migration output."""

from __future__ import annotations

from typing import Any

from rich.console import Console
from rich.panel import Panel
from rich.progress import BarColumn, MofNCompleteColumn, Progress, SpinnerColumn, TextColumn
from rich.table import Table
from rich.text import Text

from rosettastone.core.types import EvalResult

# Score thresholds for color-coding
_THRESHOLD_GREEN = 0.90
_THRESHOLD_YELLOW = 0.80


def _score_color(score: float) -> str:
    """Return a Rich color name based on score thresholds."""
    if score >= _THRESHOLD_GREEN:
        return "green"
    if score >= _THRESHOLD_YELLOW:
        return "yellow"
    return "red"


def _fmt_pct(value: float) -> Text:
    """Format a float as a percentage with threshold-based color."""
    color = _score_color(value)
    return Text(f"{value:.0%}", style=color)


class MigrationDisplay:
    """Rich-based display helper for migration pipeline output."""

    def __init__(self, console: Console | None = None) -> None:
        self.console = console or Console()

    # ------------------------------------------------------------------
    # Progress
    # ------------------------------------------------------------------

    def create_progress(self) -> Progress:
        """Return a Rich Progress configured for pipeline step tracking."""
        return Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            MofNCompleteColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            console=self.console,
            transient=False,
        )

    # ------------------------------------------------------------------
    # Summary table
    # ------------------------------------------------------------------

    def show_summary_table(
        self,
        results: list[EvalResult],
        per_type_scores: dict[str, Any],
    ) -> None:
        """Render a per-output-type breakdown table.

        Args:
            results: List of evaluation results.
            per_type_scores: Mapping of output type label → dict with keys
                ``win_rate``, ``sample_count``, ``avg_score``, ``threshold``,
                and optionally ``confidence_interval``.
        """
        table = Table(
            title="Evaluation Summary",
            show_header=True,
            header_style="bold",
            box=None,
        )
        table.add_column("Output Type", style="bold")
        table.add_column("Win Rate", justify="right")
        table.add_column("Sample Count", justify="right")
        table.add_column("Avg Score", justify="right")
        table.add_column("Threshold", justify="right")
        table.add_column("95% CI", justify="right")

        if not results and not per_type_scores:
            table.add_row("—", "—", "—", "—", "—", "—")
            self.console.print(table)
            return

        for output_type, stats in per_type_scores.items():
            win_rate: float = stats.get("win_rate", 0.0)
            sample_count: int = stats.get("sample_count", 0)
            avg_score: float = stats.get("avg_score", 0.0)
            threshold: float = stats.get("threshold", _THRESHOLD_GREEN)
            ci = stats.get("confidence_interval", (0.0, 0.0))
            if isinstance(ci, (list, tuple)) and len(ci) >= 2 and (ci[0] != 0.0 or ci[1] != 0.0):
                ci_str = f"[{ci[0]:.0%}–{ci[1]:.0%}]"
            else:
                ci_str = "—"

            table.add_row(
                str(output_type),
                _fmt_pct(win_rate),
                str(sample_count),
                _fmt_pct(avg_score),
                _fmt_pct(threshold),
                ci_str,
            )

        # Aggregate row when there is more than one type
        if len(per_type_scores) > 1 and results:
            total = len(results)
            wins = sum(1 for r in results if r.is_win)
            overall_win_rate = wins / total if total > 0 else 0.0
            overall_avg = sum(r.composite_score for r in results) / total if total > 0 else 0.0
            table.add_section()
            table.add_row(
                "[bold]Overall[/bold]",
                _fmt_pct(overall_win_rate),
                str(total),
                _fmt_pct(overall_avg),
                "—",
                "—",
            )

        self.console.print(table)

    # ------------------------------------------------------------------
    # Stage timing table
    # ------------------------------------------------------------------

    def show_timing_table(self, stage_timing: dict[str, float]) -> None:
        """Render a stage timing breakdown table sorted by duration descending.

        Args:
            stage_timing: Mapping of stage name → elapsed seconds.
        """
        if not stage_timing:
            return

        table = Table(
            title="Stage Timing",
            show_header=True,
            header_style="bold",
            box=None,
        )
        table.add_column("Stage", style="bold")
        table.add_column("Duration", justify="right")

        total = sum(stage_timing.values())
        sorted_stages = sorted(stage_timing.items(), key=lambda x: x[1], reverse=True)
        for stage, secs in sorted_stages:
            share = f"({secs / total:.0%})" if total > 0 else ""
            table.add_row(stage, f"{secs:.1f}s {share}")

        self.console.print(table)

    # ------------------------------------------------------------------
    # Recommendation panel
    # ------------------------------------------------------------------

    def show_recommendation(self, recommendation: str, reasoning: str) -> None:
        """Render a styled panel for the migration recommendation.

        Args:
            recommendation: One of ``"GO"``, ``"NO_GO"``, or ``"CONDITIONAL"``.
            reasoning: Human-readable explanation of the recommendation.
        """
        upper = recommendation.upper()
        if upper == "GO":
            border_style = "green"
            title = "[bold green]GO[/bold green]"
        elif upper == "NO_GO":
            border_style = "red"
            title = "[bold red]NO GO[/bold red]"
        else:
            border_style = "yellow"
            title = "[bold yellow]CONDITIONAL[/bold yellow]"

        self.console.print(
            Panel(
                reasoning,
                title=title,
                border_style=border_style,
                padding=(1, 2),
            )
        )

    # ------------------------------------------------------------------
    # Cost summary
    # ------------------------------------------------------------------

    def show_cost_summary(self, costs: dict[str, float]) -> None:
        """Render a per-phase cost breakdown table.

        Args:
            costs: Mapping of phase name → cost in USD.
        """
        table = Table(
            title="Cost Summary",
            show_header=True,
            header_style="bold",
            box=None,
        )
        table.add_column("Phase", style="bold")
        table.add_column("Cost (USD)", justify="right")

        total = 0.0
        for phase, cost in costs.items():
            table.add_row(phase, f"${cost:.4f}")
            total += cost

        if costs:
            table.add_section()
            table.add_row("[bold]Total[/bold]", f"[bold]${total:.4f}[/bold]")

        self.console.print(table)

    # ------------------------------------------------------------------
    # Prompt evolution
    # ------------------------------------------------------------------

    def show_prompt_evolution(
        self,
        optimized_prompt: str,
        baseline_score: float,
        confidence_score: float,
        improvement: float,
        sample_comparisons: list[dict[str, Any]] | None = None,
    ) -> None:
        """Render a before/after prompt evolution panel.

        Shows the GEPA-generated system instruction alongside score impact and
        top improvements from the test set.
        """
        from rich.rule import Rule

        self.console.print()
        self.console.print(Rule("[bold]Prompt Evolution[/bold]", style="blue"))

        # Before / after scores
        before_color = _score_color(baseline_score)
        after_color = _score_color(confidence_score)
        delta_str = f"{'+' if improvement >= 0 else ''}{improvement:.0%}"
        delta_color = "green" if improvement >= 0 else "red"

        self.console.print(
            f"  [bold]Before[/bold]  [{before_color}]{baseline_score:.0%}[/{before_color}]"
            f"  →  [{after_color}]{confidence_score:.0%}[/{after_color}]"
            f"  ([{delta_color}]{delta_str}[/{delta_color}])"
        )
        self.console.print()

        # Optimized prompt (truncated)
        prompt_preview = optimized_prompt.strip()
        if len(prompt_preview) > 400:
            prompt_preview = prompt_preview[:397] + "..."
        self.console.print(
            Panel(
                prompt_preview,
                title="[bold blue]GEPA-Optimized System Instruction[/bold blue]",
                border_style="blue",
                padding=(1, 2),
                subtitle="[dim]full prompt in optimized_prompt.txt[/dim]",
            )
        )

        # Top improvements table
        if sample_comparisons:
            self.console.print()
            table = Table(
                title=f"Top {len(sample_comparisons)} Improvements",
                show_header=True,
                header_style="bold",
                box=None,
            )
            table.add_column("#", style="dim", width=4)
            table.add_column("Type", style="bold")
            table.add_column("Before", justify="right")
            table.add_column("After", justify="right")
            table.add_column("Delta", justify="right")
            table.add_column("Win?", justify="center")

            for s in sample_comparisons:
                b_score = s["baseline_score"]
                v_score = s["optimized_score"]
                delta = s["delta"]
                delta_color_cell = "green" if delta >= 0 else "red"
                win_str = (
                    "[green]no→YES[/green]"
                    if (not s["is_win_before"] and s["is_win_after"])
                    else (
                        "[green]yes→yes[/green]"
                        if (s["is_win_before"] and s["is_win_after"])
                        else (
                            "[red]YES→no[/red]"
                            if (s["is_win_before"] and not s["is_win_after"])
                            else "[dim]no→no[/dim]"
                        )
                    )
                )
                table.add_row(
                    str(s["index"]),
                    str(s["output_type"]),
                    _fmt_pct(b_score),
                    _fmt_pct(v_score),
                    Text(
                        f"{'+' if delta >= 0 else ''}{delta:.3f}",
                        style=delta_color_cell,
                    ),
                    win_str,
                )
            self.console.print(table)

    # ------------------------------------------------------------------
    # Safety warnings
    # ------------------------------------------------------------------

    def show_variance_warning(self, non_deterministic_count: int) -> None:
        """Render an amber warning panel if any test pairs showed high score variance.

        Args:
            non_deterministic_count: Number of test pairs with high score variance.
        """
        if non_deterministic_count <= 0:
            return
        self.console.print(
            Panel(
                f"{non_deterministic_count} test pair{'s' if non_deterministic_count != 1 else ''}"
                " showed high score variance across evaluation runs",
                title="[bold yellow]Evaluation Reliability Warning[/bold yellow]",
                border_style="yellow",
                padding=(1, 2),
            )
        )

    # ------------------------------------------------------------------
    # Safety warnings
    # ------------------------------------------------------------------

    def show_safety_warnings(self, warnings: list[Any]) -> None:
        """Render a panel listing safety warnings with severity-based styling.

        Each warning may be a plain string or an object/dict with ``severity``
        and ``message`` attributes/keys.  Unknown formats are rendered as-is.

        Severity levels: ``HIGH`` → red, ``MEDIUM`` → yellow, ``LOW`` → dim.
        """
        if not warnings:
            return

        lines: list[str] = []
        for w in warnings:
            severity, message = _extract_warning(w)
            color = _severity_color(severity)
            prefix = f"[{color}][{severity}][/{color}]" if severity else ""
            lines.append(f"{prefix} {message}".strip())

        body = "\n".join(lines)
        self.console.print(
            Panel(
                body,
                title="[bold yellow]Safety Warnings[/bold yellow]",
                border_style="yellow",
                padding=(1, 2),
            )
        )


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------


def _extract_warning(w: object) -> tuple[str, str]:
    """Return (severity, message) from a warning value of unknown type."""
    if isinstance(w, str):
        return "", w
    if isinstance(w, dict):
        return str(w.get("severity", "")).upper(), str(w.get("message", w))
    # Try attribute access (dataclass / object)
    severity = str(getattr(w, "severity", "")).upper()
    message = str(getattr(w, "message", w))
    return severity, message


def _severity_color(severity: str) -> str:
    """Map a severity string to a Rich color name."""
    mapping = {"HIGH": "red", "MEDIUM": "yellow", "LOW": "dim"}
    return mapping.get(severity.upper(), "white")
