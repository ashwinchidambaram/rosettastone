"""Rich-based CLI display components for RosettaStone migration output."""

from __future__ import annotations

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
        per_type_scores: dict,
    ) -> None:
        """Render a per-output-type breakdown table.

        Args:
            results: List of evaluation results.
            per_type_scores: Mapping of output type label → dict with keys
                ``win_rate``, ``sample_count``, ``avg_score``, ``threshold``.
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

        if not results and not per_type_scores:
            table.add_row("—", "—", "—", "—", "—")
            self.console.print(table)
            return

        for output_type, stats in per_type_scores.items():
            win_rate: float = stats.get("win_rate", 0.0)
            sample_count: int = stats.get("sample_count", 0)
            avg_score: float = stats.get("avg_score", 0.0)
            threshold: float = stats.get("threshold", _THRESHOLD_GREEN)

            table.add_row(
                str(output_type),
                _fmt_pct(win_rate),
                str(sample_count),
                _fmt_pct(avg_score),
                _fmt_pct(threshold),
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
            )

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
    # Safety warnings
    # ------------------------------------------------------------------

    def show_safety_warnings(self, warnings: list) -> None:
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
