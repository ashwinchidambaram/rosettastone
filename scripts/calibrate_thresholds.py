#!/usr/bin/env python3
"""Calibrate thresholds from human-labeled calibration data.

Usage:
    uv run python scripts/calibrate_thresholds.py --input calibration_data.json
"""
from __future__ import annotations

import json
from pathlib import Path

import typer

app = typer.Typer()


@app.command()
def main(
    input_path: Path = typer.Argument(..., help="Path to labeled calibration dataset JSON"),
    output: Path = typer.Option(Path("calibrated_thresholds.json"), "--output", "-o"),
    report: bool = typer.Option(True, "--report/--no-report", help="Print calibration report"),
) -> None:
    """Calibrate thresholds from a labeled calibration dataset."""
    from rosettastone.calibration.calibrator import ThresholdCalibrator
    from rosettastone.calibration.types import CalibrationDataset

    dataset = CalibrationDataset.model_validate_json(input_path.read_text())
    calibrator = ThresholdCalibrator()

    try:
        thresholds = calibrator.fit(dataset)
    except ImportError as e:
        typer.echo(f"[error] {e}", err=True)
        raise typer.Exit(code=1)

    output.write_text(json.dumps(thresholds, indent=2))
    typer.echo(f"Thresholds → {output}")

    if report:
        typer.echo(calibrator.report(dataset, thresholds))


if __name__ == "__main__":
    app()
