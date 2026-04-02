#!/usr/bin/env python3
"""Generate synthetic calibration pairs for threshold calibration.

Usage:
    uv run python scripts/generate_calibration_pairs.py --output calibration_data.json
"""
from __future__ import annotations

import json
from pathlib import Path

import typer

app = typer.Typer()


@app.command()
def main(
    output: Path = typer.Option(Path("calibration_data.json"), "--output", "-o"),
    n_per_type: int = typer.Option(100, "--n-per-type", help="Pairs per output type"),
    seed: int = typer.Option(42, "--seed", help="RNG seed"),
) -> None:
    """Generate synthetic calibration pairs for all output types."""
    from rosettastone.calibration.collector import generate_synthetic_pairs
    from rosettastone.calibration.types import CalibrationDataset

    all_pairs = []
    for output_type in ["json", "classification", "short_text", "long_text"]:
        pairs = generate_synthetic_pairs(output_type, n_pairs=n_per_type, seed=seed)
        all_pairs.extend(pairs)

    dataset = CalibrationDataset(pairs=all_pairs)
    output.write_text(dataset.model_dump_json(indent=2))
    typer.echo(f"Generated {len(all_pairs)} pairs → {output}")


if __name__ == "__main__":
    app()
