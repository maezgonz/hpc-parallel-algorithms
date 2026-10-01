"""Dark-mode heatmap rendering of solver fields exported as CSV."""

from __future__ import annotations

import argparse
import os
import sys

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd


def load_field(csv_path: str) -> pd.DataFrame:
    """Load a solver field exported as a square CSV grid.

    Args:
        csv_path: Path to a CSV file with one row per grid row.

    Returns:
        DataFrame of shape (n, n) with numeric cell values.

    Raises:
        ValueError: If the CSV is empty or non-square.
    """
    frame = pd.read_csv(csv_path, header=None)
    if frame.empty:
        raise ValueError("field CSV is empty")
    if frame.shape[0] != frame.shape[1]:
        raise ValueError(f"field CSV must be square, got {frame.shape}")
    return frame


def plot_field(frame: pd.DataFrame, title: str, output: str) -> None:
    """Render a dark-mode heatmap of the field.

    Args:
        frame: Square field DataFrame as produced by :func:`load_field`.
        title: Title for the plot.
        output: Destination path for the PNG figure.
    """
    plt.style.use("dark_background")
    fig, ax = plt.subplots(figsize=(8, 7), dpi=120)

    image = ax.imshow(frame.values, origin="upper", cmap="inferno", aspect="equal")
    ax.set_title(title)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    fig.colorbar(image, ax=ax, label="Value")

    fig.tight_layout()
    directory = os.path.dirname(output)
    if directory:
        os.makedirs(directory, exist_ok=True)
    fig.savefig(output)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Dark-mode heatmap of solver fields")
    parser.add_argument("--csv", required=True, help="Field CSV (square grid)")
    parser.add_argument("--title", default="Solver field", help="Title for the plot")
    parser.add_argument("--output", default="benchmarks/field.png", help="Output PNG path")
    return parser.parse_args()


def main() -> int:
    """Entry point. Returns a process exit code."""
    args = parse_args()
    try:
        frame = load_field(args.csv)
        plot_field(frame, args.title, args.output)
    except (ValueError, OSError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    print(f"grid={frame.shape[0]}x{frame.shape[1]} figure={args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
