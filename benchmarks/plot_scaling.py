"""Strong-scaling analysis from benchmark CSV files with dark-mode plots."""

from __future__ import annotations

import argparse
import os
import sys

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd


def load_scaling(csv_path: str) -> pd.DataFrame:
    """Load benchmark results and derive speedup and efficiency columns.

    Args:
        csv_path: Path to a CSV with ``threads`` and ``wall_time`` columns.

    Returns:
        DataFrame sorted by threads with ``speedup`` and ``efficiency``.

    Raises:
        ValueError: If columns are missing or values are non-positive.
    """
    frame = pd.read_csv(csv_path).sort_values("threads").reset_index(drop=True)
    required = {"threads", "wall_time"}
    if not required.issubset(frame.columns):
        raise ValueError(f"CSV must contain columns {sorted(required)}")
    if (frame["threads"] <= 0).any() or (frame["wall_time"] <= 0).any():
        raise ValueError("threads and wall_time must be positive")

    base = frame.loc[frame["threads"].idxmin(), "wall_time"]
    frame["speedup"] = base / frame["wall_time"]
    frame["efficiency"] = frame["speedup"] / frame["threads"]
    return frame


def plot_scaling(frame: pd.DataFrame, title: str, output: str) -> None:
    """Render dark-mode speedup and parallel-efficiency plots.

    Args:
        frame: Scaling DataFrame as produced by :func:`load_scaling`.
        title: Title for the speedup plot.
        output: Destination path for the PNG figure.
    """
    plt.style.use("dark_background")
    fig, (ax_speed, ax_eff) = plt.subplots(1, 2, figsize=(12, 5), dpi=120)

    ax_speed.plot(frame["threads"], frame["speedup"], "o-", color="#58a6ff", label="Measured")
    ax_speed.plot(frame["threads"], frame["threads"], "--", color="#8b949e", alpha=0.6, label="Ideal")
    ax_speed.set_xlabel("Threads")
    ax_speed.set_ylabel("Speedup")
    ax_speed.set_title(title)
    ax_speed.grid(alpha=0.2)
    ax_speed.legend(framealpha=0.2)

    ax_eff.plot(frame["threads"], frame["efficiency"], "o-", color="#f78166")
    ax_eff.axhline(1.0, color="#8b949e", linestyle="--", alpha=0.6)
    ax_eff.set_xlabel("Threads")
    ax_eff.set_ylabel("Efficiency")
    ax_eff.set_title("Parallel efficiency")
    ax_eff.set_ylim(0, 1.1)
    ax_eff.grid(alpha=0.2)

    fig.tight_layout()
    directory = os.path.dirname(output)
    if directory:
        os.makedirs(directory, exist_ok=True)
    fig.savefig(output)
    plt.close(fig)


def markdown_table(frame: pd.DataFrame) -> str:
    """Format the scaling results as a markdown table.

    Args:
        frame: Scaling DataFrame as produced by :func:`load_scaling`.

    Returns:
        Markdown table string ready to paste into the README.
    """
    rows = ["| Threads | Wall time (s) | Speedup | Efficiency |", "|---:|---:|---:|---:|"]
    for _, row in frame.iterrows():
        rows.append(
            f"| {int(row['threads'])} | {row['wall_time']:.3f} "
            f"| {row['speedup']:.2f}x | {row['efficiency']:.0%} |"
        )
    return "\n".join(rows)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Strong-scaling analysis with dark-mode plots")
    parser.add_argument("--csv", required=True, help="Benchmark CSV with threads,wall_time")
    parser.add_argument("--title", default="Strong scaling", help="Title for the speedup plot")
    parser.add_argument("--output", default="benchmarks/scaling.png", help="Output PNG path")
    return parser.parse_args()


def main() -> int:
    """Entry point. Returns a process exit code."""
    args = parse_args()
    try:
        frame = load_scaling(args.csv)
        plot_scaling(frame, args.title, args.output)
    except (ValueError, OSError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    print(markdown_table(frame))
    print(f"figure={args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
