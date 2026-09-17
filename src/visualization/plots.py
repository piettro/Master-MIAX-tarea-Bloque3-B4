"""Static report figures (PNG).

Every figure required by the brief is produced here:

* training curves for every trained network (one grid per cell);
* one results chart per ``(V, H)`` combination (16 charts);
* one chart per output window gathering all results;
* the competition matrix, the research ablation and the 2025 portfolios.

Styling follows the course feedback: bar axes are zoomed to the informative
range instead of starting at zero, and curve axes ignore the first epochs,
whose large losses would otherwise flatten the convergence phase.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Mapping

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402

logger = logging.getLogger(__name__)

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_SECONDARY = "#52514e"
INK_MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"
NEUTRAL = "#f0efec"

# Fixed categorical order: colour follows the entity, never its rank.
FAMILY_COLORS = {
    "dense": "#2a78d6",
    "recurrent": "#eb6834",
    "convolutional": "#1baf7a",
    "mixed": "#4a3aa7",
    "baseline": INK_MUTED,
}
PARTITION_COLORS = {"train": "#2a78d6", "val": "#eb6834", "test": "#1baf7a"}
MARKERS = ("o", "s", "^", "D", "v", "P")
# Models whose validation MAE exceeds the cell's best by this factor are
# drawn off-scale, so they do not flatten the differences that matter.
OFF_SCALE_FACTOR = 1.25

plt.rcParams.update({
    "figure.facecolor": SURFACE,
    "axes.facecolor": SURFACE,
    "axes.edgecolor": AXIS,
    "axes.labelcolor": INK_SECONDARY,
    "axes.titlecolor": INK,
    "axes.grid": True,
    "grid.color": GRID,
    "grid.linewidth": 0.6,
    "xtick.color": INK_MUTED,
    "ytick.color": INK_MUTED,
    "text.color": INK,
    "legend.frameon": False,
    "font.size": 9,
    "axes.titlesize": 10,
    "savefig.dpi": 130,
    "savefig.bbox": "tight",
})


def _save(fig: plt.Figure, path: Path) -> None:
    """Write a figure to disk and release its memory.

    Args:
        fig: Figure to save.
        path: Destination PNG path; parents are created.

    Raises:
        OSError: If the file cannot be written.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        fig.savefig(path)
    except OSError as exc:
        raise OSError(f"Cannot save figure {path}: {exc}") from exc
    finally:
        plt.close(fig)
    logger.debug("Saved figure %s", path)


def _zoomed_limits(
    values: np.ndarray, pad: float = 0.15
) -> tuple[float, float]:
    """Axis limits spanning the data with padding, not anchored at zero.

    Args:
        values: Finite values to show.
        pad: Padding as a share of the data range.

    Returns:
        ``(low, high)``.
    """
    low, high = float(np.nanmin(values)), float(np.nanmax(values))
    span = high - low or abs(high) * 0.05 or 1e-6
    return low - pad * span, high + pad * span


def plot_training_curves(
    histories: Mapping[str, pd.DataFrame],
    title: str,
    path: Path,
    skip_epochs: int = 3,
) -> None:
    """Grid of train/validation MAE curves, one panel per model.

    Args:
        histories: Model name to history with ``epoch``, ``loss`` and
            ``val_loss`` columns (log-return units).
        title: Figure title.
        path: Output PNG path.
        skip_epochs: Leading epochs excluded from the y-range.

    Raises:
        ValueError: If ``histories`` is empty.
    """
    if not histories:
        raise ValueError("No histories to plot")
    n = len(histories)
    cols = min(4, n)
    rows = int(np.ceil(n / cols))
    fig, axes = plt.subplots(
        rows, cols, figsize=(3.4 * cols, 2.6 * rows), squeeze=False
    )
    for ax, (name, history) in zip(axes.flat, histories.items()):
        ax.plot(history["epoch"], history["loss"], lw=1.5,
                color=PARTITION_COLORS["train"], label="train")
        ax.plot(history["epoch"], history["val_loss"], lw=1.5,
                color=PARTITION_COLORS["val"], label="validation")
        best = int(history["val_loss"].idxmin())
        ax.axvline(history["epoch"].iloc[best], color=INK_MUTED, lw=0.8,
                   ls=":")
        tail = history.iloc[min(skip_epochs, len(history) - 1):]
        ax.set_ylim(*_zoomed_limits(
            tail[["loss", "val_loss"]].to_numpy(), pad=0.1
        ))
        ax.set_title(f"{name} (best epoch {best + 1})")
        ax.set_xlabel("epoch")
        ax.set_ylabel("MAE")
    for ax in list(axes.flat)[n:]:
        ax.set_visible(False)
    handles, labels = axes.flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper right", ncols=2)
    fig.suptitle(f"{title} (dotted line: restored best epoch)")
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    _save(fig, path)


def plot_cell_results(cell: pd.DataFrame, title: str, path: Path) -> None:
    """Train/validation/test MAE of every model in one window combination.

    Models are sorted by validation MAE; the validation-selected winner is
    starred and its test MAE annotated.

    Args:
        cell: Results rows of a single ``(V, H)`` cell.
        title: Figure title.
        path: Output PNG path.

    Raises:
        ValueError: If ``cell`` is empty.
    """
    if cell.empty:
        raise ValueError("Cell results are empty")
    cell = cell.sort_values("mae_val", ascending=False).reset_index(drop=True)
    positions = np.arange(len(cell))
    height = 0.26
    columns = ("mae_train", "mae_val", "mae_test")
    in_scale = cell["mae_val"] <= OFF_SCALE_FACTOR * cell["mae_val"].min()
    low, high = _zoomed_limits(
        cell.loc[in_scale, list(columns)].to_numpy(), pad=0.25
    )
    fig, ax = plt.subplots(figsize=(7.5, 0.42 * len(cell) + 1.4))
    for offset, partition, column in zip(
        (height, 0.0, -height), ("train", "val", "test"), columns
    ):
        ax.barh(positions + offset, np.minimum(cell[column], high) - low,
                left=low, height=height * 0.9,
                color=PARTITION_COLORS[partition], label=partition)
    for position, row in zip(positions, cell.itertuples()):
        if row.mae_val > OFF_SCALE_FACTOR * cell["mae_val"].min():
            ax.text(high, position, f"off scale (val {row.mae_val:.4f}) ",
                    ha="right", va="center", fontsize=7, color=SURFACE)
    ax.set_xlim(low, high)
    labels = [
        f"{m}  [{f}]" + ("  *" if i == len(cell) - 1 else "")
        for i, (m, f) in enumerate(zip(cell["model"], cell["family"]))
    ]
    ax.set_yticks(positions, labels)
    winner = cell.iloc[-1]
    ax.set_title(
        f"{title}\n* selected on validation, test MAE "
        f"{winner['mae_test']:.6f}"
    )
    ax.set_xlabel("MAE (log-return units)")
    ax.grid(axis="y", visible=False)
    ax.legend(loc="lower left", ncols=3, bbox_to_anchor=(0, -0.22))
    _save(fig, path)


def plot_output_window_summary(
    results: pd.DataFrame,
    output_window: int,
    path: Path,
) -> None:
    """All models' validation and test MAE across input windows for one H.

    Args:
        results: Full competition results.
        output_window: ``H`` to plot.
        path: Output PNG path.

    Raises:
        ValueError: If no row matches ``output_window``.
    """
    subset = results[results["output_window"] == output_window]
    if subset.empty:
        raise ValueError(f"No results for H={output_window}")
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6), sharey=True)
    cutoff = OFF_SCALE_FACTOR * subset["mae_val"].min()
    shown = subset.groupby("model")["mae_val"].transform("max") <= cutoff
    family_counts: dict[str, int] = {}
    for model, rows in subset.groupby("model", sort=False):
        rows = rows.sort_values("input_window")
        family = rows["family"].iloc[0]
        index = family_counts.get(family, 0)
        family_counts[family] = index + 1
        style = {
            "color": FAMILY_COLORS[family],
            "marker": MARKERS[index % len(MARKERS)],
            "lw": 1.5,
            "ms": 5,
            "ls": "--" if family == "baseline" else "-",
            "label": f"{model} [{family}]",
        }
        if rows["mae_val"].max() > cutoff:
            style["label"] += " - off scale, see tables"
            style["alpha"] = 0.0
        for ax, column in zip(axes, ("mae_val", "mae_test")):
            ax.plot(rows["input_window"], rows[column], **style)
    for ax, name in zip(axes, ("validation", "test")):
        ax.set_xscale("log")
        windows = sorted(subset["input_window"].unique())
        ax.set_xticks(windows, [str(w) for w in windows])
        ax.minorticks_off()
        ax.set_xlabel("input window V (days, log scale)")
        ax.set_title(f"{name} MAE")
    axes[0].set_ylabel("MAE (log-return units)")
    axes[0].set_ylim(*_zoomed_limits(
        subset.loc[shown, ["mae_val", "mae_test"]].to_numpy(), pad=0.08
    ))
    axes[1].legend(loc="center left", bbox_to_anchor=(1.02, 0.5))
    fig.suptitle(f"All models, output window H={output_window}")
    fig.tight_layout()
    _save(fig, path)


def plot_competition_matrix(
    best: pd.DataFrame,
    results: pd.DataFrame,
    path: Path,
) -> None:
    """Competition matrix coloured by improvement over the best baseline.

    Each cell shows the selected model and its test MAE. The colour is the
    percentage change of that test MAE relative to the validation-selected
    baseline of the same cell (blue: better, orange: worse), which makes
    cells with very different horizons comparable.

    Args:
        best: Per-cell winners.
        results: Full results (to find each cell's best baseline).
        path: Output PNG path.
    """
    baselines = results[results["family"] == "baseline"]
    reference = baselines.loc[
        baselines.groupby(["input_window", "output_window"])[
            "mae_val"
        ].idxmin()
    ].set_index(["input_window", "output_window"])["mae_test"]
    merged = best.set_index(["input_window", "output_window"])
    change = 100 * (merged["mae_test"] / reference.loc[merged.index] - 1)

    inputs = sorted(best["input_window"].unique())
    outputs = sorted(best["output_window"].unique())
    grid = np.full((len(inputs), len(outputs)), np.nan)
    for (V, H), value in change.items():
        grid[inputs.index(V), outputs.index(H)] = value

    limit = max(1.0, float(np.nanmax(np.abs(grid))))
    cmap = LinearSegmentedColormap.from_list(
        "improvement", ["#2a78d6", NEUTRAL, "#eb6834"]
    )
    fig, ax = plt.subplots(figsize=(1.9 * len(outputs) + 2, 1.2 * len(inputs)
                                    + 1.2))
    image = ax.imshow(grid, cmap=cmap, vmin=-limit, vmax=limit, aspect="auto")
    for (V, H), row in merged.iterrows():
        i, j = inputs.index(V), outputs.index(H)
        ax.text(j, i, f"{row['mae_test']:.6f}\n{row['model']}\n"
                f"{change.loc[(V, H)]:+.1f}%", ha="center", va="center",
                fontsize=8, color=INK)
    ax.set_xticks(range(len(outputs)), [f"H={h}" for h in outputs])
    ax.set_yticks(range(len(inputs)), [f"V={v}" for v in inputs])
    ax.grid(False)
    colorbar = fig.colorbar(image, ax=ax)
    colorbar.set_label("test MAE vs best baseline (%)")
    ax.set_title("Competition: test MAE of the validation-selected model")
    _save(fig, path)


def plot_research(research: pd.DataFrame, path: Path) -> None:
    """Test MAE change of each research variant versus the competition model.

    Args:
        research: Research results with ``test_change_pct``.
        path: Output PNG path.
    """
    variants = [
        v for v in research["variant"].unique() if "competition" not in v
    ]
    cells = research[["input_window", "output_window"]].drop_duplicates()
    labels = [f"V={v} H={h}" for v, h in cells.to_numpy()]
    positions = np.arange(len(cells))
    width = 0.8 / max(1, len(variants))
    colors = ("#2a78d6", "#eb6834", "#1baf7a", "#4a3aa7")
    fig, ax = plt.subplots(figsize=(max(8, 0.6 * len(cells) + 3), 4.2))
    for k, variant in enumerate(variants):
        rows = research[research["variant"] == variant].set_index(
            ["input_window", "output_window"]
        )
        values = [
            rows.loc[tuple(c), "test_change_pct"] for c in cells.to_numpy()
        ]
        significant = [
            rows.loc[tuple(c), "dm_p_value"] < 0.05 for c in cells.to_numpy()
        ]
        bars = ax.bar(positions + (k - (len(variants) - 1) / 2) * width,
                      values, width * 0.9, color=colors[k % len(colors)],
                      label=variant)
        for bar, flag in zip(bars, significant):
            if flag:
                bar.set_hatch("//")
    ax.axhline(0, color=AXIS, lw=1)
    ax.set_xticks(positions, labels, rotation=45, ha="right")
    ax.set_ylabel("test MAE change vs competition (%)")
    ax.set_title("Research ablation (hatched: Diebold-Mariano p < 0.05)")
    ax.legend()
    _save(fig, path)


def plot_portfolios(curves: pd.DataFrame, title: str, path: Path) -> None:
    """Cumulative wealth of each portfolio.

    Args:
        curves: Daily wealth, one column per portfolio.
        title: Figure title.
        path: Output PNG path.
    """
    colors = ("#2a78d6", "#eb6834", "#1baf7a", "#4a3aa7")
    fig, ax = plt.subplots(figsize=(9, 4.5))
    for k, column in enumerate(curves.columns):
        ax.plot(curves.index, 100 * (curves[column] - 1), lw=2,
                color=colors[k % len(colors)], label=column)
        ax.annotate(f"{100 * (curves[column].iloc[-1] - 1):+.1f}%",
                    xy=(curves.index[-1], 100 * (curves[column].iloc[-1] - 1)),
                    xytext=(4, 0), textcoords="offset points", fontsize=8,
                    color=INK_SECONDARY, va="center")
    ax.axhline(0, color=AXIS, lw=1)
    ax.set_ylabel("cumulative return (%)")
    ax.set_title(title)
    ax.legend(loc="upper left")
    _save(fig, path)
