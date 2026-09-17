"""Single entry point of the workshop pipeline.

Usage::

    python main.py                       # full run (several hours on CPU)
    python main.py --quick               # smoke run, a few minutes
    python main.py --stages report       # re-draw figures and tables only
    python main.py --stages competition --no-resume

Stages run in order: ``competition`` -> ``report`` -> ``research`` ->
``portfolio``. Each later stage reads the competition tables from disk, so
stages can be re-run independently.
"""

from __future__ import annotations

import argparse
import logging
import os
import sys

# Silence TensorFlow's C++ start-up banner before it is imported anywhere.
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import pandas as pd  # noqa: E402

from src.data.loaders import (  # noqa: E402
    compute_log_returns,
    load_close_prices,
)
from src.experiments.competition import (  # noqa: E402
    RESULTS_FILE,
    SEEDS_FILE,
    history_path,
    run_competition,
)
from src.evaluation.selection import select_best  # noqa: E402
from src.reporting.tables import (  # noqa: E402
    write_cell_tables,
    write_competition_matrix,
    write_frame,
)
from src.utils.config import ProjectConfig, quick_config  # noqa: E402
from src.utils.logging_config import configure_logging  # noqa: E402

logger = logging.getLogger("main")

STAGES = ("competition", "report", "research", "portfolio")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments.

    Args:
        argv: Arguments to parse; defaults to ``sys.argv[1:]``.

    Returns:
        Parsed namespace.
    """
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--stages", nargs="+", choices=STAGES, default=list(STAGES),
        help="stages to run (default: all)",
    )
    parser.add_argument(
        "--quick", action="store_true",
        help="2x2 windows, few epochs; writes to outputs_quick/",
    )
    parser.add_argument(
        "--refresh-data", action="store_true",
        help="re-download prices from Yahoo Finance (overwrites snapshot)",
    )
    parser.add_argument(
        "--no-resume", action="store_true",
        help="ignore partial competition results and start over",
    )
    parser.add_argument(
        "--log-level", default="INFO",
        choices=("DEBUG", "INFO", "WARNING", "ERROR"),
    )
    return parser.parse_args(argv)


def load_competition_tables(
    config: ProjectConfig,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Read the competition results written by a previous run.

    Args:
        config: Project configuration.

    Returns:
        ``(results, seed_robustness)``.

    Raises:
        FileNotFoundError: If the competition stage has not been run.
    """
    results_path = config.paths.tables / RESULTS_FILE
    if not results_path.exists():
        raise FileNotFoundError(
            f"{results_path} not found; run `python main.py --stages "
            f"competition` first"
        )
    seeds_path = config.paths.tables / SEEDS_FILE
    seeds = pd.read_csv(seeds_path) if seeds_path.exists() else pd.DataFrame()
    return pd.read_csv(results_path), seeds


def run_report(config: ProjectConfig) -> None:
    """Write competition tables and figures from saved results.

    Args:
        config: Project configuration.
    """
    from src.visualization import plots

    results, seeds = load_competition_tables(config)
    best = select_best(results, by="mae_val")
    best_networks = select_best(
        results, by="mae_val",
        families=("dense", "recurrent", "convolutional", "mixed"),
    )
    tables, figures = config.paths.tables, config.paths.figures

    write_cell_tables(results, tables)
    matrix_text = write_competition_matrix(best, seeds, tables, best_networks)
    logger.info("Competition matrix (test MAE, selected on validation):\n%s",
                matrix_text)

    for (V, H), cell in results.groupby(["input_window", "output_window"]):
        plots.plot_cell_results(
            cell, f"V={V}, H={H}: MAE by model",
            figures / "cells" / f"results_V{V}_H{H}.png",
        )
        histories = {}
        for model in cell.loc[cell["family"] != "baseline", "model"]:
            path = history_path(config.paths.histories, V, H, model)
            if path.exists():
                histories[model] = pd.read_csv(path)
        if histories:
            plots.plot_training_curves(
                histories, f"Training curves V={V}, H={H}",
                figures / "training_curves" / f"curves_V{V}_H{H}.png",
            )
    for H in sorted(results["output_window"].unique()):
        plots.plot_output_window_summary(
            results, H, figures / f"summary_output_window_H{H}.png"
        )
    plots.plot_competition_matrix(
        best, results, figures / "competition_matrix.png"
    )


def main(argv: list[str] | None = None) -> int:
    """Run the requested pipeline stages.

    Args:
        argv: Command-line arguments.

    Returns:
        Process exit code.
    """
    args = parse_args(argv)
    config = quick_config() if args.quick else ProjectConfig()
    configure_logging(getattr(logging, args.log_level), config.paths.log_file)
    logger.info("Stages %s | outputs in %s", args.stages, config.paths.outputs)

    try:
        prices = load_close_prices(
            config.paths.raw_prices, config.data.tickers,
            config.data.start_date, refresh=args.refresh_data,
        )
        returns = compute_log_returns(prices)
        logger.info("Loaded %d days x %d assets (%s to %s)", *returns.shape,
                    returns.index[0].date(), returns.index[-1].date())

        if "competition" in args.stages:
            run_competition(returns, config, resume=not args.no_resume)
        if "report" in args.stages:
            run_report(config)
        if "research" in args.stages:
            from src.experiments.research import run_research
            from src.visualization.plots import plot_research

            results, _ = load_competition_tables(config)
            research = run_research(prices, returns, results, config)
            write_frame(research.results, config.paths.tables / "research")
            plot_research(
                research.results,
                config.paths.figures / "research_ablation.png",
            )
        if "portfolio" in args.stages:
            from src.experiments.portfolio import run_portfolio
            from src.visualization.plots import (
                plot_portfolios,
                plot_training_curves,
            )

            results, _ = load_competition_tables(config)
            portfolio = run_portfolio(prices, returns, results, config)
            write_frame(portfolio.metrics, config.paths.tables
                        / "portfolio_2025", index=True)
            logger.info("2025 portfolios:\n%s",
                        portfolio.metrics.round(4).to_string())
            plot_portfolios(
                portfolio.equity_curves, "2025: portfolios with and without "
                "model predictions",
                config.paths.figures / "portfolio_2025.png",
            )
            plot_training_curves(
                {portfolio.model_name: portfolio.network.history},
                f"Portfolio model {portfolio.model_name} "
                f"V={portfolio.input_window} (targets before 2025)",
                config.paths.figures / "portfolio_model_curve.png",
            )
    except (FileNotFoundError, ValueError) as exc:
        logger.error("%s", exc)
        return 1
    logger.info("Done.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
