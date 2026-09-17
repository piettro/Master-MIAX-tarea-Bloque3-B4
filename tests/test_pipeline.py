"""End-to-end integration test on a tiny synthetic market.

Runs every stage of ``main.py`` -- competition, report, research and the
2025 portfolio -- with two epochs and two networks, into a temporary
directory.
"""

from __future__ import annotations

from dataclasses import replace

import pandas as pd

from src.models.registry import get_spec
from src.utils.config import ProjectConfig


def _tiny_config(tmp_path) -> ProjectConfig:
    base = ProjectConfig()
    return replace(
        base,
        paths=replace(base.paths, outputs=tmp_path / "outputs"),
        data=replace(
            base.data, tickers=("AAA", "BBB", "CCC", "DDD"),
            input_windows=(5,), output_windows=(1, 90),
        ),
        training=replace(
            base.training, max_epochs=2, patience=1, n_seeds_final=2,
            batch_size=64,
        ),
        research=replace(
            base.research, ffd_threshold=1e-2, d_grid=(0.5, 1.0),
            adf_max_lag=5, stationary_share=0.5,
        ),
    )


def test_full_pipeline_runs(tmp_path, synthetic_prices, synthetic_returns):
    import main
    from src.experiments.competition import run_competition
    from src.experiments.portfolio import run_portfolio
    from src.experiments.research import run_research

    config = _tiny_config(tmp_path)
    specs = (get_spec("mlp_small"), get_spec("cnn"))

    outputs = run_competition(synthetic_returns, config, specs=specs)
    assert len(outputs.results) == 2 * (5 + len(specs))
    assert {"mae_train", "mae_val", "mae_test", "n_params",
            "dm_p_value"} <= set(outputs.results.columns)
    assert len(outputs.best) == 2
    assert (outputs.seed_robustness["n_seeds"] == 2).all()

    # Resuming must not retrain or duplicate finished cells.
    resumed = run_competition(synthetic_returns, config, specs=specs)
    assert len(resumed.results) == len(outputs.results)

    main.run_report(config)
    figures = config.paths.figures
    assert (figures / "competition_matrix.png").exists()
    assert (figures / "cells" / "results_V5_H90.png").exists()
    assert (figures / "training_curves" / "curves_V5_H1.png").exists()
    assert (config.paths.tables / "competition_matrix.txt").exists()

    results = pd.read_csv(config.paths.tables / "competition_results.csv")
    research = run_research(
        synthetic_prices, synthetic_returns, results, config
    )
    assert set(research.results["variant"]) == {
        "log_returns (competition)", "ffd_inputs", "embargo",
    }

    portfolio = run_portfolio(
        synthetic_prices, synthetic_returns, results, config
    )
    assert len(portfolio.metrics) == 2
    assert portfolio.equity_curves.index.min() >= pd.Timestamp("2025-01-01")
    assert portfolio.metrics.loc["equal_weight_buy_and_hold",
                                 "n_rebalances"] == 1
