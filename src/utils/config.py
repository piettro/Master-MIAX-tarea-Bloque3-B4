"""Central configuration: every path, constant and hyper-parameter.

Nothing else in the package hard-codes a magic number. Stages receive the
relevant frozen dataclass, which keeps them testable with small overrides
(see :func:`quick_config`).
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class PathConfig:
    """File-system layout of the project.

    Attributes:
        root: Project root directory.
        raw_prices: Parquet snapshot of daily adjusted close prices.
        outputs: Root of every generated artefact.
    """

    root: Path = PROJECT_ROOT
    raw_prices: Path = PROJECT_ROOT / "data" / "raw" / "close_prices.parquet"
    outputs: Path = PROJECT_ROOT / "outputs"

    @property
    def tables(self) -> Path:
        """Directory holding CSV and Markdown result tables."""
        return self.outputs / "tables"

    @property
    def figures(self) -> Path:
        """Directory holding every PNG figure."""
        return self.outputs / "figures"

    @property
    def histories(self) -> Path:
        """Directory holding per-training loss histories as CSV."""
        return self.outputs / "histories"

    @property
    def log_file(self) -> Path:
        """Plain-text log of the last run."""
        return self.outputs / "run.log"


@dataclass(frozen=True)
class DataConfig:
    """Data source and windowing parameters (assignment brief, section 3).

    Attributes:
        tickers: The 23 S&P 500 constituents defined by the professor's
            starter notebook.
        start_date: First date requested from Yahoo Finance.
        input_windows: Look-back lengths ``V`` in trading days.
        output_windows: Forecast horizons ``H`` in trading days; the target
            is the mean log return over the next ``H`` days.
        test_fraction: Share of windows reserved for test. The split is
            chronological (``shuffle=False``) exactly as in the provided
            ``train_test_split`` call, so the competition test set is the
            one the professor distributes.
        validation_fraction: Share of the *training* windows (their
            chronological tail) used for validation and model selection.
        embargo_validation: If True, drop the training windows whose raw
            data overlaps the first validation window. Off in the
            competition to mirror the professor's split; switched on as a
            single-change ablation in the research stage.
    """

    tickers: tuple[str, ...] = (
        "AEP", "BA", "CAT", "CNP", "CVX", "DIS", "DTE", "ED", "GD", "GE",
        "HON", "HPQ", "IBM", "IP", "JNJ", "KO", "KR", "MMM", "MO", "MRK",
        "MSI", "PG", "XOM",
    )
    start_date: str = "1945-01-01"
    input_windows: tuple[int, ...] = (5, 10, 30, 90)
    output_windows: tuple[int, ...] = (1, 5, 30, 90)
    test_fraction: float = 0.10
    # NOTE: improved over professor's solution -- with H=90 the targets of
    # neighbouring windows overlap by 89 days, so a 5% validation tail holds
    # only ~8 independent 90-day blocks; 10% halves the selection noise.
    validation_fraction: float = 0.10
    embargo_validation: bool = False


@dataclass(frozen=True)
class TrainingConfig:
    """Optimisation hyper-parameters shared by every neural network.

    Attributes:
        learning_rate: Adam step size. Targets are rescaled by one scalar,
            so the default Keras rate converges without the 1e-5 workaround
            of the original submission.
        batch_size: Mini-batch size.
        max_epochs: Upper bound on epochs; early stopping ends most runs.
        patience: Epochs without validation improvement before stopping.
            Kept generous: the correction lecture criticised curves that
            were still descending when training stopped.
        min_delta: Minimum validation MAE decrease (in scaled units) that
            counts as an improvement.
        use_reduce_lr: Enable ReduceLROnPlateau. Off by default because the
            lecture recommended first finding a learning rate that yields a
            clean curve, then adding automatic schedules only if needed.
        reduce_lr_factor: Multiplicative decay applied on plateau.
        reduce_lr_patience: Epochs of plateau before decaying.
        min_learning_rate: Floor for the decayed learning rate.
        seed: Global random seed.
        n_seeds_final: Independent re-trainings of each selected model, used
            to report the seed dispersion of the winning test MAE.
        max_params_to_samples: Rule of thumb from the lecture (~10 samples
            per parameter); larger models are flagged in the tables.
        verbose: Keras ``fit`` verbosity.
    """

    learning_rate: float = 1e-3
    batch_size: int = 128
    max_epochs: int = 300
    patience: int = 25
    min_delta: float = 1e-4
    use_reduce_lr: bool = False
    reduce_lr_factor: float = 0.5
    reduce_lr_patience: int = 10
    min_learning_rate: float = 1e-6
    seed: int = 42
    n_seeds_final: int = 3
    max_params_to_samples: float = 0.10
    verbose: int = 0


@dataclass(frozen=True)
class ResearchConfig:
    """Parameters of the research (pre-processing) stage.

    Attributes:
        ffd_threshold: Weight truncation of fixed-width fractional
            differentiation.
        d_grid: Orders tested when searching the minimum stationary ``d``.
        adf_significance: ADF p-value threshold for stationarity.
        adf_max_lag: Fixed ADF lag order, so statistics are comparable.
        stationary_share: Share of assets that must pass the ADF test for a
            common ``d`` to be accepted.
    """

    ffd_threshold: float = 1e-4
    d_grid: tuple[float, ...] = (
        0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0,
    )
    adf_significance: float = 0.05
    adf_max_lag: int = 10
    stationary_share: float = 0.90


@dataclass(frozen=True)
class PortfolioConfig:
    """2025 back-test parameters (assignment brief, research task).

    Attributes:
        output_window: Horizon of the model that drives the portfolio.
        start: First trading day of the evaluation year.
        end: Last calendar day of the evaluation year.
        rebalance_every: Trading days between rebalances.
        top_k: Number of assets held by the prediction-driven portfolio.
        transaction_cost_bps: One-way cost charged on traded notional.
        trading_days_per_year: Annualisation constant.
    """

    output_window: int = 90
    start: str = "2025-01-01"
    end: str = "2025-12-31"
    rebalance_every: int = 21
    top_k: int = 5
    transaction_cost_bps: float = 5.0
    trading_days_per_year: int = 252


@dataclass(frozen=True)
class EvaluationConfig:
    """Statistical comparison settings.

    Attributes:
        significance: Level of the Diebold-Mariano test.
    """

    significance: float = 0.05


@dataclass(frozen=True)
class ProjectConfig:
    """Aggregate configuration passed to ``main.py`` stages."""

    paths: PathConfig = field(default_factory=PathConfig)
    data: DataConfig = field(default_factory=DataConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    research: ResearchConfig = field(default_factory=ResearchConfig)
    portfolio: PortfolioConfig = field(default_factory=PortfolioConfig)
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)
    quick: bool = False


def quick_config(base: ProjectConfig | None = None) -> ProjectConfig:
    """Derive a fast smoke-test configuration from ``base``.

    Two window sizes, few epochs and a single seed: every stage and figure
    is exercised in minutes, but the numbers are not competition results.

    Args:
        base: Configuration to shrink; defaults to :class:`ProjectConfig`.

    Returns:
        A new configuration writing to ``outputs_quick/``.
    """
    base = base or ProjectConfig()
    return replace(
        base,
        paths=replace(base.paths, outputs=base.paths.root / "outputs_quick"),
        data=replace(base.data, input_windows=(5, 30), output_windows=(1, 90)),
        training=replace(
            base.training, max_epochs=8, patience=3, n_seeds_final=1
        ),
        quick=True,
    )
