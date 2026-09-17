# Architecture

## 1. Project overview

The project forecasts the **mean daily log return of 23 S&P 500 stocks over
the next H days** (H = 1, 5, 30, 90) from the previous V days (V = 5, 10, 30,
90), minimising MAE. For each of the 16 window combinations it trains 8 neural
networks from 4 families (dense, recurrent, convolutional, mixed) and 5
baselines, selects the best model per cell on validation, and reports its
test MAE. It also runs a one-change-at-a-time pre-processing study and a
2025 back-test of a portfolio driven by the best 90-day model.

## 2. Module dependency diagram

```mermaid
graph TD
    main[main.py] --> cfg[src/utils/config.py]
    main --> log[src/utils/logging_config.py]
    main --> loaders[src/data/loaders.py]
    main --> comp[src/experiments/competition.py]
    main --> research[src/experiments/research.py]
    main --> portfolio[src/experiments/portfolio.py]
    main --> tables[src/reporting/tables.py]
    main --> plots[src/visualization/plots.py]

    comp --> windowing[src/data/windowing.py]
    comp --> splits[src/data/splits.py]
    comp --> runner[src/experiments/runner.py]
    comp --> baselines[src/models/baselines.py]
    comp --> registry[src/models/registry.py]
    comp --> signif[src/evaluation/significance.py]
    comp --> selection[src/evaluation/selection.py]

    research --> fracdiff[src/preprocessing/fracdiff.py]
    research --> runner
    research --> comp
    research --> signif

    portfolio --> backtest[src/evaluation/backtest.py]
    portfolio --> trainer
    portfolio --> selection

    runner --> trainer[src/training/trainer.py]
    runner --> metrics[src/evaluation/metrics.py]
    trainer --> scaling[src/data/scaling.py]
    trainer --> registry
    trainer --> seeding[src/utils/seeding.py]
    registry --> arch[src/models/architectures.py]
    tables --> selection
```

| Package | Responsibility |
|---|---|
| `src/utils` | Frozen configuration dataclasses, logging set-up, seeding |
| `src/data` | Price snapshot I/O, log returns, sliding windows, chronological splits, scalers |
| `src/preprocessing` | Fixed-width fractional differentiation and ADF-based choice of `d` |
| `src/models` | Baselines, Keras architectures, named model catalogue |
| `src/training` | Compile, fit with early stopping, loss history, convergence flag |
| `src/evaluation` | MAE metrics, Diebold-Mariano test, validation-based selection, back-tester |
| `src/experiments` | The three stages: competition, research, portfolio |
| `src/reporting`, `src/visualization` | CSV / Markdown / text tables and PNG figures |

## 3. Data flow

```mermaid
flowchart LR
    A[(close_prices.parquet)] --> B[log returns]
    B --> C["create_windows(V, H)<br/>X: N x V x 23<br/>y: N x 23 (mean of next H)"]
    C --> D["chronological split<br/>train | validation | test (last 10%)"]
    D --> E[InputScaler / TargetScaler<br/>fit on train only]
    E --> F[5 baselines + 8 networks<br/>MAE loss, early stopping on val]
    F --> G[MAE train / val / test<br/>in log-return units]
    G --> H[select best per cell<br/>on VALIDATION MAE]
    H --> I[competition matrix<br/>test MAE, V x H]
    G --> J[Diebold-Mariano<br/>vs best baseline]

    A --> K[log prices] --> L[FFD d* chosen on<br/>pre-validation data]
    L --> M[research: same split,<br/>FFD inputs, same targets]
    H --> M
    M --> N[research table:<br/>MAE change + DM p-value]

    H --> O[best H=90 network]
    O --> P[retrain on targets<br/>ending before 2025]
    P --> Q[forecast at each<br/>rebalance date]
    A --> R[simple returns 2025]
    Q --> S[top-k long-only<br/>portfolio]
    R --> S
    R --> T[equal-weight<br/>buy and hold]
    S --> U[2025 metrics & equity curves]
    T --> U
```

### Split geometry of one cell

```
windows:  0 ........................................................ N-1
          |------------- train -------------|--val--|---- test ----|
                                            ^       ^
                         (embargo of V+H-1  |       ceil(0.1 N) windows,
                          windows here in   |       identical to the
                          the research      |       professor's
                          variant)          |       train_test_split(
                                            |       shuffle=False)
```

## 4. Class diagram

```mermaid
classDiagram
    class ProjectConfig {
        PathConfig paths
        DataConfig data
        TrainingConfig training
        ResearchConfig research
        PortfolioConfig portfolio
        EvaluationConfig evaluation
    }
    class Baseline {
        <<abstract>>
        name
        family
        fit(X, y)
        predict(X)
        n_params
    }
    Baseline <|-- BuyAndHoldBaseline
    Baseline <|-- ZeroReturnBaseline
    Baseline <|-- WindowMeanBaseline
    Baseline <|-- HistoricalMeanBaseline
    Baseline <|-- LinearRegressionBaseline

    class ModelSpec {
        name
        family
        builder
        kwargs
        build(input_shape, n_outputs)
    }
    class TrainedNetwork {
        model
        input_scaler
        target_scaler
        history
        best_epoch
        hit_epoch_limit
        predict(X)
    }
    class InputScaler {
        fit(X)
        transform(X)
    }
    class TargetScaler {
        fit(y)
        transform(y)
        inverse_transform(y)
    }
    class SplitIndices {
        train
        validation
        test
    }
    class DataSplit {
        X_train y_train
        X_val y_val
        X_test y_test
    }
    class ModelOutcome {
        row
        test_predictions
        test_losses
        network
    }
    class BacktestResult {
        daily_returns
        weights
        turnover
        equity_curve
    }
    TrainedNetwork --> InputScaler
    TrainedNetwork --> TargetScaler
    ModelSpec ..> TrainedNetwork : train_network()
    SplitIndices ..> DataSplit : apply_split()
    ModelOutcome --> TrainedNetwork
    ProjectConfig ..> ModelOutcome : run_network()
```

## 5. Key design decisions

| Decision | Reason |
|---|---|
| Test set = last `ceil(10%)` windows, chronological | Reproduces the professor's `train_test_split(test_size=0.1, shuffle=False)` exactly (unit-tested) |
| Selection on validation, report test | Correction lecture: choosing on test is cheating |
| Target centred per asset, scaled by **one** scalar | MAE on scaled targets equals MAE in log-return units divided by a constant, so the objective is unchanged; per-asset scales would re-weight assets |
| MAE always in log-return units, including research | Errors for different pre-processing must be comparable (lecture critique of transformed targets) |
| No ReduceLROnPlateau by default | Lecture: first find a learning rate that gives a clean curve |
| Patience 25, best weights restored, `hit_epoch_limit` flag | Lecture criticised curves stopped while still descending |
| Diebold-Mariano with Newey-West lag `H-1` | Overlapping targets make loss differences autocorrelated |
| Research changes one thing at a time | Lecture: otherwise no conclusion can be drawn |
| Portfolio model retrained on targets ending before 2025 | No look-ahead, and uses the most recent pre-2025 data |

## 6. How to run

```bash
python -m venv .venv
.venv\Scripts\activate            # Windows  (source .venv/bin/activate elsewhere)
pip install -r requirements.txt
python main.py --quick            # ~5 minutes, smoke test into outputs_quick/
python main.py                    # full run into outputs/ (several hours on CPU)
pytest                            # 48 tests, ~1 minute
```
