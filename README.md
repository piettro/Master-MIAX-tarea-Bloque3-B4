# Neural Networks for Multi-Asset Forecasting

> Master's in AI & Quantum Computing Applied to Financial Markets (MIAX)
> Course: Basic Artificial Intelligence | Assignment: Workshop B3-T4/T5/T6 --
> Neural Networks for Forecasting

## Overview

The project forecasts the **average daily log return of 23 S&P 500 stocks
over the next H trading days** from the previous V days, training every model
to minimise the mean absolute error (MAE), as the assignment brief requires.

- **Competition.** All 16 combinations of input windows V ∈ {5, 10, 30, 90}
  and output windows H ∈ {1, 5, 30, 90} are covered. Each cell trains 8 neural
  networks from four families and 5 simple baselines. The families are dense,
  recurrent (LSTM/GRU), convolutional (causal Conv1D), and mixed
  (Conv→GRU→Dense, plus a multi-branch CNN/GRU/Dense model). The data and
  chronological 90/10 test split are exactly those of the professor's starter
  notebook. Train, validation and test MAE and the parameter count are reported
  for every model. The winner of each cell is **selected on validation** and
  its **test** MAE forms the 4×4 competition matrix.
- **Statistical comparison.** Every model is compared with the cell's best
  baseline using a Diebold-Mariano test with a Newey-West variance, which
  handles the overlap between H-day targets. Each cell's winning network is
  re-trained with extra seeds.
- **Research.** A one-change-at-a-time ablation of ideas from the financial
  pre-processing workshop (López de Prado):
  1. fractionally differentiated log prices as inputs;
  2. an embargo between the training and validation windows.

  Every variant is scored on the same test set, in the same units.
- **Portfolios.** The best 90-day network is re-trained on data available
  before 2025. It drives a top-5 long-only portfolio that is compared over 2025
  with an equal-weight buy-and-hold portfolio that uses no predictions.

## Project Structure

```
.
├── main.py                     # single entry point (CLI)
├── src/
│   ├── utils/                  # config.py (all constants), logging, seeding
│   ├── data/                   # loaders, windowing, chronological splits, scalers
│   ├── preprocessing/          # fractional differentiation + ADF selection of d
│   ├── models/                 # baselines, Keras architectures, model registry
│   ├── training/               # trainer: Adam + MAE, early stopping, histories
│   ├── evaluation/             # metrics, Diebold-Mariano, selection, back-tester
│   ├── experiments/            # competition, research and portfolio stages
│   ├── reporting/              # CSV / Markdown / text tables
│   └── visualization/          # PNG figures
├── tests/                      # 48 pytest tests incl. an end-to-end pipeline test
├── docs/architecture.md        # Mermaid diagrams and design decisions
├── data/raw/close_prices.parquet   # versioned price snapshot (see data/README.md)
├── outputs/                    # generated tables, figures and loss histories
├── professor_solution/         # professor's data starter notebooks (verbatim)
├── materials/                  # assignment brief and lecture transcripts
├── notebooks/                  # exploration only
├── requirements.txt
└── setup.cfg                   # flake8 and pytest settings
```

## Setup & Installation

Python 3.10–3.12 is supported. TensorFlow 2.16 does not support Python 3.13 or
newer.

```bash
# 1. Clone the repo
git clone <repo-url>
cd <project-folder>

# 2. Create and activate a virtual environment
python -m venv .venv
source .venv/bin/activate      # Linux / macOS
.venv\Scripts\activate         # Windows

# 3. Install dependencies
pip install -r requirements.txt
```

## Usage

```bash
python main.py                  # full pipeline into outputs/ (several hours on CPU)
python main.py --quick          # smoke run (2x2 windows, 8 epochs) into outputs_quick/
pytest                          # unit + integration tests (~1 minute)
flake8 src main.py tests        # PEP 8 check (79 columns)
```

| Option | Effect |
|---|---|
| `--stages competition report research portfolio` | Run a subset of stages. Later stages read the competition tables from disk. |
| `--quick` | Small configuration for a fast end-to-end check. |
| `--no-resume` | Retrain every competition cell instead of skipping cells already saved. |
| `--refresh-data` | Re-download prices from Yahoo Finance, overwriting the snapshot. |
| `--log-level DEBUG` | More verbose logging (also written to `outputs/run.log`). |

Every hyper-parameter lives in `src/utils/config.py`: windows, split
fractions, learning rate, patience, seeds, the fractional-differentiation grid,
rebalancing frequency, costs and so on. The model catalogue is in
`src/models/registry.py`.

## Results

RESULTS_PLACEHOLDER

## Architecture

See [docs/architecture.md](docs/architecture.md) for diagrams.

## References

- Professor's solution: `professor_solution/`
- Assignment materials: `materials/`
- M. López de Prado, *Advances in Financial Machine Learning*, Wiley, 2018
  (ch. 5 fractional differentiation, ch. 7 purging and embargo).
- F. X. Diebold and R. S. Mariano, "Comparing Predictive Accuracy", *JBES*,
  1995; D. Harvey, S. Leybourne and P. Newbold, *IJF*, 1997.
