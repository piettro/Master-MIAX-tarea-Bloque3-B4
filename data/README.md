# data/

`raw/close_prices.parquet` is a snapshot of the adjusted daily closes of the
23 S&P 500 assets used in the workshop (1962-01-02 to 2026-05-15), downloaded
with `yfinance` exactly as the professor's starter notebook does
(`auto_adjust=True`, columns with missing values dropped).

It is versioned because Yahoo Finance revises history retroactively, so a live
download does not reproduce past results. To refresh it:

```bash
python main.py --refresh-data --stages report
```

If the file is missing, `main.py` downloads it automatically; if that fails,
place a Parquet file with a date index and one column per ticker at
`data/raw/close_prices.parquet`.
