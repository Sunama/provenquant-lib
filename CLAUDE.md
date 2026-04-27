# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## About

`provenquant` is a Python library that bridges academic financial ML research (primarily from *Advances in Financial Machine Learning* by Lopez de Prado) to production trading, with a freqtrade adapter.

## Commands

```bash
# Install dependencies (requires Poetry)
poetry install

# Run all tests
poetry run pytest

# Run a single test file
poetry run pytest tests/core/test_labeling.py

# Run a specific test
poetry run pytest tests/core/test_labeling.py::test_name -v
```

## Architecture

The library is organized into three top-level packages:

### `provenquant/core/`
The main algorithmic layer. Each module maps to a distinct ML pipeline stage:

- **`bars.py`** — Alternative bar types: dollar bars, tick bars, volume bars, dollar/volume imbalance bars, and timeframe resampling. Bars always index on `end_time` and carry `start_time`, `cum_ticks`, `cum_dollar`, `buy_volume`, `sell_volume`.
- **`labeling.py`** — Triple barrier labeling pipeline. Flow: CUSUM filter (`filtrate_tripple_label_barrier` / `filtrate_dynamic_tripple_label_barrier`) → label assignment (`get_tripple_label_barrier` / `get_dynamic_tripple_label_barrier`). Labels map to `{-1: 0, 0: 1, 1: 2}` via `mapped_label`. Also contains OU process fitting (`fit_ou_ols`), simulation (`simulate_ou_process`), and Optuna-based barrier optimization (`optimize_triple_barriers`).
- **`feature_engineer.py`** — Fractional differentiation (FFD): `get_frac_diff` for a single series, `get_frac_diffs` for a list, `get_frac_diff_df` for a DataFrame. Supports multi-process parallelism via `num_threads`.
- **`feature_selection.py`** — Feature importance via MDA (`calculate_mda_feature_importances`) and SFI (`calculate_sfi_feature_importances`), orthogonal features via PCA (`orthogonal_features`), stationarity testing (`stationary_test` uses ADF + KPSS), and backward feature elimination.
- **`sample_weight.py`** — Sample weighting: `compute_average_uniqueness` (concurrency-based), `compute_abs_return_uniqueness`, `compute_time_decay` (linear), `compute_time_decay_exponential`.
- **`cross_validation/`** — `PurgedKFold` and `CombinatorialPurgedKFold` that respect the temporal structure of financial data by purging and embargoing samples around test folds.
- **`splitter/`** — `SlidingWindowSplitter` for walk-forward backtesting with configurable train/test/OOS windows and optional purging/embargo gaps.
- **`metric/`** — `calculate_psr` (Probabilistic Sharpe Ratio) and `calculate_psi` (Population Stability Index).

### `provenquant/utils/`
DataFrame utilities:

- **`merge.py`** — Multi-timeframe merging: `larger_timeframe_merge_to_smaller_timeframe_dataframe` (leak-safe `merge_asof`), `smaller_timeframe_merge_sum_to_larger_timeframe_dataframe`, `match_merge_dataframe`, `match_merge_series`.
- **`batch_dataframe.py`** — Batch processing helpers.

### `provenquant/adapters/`
Production integration:

- **`freqtrade.py`** — `get_entry_row` and `is_exit_with_vertical_barrier` for wiring triple-barrier exits into a freqtrade strategy's `custom_exit` callback.

## Conventions

- All functions accept either a datetime index (`time_col='index'`) or a named datetime column (`time_col='date'` etc.), which is the freqtrade convention.
- DataFrames passed to labeling functions must already have a `t1` column (the vertical barrier timestamp) produced by the filtrate step.
- `conftest.py` forces `multiprocessing` start method to `spawn` to avoid fork issues with numpy in tests.
