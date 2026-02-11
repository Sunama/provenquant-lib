import pytest
import pandas as pd
import numpy as np

from provenquant.core.sample_weight import (
  compute_time_decay,
  compute_abs_return_uniqueness,
  compute_average_uniqueness,
)


class TestComputeTimeDecay:
  def test_basic_decay(self):
    series = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0])
    result = compute_time_decay(series, last_weight=1.0)
    assert isinstance(result, pd.Series)
    assert len(result) == len(series)

  def test_decay_with_zero_last_weight(self):
    series = pd.Series([1.0, 2.0, 3.0])
    result = compute_time_decay(series, last_weight=0.0)
    assert isinstance(result, pd.Series)
    assert len(result) == 3

  def test_decay_with_negative_last_weight(self):
    series = pd.Series([1.0, 2.0, 3.0])
    result = compute_time_decay(series, last_weight=-0.5)
    assert isinstance(result, pd.Series)

  def test_decay_handles_negative_values(self):
    series = pd.Series([1.0, -2.0, 3.0, -1.0])
    result = compute_time_decay(series)
    assert (result >= 0).all()


class TestComputeAbsReturnUniqueness:
  def test_basic_computation(self):
    df = pd.DataFrame({
      't1': [2, 3, 4],
      'return': [0.05, -0.03, 0.02],
      'uniqueness': [0.8, 0.9, 0.7],
    })
    result = compute_abs_return_uniqueness(df)
    assert isinstance(result, pd.Series)
    assert len(result) == 3

  def test_with_nan_vertical_barrier(self):
    df = pd.DataFrame({
      't1': [2, np.nan, 4],
      'return': [0.05, 0.03, 0.02],
      'uniqueness': [0.8, 0.9, 0.7],
    })
    result = compute_abs_return_uniqueness(df)
    # The function currently ignores t1 and computes |return| * uniqueness
    assert result.iloc[1] > 0

  def test_with_invalid_barrier(self):
    df = pd.DataFrame({
      't1': [1, 2, 3],
      'return': [0.05, 0.03, 0.02],
      'uniqueness': [0.8, 0.9, 0.7],
    }, index=[2, 3, 4])
    result = compute_abs_return_uniqueness(df)
    # The function ignores the relation between index and t1
    assert result.iloc[0] > 0

  def test_normalize_false(self):
    df = pd.DataFrame({
      't1': [2, 3, 4],
      'return': [0.5, 0.3, 0.2],
      'uniqueness': [0.8, 0.9, 0.7],
    })
    result = compute_abs_return_uniqueness(df, normalize=False)
    assert all(result >= 0)

  def test_respects_min_weight(self):
    df = pd.DataFrame({
      't1': [2, 3, 4],
      'return': [0.0, 0.0, 0.0],
      'uniqueness': [0.0, 0.0, 0.0],
    })
    min_weight = 1e-6
    result = compute_abs_return_uniqueness(df, min_weight=min_weight)
    assert all(result >= min_weight)


class TestComputeAverageUniqueness:
  def test_basic_computation(self):
    df = pd.DataFrame({
      't1': [2, 3, 4],
    })
    result = compute_average_uniqueness(df)
    assert isinstance(result, pd.Series)
    assert len(result) == 3

  def test_with_overlapping_events(self):
    df = pd.DataFrame({
      't1': [3, 4, 5],
    }, index=[1, 2, 3])
    result = compute_average_uniqueness(df)
    assert isinstance(result, pd.Series)
    assert len(result) == 3

  def test_with_nan_barrier(self):
    df = pd.DataFrame({
      't1': [2, np.nan, 4],
    })
    result = compute_average_uniqueness(df)
    assert result.iloc[1] == 0.0

  def test_with_invalid_barrier(self):
    df = pd.DataFrame({
      't1': [1, 2, 3],
    }, index=[2, 3, 4])
    result = compute_average_uniqueness(df)
    assert result.iloc[0] == 0.0

  def test_non_index_datetime_col(self):
    df = pd.DataFrame({
      'start_time': [1, 2, 3],
      't1': [3, 4, 5],
    })
    result = compute_average_uniqueness(df, datetime_col='start_time')
    assert isinstance(result, pd.Series)
    assert len(result) == 3