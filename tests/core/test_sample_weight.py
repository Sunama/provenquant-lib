import pytest
import pandas as pd
import numpy as np

from provenquant.core.sample_weight import (
  compute_time_decay,
  compute_time_decay_exponential,
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


class TestComputeTimeDecayExponential:
  def test_basic_decay(self):
    series = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0])
    result = compute_time_decay_exponential(series, last_weight=1.0, decay_factor=0.5)
    # With last_weight=1.0, all weights should be 1.0 regardless of decay_factor
    assert isinstance(result, pd.Series)
    assert len(result) == len(series)
    assert np.allclose(result, 1.0)

  def test_decay_effect(self):
    series = pd.Series([1.0, 1.0, 1.0, 1.0, 1.0])
    result = compute_time_decay_exponential(series, last_weight=0.0, decay_factor=0.5)
    # The newest element (the last one) should have weight 1.0
    assert result.iloc[-1] == 1.0
    # The weights should be monotonically non-decreasing (going forward in time)
    assert (result.diff().iloc[1:] >= 0).all()
    # Older elements should be smaller than 1.0
    assert result.iloc[0] < 1.0

  def test_decay_factor_effect(self):
    series = pd.Series([1.0, 1.0, 1.0, 1.0, 1.0])
    # Higher decay factor should result in smaller weights for older samples
    result_high = compute_time_decay_exponential(series, last_weight=0.0, decay_factor=1.0)
    result_low = compute_time_decay_exponential(series, last_weight=0.0, decay_factor=0.1)
    
    # Oldest sample (index 0) should have lower weight for higher decay factor
    assert result_high.iloc[0] < result_low.iloc[0]
    # Newest sample is always 1.0
    assert result_high.iloc[-1] == 1.0
    assert result_low.iloc[-1] == 1.0

  def test_respects_last_weight_floor(self):
    series = pd.Series([1.0, 1.0, 1.0, 1.0, 1.0])
    last_weight = 0.5
    result = compute_time_decay_exponential(series, last_weight=last_weight, decay_factor=5.0)
    
    # For a very high decay factor, the weights should approach last_weight but not go below it
    assert (result >= last_weight).all()
    # The oldest sample (most decayed) should be close to last_weight if factor is high.
    # Actually, it will be (1-L)*exp(-d*(T-u0)) + L
    assert result.iloc[0] >= last_weight


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