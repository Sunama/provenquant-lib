import pytest
import pandas as pd
import numpy as np
from provenquant.core.feature_engineer import get_frac_diff, get_frac_diff_df, get_frac_diffs, _process_column

class TestGetFracDiff:
  """Tests for get_frac_diff function."""
  
  def test_basic_fractional_diff(self):
    """Test basic fractional differencing with simple series."""
    series = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0])
    result = get_frac_diff(series, d=0.5)
    
    assert isinstance(result, pd.Series)
    assert len(result) == len(series)
    assert result.index.equals(series.index)
  
  def test_zero_differencing(self):
    """Test with d=0 (no differencing)."""
    series = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0])
    result = get_frac_diff(series, d=0)
    
    pd.testing.assert_series_equal(result, series)
  
  def test_single_element(self):
    """Test with single element series."""
    series = pd.Series([5.0])
    result = get_frac_diff(series, d=0.5)
    
    assert len(result) == 1
    assert result.iloc[0] == 5.0
  
  def test_with_nan_values(self):
    """Test handling of NaN values."""
    series = pd.Series([1.0, np.nan, 3.0, 4.0, 5.0])
    result = get_frac_diff(series, d=0.5)
    
    assert len(result) == len(series)
    assert isinstance(result, pd.Series)
  
  def test_negative_d(self):
    """Test with negative differencing order."""
    series = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0])
    result = get_frac_diff(series, d=-0.5)
    
    assert isinstance(result, pd.Series)
    assert len(result) == len(series)


class TestGetFracDiffs:
  """Tests for get_frac_diffs function."""
  
  def test_single_series_sequential(self):
    """Test fractional differencing on a single series."""
    series = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0])
    result = get_frac_diffs([series], d=0.5, num_threads=1)
    
    assert isinstance(result, list)
    assert len(result) == 1
    assert isinstance(result[0], pd.Series)
    assert len(result[0]) == len(series)
  
  def test_multiple_series_sequential(self):
    """Test fractional differencing on multiple series sequentially."""
    series1 = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0])
    series2 = pd.Series([10.0, 20.0, 30.0, 40.0, 50.0])
    series3 = pd.Series([100.0, 200.0, 300.0, 400.0, 500.0])
    
    result = get_frac_diffs([series1, series2, series3], d=0.5, num_threads=1)
    
    assert isinstance(result, list)
    assert len(result) == 3
    for i, res in enumerate(result):
      assert isinstance(res, pd.Series)
      assert len(res) == 5
  
  def test_multiple_series_parallel(self):
    """Test fractional differencing on multiple series with parallel processing."""
    series1 = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0])
    series2 = pd.Series([10.0, 20.0, 30.0, 40.0, 50.0])
    series3 = pd.Series([100.0, 200.0, 300.0, 400.0, 500.0])
    
    result = get_frac_diffs([series1, series2, series3], d=0.5, num_threads=2)
    
    assert isinstance(result, list)
    assert len(result) == 3
    for i, res in enumerate(result):
      assert isinstance(res, pd.Series)
      assert len(res) == 5
  
  def test_empty_list(self):
    """Test with empty series list."""
    result = get_frac_diffs([], d=0.5, num_threads=1)
    
    assert isinstance(result, list)
    assert len(result) == 0
  
  def test_different_length_series(self):
    """Test with series of different lengths."""
    series1 = pd.Series([1.0, 2.0, 3.0])
    series2 = pd.Series([10.0, 20.0, 30.0, 40.0, 50.0])
    
    result = get_frac_diffs([series1, series2], d=0.5, num_threads=1)
    
    assert len(result) == 2
    assert len(result[0]) == 3
    assert len(result[1]) == 5
  
  def test_sequential_vs_parallel_consistency(self):
    """Test that sequential and parallel processing produce same results."""
    series1 = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0])
    series2 = pd.Series([10.0, 20.0, 30.0, 40.0, 50.0])
    
    result_seq = get_frac_diffs([series1, series2], d=0.5, num_threads=1)
    result_par = get_frac_diffs([series1, series2], d=0.5, num_threads=2)
    
    assert len(result_seq) == len(result_par)
    for seq, par in zip(result_seq, result_par):
      pd.testing.assert_series_equal(seq, par)
  
  def test_zero_differencing(self):
    """Test with d=0 (no differencing)."""
    series1 = pd.Series([1.0, 2.0, 3.0])
    series2 = pd.Series([4.0, 5.0, 6.0])
    
    result = get_frac_diffs([series1, series2], d=0, num_threads=1)
    
    pd.testing.assert_series_equal(result[0], series1)
    pd.testing.assert_series_equal(result[1], series2)
  
  def test_with_named_series(self):
    """Test with named series."""
    series1 = pd.Series([1.0, 2.0, 3.0], name='price')
    series2 = pd.Series([10.0, 20.0, 30.0], name='volume')
    
    result = get_frac_diffs([series1, series2], d=0.5, num_threads=1)
    
    assert len(result) == 2
    assert isinstance(result[0], pd.Series)
    assert isinstance(result[1], pd.Series)


class TestProcessColumn:
  """Tests for _process_column helper function."""
  
  def test_process_single_column(self):
    """Test processing a single column."""
    df = pd.DataFrame({'price': [1.0, 2.0, 3.0, 4.0, 5.0]})
    new_col_name, result_series = _process_column('price', df, d=0.5, prefix='', postfix='_diff')
    
    assert new_col_name == 'price_diff'
    assert isinstance(result_series, pd.Series)
    assert len(result_series) == len(df)
  
  def test_process_column_with_prefix(self):
    """Test processing with custom prefix."""
    df = pd.DataFrame({'value': [1.0, 2.0, 3.0]})
    new_col_name, _ = _process_column('value', df, d=0.5, prefix='frac_', postfix='')
    
    assert new_col_name == 'frac_value'


class TestGetFracDiffDf:
  """Tests for get_frac_diff_df function."""
  
  def test_single_column_sequential(self):
    """Test fractional differencing on single column with sequential processing."""
    df = pd.DataFrame({'price': [1.0, 2.0, 3.0, 4.0, 5.0]})
    result = get_frac_diff_df(df, cols=['price'], d=0.5, num_threads=1)
    
    assert 'price_frac_diff' in result.columns
    assert len(result) == len(df)
    assert 'price' in result.columns
  
  def test_multiple_columns_sequential(self):
    """Test fractional differencing on multiple columns."""
    df = pd.DataFrame({
      'price': [1.0, 2.0, 3.0, 4.0, 5.0],
      'volume': [100.0, 200.0, 300.0, 400.0, 500.0]
    })
    result = get_frac_diff_df(df, cols=['price', 'volume'], d=0.5, num_threads=1)
    
    assert 'price_frac_diff' in result.columns
    assert 'volume_frac_diff' in result.columns
    assert len(result) == len(df)
  
  def test_custom_prefix_postfix(self):
    """Test with custom prefix and postfix."""
    df = pd.DataFrame({'price': [1.0, 2.0, 3.0]})
    result = get_frac_diff_df(df, cols=['price'], d=0.5, prefix='new_', postfix='_modified', num_threads=1)
    
    assert 'new_price_modified' in result.columns
    assert 'price' in result.columns
  
  def test_preserves_original_columns(self):
    """Test that original columns are preserved."""
    df = pd.DataFrame({'price': [1.0, 2.0, 3.0, 4.0, 5.0]})
    result = get_frac_diff_df(df, cols=['price'], d=0.5, num_threads=1)
    
    pd.testing.assert_series_equal(result['price'], df['price'])
  
  def test_empty_columns_list(self):
    """Test with empty columns list."""
    df = pd.DataFrame({'price': [1.0, 2.0, 3.0]})
    result = get_frac_diff_df(df, cols=[], d=0.5, num_threads=1)
    
    pd.testing.assert_frame_equal(result, df)
  
  def test_parallel_processing(self):
    """Test parallel processing with multiple processes."""
    df = pd.DataFrame({
      'price': [1.0, 2.0, 3.0, 4.0, 5.0],
      'volume': [100.0, 200.0, 300.0, 400.0, 500.0]
    })
    result = get_frac_diff_df(df, cols=['price', 'volume'], d=0.5, num_threads=2)
    
    assert 'price_frac_diff' in result.columns
    assert 'volume_frac_diff' in result.columns
    assert len(result) == len(df)
  
  def test_different_d_values(self):
    """Test with different differencing orders."""
    df = pd.DataFrame({'price': [1.0, 2.0, 3.0, 4.0, 5.0]})
    
    result_d_0 = get_frac_diff_df(df, cols=['price'], d=0, num_threads=1)
    result_d_1 = get_frac_diff_df(df, cols=['price'], d=1, num_threads=1)
    result_d_0_5 = get_frac_diff_df(df, cols=['price'], d=0.5, num_threads=1)
    
    assert len(result_d_0) == len(result_d_1) == len(result_d_0_5)