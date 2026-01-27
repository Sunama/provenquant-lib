import pytest
import pandas as pd
import numpy as np
import tempfile
import os
from pathlib import Path
from provenquant.utils.batch_dataframe import BatchDataframe

@pytest.fixture
def temp_dir():
  """Create a temporary directory for test files."""
  with tempfile.TemporaryDirectory() as tmpdir:
    yield tmpdir


@pytest.fixture
def sample_dataframe():
  """Create a sample DataFrame with datetime index."""
  dates = pd.date_range('2024-01-01', periods=100, freq='h')
  df = pd.DataFrame({
    'value': np.random.randn(100),
    'price': np.random.rand(100) * 100
  }, index=dates)
  return df


class TestBatchDataframe:
  
  def test_init(self, temp_dir):
    """Test BatchDataframe initialization."""
    batch_df = BatchDataframe(dir_path=temp_dir, batch_size='1D')
    assert batch_df.dir_path == temp_dir
    assert batch_df.batch_size == '1D'
  
  def test_save_dataframe(self, temp_dir, sample_dataframe):
    """Test saving DataFrame in batches."""
    batch_df = BatchDataframe(dir_path=temp_dir, batch_size='1D')
    batch_df.save_dataframe(sample_dataframe)
    
    files = os.listdir(temp_dir)
    assert len(files) > 0
    assert any(f.endswith('.parquet') for f in files)
  
  def test_load_dataframe(self, temp_dir, sample_dataframe):
    """Test loading DataFrame from batches."""
    batch_df = BatchDataframe(dir_path=temp_dir, batch_size='1D')
    batch_df.save_dataframe(sample_dataframe)
    
    from_dt = sample_dataframe.index.min()
    to_dt = sample_dataframe.index.max()
    loaded_df = batch_df.load_dataframe(from_dt, to_dt)
    
    assert len(loaded_df) > 0
    assert loaded_df.index.min() >= from_dt
    assert loaded_df.index.max() <= to_dt
  
  def test_load_dataframe_partial_range(self, temp_dir, sample_dataframe):
    """Test loading DataFrame with partial datetime range."""
    batch_df = BatchDataframe(dir_path=temp_dir, batch_size='1D')
    batch_df.save_dataframe(sample_dataframe)
    
    from_dt = sample_dataframe.index[10]
    to_dt = sample_dataframe.index[50]
    loaded_df = batch_df.load_dataframe(from_dt, to_dt)
    
    assert len(loaded_df) <= len(sample_dataframe)
    assert len(loaded_df) > 0
    assert loaded_df.index.min() >= from_dt
  
  def test_load_empty_directory(self, temp_dir):
    """Test loading from empty directory."""
    batch_df = BatchDataframe(dir_path=temp_dir)
    from_dt = pd.Timestamp('2024-01-01')
    to_dt = pd.Timestamp('2024-01-02')
    
    result = batch_df.load_dataframe(from_dt, to_dt)
    assert result.empty
  
  def test_batch_size_default(self, temp_dir):
    """Test default batch size."""
    batch_df = BatchDataframe(dir_path=temp_dir)
    assert batch_df.batch_size == '1D'
  
  def test_last_datetime_with_data(self, temp_dir, sample_dataframe):
    """Test last_datetime() returns the last datetime when batches have data."""
    batch_df = BatchDataframe(dir_path=temp_dir, batch_size='1D')
    batch_df.save_dataframe(sample_dataframe)
    
    last_dt = batch_df.last_datetime()
    expected_last = sample_dataframe.index.max()
    
    assert last_dt is not None
    assert last_dt == expected_last
  
  def test_last_datetime_empty_directory(self, temp_dir):
    """Test last_datetime() returns None when directory is empty."""
    batch_df = BatchDataframe(dir_path=temp_dir)
    last_dt = batch_df.last_datetime()
    
    assert last_dt is None
  
  def test_last_datetime_multiple_batches(self, temp_dir):
    """Test last_datetime() with multiple batch files."""
    batch_df = BatchDataframe(dir_path=temp_dir, batch_size='1D')
    
    # Create multiple DataFrames with different date ranges
    dates1 = pd.date_range('2024-01-01', periods=24, freq='h')
    df1 = pd.DataFrame({
      'value': np.random.randn(24),
      'price': np.random.rand(24) * 100
    }, index=dates1)
    
    dates2 = pd.date_range('2024-01-03', periods=24, freq='h')
    df2 = pd.DataFrame({
      'value': np.random.randn(24),
      'price': np.random.rand(24) * 100
    }, index=dates2)
    
    # Save both DataFrames
    batch_df.save_dataframe(df1)
    batch_df.save_dataframe(df2)
    
    last_dt = batch_df.last_datetime()
    expected_last = df2.index.max()
    
    assert last_dt is not None
    assert last_dt == expected_last