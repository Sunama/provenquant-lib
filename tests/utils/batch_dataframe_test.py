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
    assert batch_df.batches == {}
  
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