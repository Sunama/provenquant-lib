import pytest
import pandas as pd
from provenquant.core.splitter.sliding_window_split import SlidingWindowSplitter

class TestSlidingWindowSplitter:
  
  def test_basic_split(self):
    """Test basic splitting functionality with simple parameters."""
    splitter = SlidingWindowSplitter(
      n_splits=2,
      train_day=10,
      val_day=5
    )
    
    start_date = pd.Timestamp("2024-01-01")
    end_date = pd.Timestamp("2024-02-01")
    
    splits = list(splitter.split(start_date, end_date))
    assert len(splits) == 2
    
    # Check chronological order
    for i in range(len(splits) - 1):
      assert splits[i][0][0] < splits[i+1][0][0]
  
  def test_split_with_embargo(self):
    """Test splitting with embargo days."""
    splitter = SlidingWindowSplitter(
      n_splits=1,
      train_day=10,
      val_day=5,
      embargo_day=2
    )
    
    start_date = pd.Timestamp("2024-01-01")
    end_date = pd.Timestamp("2024-02-01")
    
    splits = list(splitter.split(start_date, end_date))
    assert len(splits) == 1
    
    (train_start, train_end), (test_start, test_end) = splits[0]
    # Verify embargo gap between train_end and test_start
    gap = test_start - train_end
    assert gap.days == 2
  
  def test_split_with_purging(self):
    """Test splitting with purging days."""
    splitter = SlidingWindowSplitter(
      n_splits=1,
      train_day=10,
      val_day=5,
      purging_day=3
    )
    
    start_date = pd.Timestamp("2024-01-01")
    end_date = pd.Timestamp("2024-02-01")
    
    splits = list(splitter.split(start_date, end_date))
    assert len(splits) == 1
    
    (train_start, train_end), (test_start, test_end) = splits[0]
    gap = test_start - train_end
    assert gap.days == 3
  
  def test_no_splits_when_insufficient_data(self):
    """Test that no splits are returned when insufficient data."""
    splitter = SlidingWindowSplitter(
      n_splits=10,
      train_day=100,
      val_day=10
    )
    
    start_date = pd.Timestamp("2024-01-01")
    end_date = pd.Timestamp("2024-01-10")
    
    splits = list(splitter.split(start_date, end_date))
    assert len(splits) == 0
  
  def test_validation_period_length(self):
    """Test that validation period has correct length."""
    splitter = SlidingWindowSplitter(
      n_splits=1,
      train_day=10,
      val_day=7
    )
    
    start_date = pd.Timestamp("2024-01-01")
    end_date = pd.Timestamp("2024-02-01")
    
    splits = list(splitter.split(start_date, end_date))
    (_, _), (test_start, test_end) = splits[0]
    
    assert (test_end - test_start).days == 7
  
  def test_training_period_length(self):
    """Test that training period has correct length."""
    splitter = SlidingWindowSplitter(
      n_splits=1,
      train_day=15,
      val_day=5
    )
    
    start_date = pd.Timestamp("2024-01-01")
    end_date = pd.Timestamp("2024-02-01")
    
    splits = list(splitter.split(start_date, end_date))
    (train_start, train_end), (_, _) = splits[0]
    
    assert (train_end - train_start).days == 15