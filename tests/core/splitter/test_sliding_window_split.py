import pytest
import pandas as pd
from datetime import timedelta
from provenquant.core.splitter.sliding_window_split import SlidingWindowSplitter

class TestSlidingWindowSplitter:
    
    def test_basic_split(self):
        """Test basic splitting functionality with simple parameters."""
        splitter = SlidingWindowSplitter(
            n_splits=3,
            train_day=30,
            test_day=10,
            oos_day=5
        )
        
        start_date = pd.Timestamp("2024-01-01")
        end_date = pd.Timestamp("2024-06-01")
        
        splits = list(splitter.split(start_date, end_date))
        assert len(splits) == 3
        
        # Check chronological order
        for i in range(len(splits) - 1):
            assert splits[i][0][0] < splits[i+1][0][0] # train_start
            assert splits[i][1][0] < splits[i+1][1][0] # test_start
            assert splits[i][2][0] < splits[i+1][2][0] # oos_start

    def test_split_structure(self):
        """Test that the split returns the correct structure of tuples."""
        splitter = SlidingWindowSplitter(
            n_splits=1,
            train_day=30,
            test_day=10,
            oos_day=5
        )
        
        start_date = pd.Timestamp("2024-01-01")
        end_date = pd.Timestamp("2024-06-01")
        
        splits = list(splitter.split(start_date, end_date))
        assert len(splits) == 1
        
        split = splits[0]
        assert len(split) == 3 # (train, test, oos)
        assert len(split[0]) == 2 # (train_start, train_end)
        assert len(split[1]) == 2 # (test_start, test_end)
        assert len(split[2]) == 2 # (oos_start, oos_end)

    def test_period_lengths(self):
        """Test that train, test, and oos periods have correct lengths."""
        train_day = 30
        test_day = 10
        oos_day = 5
        splitter = SlidingWindowSplitter(
            n_splits=1,
            train_day=train_day,
            test_day=test_day,
            oos_day=oos_day
        )
        
        start_date = pd.Timestamp("2024-01-01")
        end_date = pd.Timestamp("2024-03-01")
        
        splits = list(splitter.split(start_date, end_date))
        (train_start, train_end), (test_start, test_end), (oos_start, oos_end) = splits[0]
        
        assert (train_end - train_start).days == train_day
        assert (test_end - test_start).days == test_day
        assert (oos_end - oos_start).days == oos_day

    def test_purging_effect(self):
        """Test that purging_day creates a gap between train and test."""
        purging_day = 3
        splitter = SlidingWindowSplitter(
            n_splits=1,
            train_day=30,
            test_day=10,
            purging_day=purging_day
        )
        
        start_date = pd.Timestamp("2024-01-01")
        end_date = pd.Timestamp("2024-03-01")
        
        splits = list(splitter.split(start_date, end_date))
        (train_start, train_end), (test_start, test_end), _ = splits[0]
        
        # Implementation: train_end = test_start - timedelta(days=self.purging_day)
        assert (test_start - train_end).days == purging_day

    def test_no_splits_insufficient_data(self):
        """Test that no splits are returned if there's not enough data."""
        splitter = SlidingWindowSplitter(
            n_splits=1,
            train_day=100,
            test_day=10
        )
        
        # Only 50 days of data
        start_date = pd.Timestamp("2024-01-01")
        end_date = pd.Timestamp("2024-02-20")
        
        splits = list(splitter.split(start_date, end_date))
        assert len(splits) == 0

    def test_backward_calculation_logic(self):
        """
        Verify that splits are calculated backwards from end_date correctly.
        """
        end_date = pd.Timestamp("2024-06-01")
        splitter = SlidingWindowSplitter(
            n_splits=1,
            train_day=30,
            test_day=10,
            oos_day=5,
            purging_day=2
        )
        
        splits = list(splitter.split(pd.Timestamp("2024-01-01"), end_date))
        (train_start, train_end), (test_start, test_end), (oos_start, oos_end) = splits[0]
        
        assert oos_end == end_date
        assert oos_start == end_date - timedelta(days=5)
        assert test_end == oos_start
        assert test_start == test_end - timedelta(days=10)
        assert train_end == test_start - timedelta(days=2)
        assert train_start == train_end - timedelta(days=30)

    def test_embargo_effect(self):
        """
        Test that embargo_day affects the distance between splits.
        step_day = test_day + oos_day + embargo_day + purging_day
        """
        splitter = SlidingWindowSplitter(
            n_splits=2,
            train_day=30,
            test_day=10,
            oos_day=0,
            embargo_day=5,
            purging_day=0
        )
        
        start_date = pd.Timestamp("2024-01-01")
        end_date = pd.Timestamp("2024-06-01")
        
        splits = list(splitter.split(start_date, end_date))
        assert len(splits) == 2
        
        s1 = splits[1] # Later split
        s0 = splits[0] # Earlier split
        
        # step_day = 10 + 0 + 5 + 0 = 15
        assert (s1[2][1] - s0[2][1]).days == 15
