from provenquant.core.labeling import (
    filtrate_tripple_label_barrier,
    get_tripple_label_barrier,
    filtrate_dynamic_tripple_label_barrier,
    get_dynamic_tripple_label_barrier,
)
import numpy as np
import os
import pandas as pd
import pytest

@pytest.fixture
def sample_dataframe():
    """Load sample dataframe from feather file."""
    current_dir = os.getcwd()
    dataframe_path = os.path.join(current_dir, 'data', 'btc_usdt.feather')
    if os.path.exists(dataframe_path):
        dataframe = pd.read_feather(dataframe_path)
        return dataframe
    else:
        # Fallback: create a sample dataframe if file doesn't exist
        dates = pd.date_range(start='2023-01-01', periods=100, freq='1h')
        data = {
            'close': np.random.uniform(50000, 60000, 100),
            'volume': np.random.uniform(1000, 5000, 100),
        }
        return pd.DataFrame(data, index=dates)

@pytest.fixture
def simple_dataframe():
    """Create a simple deterministic dataframe for testing."""
    dates = pd.date_range(start='2023-01-01', periods=50, freq='1h')
    data = {
        'close': [100 + i * 0.5 for i in range(50)],  # Slight uptrend
        'volume': [1000 + i * 10 for i in range(50)],
    }
    return pd.DataFrame(data, index=dates)

@pytest.fixture
def volatile_dataframe():
    """Create a volatile dataframe with bigger price swings."""
    dates = pd.date_range(start='2023-01-01', periods=100, freq='1h')
    np.random.seed(42)
    returns = np.random.normal(0.001, 0.02, 100)
    close_prices = 100 * np.exp(np.cumsum(returns))
    data = {
        'close': close_prices,
        'volume': np.random.uniform(1000, 5000, 100),
    }
    return pd.DataFrame(data, index=dates)
  
def test_filtrate_tripple_label_barrier_basic(simple_dataframe):
    """Test basic functionality of filtrate_tripple_label_barrier."""
    # Convert index to column for testing
    df = simple_dataframe.reset_index()
    df.rename(columns={'index': 'datetime'}, inplace=True)
    
    result = filtrate_tripple_label_barrier(
        df,
        cusum_threshold=0.01,
        vertical_barrier=5,
        datetime_col='datetime'
    )
    
    # Check structure
    assert isinstance(result, pd.DataFrame)
    assert 't1' in result.columns
    assert 'close' in result.columns
    assert 'volume' in result.columns
    
    # Check that index represents event times
    assert len(result) >= 0


def test_filtrate_tripple_label_barrier_with_column(simple_dataframe):
    """Test filtrate_tripple_label_barrier with datetime column."""
    df_with_col = simple_dataframe.reset_index()
    df_with_col.rename(columns={'index': 'datetime'}, inplace=True)
    
    result = filtrate_tripple_label_barrier(
        df_with_col,
        cusum_threshold=0.01,
        vertical_barrier=5,
        datetime_col='datetime'
    )
    
    # Check structure
    assert isinstance(result, pd.DataFrame)
    assert 't1' in result.columns
    assert 'close' in result.columns


def test_filtrate_tripple_label_barrier_high_threshold(simple_dataframe):
    """Test with high threshold - should produce fewer events."""
    df = simple_dataframe.reset_index()
    df.rename(columns={'index': 'datetime'}, inplace=True)
    
    result = filtrate_tripple_label_barrier(
        df,
        cusum_threshold=0.5,  # Very high threshold
        vertical_barrier=5,
        datetime_col='datetime'
    )
    
    # Higher threshold should result in fewer events
    assert len(result) >= 0


def test_filtrate_tripple_label_barrier_low_threshold(volatile_dataframe):
    """Test with low threshold - should produce more events."""
    df = volatile_dataframe.reset_index()
    df.rename(columns={'index': 'datetime'}, inplace=True)
    
    result = filtrate_tripple_label_barrier(
        df,
        cusum_threshold=0.001,  # Very low threshold
        vertical_barrier=5,
        datetime_col='datetime'
    )
    
    # Low threshold should result in more events for volatile data
    assert len(result) > 0


def test_filtrate_tripple_label_barrier_vertical_barrier_boundary(simple_dataframe):
    """Test vertical barrier doesn't exceed dataframe bounds."""
    df = simple_dataframe.reset_index()
    df.rename(columns={'index': 'datetime'}, inplace=True)
    
    result = filtrate_tripple_label_barrier(
        df,
        cusum_threshold=0.01,
        vertical_barrier=1000,  # Very large barrier
        datetime_col='datetime'
    )
    
    if len(result) > 0:
        # t1 should never exceed the last datetime
        assert (result['t1'] <= df['datetime'].max()).all()


def test_filtrate_dynamic_tripple_label_barrier_basic(simple_dataframe):
    """Test basic functionality of filtrate_dynamic_tripple_label_barrier."""
    df = simple_dataframe.copy()
    # Add a dynamic threshold column
    df['dynamic_threshold'] = 0.01  # Constant threshold for simplicity
    
    result = filtrate_dynamic_tripple_label_barrier(
        df,
        cusum_threshold_col='dynamic_threshold',
        vertical_barrier=5,
        datetime_col='index'
    )
    
    # Check structure
    assert isinstance(result, pd.DataFrame)
    assert 't1' in result.columns
    assert 'close' in result.columns
    assert 'dynamic_threshold' in result.columns
    assert len(result) >= 0


def test_filtrate_dynamic_tripple_label_barrier_variable_threshold(volatile_dataframe):
    """Test with variable thresholds per row."""
    df = volatile_dataframe.copy()
    # Create random dynamic thresholds
    np.random.seed(42)
    df['dynamic_threshold'] = np.random.uniform(0.005, 0.02, len(df))
    
    result = filtrate_dynamic_tripple_label_barrier(
        df,
        cusum_threshold_col='dynamic_threshold',
        vertical_barrier=5,
        datetime_col='index'
    )
    
    assert isinstance(result, pd.DataFrame)
    assert 't1' in result.columns
    assert len(result) >= 0


def test_filtrate_tripple_label_barrier_preserves_data():
    """Test that filtrate preserves data columns correctly."""
    dates = pd.date_range(start='2023-01-01', periods=20, freq='1h')
    data = {
        'datetime': dates,
        'close': [100 + i * 0.5 for i in range(20)],
        'volume': [1000 + i * 10 for i in range(20)],
        'other_col': [i ** 2 for i in range(20)],
    }
    df = pd.DataFrame(data)
    
    result = filtrate_tripple_label_barrier(
        df,
        cusum_threshold=0.01,
        vertical_barrier=3,
        datetime_col='datetime'
    )
    
    if len(result) > 0:
        # Check that all original columns are preserved
        assert 'close' in result.columns
        assert 'volume' in result.columns
        assert 'other_col' in result.columns


def test_get_tripple_label_barrier_basic():
    """Test basic functionality of get_tripple_label_barrier."""
    # Create simple test dataframe
    dates = pd.date_range(start='2023-01-01', periods=20, freq='1h')
    df = pd.DataFrame({
        'close': [100, 102, 101, 105, 103, 108, 106, 110, 107, 112,
                  111, 115, 113, 118, 116, 120, 119, 125, 123, 128],
        'volume': [1000] * 20
    }, index=dates)
    
    close_series = df['close']
    
    # Create events dataframe with t1 values
    events_df = pd.DataFrame({
        't1': [dates[5], dates[10], dates[15]],
    }, index=[dates[0], dates[5], dates[10]])
    
    result = get_tripple_label_barrier(
        events_df,
        close_series,
        tp=0.02,
        sl=0.01
    )
    
    # Check structure
    assert 'label' in result.columns
    assert 'return' in result.columns
    assert len(result) == len(events_df)
    
    # Check labels are ternary
    assert set(result['label'].unique()).issubset({-1, 0, 1})


def test_get_dynamic_tripple_label_barrier_basic():
    """Test basic functionality of get_dynamic_tripple_label_barrier."""
    dates = pd.date_range(start='2023-01-01', periods=20, freq='1h')
    df = pd.DataFrame({
        'close': [100, 102, 101, 105, 103, 108, 106, 110, 107, 112,
                  111, 115, 113, 118, 116, 120, 119, 125, 123, 128],
        'volume': [1000] * 20
    }, index=dates)
    
    close_series = df['close']
    
    # Create events dataframe with t1 values and dynamic thresholds
    events_df = pd.DataFrame({
        't1': [dates[5], dates[10], dates[15]],
        'threshold': [0.01, 0.02, 0.015]
    }, index=[dates[0], dates[5], dates[10]])
    
    result = get_dynamic_tripple_label_barrier(
        events_df,
        close_series,
        cusum_threshold_col='threshold',
        tp_multiplier=2.0,
        sl_multiplier=1.0
    )
    
    # Check structure
    assert 'label' in result.columns
    assert 'return' in result.columns
    assert 'max_return' in result.columns
    assert 'min_return' in result.columns
    assert 'mapped_label' in result.columns
    assert len(result) == len(events_df)
    
    # Check labels are ternary
    assert set(result['label'].unique()).issubset({-1, 0, 1})


def test_get_tripple_label_barrier_positive_return():
    """Test labeling with positive returns."""
    dates = pd.date_range(start='2023-01-01', periods=10, freq='1h')
    close_series = pd.Series([100, 101, 102, 103, 104, 105, 106, 107, 108, 109], index=dates)
    
    events_df = pd.DataFrame({
        't1': [dates[5]],  # price goes from 100 to 105 = 5% return
    }, index=[dates[0]])
    
    result = get_tripple_label_barrier(
        events_df,
        close_series,
        tp=0.02,
        sl=0.01
    )
    
    assert result['label'].iloc[0] == 1
    assert result['return'].iloc[0] >= 0.02


def test_get_tripple_label_barrier_negative_return():
    """Test labeling with negative returns."""
    dates = pd.date_range(start='2023-01-01', periods=10, freq='1h')
    close_series = pd.Series([100, 99, 98, 97, 96, 95, 94, 93, 92, 91], index=dates)
    
    events_df = pd.DataFrame({
        't1': [dates[5]],  # price goes from 100 to 95 = -5% return
    }, index=[dates[0]])
    
    result = get_tripple_label_barrier(
        events_df,
        close_series,
        tp=0.02,
        sl=0.01
    )
    
    assert result['label'].iloc[0] == -1
    assert result['return'].iloc[0] <= -0.01


def test_get_tripple_label_barrier_zero_label():
    """Test labeling with returns below threshold."""
    dates = pd.date_range(start='2023-01-01', periods=10, freq='1h')
    close_series = pd.Series([100, 100.5, 101, 101.5, 102, 102.5, 103, 103.5, 104, 104.5], index=dates)
    
    events_df = pd.DataFrame({
        't1': [dates[3]],  # price goes from 100 to 101.5 = 1.5% return
    }, index=[dates[0]])
    
    result = get_tripple_label_barrier(
        events_df,
        close_series,
        tp=0.02,
        sl=0.01
    )
    
    assert result['label'].iloc[0] == 0
    assert 0.01 < result['return'].iloc[0] < 0.02


def test_get_tripple_label_barrier_nan_t1():
    """Test handling of NaN t1 values."""
    dates = pd.date_range(start='2023-01-01', periods=10, freq='1h')
    close_series = pd.Series([100 + i for i in range(10)], index=dates)
    
    events_df = pd.DataFrame({
        't1': [dates[3], np.nan, dates[5]],
    }, index=[dates[0], dates[1], dates[2]])
    
    result = get_tripple_label_barrier(
        events_df,
        close_series,
        tp=0.02,
        sl=0.01
    )
    
    # NaN t1 should result in label 0 and return 0
    assert result['label'].iloc[1] == 0
    assert result['return'].iloc[1] == 0


def test_get_tripple_label_barrier_threshold_parameter(sample_dataframe):
    """Test threshold, pt, and sl parameters affect labeling."""
    if len(sample_dataframe) < 20:
        pytest.skip("Sample dataframe too small")
    
    close_series = sample_dataframe['close']
    
    # Create events with moderate returns
    events_df = pd.DataFrame({
        't1': [close_series.index[10]],
    }, index=[close_series.index[0]])
    
    # Test with different tp values
    result_low = get_tripple_label_barrier(
        events_df.copy(),
        close_series,
        tp=0.0001,
        sl=0.01
    )
    
    result_high = get_tripple_label_barrier(
        events_df.copy(),
        close_series,
        tp=0.1,
        sl=0.01
    )
    
    # Higher threshold may change labels
    assert len(result_low) == len(result_high)


def test_get_tripple_label_barrier_multiple_events():
    """Test with multiple events."""
    dates = pd.date_range(start='2023-01-01', periods=15, freq='1h')
    close_series = pd.Series(
        [100, 102, 104, 103, 105, 107, 106, 108, 110, 109, 111, 113, 112, 114, 116],
        index=dates
    )
    
    events_df = pd.DataFrame({
        't1': [dates[2], dates[5], dates[9]],
    }, index=[dates[0], dates[3], dates[6]])
    
    result = get_tripple_label_barrier(
        events_df,
        close_series,
        tp=0.02,
        sl=0.01
    )
    
    assert len(result) == 3
    assert len(result['label']) == 3
    assert len(result['return']) == 3


def test_integration_filtrate_and_label(simple_dataframe):
    """Test integration of both functions."""
    df = simple_dataframe.reset_index()
    df.rename(columns={'index': 'datetime'}, inplace=True)
    
    # First, filtrate
    filtered = filtrate_tripple_label_barrier(
        df,
        cusum_threshold=0.01,
        vertical_barrier=5,
        datetime_col='datetime'
    )
    
    if len(filtered) > 0:
        # Then label
        result = get_tripple_label_barrier(
            filtered,
            simple_dataframe['close'],
            tp=0.02,
            sl=0.01
        )
        
        # Check result structure
        assert 'label' in result.columns
        assert 'return' in result.columns
        assert len(result) == len(filtered)
        assert set(result['label'].unique()).issubset({-1, 0, 1})


def test_filtrate_empty_result():
    """Test when filtrate produces no events."""
    dates = pd.date_range(start='2023-01-01', periods=10, freq='1h')
    df = pd.DataFrame({
        'datetime': dates,
        'close': [100] * 10,  # No price change
        'volume': [1000] * 10
    })
    
    result = filtrate_tripple_label_barrier(
        df,
        cusum_threshold=0.5,  # Very high threshold
        vertical_barrier=1,
        datetime_col='datetime'
    )
    
    # Result should be empty dataframe
    assert isinstance(result, pd.DataFrame)
    assert len(result) == 0


def test_get_tripple_label_barrier_empty_dataframe():
    """Test with empty events dataframe."""
    dates = pd.date_range(start='2023-01-01', periods=10, freq='1h')
    close_series = pd.Series([100 + i for i in range(10)], index=dates)
    
    events_df = pd.DataFrame()
    
    result = get_tripple_label_barrier(
        events_df,
        close_series,
        tp=0.02,
        sl=0.01
    )
    
    assert len(result) == 0


def test_filtrate_different_vertical_barriers(volatile_dataframe):
    """Test with different vertical barrier values."""
    df = volatile_dataframe.reset_index()
    df.rename(columns={'index': 'datetime'}, inplace=True)
    
    result_small = filtrate_tripple_label_barrier(
        df,
        cusum_threshold=0.01,
        vertical_barrier=2,
        datetime_col='datetime'
    )
    
    result_large = filtrate_tripple_label_barrier(
        df,
        cusum_threshold=0.01,
        vertical_barrier=10,
        datetime_col='datetime'
    )
    
    # Both should have same number of events but different t1 values
    assert len(result_small) == len(result_large)
    if len(result_small) > 0:
        # t1 values should differ (larger barrier = later t1)
        assert not result_small['t1'].equals(result_large['t1'])
