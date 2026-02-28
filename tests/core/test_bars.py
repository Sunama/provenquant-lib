import pandas as pd
import numpy as np
import pytest
from provenquant.core.bars import (
    convert_standard_bars_to_larger_timeframe,
    get_dollar_bars,
    get_dollar_imbalance_bars,
    get_tick_bars,
    get_volume_bars,
    get_volume_imbalance_bars
)

@pytest.fixture
def sample_tick_data():
    """Create a sample tick data for testing."""
    dates = pd.date_range(start='2023-01-01', periods=100, freq='s')
    data = {
        'open': np.linspace(100, 110, 100),
        'high': np.linspace(101, 111, 100),
        'low': np.linspace(99, 109, 100),
        'close': np.linspace(100, 110, 100),
        'volume': [10] * 100
    }
    df = pd.DataFrame(data, index=dates)
    return df

def test_convert_standard_bars_to_larger_timeframe(sample_tick_data):
    # Test converting 1s bars to 10s bars
    resampled = convert_standard_bars_to_larger_timeframe(sample_tick_data, '10s')
    
    assert len(resampled) == 10
    assert 'open' in resampled.columns
    assert 'high' in resampled.columns
    assert 'low' in resampled.columns
    assert 'close' in resampled.columns
    assert 'volume' in resampled.columns
    assert resampled.iloc[0]['volume'] == 100  # 10 * 10

def test_get_tick_bars(sample_tick_data):
    # Each bar should have 10 ticks
    tick_bars = get_tick_bars(sample_tick_data, tick_bar_size=10)
    
    assert len(tick_bars) == 10
    assert (tick_bars['cum_ticks'] == 10).all()
    assert tick_bars.iloc[0]['volume'] == 100

def test_get_volume_bars(sample_tick_data):
    # Each bar should have at least 50 volume
    volume_bars = get_volume_bars(sample_tick_data, volume_bar_size=50)
    
    assert len(volume_bars) == 20  # 100 ticks * 10 volume / 50 = 20 bars
    assert (volume_bars['volume'] >= 50).all()

def test_get_dollar_bars(sample_tick_data):
    # Dollar values: price * volume
    # Price is approx 100, volume is 10, so dollar value is approx 1000 per tick
    threshold = 5000 
    dollar_bars = get_dollar_bars(sample_tick_data, threshold=threshold)
    
    assert len(dollar_bars) > 0
    assert (dollar_bars['cum_dollar'] >= threshold).all()

def test_get_dollar_imbalance_bars(sample_tick_data):
    # With increasing price, imbalance should be positive
    imbalance_bars = get_dollar_imbalance_bars(
        sample_tick_data, 
        expected_imbalance_window=10, 
        exp_num_ticks_initial=1000
    )
    
    assert len(imbalance_bars) > 0
    assert 'buy_volume' in imbalance_bars.columns
    assert 'sell_volume' in imbalance_bars.columns

def test_get_volume_imbalance_bars(sample_tick_data):
    imbalance_bars = get_volume_imbalance_bars(
        sample_tick_data, 
        expected_imbalance_window=10, 
        exp_num_ticks_initial=10
    )
    
    assert len(imbalance_bars) > 0
    assert 'buy_volume' in imbalance_bars.columns
    assert 'sell_volume' in imbalance_bars.columns

def test_get_dollar_bars_multiprocessing(sample_tick_data):
    threshold = 5000
    dollar_bars_seq = get_dollar_bars(sample_tick_data, threshold=threshold, num_threads=1)
    dollar_bars_parallel = get_dollar_bars(sample_tick_data, threshold=threshold, num_threads=2)
    
    # Results should be identical or very similar (depending on how chunks are handled)
    # The current implementation handles carry_over_state, so they should match
    pd.testing.assert_frame_equal(dollar_bars_seq, dollar_bars_parallel)
