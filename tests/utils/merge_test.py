import pytest
import numpy as np
import pandas as pd

from provenquant.utils.merge import (
    match_merge_dataframe,
    match_merge_series,
    larger_timeframe_merge_to_smaller_timeframe_dataframe,
    smaller_timeframe_merge_sum_to_larger_timeframe_dataframe,
)
def test_match_merge_dataframe_basic():
    """Test basic functionality of match_merge_dataframe."""
    from_df = pd.DataFrame({
        'datetime': pd.date_range('2024-01-01', periods=3),
        'value1': [10, 20, 30],
        'value2': [100, 200, 300]
    })
    to_df = pd.DataFrame({
        'datetime': pd.date_range('2024-01-01', periods=3),
        'existing': ['a', 'b', 'c']
    })
    
    result = match_merge_dataframe(from_df, to_df, ['value1', 'value2'], 'datetime', 'datetime')
    
    assert 'value1' in result.columns
    assert 'value2' in result.columns
    assert list(result['value1']) == [10, 20, 30]
    assert list(result['value2']) == [100, 200, 300]


def test_match_merge_dataframe_with_missing_dates():
    """Test match_merge_dataframe when some dates don't match."""
    from_df = pd.DataFrame({
        'datetime': pd.to_datetime(['2024-01-01', '2024-01-03']),
        'value': [10, 30]
    })
    to_df = pd.DataFrame({
        'datetime': pd.to_datetime(['2024-01-01', '2024-01-02', '2024-01-03']),
        'existing': ['a', 'b', 'c']
    })
    
    result = match_merge_dataframe(from_df, to_df, ['value'], 'datetime', 'datetime')
    
    assert result['value'].iloc[0] == 10
    assert np.isnan(result['value'].iloc[1])
    assert result['value'].iloc[2] == 30


def test_match_merge_series_with_time_column():
    """Test match_merge_series with datetime column."""
    from_series = pd.Series([10, 20, 30], index=pd.date_range('2024-01-01', periods=3))
    to_df = pd.DataFrame({
        'datetime': pd.date_range('2024-01-01', periods=3),
        'existing': ['a', 'b', 'c']
    })
    
    result = match_merge_series(from_series, to_df, 'datetime', 'new_col')
    
    assert 'new_col' in result.columns
    assert list(result['new_col']) == [10, 20, 30]


def test_match_merge_series_with_index():
    """Test match_merge_series using index."""
    from_series = pd.Series([10, 20, 30], index=pd.date_range('2024-01-01', periods=3))
    to_df = pd.DataFrame({
        'existing': ['a', 'b', 'c']
    }, index=pd.date_range('2024-01-01', periods=3))
    
    result = match_merge_series(from_series, to_df, 'index', 'new_col')
    
    assert 'new_col' in result.columns
    assert list(result['new_col']) == [10, 20, 30]


def test_larger_timeframe_merge_basic():
    """Test basic functionality of larger_timeframe_merge_to_smaller_timeframe_dataframe."""
    large_tf_df = pd.DataFrame({
        'datetime': pd.to_datetime(['2024-01-01 00:00', '2024-01-01 01:00', '2024-01-01 02:00']),
        'large_value': [100, 200, 300]
    })
    small_tf_df = pd.DataFrame({
        'datetime': pd.to_datetime(
            ['2024-01-01 00:00', '2024-01-01 00:30', '2024-01-01 01:00', '2024-01-01 01:30', '2024-01-01 02:00']
        ),
        'small_value': [1, 2, 3, 4, 5]
    })
    
    result = larger_timeframe_merge_to_smaller_timeframe_dataframe(
        large_tf_df, small_tf_df, 'datetime', 'datetime', ['large_value']
    )
    
    # NaN compares false with ==, so check with isnan for missing values
    assert np.isnan(result['large_value'].iloc[0])
    assert np.isnan(result['large_value'].iloc[1])
    assert result['large_value'].iloc[2] == 100.0
    assert result['large_value'].iloc[3] == 100.0
    assert result['large_value'].iloc[4] == 200.0


def test_larger_timeframe_merge_index_to_index_named():
    """Test larger_timeframe_merge_to_smaller_timeframe_dataframe with named index to named index merging."""
    large_tf_df = pd.DataFrame({
        'large_value': [100, 200, 300]
    }, index=pd.date_range('2024-01-01 00:00', periods=3, freq='1h'))
    large_tf_df.index.name = 'time'
    
    small_tf_df = pd.DataFrame({
        'small_value': [1, 2, 3, 4, 5]
    }, index=pd.date_range('2024-01-01 00:00', periods=5, freq='30min'))
    small_tf_df.index.name = 'time'
    
    result = larger_timeframe_merge_to_smaller_timeframe_dataframe(
        large_tf_df, small_tf_df, 'index', 'index', ['large_value']
    )
    
    assert 'large_value' in result.columns
    assert result.index.name == 'time'
    assert np.isnan(result['large_value'].iloc[0])
    assert result['large_value'].iloc[2] == 100.0


def test_larger_timeframe_merge_index_to_index_unnamed():
    """Test larger_timeframe_merge_to_smaller_timeframe_dataframe with unnamed index to unnamed index merging."""
    large_tf_df = pd.DataFrame({
        'large_value': [100, 200, 300]
    }, index=pd.date_range('2024-01-01 00:00', periods=3, freq='1h'))
    large_tf_df.index.name = None
    
    small_tf_df = pd.DataFrame({
        'small_value': [1, 2, 3, 4, 5]
    }, index=pd.date_range('2024-01-01 00:00', periods=5, freq='30min'))
    small_tf_df.index.name = None
    
    # This should reproduce the KeyError: Requested level (_small_dt) does not match index name (None)
    result = larger_timeframe_merge_to_smaller_timeframe_dataframe(
        large_tf_df, small_tf_df, 'index', 'index', ['large_value']
    )
    
    assert 'large_value' in result.columns
    assert np.isnan(result['large_value'].iloc[0])
    assert np.isnan(result['large_value'].iloc[1])
    assert result['large_value'].iloc[2] == 100.0
    assert result['large_value'].iloc[3] == 100.0
    assert result['large_value'].iloc[4] == 200.0


def test_match_merge_dataframe_empty_from_df():
    """Test match_merge_dataframe with empty from_df."""
    from_df = pd.DataFrame({'datetime': [], 'value': []})
    to_df = pd.DataFrame({
        'datetime': pd.date_range('2024-01-01', periods=3),
        'existing': ['a', 'b', 'c']
    })
    
    result = match_merge_dataframe(from_df, to_df, ['value'], 'datetime', 'datetime')
    
    assert 'value' in result.columns
    assert all(np.isnan(result['value']))

def test_smaller_timeframe_merge_sum_to_larger_timeframe_basic():
    """Test basic functionality of smaller_timeframe_merge_sum_to_larger_timeframe_dataframe."""
    large_tf_df = pd.DataFrame({
        'datetime': pd.to_datetime(['2024-01-01 00:00', '2024-01-01 01:00', '2024-01-01 02:00']),
        'large_id': [1, 2, 3]
    })
    small_tf_df = pd.DataFrame({
        'datetime': pd.to_datetime(
            ['2024-01-01 00:00', '2024-01-01 00:15', '2024-01-01 00:30', '2024-01-01 00:45',
             '2024-01-01 01:00', '2024-01-01 01:15', '2024-01-01 01:30', '2024-01-01 01:45']
        ),
        'small_value': [10, 20, 30, 40, 50, 60, 70, 80]
    })
    
    result = smaller_timeframe_merge_sum_to_larger_timeframe_dataframe(
        small_tf_df, large_tf_df, 'datetime', 'datetime', ['small_value']
    )
    
    # Only closed intervals are aggregated (drops last unclosed bar)
    # First interval: [00:00, 01:00) should sum values from 00:00, 00:15, 00:30, 00:45 = 10+20+30+40 = 100
    assert result['small_value'].iloc[0] == 100.0
    # Second interval: [01:00, 02:00) should sum values from 01:00, 01:15, 01:30, 01:45 = 50+60+70+80 = 260
    assert result['small_value'].iloc[1] == 260.0


def test_smaller_timeframe_merge_sum_to_larger_timeframe_multiple_columns():
    """Test aggregation with multiple columns."""
    large_tf_df = pd.DataFrame({
        'datetime': pd.to_datetime(['2024-01-01 00:00', '2024-01-01 01:00']),
        'large_id': [1, 2]
    })
    small_tf_df = pd.DataFrame({
        'datetime': pd.to_datetime(['2024-01-01 00:00', '2024-01-01 00:30', '2024-01-01 01:00', '2024-01-01 01:30']),
        'value1': [10, 20, 30, 40],
        'value2': [100, 200, 300, 400]
    })
    
    result = smaller_timeframe_merge_sum_to_larger_timeframe_dataframe(
        small_tf_df, large_tf_df, 'datetime', 'datetime', ['value1', 'value2']
    )
    
    assert result['value1'].iloc[0] == 30.0  # 10 + 20
    assert result['value1'].iloc[1] == 70.0  # 30 + 40
    assert result['value2'].iloc[0] == 300.0  # 100 + 200
    assert result['value2'].iloc[1] == 700.0  # 300 + 400


def test_smaller_timeframe_merge_sum_no_matching_data():
    """Test when no small timeframe data falls within large timeframe intervals."""
    large_tf_df = pd.DataFrame({
        'datetime': pd.to_datetime(['2024-01-01 00:00', '2024-01-01 01:00']),
        'large_id': [1, 2]
    })
    small_tf_df = pd.DataFrame({
        'datetime': pd.to_datetime(['2024-01-02 00:00', '2024-01-02 01:00']),
        'small_value': [10, 20]
    })
    
    result = smaller_timeframe_merge_sum_to_larger_timeframe_dataframe(
        small_tf_df, large_tf_df, 'datetime', 'datetime', ['small_value']
    )
    
    # No small data falls in intervals, all should be 0
    assert result['small_value'].iloc[0] == 0.0
    assert result['small_value'].iloc[1] == 0.0


def test_match_merge_dataframe_with_index():
    """Test match_merge_dataframe using index."""
    from_df = pd.DataFrame({
        'value1': [10, 20, 30],
        'value2': [100, 200, 300]
    }, index=pd.date_range('2024-01-01', periods=3))
    
    to_df = pd.DataFrame({
        'existing': ['a', 'b', 'c']
    }, index=pd.date_range('2024-01-01', periods=3))
    
    result = match_merge_dataframe(from_df, to_df, ['value1', 'value2'], 'index', 'index')
    
    assert 'value1' in result.columns
    assert 'value2' in result.columns
    assert list(result['value1']) == [10, 20, 30]
    assert list(result['value2']) == [100, 200, 300]


def test_match_merge_dataframe_mixed_index_and_column():
    """Test match_merge_dataframe with index on from_df and column on to_df."""
    from_df = pd.DataFrame({
        'value': [10, 20, 30]
    }, index=pd.date_range('2024-01-01', periods=3))
    
    to_df = pd.DataFrame({
        'datetime': pd.date_range('2024-01-01', periods=3),
        'existing': ['a', 'b', 'c']
    })
    
    result = match_merge_dataframe(from_df, to_df, ['value'], 'index', 'datetime')
    
    assert 'value' in result.columns
    assert list(result['value']) == [10, 20, 30]


def test_larger_timeframe_merge_with_index():
    """Test larger_timeframe_merge_to_smaller_timeframe_dataframe using index."""
    large_tf_df = pd.DataFrame({
        'large_value': [100, 200, 300]
    }, index=pd.to_datetime(['2024-01-01 00:00', '2024-01-01 01:00', '2024-01-01 02:00']))
    
    small_tf_df = pd.DataFrame({
        'small_value': [1, 2, 3, 4, 5]
    }, index=pd.to_datetime(['2024-01-01 00:00', '2024-01-01 00:30', '2024-01-01 01:00', '2024-01-01 01:30', '2024-01-01 02:00']))
    
    result = larger_timeframe_merge_to_smaller_timeframe_dataframe(
        large_tf_df, small_tf_df, 'index', 'index', ['large_value']
    )
    
    # Check that result maintains original index
    assert list(result.index) == list(small_tf_df.index)
    # Check values
    assert np.isnan(result['large_value'].iloc[0])
    assert np.isnan(result['large_value'].iloc[1])
    assert result['large_value'].iloc[2] == 100.0
    assert result['large_value'].iloc[3] == 100.0
    assert result['large_value'].iloc[4] == 200.0


def test_larger_timeframe_merge_preserves_index():
    """Test that larger_timeframe_merge_to_smaller_timeframe_dataframe preserves original index."""
    custom_index = ['row_a', 'row_b', 'row_c', 'row_d', 'row_e']
    
    large_tf_df = pd.DataFrame({
        'datetime': pd.to_datetime(['2024-01-01 00:00', '2024-01-01 01:00', '2024-01-01 02:00']),
        'large_value': [100, 200, 300]
    })
    small_tf_df = pd.DataFrame({
        'datetime': pd.to_datetime(['2024-01-01 00:00', '2024-01-01 00:30', '2024-01-01 01:00', '2024-01-01 01:30', '2024-01-01 02:00']),
        'small_value': [1, 2, 3, 4, 5]
    }, index=custom_index)
    
    result = larger_timeframe_merge_to_smaller_timeframe_dataframe(
        large_tf_df, small_tf_df, 'datetime', 'datetime', ['large_value']
    )
    
    # Check that result maintains original index
    assert list(result.index) == custom_index


def test_smaller_timeframe_merge_sum_with_index():
    """Test smaller_timeframe_merge_sum_to_larger_timeframe_dataframe using index."""
    large_tf_df = pd.DataFrame({
        'large_id': [1, 2]
    }, index=pd.to_datetime(['2024-01-01 00:00', '2024-01-01 01:00']))
    
    small_tf_df = pd.DataFrame({
        'small_value': [10, 20, 30, 40, 50, 60, 70, 80]
    }, index=pd.to_datetime(['2024-01-01 00:00', '2024-01-01 00:15', '2024-01-01 00:30', '2024-01-01 00:45',
                              '2024-01-01 01:00', '2024-01-01 01:15', '2024-01-01 01:30', '2024-01-01 01:45']))
    
    result = smaller_timeframe_merge_sum_to_larger_timeframe_dataframe(
        small_tf_df, large_tf_df, 'index', 'index', ['small_value']
    )
    
    # First interval: [00:00, 01:00) should sum 10+20+30+40 = 100
    assert result['small_value'].iloc[0] == 100.0
    # Second interval: [01:00, 02:00) should sum 50+60+70+80 = 260
    assert result['small_value'].iloc[1] == 260.0


def test_smaller_timeframe_merge_sum_with_timezone_aware_index():
    """Test smaller_timeframe_merge_sum_to_larger_timeframe_dataframe with timezone-aware index."""
    large_tf_df = pd.DataFrame({
        'large_id': [1, 2]
    }, index=pd.to_datetime(['2024-01-01 00:00', '2024-01-01 01:00'], utc=True))
    
    small_tf_df = pd.DataFrame({
        'small_value': [10, 20, 30, 40]
    }, index=pd.to_datetime(['2024-01-01 00:00', '2024-01-01 00:30', '2024-01-01 01:00', '2024-01-01 01:30'], utc=True))
    
    result = smaller_timeframe_merge_sum_to_larger_timeframe_dataframe(
        small_tf_df, large_tf_df, 'index', 'index', ['small_value']
    )
    
    assert result['small_value'].iloc[0] == 30.0  # 10 + 20
    assert result['small_value'].iloc[1] == 70.0  # 30 + 40


def test_larger_timeframe_merge_with_timezone_aware_mixed():
    """Test larger_timeframe_merge_to_smaller_timeframe_dataframe with timezone-aware index and column."""
    large_tf_df = pd.DataFrame({
        'large_value': [100, 200, 300]
    }, index=pd.to_datetime(['2024-01-01 00:00', '2024-01-01 01:00', '2024-01-01 02:00'], utc=True))
    
    small_tf_df = pd.DataFrame({
        'datetime': pd.to_datetime(['2024-01-01 00:00', '2024-01-01 00:30', '2024-01-01 01:00', '2024-01-01 01:30', '2024-01-01 02:00'], utc=True),
        'small_value': [1, 2, 3, 4, 5]
    })
    
    result = larger_timeframe_merge_to_smaller_timeframe_dataframe(
        large_tf_df, small_tf_df, 'index', 'datetime', ['large_value']
    )
    
    # Check that result can handle timezone-aware datetimes
    assert np.isnan(result['large_value'].iloc[0])
    assert np.isnan(result['large_value'].iloc[1])
    assert result['large_value'].iloc[2] == 100.0

