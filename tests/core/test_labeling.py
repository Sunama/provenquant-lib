from provenquant.core.labeling import (
    get_horizontal_barrier_events,
    add_vertical_barrier_to_horizontal_barrier_events,
    get_binary_labels,
    get_triple_barrier_labels,
)
import numpy as np
import os
import pandas as pd
import pytest

@pytest.fixture
def sample_dataframe():
    current_dir = os.getcwd()
    dataframe_path = os.path.join(current_dir, 'data', 'btc_usdt.feather')
    dataframe = pd.read_feather(dataframe_path)
    
    return dataframe

@pytest.fixture
def small_dataframe_with_date():
    dates = pd.date_range(start='2023-01-01', periods=6, freq='h')
    data = {
        'date': dates,
        'close': [100, 103, 104, 101, 99, 102],
        'volume': [200, 210, 220, 230, 240, 250],
    }
    return pd.DataFrame(data)
  
def test_get_horizontal_barrier_events(sample_dataframe):
    threshold = 3.0
    events = get_horizontal_barrier_events(sample_dataframe, threshold)
    
    # Check if events are a subset of the original dataframe
    assert all(event in sample_dataframe.index for event in events.index)
    
    # Check if the number of events is reasonable
    assert len(events) >= 0 and len(events) < len(sample_dataframe)

def test_add_vertical_barrier_to_horizontal_barrier_events(sample_dataframe):
    threshold = 0.005
    vertical_barrier_duration = pd.Timedelta(days=1)
    
    events = get_horizontal_barrier_events(sample_dataframe, threshold, datetime_col='date')
    if len(events) == 0:
        events = pd.DataFrame(index=[sample_dataframe['date'].iloc[0]])
    events_with_vertical = add_vertical_barrier_to_horizontal_barrier_events(
        sample_dataframe, events, vertical_barrier_duration, datetime_col='date'
    )
    
    # Check if vertical barriers are added
    assert 'vertical_barrier' in events_with_vertical.columns
    
    # Check if vertical barriers are within the dataframe date range
    assert events_with_vertical['vertical_barrier'].min() >= sample_dataframe['date'].min()
    assert events_with_vertical['vertical_barrier'].max() <= sample_dataframe['date'].max()
    # Check if ret column is added and contains values
    assert 'ret' in events_with_vertical.columns
    assert len(events_with_vertical['ret']) > 0

def test_get_binary_labels_long(sample_dataframe):
    """Test get_binary_labels with long side."""
    threshold = 0.005
    vertical_barrier_duration = pd.Timedelta(days=1)
    
    events = get_horizontal_barrier_events(sample_dataframe, threshold, datetime_col='date')
    if len(events) == 0:
        events = pd.DataFrame(index=[sample_dataframe['date'].iloc[0]])
    events_with_barriers = add_vertical_barrier_to_horizontal_barrier_events(
        sample_dataframe, events, vertical_barrier_duration, datetime_col='date'
    )
    
    labels = get_binary_labels(events_with_barriers, side='long')
    
    # Check if labels are binary (0 or 1)
    assert all(label in [0, 1] for label in labels)
    assert len(labels) > 0
    
    # Check if labels match the return values
    for i, ret in enumerate(events_with_barriers['ret']):
        if ret > 0:
            assert labels.iloc[i] == 1
        else:
            assert labels.iloc[i] == 0

def test_get_binary_labels_short(sample_dataframe):
    """Test get_binary_labels with short side."""
    threshold = 0.005
    vertical_barrier_duration = pd.Timedelta(days=1)
    
    events = get_horizontal_barrier_events(sample_dataframe, threshold, datetime_col='date')
    if len(events) == 0:
        events = pd.DataFrame(index=[sample_dataframe['date'].iloc[0]])
    events_with_barriers = add_vertical_barrier_to_horizontal_barrier_events(
        sample_dataframe, events, vertical_barrier_duration, datetime_col='date'
    )
    
    labels = get_binary_labels(events_with_barriers, side='short')
    
    # Check if labels are binary (0 or 1)
    assert all(label in [0, 1] for label in labels)
    assert len(labels) > 0
    
    # Check if labels match the return values for short side
    for i, ret in enumerate(events_with_barriers['ret']):
        if ret < 0:
            assert labels.iloc[i] == 1
        else:
            assert labels.iloc[i] == 0

def test_get_binary_labels_invalid_side(sample_dataframe):
    """Test get_binary_labels with invalid side parameter."""
    threshold = 3.0
    vertical_barrier_duration = pd.Timedelta(days=3)
    
    events = get_horizontal_barrier_events(sample_dataframe, threshold)
    events_with_barriers = add_vertical_barrier_to_horizontal_barrier_events(
        sample_dataframe, events, vertical_barrier_duration
    )
    
    # Should raise ValueError for invalid side
    with pytest.raises(ValueError, match="side must be either 'long' or 'short'"):
        get_binary_labels(events_with_barriers, side='invalid')

def test_get_triple_barrier_labels_long(sample_dataframe):
    """Test get_triple_barrier_labels with long side."""
    threshold = 0.005
    vertical_barrier_duration = pd.Timedelta(days=1)
    pt = 2.0
    sl = 1.0
    
    events = get_horizontal_barrier_events(sample_dataframe, threshold, datetime_col='date')
    events_with_barriers = add_vertical_barrier_to_horizontal_barrier_events(
        sample_dataframe, events, vertical_barrier_duration, datetime_col='date'
    )
    
    labels, rets = get_triple_barrier_labels(
        sample_dataframe,
        events_with_barriers,
        threshold=threshold,
        pt=pt,
        sl=sl,
        side='long',
        datetime_col='date'
    )
    
    # Check if labels are ternary (-1, 0, or 1)
    assert all(label in [-1, 0, 1] for label in labels)
    assert len(rets) == len(labels)
    
    # Labels consistent with computed returns
    upper = threshold * pt
    lower = -threshold * sl
    for i, ret in enumerate(rets):
        if ret >= upper:
            assert labels.iloc[i] == 1
        elif ret <= lower:
            assert labels.iloc[i] == -1
        else:
            assert labels.iloc[i] == 0

def test_get_triple_barrier_labels_short(sample_dataframe):
    """Test get_triple_barrier_labels with short side."""
    threshold = 0.005
    vertical_barrier_duration = pd.Timedelta(days=1)
    pt = 2.0
    sl = 1.0
    
    events = get_horizontal_barrier_events(sample_dataframe, threshold, datetime_col='date')
    events_with_barriers = add_vertical_barrier_to_horizontal_barrier_events(
        sample_dataframe, events, vertical_barrier_duration, datetime_col='date'
    )
    
    labels, rets = get_triple_barrier_labels(
        sample_dataframe,
        events_with_barriers,
        threshold=threshold,
        pt=pt,
        sl=sl,
        side='short',
        datetime_col='date'
    )
    
    # Check if labels are ternary (-1, 0, or 1)
    assert all(label in [-1, 0, 1] for label in labels)
    assert len(rets) == len(labels)
    
    upper = threshold * pt
    lower = -threshold * sl
    for i, ret in enumerate(rets):
        if ret <= -upper:
            assert labels.iloc[i] == 1
        elif ret >= -lower:
            assert labels.iloc[i] == -1
        else:
            assert labels.iloc[i] == 0

def test_get_triple_barrier_labels_invalid_side(sample_dataframe):
    """Test get_triple_barrier_labels with invalid side parameter."""
    threshold = 0.005
    vertical_barrier_duration = pd.Timedelta(days=1)
    
    events = get_horizontal_barrier_events(sample_dataframe, threshold, datetime_col='date')
    events_with_barriers = add_vertical_barrier_to_horizontal_barrier_events(
        sample_dataframe, events, vertical_barrier_duration, datetime_col='date'
    )
    
    # Ensure there is at least one event; otherwise force one manually
    if len(events_with_barriers) == 0:
        forced_event_time = sample_dataframe['date'].iloc[0]
        events_with_barriers = pd.DataFrame(index=[forced_event_time])
        events_with_barriers = add_vertical_barrier_to_horizontal_barrier_events(
            sample_dataframe, events_with_barriers, vertical_barrier_duration, datetime_col='date'
        )
    
    # Should raise ValueError for invalid side
    with pytest.raises(ValueError, match="side must be either 'long' or 'short'"):
        get_triple_barrier_labels(
            sample_dataframe,
            events_with_barriers,
            threshold=0.005,
            pt=2,
            sl=1,
            side='invalid',
            datetime_col='date'
        )


def test_get_triple_barrier_labels_notebook_like(small_dataframe_with_date):
    """Notebook-style pipeline with datetime_col and deterministic outcomes."""
    event_times = [small_dataframe_with_date['date'].iloc[0], small_dataframe_with_date['date'].iloc[3]]
    events = pd.DataFrame(index=event_times)

    events = add_vertical_barrier_to_horizontal_barrier_events(
        small_dataframe_with_date,
        events,
        vertical_barrier_duration=pd.Timedelta('2h'),
        datetime_col='date',
    )

    labels, rets = get_triple_barrier_labels(
        small_dataframe_with_date,
        events,
        threshold=0.01,
        pt=1.0,
        sl=1.0,
        side='long',
        datetime_col='date',
    )

    # Alignment and validity
    assert len(labels) == len(events) == len(rets)
    assert all(labels.index == events.index)
    assert all(rets.index == events.index)
    assert set(labels.unique()).issubset({-1, 0, 1})
    assert rets.notna().all()

    # Deterministic: first event hits profit-take (~+3%), second hits stop-loss (~-2%)
    assert labels.iloc[0] == 1
    assert labels.iloc[1] == -1
    assert rets.iloc[0] >= 0.03 - 1e-6
    assert rets.iloc[1] <= -0.019 - 1e-6