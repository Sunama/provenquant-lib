import pandas as pd
import numpy as np

def get_horizontal_barrier_events(
    dataframe: pd.DataFrame,
    threshold: float,
    datetime_col: str = 'index',
) -> pd.DataFrame:
    """Reduce dataframe by use CUSUM filter to get horizontal barrier events.

    Args:
        dataframe (pd.DataFrame): DataFrame that contains close prices.
        threshold (float): Volatility threshold for CUSUM filter in percentage.
        datetime_col (str): Name of the datetime column. Defaults to 'index'.

    Returns:
        pd.DataFrame: DataFrame with horizontal barrier events.
    """
    
    pct_changes = dataframe['close'].pct_change()
    datetimes = dataframe.index if datetime_col == 'index' else dataframe[datetime_col]
    t_events = []
    s_pos, s_neg = 0, 0

    for i in range(1, len(pct_changes)):
        pct_change = pct_changes.iloc[i]
        s_pos = max(0, s_pos + pct_change)
        s_neg = min(0, s_neg + pct_change)

        if s_pos > threshold:
            t_events.append(datetimes[i])
            s_pos = 0
        elif s_neg < -threshold:
            t_events.append(datetimes[i])
            s_neg = 0

    events = pd.DataFrame(index=t_events)
    
    return events

def add_vertical_barrier_to_horizontal_barrier_events(
    dataframe: pd.DataFrame,
    events: pd.DataFrame,
    vertical_barrier_duration: pd.Timedelta,
    datetime_col: str = 'index',
) -> pd.DataFrame:
    """Add vertical barrier to horizontal barrier events.

    Args:
        dataframe (pd.DataFrame): Raw DataFrame that contains close prices.
        events (pd.DataFrame): horizontal barrier events.
        vertical_barrier_duration (pd.Timedelta): Duration for vertical barrier.
        datetime_col (str): Name of the datetime column. Defaults to 'index'.

    Returns:
        pd.DataFrame: DataFrame with vertical_barrier and ret added to events.
    """
    
    datetimes = dataframe.index if datetime_col == 'index' else dataframe[datetime_col]
    last_datetime = datetimes[-1] if datetime_col == 'index' else datetimes.iloc[-1]
    
    rets = []
    vertical_barriers = []

    for event_time in events.index:
        vertical_barrier_time = event_time + vertical_barrier_duration
        if vertical_barrier_time > last_datetime:
            vertical_barrier_time = last_datetime
        vertical_barriers.append(vertical_barrier_time)
        
        if datetime_col == 'index':
            initial_price = dataframe.loc[event_time, 'close']
            final_price = dataframe.loc[vertical_barrier_time, 'close']
        else:
            initial_price = dataframe.loc[dataframe[datetime_col] == event_time, 'close'].values[0]
            final_price = dataframe.loc[dataframe[datetime_col] == vertical_barrier_time, 'close'].values[0]
        ret = (final_price - initial_price) / initial_price
        rets.append(ret)

    events['vertical_barrier'] = vertical_barriers
    events['ret'] = rets
    
    return events

def get_binary_labels(
    events: pd.DataFrame,
    min_ret: float = 0.0,
    side: str = 'long',
) -> pd.Series:
    """Get binary labels from events DataFrame.

    Args:
        events (pd.DataFrame): DataFrame with 'ret' column.
        min_ret (float): Minimum return to consider for labeling.
        side (str): 'long' or 'short' to indicate the trade side.
                    Defaults to 'long'.
    Returns:
        pd.Series: Series containing binary labels.
    """
    
    if side not in ('long', 'short'):
        raise ValueError("side must be either 'long' or 'short'")

    labels = []
    for ret in events['ret']:
        if abs(ret) < min_ret:
            labels.append(0)
            continue

        if side == 'long':
            label = 1 if ret > 0 else 0
        else:  # side == 'short'
            label = 1 if ret < 0 else 0

        labels.append(label)
        
    return pd.Series(labels, index=events.index)

def get_triple_barrier_labels(
    dataframe: pd.DataFrame,
    events: pd.DataFrame,
    threshold: float,
    pt: float,
    sl: float,
    side: str = 'long',
    datetime_col: str = 'index',
) -> tuple[pd.Series, pd.Series]:
    """Get triple barrier labels from events DataFrame.

    Args:
        dataframe (pd.DataFrame): Raw DataFrame that contains close prices.
        events (pd.DataFrame): DataFrame with event times.
        threshold (float): Threshold for determining significant returns.
        pt (float): Multiple of threshold for profit-taking barrier.
        sl (float): Multiple of threshold for stop-loss barrier.
        side (str): 'long' or 'short' to indicate the trade side.
                    Defaults to 'long'.
        datetime_col (str): Name of the datetime column. Defaults to 'index'.

    Returns:
        pd.Series: Series containing triple barrier labels.
    """
    
    upper_barrier = threshold * pt
    lower_barrier = -threshold * sl
    
    labels = []
    rets = []
    for event_time in events.index:
        if datetime_col == 'index':
            initial_price = dataframe.loc[event_time, 'close']
            event_idx = dataframe.index.get_loc(event_time)
        else:
            initial_price = dataframe.loc[dataframe[datetime_col] == event_time, 'close'].values[0]
            event_idx = dataframe[dataframe[datetime_col] == event_time].index[0]
        
        # Get vertical barrier time if available
        if 'vertical_barrier' in events.columns:
            vertical_barrier = events.loc[event_time, 'vertical_barrier']
            if datetime_col == 'index':
                end_idx = dataframe.index.get_loc(vertical_barrier)
            else:
                end_idx = dataframe[dataframe[datetime_col] == vertical_barrier].index[0]
        else:
            end_idx = len(dataframe) - 1
        
        # Scan through prices until hitting a barrier or vertical barrier
        label = 0
        final_ret = 0
        for i in range(event_idx + 1, end_idx + 1):
            if datetime_col == 'index':
                current_price = dataframe.iloc[i]['close']
            else:
                current_price = dataframe.iloc[i]['close']
            
            ret = (current_price - initial_price) / initial_price
            final_ret = ret
            
            if side == 'long':
                if ret >= upper_barrier:
                    label = 1
                    break
                elif ret <= lower_barrier:
                    label = -1
                    break
            elif side == 'short':
                if ret <= -upper_barrier:
                    label = 1
                    break
                elif ret >= -lower_barrier:
                    label = -1
                    break
            else:
                raise ValueError("side must be either 'long' or 'short'")
        
        labels.append(label)
        rets.append(final_ret)
        
    return pd.Series(labels, index=events.index), pd.Series(rets, index=events.index)

