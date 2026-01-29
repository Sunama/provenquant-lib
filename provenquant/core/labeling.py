import pandas as pd
import numpy as np

def filtrate_tripple_label_barrier(
    dataframe: pd.DataFrame,
    cusum_threshold: float,
    vertical_barrier: int,
    datetime_col: str = 'index',
) -> pd.DataFrame:
    """Filtrate triple barrier labels from raw DataFrame.
       Use this function before applying triple barrier labeling or in
       production based that we don't need labels and returns yet.

    Args:
        dataframe (pd.DataFrame): Raw DataFrame that contains close prices.
        cusum_threshold (float): Threshold for CUSUM filter in percentage.
        vertical_barrier (int): Ticks for vertical barrier.
        datetime_col (str): Name of the datetime column. Defaults to 'index'.

    Returns:
        pd.DataFrame: DataFrame with t1.
    """
    # CUSUM Filter
    if datetime_col != 'index':
        close_prices = dataframe.set_index(datetime_col)['close']
    else:
        close_prices = dataframe['close']
    
    diff = close_prices.pct_change().dropna()
    
    pos_cusum, neg_cusum = 0, 0
    t_events = []
    for idx in diff.index[1:]:
        pos_cusum = max(0, pos_cusum + diff.loc[idx])
        neg_cusum = min(0, neg_cusum + diff.loc[idx])
        
        if pos_cusum > cusum_threshold:
            t_events.append(idx)
            pos_cusum = 0
        elif neg_cusum < -cusum_threshold:
            t_events.append(idx)
            neg_cusum = 0
    t_events = pd.DatetimeIndex(t_events)
    
    # Vertical Barrier
    # Build t1 values using a list first, then create Series with proper dtype
    t1_values = []
    for event_time in t_events:
        t1_value = close_prices.index[
            close_prices.index.get_loc(event_time) + vertical_barrier
            ] if (close_prices.index.get_loc(event_time) + vertical_barrier) < len(close_prices) else close_prices.index[-1]
        t1_values.append(t1_value)
    
    t1 = pd.Series(t1_values, index=t_events, dtype=close_prices.index.dtype)
    df = pd.DataFrame(index=t_events)
    df['t1'] = t1
    
    # Add another columns in dataframe to df
    if datetime_col == 'index':
        # datetime is already the index
        for col in dataframe.columns:
            df[col] = dataframe.loc[t_events][col]
    else:
        # datetime is a column, need to set it as index first
        for col in dataframe.columns:
            if col != datetime_col:
                df[col] = dataframe.set_index(datetime_col).loc[t_events][col]
    
    return df

def get_tripple_label_barrier(
    dataframe: pd.DataFrame,
    close_series: pd.Series,
    threshold: float = 0.01,
    pt: float = 2,
    sl: float = 1,
) -> pd.DataFrame:
    """Get triple barrier labels from DataFrame with t1.

    Args:
        dataframe (pd.DataFrame): DataFrame with t1.
        close_series (pd.Series): Series of close prices that have datetime index.
        threshold (float): Threshold for labeling. Defaults to 0.01.
        pt (float): Profit taking multiplier. Defaults to 2.
        sl (float): Stop loss multiplier. Defaults to 1.
        
    Returns:
        pd.DataFrame: DataFrame with labels returns and mapped_labels.
    """
    labels = []
    returns = []
    
    for event_time, row in dataframe.iterrows():
        t1 = row['t1']
        if pd.isna(t1):
            labels.append(0)
            returns.append(0)
            continue
        
        start_price = close_series.loc[event_time]
        end_price = close_series.loc[t1]
        ret = (end_price - start_price) / start_price
        returns.append(ret)
        
        if ret > threshold * pt:
            labels.append(1)
        elif ret < -threshold * sl:
            labels.append(-1)
        else:
            labels.append(0)
    
    dataframe['label'] = labels
    dataframe['return'] = returns
    dataframe['mapped_label'] = dataframe['label'].map({1: 2, -1: 1, 0: 0})
    
    return dataframe
