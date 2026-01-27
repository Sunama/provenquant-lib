import pandas as pd

def get_dollar_bars(
    dataframe: pd.DataFrame,
    dollar_bar_size: float,
    datetime_col: str='index',
) -> pd.DataFrame:
    """Generate dollar bars from tick data.

    Args:
        dataframe (pd.DataFrame): Input tick data with 'open', 'high', 'low', 'close', 'volume' columns.
        dollar_bar_size (float): Dollar threshold for each bar.
        datetime_col (str, optional): Name of the datetime column. Defaults to 'index'.
    Returns:
        pd.DataFrame: DataFrame containing dollar bars.
    """
    df = dataframe.copy()
    if datetime_col != 'index':
        df.set_index(datetime_col, inplace=True)
    
    df['dollar_value'] = df['close'] * df['volume']
    
    bars = []
    cum_dollar_value = 0.0
    bar_open = None
    bar_high = None
    bar_low = None
    bar_volume = 0
    bar_start_time = None
    
    for idx, row in df.iterrows():
        if bar_open is None:
            bar_open = row['open']
            bar_high = row['high']
            bar_low = row['low']
            bar_start_time = idx
        
        cum_dollar_value += row['dollar_value']
        bar_high = max(bar_high, row['high'])
        bar_low = min(bar_low, row['low'])
        bar_volume += row['volume']
        
        if cum_dollar_value >= dollar_bar_size:
            bar_close = row['close']
            bars.append({
                'open': bar_open,
                'high': bar_high,
                'low': bar_low,
                'close': bar_close,
                'volume': bar_volume,
                'start_time': bar_start_time,
                'end_time': idx
            })
            cum_dollar_value = 0.0
            bar_open = None
            bar_high = None
            bar_low = None
            bar_volume = 0
            bar_start_time = None
    
    bars_df = pd.DataFrame(bars)
    bars_df.set_index('end_time', inplace=True)
    
    return bars_df

def get_dollar_imbalance_bars(
  dataframe: pd.DataFrame,
  expected_imbalance_window: int,
  exp_num_ticks_initial: int,
  datetime_col: str='index',
) -> pd.DataFrame:
    """Generate dollar imbalance bars from tick data.

    Args:
        dataframe (pd.DataFrame): Input tick data with 'open', 'high', 'low', 'close', 'volume' columns.
        expected_imbalance_window (int): Window size for expected imbalance calculation.
        exp_num_ticks_initial (int): Initial expected number of ticks for the first bar.
        datetime_col (str, optional): Name of the datetime column. Defaults to 'index'.
    Returns:
        pd.DataFrame: DataFrame containing dollar imbalance bars.
    """
    df = dataframe.copy()
    if datetime_col != 'index':
        df.set_index(datetime_col, inplace=True)
    
    df['dollar_imbalance'] = ((df['close'] - df['open']).abs()) * df['volume']
    
    expected_imbalance = df['dollar_imbalance'].rolling(window=expected_imbalance_window).mean()
    expected_imbalance.iloc[expected_imbalance_window - 1] = expected_imbalance.iloc[expected_imbalance_window - 1:].mean()
    expected_imbalance.ffill(inplace=True)
    
    bars = []
    cum_imbalance = 0.0
    bar_open = None
    bar_high = None
    bar_low = None
    bar_volume = 0
    bar_start_time = None
    
    for idx, row in df.iterrows():
        if bar_open is None:
            bar_open = row['open']
            bar_high = row['high']
            bar_low = row['low']
            bar_start_time = idx
        
        cum_imbalance += row['dollar_imbalance']
        bar_high = max(bar_high, row['high'])
        bar_low = min(bar_low, row['low'])
        bar_volume += row['volume']
        
        exp_imbalance_value = expected_imbalance.loc[idx] if not pd.isna(expected_imbalance.loc[idx]) else exp_num_ticks_initial
        
        if cum_imbalance >= exp_imbalance_value:
            bar_close = row['close']
            bars.append({
                'open': bar_open,
                'high': bar_high,
                'low': bar_low,
                'close': bar_close,
                'volume': bar_volume,
                'start_time': bar_start_time,
                'end_time': idx
            })
            cum_imbalance = 0.0
            bar_open = None
            bar_high = None
            bar_low = None
            bar_volume = 0
            bar_start_time = None
    
    bars_df = pd.DataFrame(bars)
    bars_df.set_index('end_time', inplace=True)
    
    return bars_df

def get_tick_bars(
    dataframe: pd.DataFrame,
    tick_bar_size: int,
    datetime_col: str='index',
) -> pd.DataFrame:
    """Generate tick bars from tick data.

    Args:
        dataframe (pd.DataFrame): Input tick data with 'open', 'high', 'low', 'close', 'volume' columns.
        tick_bar_size (int): Number of ticks for each bar.
        datetime_col (str, optional): Name of the datetime column. Defaults to 'index'.
    Returns:
        pd.DataFrame: DataFrame containing tick bars.
    """
    df = dataframe.copy()
    if datetime_col != 'index':
        df.set_index(datetime_col, inplace=True)
    
    bars = []
    tick_count = 0
    bar_open = None
    bar_high = None
    bar_low = None
    bar_start_time = None
    
    for idx, row in df.iterrows():
        if bar_open is None:
            bar_open = row['open']
            bar_high = row['high']
            bar_low = row['low']
            bar_start_time = idx
        
        tick_count += 1
        bar_high = max(bar_high, row['high'])
        bar_low = min(bar_low, row['low'])
        
        if tick_count >= tick_bar_size:
            bar_close = row['close']
            bars.append({
                'open': bar_open,
                'high': bar_high,
                'low': bar_low,
                'close': bar_close,
                'volume': tick_count,
                'start_time': bar_start_time,
                'end_time': idx
            })
            tick_count = 0
            bar_open = None
            bar_high = None
            bar_low = None
            bar_start_time = None
    
    bars_df = pd.DataFrame(bars)
    bars_df.set_index('end_time', inplace=True)
    
    return bars_df

def get_volume_bars(
    dataframe: pd.DataFrame,
    volume_bar_size: float,
    datetime_col: str='index',
) -> pd.DataFrame:
    """Generate volume bars from tick data.

    Args:
        dataframe (pd.DataFrame): Input tick data with 'open', 'high', 'low', 'close', 'volume' columns.
        volume_bar_size (float): Volume threshold for each bar.
        datetime_col (str, optional): Name of the datetime column. Defaults to 'index'.
    Returns:
        pd.DataFrame: DataFrame containing volume bars.
    """
    df = dataframe.copy()
    if datetime_col != 'index':
        df.set_index(datetime_col, inplace=True)
    
    bars = []
    cum_volume = 0.0
    bar_open = None
    bar_high = None
    bar_low = None
    bar_start_time = None
    
    for idx, row in df.iterrows():
        if bar_open is None:
            bar_open = row['open']
            bar_high = row['high']
            bar_low = row['low']
            bar_start_time = idx
        
        cum_volume += row['volume']
        bar_high = max(bar_high, row['high'])
        bar_low = min(bar_low, row['low'])
        
        if cum_volume >= volume_bar_size:
            bar_close = row['close']
            bars.append({
                'open': bar_open,
                'high': bar_high,
                'low': bar_low,
                'close': bar_close,
                'volume': cum_volume,
                'start_time': bar_start_time,
                'end_time': idx
            })
            cum_volume = 0.0
            bar_open = None
            bar_high = None
            bar_low = None
            bar_start_time = None
    
    bars_df = pd.DataFrame(bars)
    bars_df.set_index('end_time', inplace=True)
    
    return bars_df

def get_volume_imbalance_bars(
  dataframe: pd.DataFrame,
  expected_imbalance_window: int,
  exp_num_ticks_initial: int,
  datetime_col: str='index',
) -> pd.DataFrame:
    """Generate volume imbalance bars from tick data.

    Args:
        dataframe (pd.DataFrame): Input tick data with 'open', 'high', 'low', 'close', 'volume' columns.
        expected_imbalance_window (int): Window size for expected imbalance calculation.
        exp_num_ticks_initial (int): Initial expected number of ticks for the first bar.
        datetime_col (str, optional): Name of the datetime column. Defaults to 'index'.
    Returns:
        pd.DataFrame: DataFrame containing volume imbalance bars.
    """
    df = dataframe.copy()
    if datetime_col != 'index':
        df.set_index(datetime_col, inplace=True)
    
    df['volume_imbalance'] = ((df['close'] - df['open']).abs())
    
    expected_imbalance = df['volume_imbalance'].rolling(window=expected_imbalance_window).mean()
    expected_imbalance.iloc[expected_imbalance_window - 1] = expected_imbalance.iloc[expected_imbalance_window - 1:].mean()
    expected_imbalance.ffill(inplace=True)
    
    bars = []
    cum_imbalance = 0.0
    bar_open = None
    bar_high = None
    bar_low = None
    bar_volume = 0
    bar_start_time = None
    
    for idx, row in df.iterrows():
        if bar_open is None:
            bar_open = row['open']
            bar_high = row['high']
            bar_low = row['low']
            bar_start_time = idx
        
        cum_imbalance += row['volume_imbalance']
        bar_high = max(bar_high, row['high'])
        bar_low = min(bar_low, row['low'])
        bar_volume += row['volume']
        
        exp_imbalance_value = expected_imbalance.loc[idx] if not pd.isna(expected_imbalance.loc[idx]) else exp_num_ticks_initial
        
        if cum_imbalance >= exp_imbalance_value:
            bar_close = row['close']
            bars.append({
                'open': bar_open,
                'high': bar_high,
                'low': bar_low,
                'close': bar_close,
                'volume': bar_volume,
                'start_time': bar_start_time,
                'end_time': idx
            })
            cum_imbalance = 0.0
            bar_open = None
            bar_high = None
            bar_low = None
            bar_volume = 0
            bar_start_time = None
    
    bars_df = pd.DataFrame(bars)
    bars_df.set_index('end_time', inplace=True)
    
    return bars_df
