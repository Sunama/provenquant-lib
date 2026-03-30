import pandas as pd

def convert_standard_bars_to_larger_timeframe(
    dataframe: pd.DataFrame,
    timeframe: str,
    time_col: str='index',
) -> pd.DataFrame:
    """Convert standard bars to a larger timeframe (e.g., 1-minute bars to 5-minute bars).

    Args:
        dataframe (pd.DataFrame): Input standard bars with 'open', 'high', 'low', 'close', 'volume' columns and datetime index.
        timeframe (str): Resampling timeframe (e.g., '5T' for 5 minutes, '15T' for 15 minutes).
        time_col (str, optional): Name of the datetime column. Defaults to 'index'.
        
    Returns:
        pd.DataFrame: DataFrame containing resampled bars.
    """
    df = dataframe.copy()
    
    if time_col != 'index':
        # Ensure time_col is datetime type
        df[time_col] = pd.to_datetime(df[time_col])
        df.set_index(time_col, inplace=True)
    elif not isinstance(df.index, pd.DatetimeIndex):
        # If it's the index, ensure it's datetime type
        df.index = pd.to_datetime(df.index)
    
    # Map common timeframe suffixes to pandas-compatible ones
    timeframe_mapping = {
        'm': 'T',    # minutes
        'h': 'H',    # hours
        'd': 'D',    # days
        'w': 'W',    # weeks
        'M': 'ME',   # months (Month End)
    }
    
    # Check if timeframe needs mapping (e.g., '5m' -> '5T')
    import re
    match = re.match(r'(\d+)([a-zA-Z]+)', timeframe)
    if match:
        value, unit = match.groups()
        if unit in timeframe_mapping:
            timeframe = f"{value}{timeframe_mapping[unit]}"

    resampled = df.resample(timeframe).agg({
        'open': 'first',
        'high': 'max',
        'low': 'min',
        'close': 'last',
        'volume': 'sum'
    }).dropna()
    
    if time_col != 'index':
        # Restore the datetime col as index if that was the original structure
        resampled.index.name = time_col
    
    return resampled

def get_dollar_bars(
    dataframe: pd.DataFrame,
    threshold: float,
    time_col: str='index',
) -> pd.DataFrame:
    """Generate dollar bars from tick data.

    Args:
        dataframe (pd.DataFrame): Input tick data with 'open', 'high', 'low', 'close', 'volume' columns.
        threshold (float): Dollar threshold for each bar.
        time_col (str, optional): Name of the datetime column. Defaults to 'index'.
        
    Returns:
        pd.DataFrame: DataFrame containing dollar bars.
    """
    df = dataframe.copy()
    if time_col != 'index':
        df.set_index(time_col, inplace=True)
    
    # Pre-calculate
    df['dollar_value'] = df['close'] * df['volume']
    df['price_change'] = df['close'].diff()
    
    bars = []
    cum_dollar_value = 0.0
    cum_ticks = 0
    cum_buy_volume = 0.0
    cum_sell_volume = 0.0
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
        cum_ticks += 1
        bar_high = max(bar_high, row['high'])
        bar_low = min(bar_low, row['low'])
        bar_volume += row['volume']
        
        # Classify volume as buy or sell based on price change
        if row['price_change'] > 0:
            cum_buy_volume += row['volume']
        elif row['price_change'] < 0:
            cum_sell_volume += row['volume']
        else:
            cum_buy_volume += row['volume'] / 2
            cum_sell_volume += row['volume'] / 2
        
        if cum_dollar_value >= threshold:
            bar_close = row['close']
            bars.append({
                'start_time': bar_start_time,
                'open': bar_open,
                'high': bar_high,
                'low': bar_low,
                'close': bar_close,
                'volume': bar_volume,
                'cum_ticks': cum_ticks,
                'cum_dollar': cum_dollar_value,
                'buy_volume': cum_buy_volume,
                'sell_volume': cum_sell_volume,
                'end_time': idx
            })
            cum_dollar_value = 0.0
            cum_ticks = 0
            cum_buy_volume = 0.0
            cum_sell_volume = 0.0
            bar_open = None
            bar_high = None
            bar_low = None
            bar_volume = 0
            bar_start_time = None
    
    # Handle incomplete bar from last row
    if bar_open is not None:
        all_bars = bars + [{
            'start_time': bar_start_time,
            'open': bar_open,
            'high': bar_high,
            'low': bar_low,
            'close': df.iloc[-1]['close'] if len(df) > 0 else bar_open,
            'volume': bar_volume,
            'cum_ticks': cum_ticks,
            'cum_dollar': cum_dollar_value,
            'buy_volume': cum_buy_volume,
            'sell_volume': cum_sell_volume,
            'end_time': df.index[-1] if len(df) > 0 else bar_start_time
        }]
    else:
        all_bars = bars
    
    bars_df = pd.DataFrame(all_bars)
    bars_df.set_index('end_time', inplace=True)
    
    return bars_df


def get_dollar_imbalance_bars(
  dataframe: pd.DataFrame,
  expected_imbalance_window: int,
  exp_num_ticks_initial: int,
  time_col: str='index',
) -> pd.DataFrame:
    """Generate dollar imbalance bars from tick data (AFML Chapter 2).

    Args:
        dataframe (pd.DataFrame): Input tick data with 'open', 'high', 'low', 'close', 'volume' columns.
        expected_imbalance_window (int): Window size for expected imbalance calculation (EMA).
        exp_num_ticks_initial (int): Initial expected number of ticks for the first bar.
        time_col (str, optional): Name of the datetime column. Defaults to 'index'.
    Returns:
        pd.DataFrame: DataFrame containing dollar imbalance bars.
    """
    df = dataframe.copy()
    if time_col != 'index':
        df.set_index(time_col, inplace=True)
    
    # Apply tick rule: +1 for uptick, -1 for downtick
    df['price_diff'] = df['close'].diff()
    df['tick_direction'] = 0
    
    # Initialize with first non-zero tick
    last_tick = 0
    for i in df.index:
        if df.loc[i, 'price_diff'] > 0:
            last_tick = 1
        elif df.loc[i, 'price_diff'] < 0:
            last_tick = -1
        # If price_diff == 0, use last_tick
        df.loc[i, 'tick_direction'] = last_tick if last_tick != 0 else 1
    
    # Dollar imbalance using tick rule: |close * volume| * tick_direction
    df['dollar_imbalance'] = df['close'] * df['volume'] * df['tick_direction']
    
    # Calculate expected imbalance using EMA
    expected_imbalance = df['dollar_imbalance'].ewm(span=expected_imbalance_window).mean().abs()
    # Forward fill for initial NaN values
    expected_imbalance.bfill(inplace=True)
    expected_imbalance.fillna(exp_num_ticks_initial, inplace=True)
    
    bars = []
    cum_imbalance = 0.0
    cum_ticks = 0
    cum_dollar_value = 0.0
    cum_buy_volume = 0.0
    cum_sell_volume = 0.0
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
        
        # Accumulate absolute imbalance
        cum_imbalance += abs(row['dollar_imbalance'])
        cum_ticks += 1
        cum_dollar_value += row['close'] * row['volume']
        bar_high = max(bar_high, row['high'])
        bar_low = min(bar_low, row['low'])
        bar_volume += row['volume']
        
        # Classify volume as buy or sell based on tick direction
        if row['tick_direction'] > 0:
            cum_buy_volume += row['volume']
        elif row['tick_direction'] < 0:
            cum_sell_volume += row['volume']
        
        exp_imbalance_value = expected_imbalance.loc[idx] if not pd.isna(expected_imbalance.loc[idx]) else exp_num_ticks_initial
        
        if cum_imbalance >= exp_imbalance_value:
            bar_close = row['close']
            bars.append({
                'start_time': bar_start_time,
                'open': bar_open,
                'high': bar_high,
                'low': bar_low,
                'close': bar_close,
                'volume': bar_volume,
                'cum_ticks': cum_ticks,
                'cum_dollar': cum_dollar_value,
                'buy_volume': cum_buy_volume,
                'sell_volume': cum_sell_volume,
                'end_time': idx
            })
            cum_imbalance = 0.0
            cum_ticks = 0
            cum_dollar_value = 0.0
            cum_buy_volume = 0.0
            cum_sell_volume = 0.0
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
    time_col: str='index',
) -> pd.DataFrame:
    """Generate tick bars from tick data.

    Args:
        dataframe (pd.DataFrame): Input tick data with 'open', 'high', 'low', 'close', 'volume' columns.
        tick_bar_size (int): Number of ticks for each bar.
        time_col (str, optional): Name of the datetime column. Defaults to 'index'.
    Returns:
        pd.DataFrame: DataFrame containing tick bars.
    """
    df = dataframe.copy()
    if time_col != 'index':
        df.set_index(time_col, inplace=True)
    
    df['price_change'] = df['close'].diff()
    
    bars = []
    tick_count = 0
    cum_volume = 0
    cum_dollar_value = 0.0
    cum_buy_volume = 0.0
    cum_sell_volume = 0.0
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
        cum_volume += row['volume']
        cum_dollar_value += row['close'] * row['volume']
        bar_high = max(bar_high, row['high'])
        bar_low = min(bar_low, row['low'])
        
        # Classify volume as buy or sell based on price change
        if row['price_change'] > 0:
            cum_buy_volume += row['volume']
        elif row['price_change'] < 0:
            cum_sell_volume += row['volume']
        else:
            # If price doesn't change, split evenly
            cum_buy_volume += row['volume'] / 2
            cum_sell_volume += row['volume'] / 2
        
        if tick_count >= tick_bar_size:
            bar_close = row['close']
            bars.append({
                'start_time': bar_start_time,
                'open': bar_open,
                'high': bar_high,
                'low': bar_low,
                'close': bar_close,
                'volume': cum_volume,
                'cum_ticks': tick_count,
                'cum_dollar': cum_dollar_value,
                'buy_volume': cum_buy_volume,
                'sell_volume': cum_sell_volume,
                'end_time': idx
            })
            tick_count = 0
            cum_volume = 0
            cum_dollar_value = 0.0
            cum_buy_volume = 0.0
            cum_sell_volume = 0.0
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
    time_col: str='index',
) -> pd.DataFrame:
    """Generate volume bars from tick data.

    Args:
        dataframe (pd.DataFrame): Input tick data with 'open', 'high', 'low', 'close', 'volume' columns.
        volume_bar_size (float): Volume threshold for each bar.
        time_col (str, optional): Name of the datetime column. Defaults to 'index'.
    Returns:
        pd.DataFrame: DataFrame containing volume bars.
    """
    df = dataframe.copy()
    if time_col != 'index':
        df.set_index(time_col, inplace=True)
    
    df['price_change'] = df['close'].diff()
    
    bars = []
    cum_volume = 0.0
    cum_ticks = 0
    cum_dollar_value = 0.0
    cum_buy_volume = 0.0
    cum_sell_volume = 0.0
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
        cum_ticks += 1
        cum_dollar_value += row['close'] * row['volume']
        bar_high = max(bar_high, row['high'])
        bar_low = min(bar_low, row['low'])
        
        # Classify volume as buy or sell based on price change
        if row['price_change'] > 0:
            cum_buy_volume += row['volume']
        elif row['price_change'] < 0:
            cum_sell_volume += row['volume']
        else:
            # If price doesn't change, split evenly
            cum_buy_volume += row['volume'] / 2
            cum_sell_volume += row['volume'] / 2
        
        if cum_volume >= volume_bar_size:
            bar_close = row['close']
            bars.append({
                'start_time': bar_start_time,
                'open': bar_open,
                'high': bar_high,
                'low': bar_low,
                'close': bar_close,
                'volume': cum_volume,
                'cum_ticks': cum_ticks,
                'cum_dollar': cum_dollar_value,
                'buy_volume': cum_buy_volume,
                'sell_volume': cum_sell_volume,
                'end_time': idx
            })
            cum_volume = 0.0
            cum_ticks = 0
            cum_dollar_value = 0.0
            cum_buy_volume = 0.0
            cum_sell_volume = 0.0
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
  time_col: str='index',
) -> pd.DataFrame:
    """Generate volume imbalance bars from tick data (AFML Chapter 2).

    Args:
        dataframe (pd.DataFrame): Input tick data with 'open', 'high', 'low', 'close', 'volume' columns.
        expected_imbalance_window (int): Window size for expected imbalance calculation (EMA).
        exp_num_ticks_initial (int): Initial expected number of ticks for the first bar.
        time_col (str, optional): Name of the datetime column. Defaults to 'index'.
    Returns:
        pd.DataFrame: DataFrame containing volume imbalance bars.
    """
    df = dataframe.copy()
    if time_col != 'index':
        df.set_index(time_col, inplace=True)
    
    # Apply tick rule: +1 for uptick, -1 for downtick
    df['price_diff'] = df['close'].diff()
    df['tick_direction'] = 0
    
    # Initialize with first non-zero tick
    last_tick = 0
    for i in df.index:
        if df.loc[i, 'price_diff'] > 0:
            last_tick = 1
        elif df.loc[i, 'price_diff'] < 0:
            last_tick = -1
        # If price_diff == 0, use last_tick
        df.loc[i, 'tick_direction'] = last_tick if last_tick != 0 else 1
    
    # Volume imbalance using tick rule: volume * tick_direction
    df['volume_imbalance'] = df['volume'] * df['tick_direction']
    
    # Calculate expected imbalance using EMA
    expected_imbalance = df['volume_imbalance'].ewm(span=expected_imbalance_window).mean().abs()
    # Forward fill for initial NaN values
    expected_imbalance.bfill(inplace=True)
    expected_imbalance.fillna(exp_num_ticks_initial, inplace=True)
    
    bars = []
    cum_imbalance = 0.0
    cum_ticks = 0
    cum_dollar_value = 0.0
    cum_buy_volume = 0.0
    cum_sell_volume = 0.0
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
        
        # Accumulate absolute imbalance
        cum_imbalance += abs(row['volume_imbalance'])
        cum_ticks += 1
        cum_dollar_value += row['close'] * row['volume']
        bar_high = max(bar_high, row['high'])
        bar_low = min(bar_low, row['low'])
        bar_volume += row['volume']
        
        # Classify volume as buy or sell based on tick direction
        if row['tick_direction'] > 0:
            cum_buy_volume += row['volume']
        elif row['tick_direction'] < 0:
            cum_sell_volume += row['volume']
        
        exp_imbalance_value = expected_imbalance.loc[idx] if not pd.isna(expected_imbalance.loc[idx]) else exp_num_ticks_initial
        
        if cum_imbalance >= exp_imbalance_value:
            bar_close = row['close']
            bars.append({
                'start_time': bar_start_time,
                'open': bar_open,
                'high': bar_high,
                'low': bar_low,
                'close': bar_close,
                'volume': bar_volume,
                'cum_ticks': cum_ticks,
                'cum_dollar': cum_dollar_value,
                'buy_volume': cum_buy_volume,
                'sell_volume': cum_sell_volume,
                'end_time': idx
            })
            cum_imbalance = 0.0
            cum_ticks = 0
            cum_dollar_value = 0.0
            cum_buy_volume = 0.0
            cum_sell_volume = 0.0
            bar_open = None
            bar_high = None
            bar_low = None
            bar_volume = 0
            bar_start_time = None
    
    bars_df = pd.DataFrame(bars)
    bars_df.set_index('end_time', inplace=True)
    
    return bars_df
