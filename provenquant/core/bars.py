import pandas as pd
from multiprocessing import Pool
from functools import partial

def _process_dollar_bars_chunk(chunk_tuple, threshold: float):
    """Helper function to process a chunk of data for dollar bars (multiprocess compatible).
    
    Args:
        chunk_tuple (tuple): (chunk_data, chunk_id)
        threshold (float): Dollar threshold for each bar.
        
    Returns:
        tuple: (bars_list, final_state_dict)
    """
    chunk, chunk_id = chunk_tuple
    
    bars = []
    cum_dollar_value = 0.0
    cum_ticks = 0
    cum_buy_volume = 0.0
    cum_sell_volume = 0.0
    bar_open = None
    bar_high = None
    bar_low = None
    bar_volume = 0
    bar_start_date = None
    
    for idx, row in chunk.iterrows():
        if bar_open is None:
            bar_open = row['open']
            bar_high = row['high']
            bar_low = row['low']
            bar_start_date = idx
        
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
                'start_date': bar_start_date,
                'open': bar_open,
                'high': bar_high,
                'low': bar_low,
                'close': bar_close,
                'volume': bar_volume,
                'cum_ticks': cum_ticks,
                'cum_dollar': cum_dollar_value,
                'buy_volume': cum_buy_volume,
                'sell_volume': cum_sell_volume,
                'end_date': idx
            })
            cum_dollar_value = 0.0
            cum_ticks = 0
            cum_buy_volume = 0.0
            cum_sell_volume = 0.0
            bar_open = None
            bar_high = None
            bar_low = None
            bar_volume = 0
            bar_start_date = None
    
    # Return final state for handling incomplete bars at chunk boundary
    final_state = {
        'cum_dollar_value': cum_dollar_value,
        'cum_ticks': cum_ticks,
        'cum_buy_volume': cum_buy_volume,
        'cum_sell_volume': cum_sell_volume,
        'bar_open': bar_open,
        'bar_high': bar_high,
        'bar_low': bar_low,
        'bar_volume': bar_volume,
        'bar_start_date': bar_start_date
    }
    
    return bars, final_state

def convert_standard_bars_to_larger_timeframe(
    dataframe: pd.DataFrame,
    timeframe: str,
    datetime_col: str='index',
) -> pd.DataFrame:
    """Convert standard bars to a larger timeframe (e.g., 1-minute bars to 5-minute bars).

    Args:
        dataframe (pd.DataFrame): Input standard bars with 'open', 'high', 'low', 'close', 'volume' columns and datetime index.
        timeframe (str): Resampling timeframe (e.g., '5T' for 5 minutes, '15T' for 15 minutes).
        datetime_col (str, optional): Name of the datetime column. Defaults to 'index'.
        
    Returns:
        pd.DataFrame: DataFrame containing resampled bars.
    """
    df = dataframe.copy()
    
    if datetime_col != 'index':
        # Ensure datetime_col is datetime type
        df[datetime_col] = pd.to_datetime(df[datetime_col])
        df.set_index(datetime_col, inplace=True)
    elif not isinstance(df.index, pd.DatetimeIndex):
        # If it's the index, ensure it's datetime type
        df.index = pd.to_datetime(df.index)
    
    resampled = df.resample(timeframe).agg({
        'open': 'first',
        'high': 'max',
        'low': 'min',
        'close': 'last',
        'volume': 'sum'
    }).dropna()
    
    if datetime_col != 'index':
        # Restore the datetime col as index if that was the original structure
        resampled.index.name = datetime_col
    
    return resampled

def get_dollar_bars(
    dataframe: pd.DataFrame,
    threshold: float,
    datetime_col: str='index',
    num_threads: int = 1,
) -> pd.DataFrame:
    """Generate dollar bars from tick data.

    Args:
        dataframe (pd.DataFrame): Input tick data with 'open', 'high', 'low', 'close', 'volume' columns.
        threshold (float): Dollar threshold for each bar.
        datetime_col (str, optional): Name of the datetime column. Defaults to 'index'.
        num_threads (int, optional): Number of processes. Defaults to 1 (sequential). Set > 1 for multiprocessing.
        
    Returns:
        pd.DataFrame: DataFrame containing dollar bars.
    """
    df = dataframe.copy()
    if datetime_col != 'index':
        df.set_index(datetime_col, inplace=True)
    
    # Pre-calculate to ensure consistency across chunks when multiprocessing
    df['dollar_value'] = df['close'] * df['volume']
    df['price_change'] = df['close'].diff()
    
    # Split data into chunks
    num_chunks = max(1, num_threads)
    chunk_size = max(1, len(df) // num_chunks)
    chunks = [df.iloc[i:i + chunk_size] for i in range(0, len(df), chunk_size)]
    
    # Process chunks (sequentially if num_threads=1, in parallel otherwise)
    with Pool(processes=num_threads if num_threads > 1 else 1) as pool:
        process_func = partial(_process_dollar_bars_chunk, threshold=threshold)
        results = pool.map(process_func, [(chunk, i) for i, chunk in enumerate(chunks)])
    
    # Combine results from all chunks
    all_bars = []
    carry_over_state = None
    
    for chunk_id, (chunk_bars, final_state) in enumerate(results):
        # Handle incomplete bar from previous chunk
        if carry_over_state and carry_over_state['bar_open'] is not None and chunk_bars:
            first_bar = chunk_bars[0]
            first_bar['open'] = carry_over_state['bar_open']
            first_bar['high'] = max(carry_over_state['bar_high'], first_bar['high'])
            first_bar['low'] = min(carry_over_state['bar_low'], first_bar['low'])
            first_bar['volume'] += carry_over_state['bar_volume']
            first_bar['cum_ticks'] += carry_over_state['cum_ticks']
            first_bar['cum_dollar'] += carry_over_state['cum_dollar_value']
            first_bar['buy_volume'] += carry_over_state['cum_buy_volume']
            first_bar['sell_volume'] += carry_over_state['cum_sell_volume']
            first_bar['start_date'] = carry_over_state['bar_start_date']
        
        all_bars.extend(chunk_bars)
        carry_over_state = final_state
    
    # Handle incomplete bar from last chunk
    if carry_over_state and carry_over_state['bar_open'] is not None:
        all_bars.append({
            'start_date': carry_over_state['bar_start_date'],
            'open': carry_over_state['bar_open'],
            'high': carry_over_state['bar_high'],
            'low': carry_over_state['bar_low'],
            'close': df.iloc[-1]['close'] if len(df) > 0 else carry_over_state['bar_open'],
            'volume': carry_over_state['bar_volume'],
            'cum_ticks': carry_over_state['cum_ticks'],
            'cum_dollar': carry_over_state['cum_dollar_value'],
            'buy_volume': carry_over_state['cum_buy_volume'],
            'sell_volume': carry_over_state['cum_sell_volume'],
            'end_date': df.index[-1] if len(df) > 0 else carry_over_state['bar_start_date']
        })
    
    bars_df = pd.DataFrame(all_bars)
    bars_df.set_index('end_date', inplace=True)
    
    return bars_df

def get_dollar_imbalance_bars(
  dataframe: pd.DataFrame,
  expected_imbalance_window: int,
  exp_num_ticks_initial: int,
  datetime_col: str='index',
) -> pd.DataFrame:
    """Generate dollar imbalance bars from tick data (AFML Chapter 2).

    Args:
        dataframe (pd.DataFrame): Input tick data with 'open', 'high', 'low', 'close', 'volume' columns.
        expected_imbalance_window (int): Window size for expected imbalance calculation (EMA).
        exp_num_ticks_initial (int): Initial expected number of ticks for the first bar.
        datetime_col (str, optional): Name of the datetime column. Defaults to 'index'.
    Returns:
        pd.DataFrame: DataFrame containing dollar imbalance bars.
    """
    df = dataframe.copy()
    if datetime_col != 'index':
        df.set_index(datetime_col, inplace=True)
    
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
    bar_start_date = None
    
    for idx, row in df.iterrows():
        if bar_open is None:
            bar_open = row['open']
            bar_high = row['high']
            bar_low = row['low']
            bar_start_date = idx
        
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
                'start_date': bar_start_date,
                'open': bar_open,
                'high': bar_high,
                'low': bar_low,
                'close': bar_close,
                'volume': bar_volume,
                'cum_ticks': cum_ticks,
                'cum_dollar': cum_dollar_value,
                'buy_volume': cum_buy_volume,
                'sell_volume': cum_sell_volume,
                'end_date': idx
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
            bar_start_date = None
    
    bars_df = pd.DataFrame(bars)
    bars_df.set_index('end_date', inplace=True)
    
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
    bar_start_date = None
    
    for idx, row in df.iterrows():
        if bar_open is None:
            bar_open = row['open']
            bar_high = row['high']
            bar_low = row['low']
            bar_start_date = idx
        
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
                'start_date': bar_start_date,
                'open': bar_open,
                'high': bar_high,
                'low': bar_low,
                'close': bar_close,
                'volume': cum_volume,
                'cum_ticks': tick_count,
                'cum_dollar': cum_dollar_value,
                'buy_volume': cum_buy_volume,
                'sell_volume': cum_sell_volume,
                'end_date': idx
            })
            tick_count = 0
            cum_volume = 0
            cum_dollar_value = 0.0
            cum_buy_volume = 0.0
            cum_sell_volume = 0.0
            bar_open = None
            bar_high = None
            bar_low = None
            bar_start_date = None
    
    bars_df = pd.DataFrame(bars)
    bars_df.set_index('end_date', inplace=True)
    
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
    bar_start_date = None
    
    for idx, row in df.iterrows():
        if bar_open is None:
            bar_open = row['open']
            bar_high = row['high']
            bar_low = row['low']
            bar_start_date = idx
        
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
                'start_date': bar_start_date,
                'open': bar_open,
                'high': bar_high,
                'low': bar_low,
                'close': bar_close,
                'volume': cum_volume,
                'cum_ticks': cum_ticks,
                'cum_dollar': cum_dollar_value,
                'buy_volume': cum_buy_volume,
                'sell_volume': cum_sell_volume,
                'end_date': idx
            })
            cum_volume = 0.0
            cum_ticks = 0
            cum_dollar_value = 0.0
            cum_buy_volume = 0.0
            cum_sell_volume = 0.0
            bar_open = None
            bar_high = None
            bar_low = None
            bar_start_date = None
    
    bars_df = pd.DataFrame(bars)
    bars_df.set_index('end_date', inplace=True)
    
    return bars_df

def get_volume_imbalance_bars(
  dataframe: pd.DataFrame,
  expected_imbalance_window: int,
  exp_num_ticks_initial: int,
  datetime_col: str='index',
) -> pd.DataFrame:
    """Generate volume imbalance bars from tick data (AFML Chapter 2).

    Args:
        dataframe (pd.DataFrame): Input tick data with 'open', 'high', 'low', 'close', 'volume' columns.
        expected_imbalance_window (int): Window size for expected imbalance calculation (EMA).
        exp_num_ticks_initial (int): Initial expected number of ticks for the first bar.
        datetime_col (str, optional): Name of the datetime column. Defaults to 'index'.
    Returns:
        pd.DataFrame: DataFrame containing volume imbalance bars.
    """
    df = dataframe.copy()
    if datetime_col != 'index':
        df.set_index(datetime_col, inplace=True)
    
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
    bar_start_date = None
    
    for idx, row in df.iterrows():
        if bar_open is None:
            bar_open = row['open']
            bar_high = row['high']
            bar_low = row['low']
            bar_start_date = idx
        
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
                'start_date': bar_start_date,
                'open': bar_open,
                'high': bar_high,
                'low': bar_low,
                'close': bar_close,
                'volume': bar_volume,
                'cum_ticks': cum_ticks,
                'cum_dollar': cum_dollar_value,
                'buy_volume': cum_buy_volume,
                'sell_volume': cum_sell_volume,
                'end_date': idx
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
            bar_start_date = None
    
    bars_df = pd.DataFrame(bars)
    bars_df.set_index('end_date', inplace=True)
    
    return bars_df
