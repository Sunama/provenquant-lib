import pandas as pd
from concurrent.futures import ProcessPoolExecutor
from functools import partial

def get_frac_diff(
    series: pd.Series,
    d: float,
    threshold: float = 1e-5,
) -> pd.Series:
    """Compute fractionally differenced series using Fixed-Window Fractional Differentiation (FFD).
    
    Args:
        series (pd.Series): Original time series.
        d (float): Differencing order.
        threshold (float, optional): Weight threshold for window truncation. Defaults to 1e-5.
    
    Returns:
        pd.Series: Fractionally differenced series.
    """
    # Compute weights for fractionally differencing until they drop below threshold
    w = [1.0]
    k = 1
    while True:
        next_w = -w[-1] * (d - k + 1) / k
        if abs(next_w) < threshold:
            break
        w.append(next_w)
        k += 1
        # Safety break to prevent infinite loop for very small d
        if k > len(series):
            break
            
    w = pd.Series(w).values
    # Replace -0.0 with 0.0 to avoid numerical issues
    w[w == 0] = 0.0

    # Apply weights to the series using a fixed window (FFD)
    # This approach is memory efficient and supports incremental processing
    frac_diff_series = pd.Series(index=series.index, dtype=float)
    
    # We need at least the window size to have a stable value, 
    # but we will compute for all available if possible.
    window_size = len(w)
    
    for i in range(len(series)):
        # For the first few elements where we don't have enough history, 
        # the value will be less "stable" but still calculated using available history.
        curr_window = min(i + 1, window_size)
        frac_diff_series.iloc[i] = (w[:curr_window][::-1] * series.iloc[i-curr_window+1:i+1].values).sum()
    
    return frac_diff_series

def _process_column(col: str, df: pd.DataFrame, d: float, prefix: str, postfix: str) -> tuple[str, pd.Series]:
    """Helper function to process a single column with fractional differencing.
    
    Args:
        col (str): Column name to process.
        df (pd.DataFrame): Original DataFrame.
        d (float): Differencing order.
        prefix (str): Prefix for new column name.
        postfix (str): Postfix for new column name.
    
    Returns:
        tuple[str, pd.Series]: Tuple of (new_column_name, differenced_series).
    """
    new_col_name = f"{prefix}{col}{postfix}"
    frac_diff_series = get_frac_diff(df[col], d)
    return new_col_name, frac_diff_series

def get_frac_diffs(
    series_list: list[pd.Series],
    d: float,
    num_threads: int = 1,
) -> list[pd.Series]:
    """Compute fractionally differenced series for a list of series.
    
    Args:
        series_list (list[pd.Series]): List of original time series.
        d (float): Differencing order.
        num_threads (int, optional): Number of processes for parallel computation. Defaults to 1.
    
    Returns:
        list[pd.Series]: List of fractionally differenced series.
    """
    
    if num_threads == 1:
        # Sequential processing
        return [get_frac_diff(series, d) for series in series_list]
    else:
        # Parallel processing
        with ProcessPoolExecutor(max_workers=num_threads) as executor:
            results = list(executor.map(partial(get_frac_diff, d=d), series_list))
        return results

def get_frac_diff_df(
  df: pd.DataFrame,
  cols: list[str],
  d: float,
  prefix: str = '',
  postfix: str = '_frac_diff',
  num_threads: int = 1,
) -> pd.DataFrame:
    """Compute fractionally differenced DataFrame.
    
    Args:
        df (pd.DataFrame): Original DataFrame.
        cols (list[str]): Columns to be differenced.
        d (float): Differencing order.
        prefix (str, optional): Prefix for new column names. Defaults to ''.
        postfix (str, optional): Postfix for new column names. Defaults to '_frac_diff'.
        num_threads (int, optional): Number of processes for parallel computation. Defaults to 1.
    
    Returns:
        pd.DataFrame: Fractionally differenced DataFrame.
    """
    
    frac_diff_df = df.copy()
    
    if num_threads == 1:
        # Sequential processing
        for col in cols:
            frac_diff_df[f"{prefix}{col}{postfix}"] = get_frac_diff(df[col], d)
    else:
        # Parallel processing
        process_func = partial(_process_column, df=df, d=d, prefix=prefix, postfix=postfix)
        with ProcessPoolExecutor(max_workers=num_threads) as executor:
            results = executor.map(process_func, cols)
            for new_col_name, frac_diff_series in results:
                frac_diff_df[new_col_name] = frac_diff_series
    
    return frac_diff_df
