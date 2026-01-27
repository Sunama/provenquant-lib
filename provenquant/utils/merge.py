import pandas as pd
import numpy as np

def match_merge_dataframe(
    from_df: pd.DataFrame,
    to_df: pd.DataFrame,
    from_cols: list,
    from_datetime_col: str,
    to_datetime_col: str,
) -> pd.DataFrame:
    """Merge columns from one DataFrame into another based on matching datetime values.

    Args:
        from_df (pd.DataFrame): DataFrame to merge data from.
        to_df (pd.DataFrame): DataFrame to merge data into.
        from_cols (list): List of column names to merge from from_df.
        from_datetime_col (str): Datetime column name in from_df.
        to_datetime_col (str): Datetime column name in to_df.

    Returns:
        pd.DataFrame: Merged DataFrame with selected columns from from_df.
    """
    merged_data = {col: [] for col in from_cols}

    from_datetimes = from_df[from_datetime_col].values
    to_datetimes = to_df[to_datetime_col].values

    for to_dt in to_datetimes:
        mask = from_datetimes == to_dt
        if np.any(mask):
            for col in from_cols:
                merged_data[col].append(from_df.loc[mask, col].values[0])
        else:
            for col in from_cols:
                merged_data[col].append(np.nan)

    for col in from_cols:
        to_df[col] = merged_data[col]

    return to_df

def match_merge_series(
    from_series: pd.Series,
    to_df: pd.DataFrame,
    to_datetime_col: str,
    series_name: str,
) -> pd.DataFrame:
    """Merge a Series into a DataFrame based on matching datetime values.

    Args:
        from_series (pd.Series): Series to merge data from.
        to_df (pd.DataFrame): DataFrame to merge data into.
        to_datetime_col (str): Datetime column name in to_df.
        series_name (str): Name of the Series to be used as column name in to_df.

    Returns:
        pd.DataFrame: Merged DataFrame with the Series added as a new column.
    """
    dataframe = to_df.copy()
    
    if to_datetime_col != 'index':
        dataframe = pd.merge(
            dataframe,
            from_series.rename(series_name),
            left_on=to_datetime_col,
            right_index=True,
            how='left'
        )
    else:
        dataframe = pd.merge(
            dataframe,
            from_series.rename(series_name),
            left_index=True,
            right_index=True,
            how='left'
        )
    
    return dataframe

def larger_timeframe_merge_to_smaller_timeframe_dataframe(
    large_tf_df: pd.DataFrame,
    small_tf_df: pd.DataFrame,
    large_tf_datetime_col: str,
    small_tf_datetime_col: str,
    large_tf_cols: list,
) -> pd.DataFrame:
    """Merge data from a larger timeframe DataFrame into a smaller timeframe DataFrame.

    Args:
        large_tf_df (pd.DataFrame): DataFrame with larger timeframe data.
        small_tf_df (pd.DataFrame): DataFrame with smaller timeframe data.
        large_tf_datetime_col (str): Datetime column name in large_tf_df.
        small_tf_datetime_col (str): Datetime column name in small_tf_df.
        large_tf_cols (list): List of column names to merge from large_tf_df.

    Returns:
        pd.DataFrame: Merged DataFrame with data from large_tf_df added to small_tf_df.
    """
    small_df = small_tf_df.copy()
    large_df = large_tf_df.copy()

    # Keep original order, merge on sorted datetimes. We align each small row with the
    # last fully completed large interval to avoid leakage from the in-progress large bar.
    small_df['_orig_order'] = np.arange(len(small_df))
    small_sorted = small_df.sort_values(small_tf_datetime_col)

    large_sorted = large_df.sort_values(large_tf_datetime_col).copy()
    # A large bar's value becomes available only at the start of the next large bar.
    large_sorted['_available_at'] = large_sorted[large_tf_datetime_col].shift(-1)
    # Drop the last large bar because it has no subsequent timestamp (not completed yet).
    large_ready = large_sorted.dropna(subset=['_available_at'])

    merged = pd.merge_asof(
        small_sorted,
        large_ready[['_available_at'] + large_tf_cols],
        left_on=small_tf_datetime_col,
        right_on='_available_at',
        direction='backward',
        allow_exact_matches=True,
    )

    merged = merged.sort_values('_orig_order').drop(columns=['_orig_order', '_available_at'])

    return merged

def smaller_timeframe_merge_sum_to_larger_timeframe_dataframe(
    small_tf_df: pd.DataFrame,
    large_tf_df: pd.DataFrame,
    small_tf_datetime_col: str,
    large_tf_datetime_col: str,
    small_tf_cols: list,
) -> pd.DataFrame:
    """Aggregate data from a smaller timeframe DataFrame into a larger timeframe DataFrame by summing.

    Args:
        small_tf_df (pd.DataFrame): DataFrame with smaller timeframe data.
        large_tf_df (pd.DataFrame): DataFrame with larger timeframe data.
        small_tf_datetime_col (str): Datetime column name in small_tf_df.
        large_tf_datetime_col (str): Datetime column name in large_tf_df.
        small_tf_cols (list): List of column names to aggregate from small_tf_df.

    Returns:
        pd.DataFrame: Merged DataFrame with aggregated data from small_tf_df added to large_tf_df.
    """
    large_df = large_tf_df.copy()
    small_df = small_tf_df.copy()

    # Create intervals for aggregation
    large_df['_start'] = large_df[large_tf_datetime_col]
    large_df['_end'] = large_df[large_tf_datetime_col].shift(-1)

    # Infer a reasonable end for the last interval using the typical large timeframe frequency
    diffs = large_df[large_tf_datetime_col].diff().dropna()
    if len(diffs) > 0:
        try:
            # Prefer mode if regular; fallback to median
            freq = diffs.mode().iloc[0]
        except Exception:
            freq = diffs.median()
    else:
        freq = pd.Timedelta(0)

    large_df['_end'] = large_df['_end'].fillna(large_df['_start'] + freq)

    aggregated_data = {col: [] for col in small_tf_cols}

    for _, large_row in large_df.iterrows():
        start = large_row['_start']
        end = large_row['_end']

        # Include data >= start and < end
        mask = (small_df[small_tf_datetime_col] >= start) & (small_df[small_tf_datetime_col] < end)
        for col in small_tf_cols:
            aggregated_value = small_df.loc[mask, col].sum()
            aggregated_data[col].append(aggregated_value)

    for col in small_tf_cols:
        large_df[col] = aggregated_data[col]

    large_df = large_df.drop(columns=['_start', '_end'])

    return large_df
