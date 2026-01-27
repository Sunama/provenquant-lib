from datetime import datetime
import pandas as pd

def get_entry_row(
    dp,
    pair: str,
    timeframe: str,
    entry_time: datetime
) -> pd.DataFrame:
    """Retrieve the dataframe row corresponding to the trade entry time.
    
    Args:
        dp (object): Data provider object with method get_analyzed_dataframe.
        pair (str): Trading pair.
        timeframe (str): Timeframe of the data.
        entry_time (datetime): Entry time of the trade.
        
    Returns:
        pd.DataFrame: DataFrame row corresponding to the entry time.
    """
    dataframe, _ = dp.get_analyzed_dataframe(pair=pair, timeframe=timeframe)
    entry_row = dataframe.iloc[(dataframe['date'] - entry_time).abs().argsort()[:1]]
    
    return entry_row

def is_exit_with_vertical_barrier(
    dp: object,
    pair: str,
    timeframe: str,
    trade: object,
    current_time: datetime,
    vertical_barrier_col: str = 'vertical_barrier',
) -> bool:
    """Determine if the trade should exit based on the vertical barrier.

    Args:
        dp (object): Data provider object with method get_analyzed_dataframe.
        pair (str): Trading pair.
        timeframe (str): Timeframe of the data.
        trade (object): Trade object containing trade details.
        current_time (datetime): Current time to check against the vertical barrier.
        vertical_barrier_col (str, optional): Column name for the vertical barrier in the dataframe. Defaults to 'vertical_barrier'.

    Returns:
        bool: True if the trade should exit based on the vertical barrier, False otherwise.
    """
    entry_row = get_entry_row(
        dp=dp,
        pair=pair,
        timeframe=timeframe,
        entry_time=trade.open_date_utc,
    )
    
    current_time_ts = pd.to_datetime(current_time)
    if current_time_ts.tzinfo is None:
        current_time_ts = current_time_ts.tz_localize('UTC')
    vertical_barrier = pd.Timestamp(entry_row[vertical_barrier_col].values[0])
    
    if vertical_barrier.tzinfo is None:
        vertical_barrier = vertical_barrier.tz_localize('UTC')
    
    if current_time_ts > vertical_barrier:
      return True
  
    return False
