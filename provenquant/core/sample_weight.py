import pandas as pd
import numpy as np

def compute_time_decay(
    series: pd.Series,
    last_weight: float=1.0,
) -> pd.Series:
    """Apply piecewise-linear decay to observed uniqueness

    Args:
        series (pd.Series): Input series to apply decay on. Note: normally this would be closed prices.
        last_weight (float, optional): Weight to assign to the last element in the series.
                                       Defaults to 1.0.

    Returns:
        pd.Series: Series with time-decayed weights applied.
    """
    weights = series.cumsum()
    if last_weight >= 0:
        slope = (1.0 - last_weight) / weights.iloc[-1]
    else:
        slope = 1.0 / ((last_weight + 1) * weights.iloc[-1])
        
    c = 1.0 - slope * weights.iloc[-1]
    weights = weights * slope + c
    weights[weights < 0] = 0.0
    
    return weights

def compute_time_decay_exponential(
    series: pd.Series,
    last_weight: float=1.0,
    decay_factor: float=0.5,
) -> pd.Series:
    """Apply exponential decay to observed uniqueness

    Args:
        series (pd.Series): Input series to apply decay on. Note: normally this would be closed prices.
        last_weight (float, optional): Weight to assign to the last element in the series.
                                       Defaults to 1.0.
        decay_factor (float, optional): Decay factor to control the rate of decay. Defaults to 0.5.

    Returns:
        pd.Series: Series with exponential time-decayed weights applied.
    """
    weights = series.cumsum()
    total = weights.iloc[-1]
    
    # Calculate exponential decay where newest observation has weight 1.0
    # and older ones decay according to the decay_factor and last_weight (floor)
    decayed_weights = (1.0 - last_weight) * np.exp(-decay_factor * (total - weights)) + last_weight
    
    return decayed_weights.clip(lower=0.0)

def compute_abs_return_uniqueness(
    dataframe: pd.DataFrame,
    return_col: str = 'return',
    uniqueness_col: str = 'uniqueness',
    normalize: bool = True,
    min_weight: float = 1e-6,
) -> pd.Series:
    """Compute absolute return * uniqueness (u_i) for each event.
    
    Based on the method from "Advances in Financial Machine Learning".
    
    Args:
        dataframe (pd.DataFrame): DataFrame containing the events.
        return_col (str): Column name for returns. Defaults to 'return'.
        uniqueness_col (str): Column name for uniqueness values. Defaults to 'uniqueness'.
        normalize (bool): Whether to normalize uniqueness values. Defaults to True.
        min_weight (float): Minimum weight to assign to any event. Defaults to 1e-6.
                                     
    Returns:
        pd.Series: Series with absolute return * uniqueness for each event.
    """
    abs_returns = dataframe[return_col].abs()
    uniqueness = dataframe[uniqueness_col]
    
    weights = abs_returns * uniqueness
    
    weights = weights.replace([0, np.inf, -np.inf], np.nan).fillna(min_weight)
    weights = weights.clip(lower=min_weight)
    
    if normalize:
        weights = weights / weights.mean()
    
    return weights

def compute_average_uniqueness(
    dataframe: pd.DataFrame,
    t1_col: str = 't1',
    datetime_col: str = 'index',
) -> pd.Series:
    """Compute average uniqueness (u_i) for each event.
    
    Based on the standard method from "Advances in Financial Machine Learning".
    
    Args:
        dataframe (pd.DataFrame): DataFrame containing the events.
        t1_col (str): Column name for the end time of events (vertical barrier). Defaults to 't1'.
        datetime_col (str): Column name for the start time of events. Defaults to 'index'.
                                     
    Returns:
        pd.Series: Series with average uniqueness for each event.
    """
    # Create list of event spans: (start, end)
    if datetime_col == 'index':
        start_times = dataframe.index
    else:
        start_times = dataframe[datetime_col]
    
    events = list(zip(start_times, dataframe[t1_col]))
    
    # Identify all unique timestamps involved to build a reference timeline
    all_times = sorted(set(dataframe[t1_col]).union(set(start_times)))
    
    # Create concurrency series
    concurrency = pd.Series(0, index=all_times)
    
    # Track event overlaps at each timestamp
    for t0, t1 in events:
        if pd.isna(t1) or t1 <= t0:
            continue
        concurrency.loc[t0:t1] += 1
    
    # Calculate average uniqueness for each event
    uniqueness = []
    for t0, t1 in events:
        if pd.isna(t1) or t1 <= t0:
            uniqueness.append(0.0)
            continue
            
        span = concurrency.loc[t0:t1]
        if len(span) == 0:
            uniqueness.append(0.0)
            continue
            
        # Average uniqueness = mean(1 / concurrency) during the event's lifespan
        u_i = (1.0 / span).mean()
        uniqueness.append(u_i)
    
    return pd.Series(uniqueness, index=dataframe.index)


