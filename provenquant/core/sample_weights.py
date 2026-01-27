import pandas as pd

def absolute_return_attribution(
    dataframe: pd.DataFrame,
    vertical_barrier_col: str,
    return_col: str,
    molecule_col: str,
) -> pd.Series:
    """Compute absolute return attribution class weights for given vertical barriers.
    
    Args:
        dataframe (pd.DataFrame): DataFrame containing the data.
        vertical_barrier_col (str): Column name for vertical barriers (datetime).
        return_col (str): Column name for returns.
        molecule_col (str): Column name for molecule identifiers.
        
    Returns:
        pd.Series: Series with absolute return attribution class weights.
    """
    weights = pd.Series(index=dataframe.index, dtype=float)
    
    for vb_time in dataframe[vertical_barrier_col].unique():
        vb_mask = dataframe[vertical_barrier_col] == vb_time
        vb_data = dataframe.loc[vb_mask]
        
        pos_mask = vb_data[return_col] > 0
        neg_mask = vb_data[return_col] < 0
        
        n_pos = pos_mask.sum()
        n_neg = neg_mask.sum()
        
        if n_pos > 0:
            weights.loc[vb_data.index[pos_mask]] = 1.0 / n_pos
        if n_neg > 0:
            weights.loc[vb_data.index[neg_mask]] = 1.0 / n_neg
            
    return weights
    

def time_decay(
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
    weights[weights < 0] = 0.0
    
    return weights
