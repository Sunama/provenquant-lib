import pandas as pd

def get_frac_diff(
  series: pd.Series,
  d: float,
) -> pd.Series:
    """Compute fractionally differenced series.
    
    Args:
        series (pd.Series): Original time series.
        d (float): Differencing order.
    
    Returns:
        pd.Series: Fractionally differenced series.
    """
    # Compute weights for fractionally differencing
    w = [1.0]
    for k in range(1, len(series)):
        w.append(-w[-1] * (d - k + 1) / k)
    w = pd.Series(w).fillna(0)

    # Apply weights to the series
    frac_diff_series = pd.Series(index=series.index)
    for i in range(len(series)):
        if i < len(w):
            frac_diff_series.iloc[i] = (w[:i+1][::-1] * series.iloc[:i+1]).sum()
        else:
            frac_diff_series.iloc[i] = (w * series.iloc[i-len(w)+1:i+1]).sum()
    
    return frac_diff_series
