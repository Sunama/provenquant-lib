import pandas as pd
import numpy as np

def get_volatility(
    dataframe: pd.DataFrame,
    span: int = 100,
) -> pd.Series:
    """Calculate the volatility of close prices using exponentially weighted standard deviation.

    Args:
        dataframe (pd.DataFrame): DataFrame that contains close prices.
        span (int, optional): Span for the EWMA calculation. Defaults to 100.

    Returns:
        pd.Series: Series containing the volatility values.
    """
    close_prices = dataframe['close']
    volatility = close_prices.pct_change().ewm(span=span).std()
    
    return volatility
