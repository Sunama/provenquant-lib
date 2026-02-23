import numpy as np
import pandas as pd
from scipy.stats import kurtosis, skew, norm

def compute_psr(
    scores: pd.Series,
    target_sharpe: float = 0.0,
) -> float:
    """Compute the Probabilistic Sharpe Ratio (PSR) for a given series of scores.

    Args:
        scores (pd.Series): Series of scores to evaluate.
        target_sharpe (float, optional): The target Sharpe Ratio to compare against. Defaults to 0.0.
    Returns:
        float: The computed PSR value.
    """
    n = len(scores)
    if n < 3:
        return 0.0  # Need enough samples for Skewness and Kurtosis

    mean_return = scores.mean()
    std_return = scores.std(ddof=1)

    if std_return == 0:
        return 0.0

    sr = mean_return / std_return
    skewness = skew(scores)
    kurt = kurtosis(scores, fisher=True)  # Excess Kurtosis

    # Standard Error of Sharpe Ratio (Bailey & Lopez de Prado)
    sr_std = np.sqrt((1 + 0.5 * sr**2 - skewness * sr + ((kurt + 3) / 4) * sr**2) / (n - 1))

    # Compute PSR (Probability that SR > target_sharpe)
    psr_value = norm.cdf((sr - target_sharpe) / sr_std)

    return float(psr_value)
