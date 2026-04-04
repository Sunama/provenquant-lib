from scipy.stats import kurtosis, skew, norm
import numpy as np
import pandas as pd

def calculate_psr(
    scores: list,
    target_sharpe: float = 0.0,
) -> float:
    """Compute the Probabilistic Sharpe Ratio (PSR) for a given list of scores.

    Args:
        scores (list): List of scores to evaluate.
        target_sharpe (float, optional): The target Sharpe Ratio to compare against. Defaults to 0.0.
    Returns:
        float: The computed PSR value.
    """
    n = len(scores)
    if n < 3:
        return 0.0  # Need enough samples for Skewness and Kurtosis

    mean_return = np.mean(scores)
    std_return = np.std(scores)

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

def calculate_psi(
    expected: pd.Series,
    actual: pd.Series,
    bins: int = 10,
) -> float:
    """Calculate the Population Stability Index (PSI) between two distributions.
    
    Args:
        expected (pd.Series): The expected distribution (e.g., historical data).
        actual (pd.Series): The actual distribution (e.g., current data).
        bins (int, optional): The number of bins to use for the histogram. Defaults to 10.
    Returns:
        float: The calculated PSI value.
    """
    try:
        _, bin_edges = pd.qcut(expected, q=bins, retbins=True, duplicates='drop')
    except ValueError:
        bin_edges = np.histogram_bin_edges(expected, bins=bins)

    bin_edges[0] = min(expected.min(), actual.min(), bin_edges[0])
    bin_edges[-1] = max(expected.max(), actual.max(), bin_edges[-1])

    expected_counts, _ = np.histogram(expected, bins=bin_edges)
    actual_counts, _ = np.histogram(actual, bins=bin_edges)

    expected_percents = expected_counts / len(expected)
    actual_percents = actual_counts / len(actual)

    epsilon = 1e-10
    expected_percents = np.where(expected_percents == 0, epsilon, expected_percents)
    actual_percents = np.where(actual_percents == 0, epsilon, actual_percents)

    psi_values = (actual_percents - expected_percents) * np.log(actual_percents / expected_percents)
    
    return float(np.sum(psi_values))
