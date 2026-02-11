from provenquant.core.feature_selection import stationary_test
from scipy import stats
import numpy as np
import pandas as pd
import optuna

def evaluate_triple_barrier(
    path: list,
    tp: float,
    sl: float,
    vb: int
) -> tuple[float, int]:
    """Return PnL and holding bars when the first barrier is hit (path starts at 0)
    
    Args:
        path (list): List of PnL values.
        tp (float): Take profit barrier.
        sl (float): Stop loss barrier.
        vb (int): Vertical barrier in bars.
        
    Returns:
        tuple: (PnL at barrier hit, holding bars)
    """
    for i in range(1, len(path)):
        pnl = path[i]  # assume normalized to start at 0
        if pnl >= tp:
            return tp, i
        if pnl <= sl:
            return sl, i
        if i >= vb:
            return pnl, i
    return path[-1], len(path) - 1  # If no barrier is hit

def filtrate_tripple_label_barrier(
    dataframe: pd.DataFrame,
    cusum_threshold: float,
    vertical_barrier: int,
    datetime_col: str = 'index',
) -> pd.DataFrame:
    """Filtrate triple barrier labels from raw DataFrame.
       Use this function before applying triple barrier labeling or in
       production based that we don't need labels and returns yet.

    Args:
        dataframe (pd.DataFrame): Raw DataFrame that contains close prices.
        cusum_threshold (float): Threshold for CUSUM filter in percentage.
        vertical_barrier (int): Ticks for vertical barrier.
        datetime_col (str): Name of the datetime column. Defaults to 'index'.

    Returns:
        pd.DataFrame: DataFrame with t1.
    """
    # CUSUM Filter
    if datetime_col != 'index':
        close_prices = dataframe.set_index(datetime_col)['close']
    else:
        close_prices = dataframe['close']
    
    diff = close_prices.pct_change().dropna()
    
    pos_cusum, neg_cusum = 0, 0
    t_events = []
    for idx in diff.index[1:]:
        pos_cusum = max(0, pos_cusum + diff.loc[idx])
        neg_cusum = min(0, neg_cusum + diff.loc[idx])
        
        if pos_cusum > cusum_threshold:
            t_events.append(idx)
            pos_cusum = 0
        elif neg_cusum < -cusum_threshold:
            t_events.append(idx)
            neg_cusum = 0
    t_events = pd.DatetimeIndex(t_events)
    
    # Vertical Barrier
    # Build t1 values using a list first, then create Series with proper dtype
    t1_values = []
    for event_time in t_events:
        t1_value = close_prices.index[
            close_prices.index.get_loc(event_time) + vertical_barrier
            ] if (close_prices.index.get_loc(event_time) + vertical_barrier) < len(close_prices) else close_prices.index[-1]
        t1_values.append(t1_value)
    
    t1 = pd.Series(t1_values, index=t_events, dtype=close_prices.index.dtype)
    df = pd.DataFrame(index=t_events)
    df['t1'] = t1
    
    # Add another columns in dataframe to df
    if datetime_col == 'index':
        # datetime is already the index
        for col in dataframe.columns:
            df[col] = dataframe.loc[t_events][col]
    else:
        # datetime is a column, need to set it as index first
        for col in dataframe.columns:
            if col != datetime_col:
                df[col] = dataframe.set_index(datetime_col).loc[t_events][col]
    
    return df

def fit_ou_ols(series: pd.Series, dt: float = 1.0):
    """Fit OU on series
    
    Args:
        series (pd.Series): Series to fit.
        dt (float): Time step size. Defaults to 1.0.
        
    Returns:
        tuple: kappa, theta, sigma
    
    """
    # Test stationarity ก่อน
    result = stationary_test(series)
    if not result:
        print("Warning: Series is not stationary. OU fit may be invalid.")

    X = series.values[:-1]
    dX = series.values[1:] - series.values[:-1]
    if len(X) < 2:
        raise ValueError("Series is too short to fit OU process.")
    
    slope, intercept, r_value, p_value, std_err = stats.linregress(X, dX)
    kappa = -slope / dt
    theta = intercept / (kappa * dt) if kappa != 0 else 0
    residuals = dX - (intercept + slope * X)
    sigma = np.std(residuals) / np.sqrt(dt)

    return kappa, theta, sigma

def get_tripple_label_barrier(
    dataframe: pd.DataFrame,
    close_series: pd.Series,
    tp: float = 0.02,
    sl: float = 0.01,
) -> pd.DataFrame:
    """Get triple barrier labels from DataFrame with t1.

    Args:
        dataframe (pd.DataFrame): DataFrame with t1.
        close_series (pd.Series): Series of close prices that have datetime index.
        tp (float): Profit taking percentage. Defaults to 2%.
        sl (float): Stop loss percentage. Defaults to 1%.
        
    Returns:
        pd.DataFrame: DataFrame with labels, mapped_labels, returns, max_returns and
        min_returns.
    """
    labels = []
    returns = []
    max_returns = []
    min_returns = []
    
    for event_time, row in dataframe.iterrows():
        t1 = row['t1']
        if pd.isna(t1):
            labels.append(0)
            returns.append(0)
            max_returns.append(0)
            min_returns.append(0)
            continue
        
        start_time = event_time
        end_time = t1
        start_price = close_series.loc[start_time]
        exited = False
        
        window_prices = close_series.loc[start_time:end_time]
        window_returns = (window_prices - start_price) / start_price
        max_returns.append(window_returns.max())
        min_returns.append(window_returns.min())

        for t, price in window_prices.items():
            ret = (price - start_price) / start_price
            
            if ret > tp:
                labels.append(1)
                returns.append(ret)
                exited = True
                break
            elif ret < -sl:
                labels.append(-1)
                returns.append(ret)
                exited = True
                break
        
        if not exited:
            end_price = close_series.loc[end_time]
            ret = (end_price - start_price) / start_price
            labels.append(0)
            returns.append(ret)
    
    dataframe['label'] = labels
    dataframe['return'] = returns
    dataframe['max_return'] = max_returns
    dataframe['min_return'] = min_returns
    dataframe['mapped_label'] = dataframe['label'].map({1: 2, 0: 1, -1: 0})
    
    return dataframe

def optimize_triple_barriers(
    kappa: float,
    theta: float,
    sigma: float,
    x0: float = 0.0,
    n_paths: int = 5000,
    n_steps: int = 2000,
    T: float = 500.0,
    n_trials: int = 100,
    show_progress: bool = False
) -> tuple[float, float, int, float]:
    """Optimize triple barrier parameters using simulated OU process with Optuna.
    
    Args:
        kappa (float): Mean reversion speed.
        theta (float): Long-term mean.
        sigma (float): Volatility.
        x0 (float): Initial value. Defaults to 0.0.
        n_paths (int): Number of simulated paths. Defaults to 5000.
        n_steps (int): Number of time steps. Defaults to 2000.
        T (float): Total time. Defaults to 500.0.
        n_trials (int): Number of optimization trials. Defaults to 100.
        show_progress (bool): Whether to show progress bar. Defaults to False.
        
    Returns:
        tuple: Optimal (tp, sl, vb)
    """
    t, paths = simulate_ou_process(kappa=kappa, theta=theta, sigma=sigma, x0=x0,
                                   T=T, n_steps=n_steps, n_paths=n_paths)
    
    def objective(trial):
        # Suggest parameters
        tp = trial.suggest_float('tp', 1.0 * sigma, 50 * sigma)
        sl = trial.suggest_float('sl', -50 * sigma, -1.0 * sigma)
        vb = trial.suggest_int('vb', 10, 100, step=5)
        
        if abs(sl) > tp:
            return -np.inf  # Invalid case
        
        # Evaluate parameters
        returns = []
        holdings = []
        for p in paths.T:
            ret, hold = evaluate_triple_barrier(p, tp, sl, vb)
            returns.append(ret)
            holdings.append(hold)
        
        mean_ret = np.mean(returns)
        std_ret = np.std(returns) if np.std(returns) > 0 else 1e-6
        sharpe = mean_ret / std_ret  # rf=0
        
        return sharpe
    
    # Create study and optimize
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=n_trials, show_progress_bar=show_progress)
    
    # Get best trial
    best_trial = study.best_trial
    best_tp = best_trial.params['tp']
    best_sl = best_trial.params['sl']
    best_vb = best_trial.params['vb']
    
    # Calculate average holdings for best params
    holdings = []
    for p in paths.T:
        _, hold = evaluate_triple_barrier(p, best_tp, best_sl, best_vb)
        holdings.append(hold)
    
    best_sharpe = best_trial.value
    avg_hold = np.mean(holdings)
    
    print(f"Optimal: TP={best_tp:.4f} ({best_tp/sigma:.2f}σ), "
          f"SL={best_sl:.4f} ({best_sl/sigma:.2f}σ), "
          f"VB={best_vb} bars, Avg hold={avg_hold:.1f} bars")
    print(f"Best Sharpe: {best_sharpe:.4f}")
    
    return best_tp, abs(best_sl), best_vb

def simulate_ou_process(
    theta: float = 0.0,
    kappa: float = 1.0,
    sigma: float = 0.5,
    x0: float = 0.0,
    T: float = 1.0,
    n_steps: int = 1000,
    n_paths: int = 1,
    seed: int = 42
):
    """Simulate discrete Ornstein-Uhlenbeck process using Euler-Maruyama scheme.
       dX_t = kappa * (theta - X_t) dt + sigma dW_t
    
    Args:
        theta (float): Equilibrium value (long-term mean).
        kappa (float): Speed of mean reversion.
        sigma (float): Volatility of the noise.
        x0 (float): Initial value of the process.
        T (float): Total time.
        n_steps (int): Number of time steps.
        n_paths (int): Number of paths to simulate.
        seed (int): Random seed for reproducibility.
        
    Returns:
        tuple: (time array, paths array of shape [n_steps+1, n_paths])
    """
    np.random.seed(seed)
    
    dt = T / n_steps                  # ขนาด time step
    t = np.linspace(0, T, n_steps+1)  # เวลา array
    
    # สร้าง array สำหรับเก็บ paths (shape: [n_steps+1, n_paths])
    paths = np.zeros((n_steps + 1, n_paths), dtype=np.float64)
    paths[0, :] = x0
    
    # Simulate แต่ละ path
    for i in range(1, n_steps + 1):
        dW = np.random.normal(0, np.sqrt(dt), size=n_paths)  # Brownian increment
        drift = np.clip(kappa * (theta - paths[i-1, :]) * dt, -1e10, 1e10)
        diffusion = sigma * dW
        new_value = paths[i-1, :] + drift + diffusion
        # ป้องกัน NaN values
        paths[i, :] = np.where(np.isfinite(new_value), new_value, paths[i-1, :])
    
    return t, paths
