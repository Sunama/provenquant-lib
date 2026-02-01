from provenquant.core.cross_validation import PurgedKFold
from sklearn.metrics import accuracy_score, log_loss, roc_auc_score
from sklearn.model_selection import train_test_split
from statsmodels.tools.sm_exceptions import InterpolationWarning
from statsmodels.tsa.stattools import adfuller, kpss
from tqdm import tqdm
import numpy as np
import pandas as pd
import warnings

def _cv_score(
    model: object,
    X: np.ndarray,
    y: np.ndarray,
    sample_weight: np.ndarray = None,
    n_splits: int = 5,
    purge: int = 0,
    embargo: int = 0,
    scoring: str = 'neg_log_loss'
) -> list[float]:
    """Calculate cross-validated score for a given model.

    Args:
        model (object): Trained model with fit and predict methods.
        X (np.ndarray): Feature matrix.
        y (np.ndarray): Target vector.
        sample_weight (np.ndarray, optional): Sample weights. Defaults to None.
        n_splits (int, optional): Number of splits for cross-validation. Defaults to 5.
        purge (int, optional): Purge size for Purged K-Fold. Defaults to 0.
        embargo (int, optional): Embargo size for Purged K-Fold. Defaults to 0.
        scoring (str, optional): Scoring metric. That are 'neg_log_loss' and 'accuracy'
                                 supported.
                                 Defaults to 'neg_log_loss'.
        
    Returns:
        list[float]: List of cross-validated scores.
    """
    scores = []

    pkf = PurgedKFold(
        n_splits=n_splits,
        purge=purge,
        embargo=embargo
    )

    for train_index, test_index in pkf.split(X):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        if sample_weight is not None:
            sw_train = sample_weight[train_index]
            sw_test = sample_weight[test_index]
        else:
            sw_train = None
            sw_test = None

        model.fit(X_train, y_train, sample_weight=sw_train)
        if scoring == 'neg_log_loss':
            y_pred = model.predict_proba(X_test)
            score = -log_loss(y_test, y_pred, sample_weight=sw_test)
        elif scoring == 'accuracy':
            y_pred = model.predict(X_test)
            score = accuracy_score(y_test, y_pred, sample_weight=sw_test)
        elif scoring == 'auc':
            y_pred = model.predict_proba(X_test)
            if y_pred.shape[1] == 2:
                y_pred = y_pred[:, 1]
                score = roc_auc_score(y_test, y_pred, sample_weight=sw_test)
            else:
                score = roc_auc_score(y_test, y_pred, multi_class='ovr', sample_weight=sw_test)
        else:
            raise ValueError("Unsupported scoring method")

        scores.append(score)

    return scores

def backward_feature_elimination(
    model: object,
    dataframe: pd.DataFrame,
    feature_cols: list,
    threshold: float = 0.55,
    verbose: bool = False
) -> list:
    """Perform backward feature elimination based on cross-validated scores.

    Args:
        model (object): Trained model with fit and predict methods.
        dataframe (pd.DataFrame): DataFrame containing features and target.
        feature_cols (list): List of feature column names.
        threshold (float, optional): Minimum improvement threshold to keep a feature.
                                     Defaults to 0.55.
        verbose (bool, optional): Whether to print progress. Defaults to False.
    
    Returns:
        list: List of selected feature column names.
    """
    selected_features = feature_cols.copy()

    while len(selected_features) > 1:
        X = dataframe[selected_features].values
        y = np.zeros(len(dataframe))
        y[int(0.2 * len(dataframe)):] = 1
        
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2)
        
        model.fit(X_train, y_train)
        y_pred = model.predict(X_val)
        
        score = roc_auc_score(y_val, y_pred)
        
        if score < threshold:
            if verbose:
                print(f"Stopping elimination. Score {score:.4f} below threshold {threshold:.4f}.")
            break
        
        feature_importance = model.feature_importances_
        most_important_idx = np.argmax(feature_importance)
        feature_to_remove = selected_features[most_important_idx]
        selected_features.remove(feature_to_remove)
        if verbose:
            print(f"Score {score:.4f}. Removed feature: {feature_to_remove}")
        
    return selected_features
    

def calculate_mda_feature_importances(
    model: object,
    dataframe: pd.DataFrame,
    feature_cols: list,
    target_col: str,
    sample_weight_col: str = None,
    n_splits: int = 5,
    purge: int = 0,
    embargo: int = 0,
    scoring: str = 'neg_log_loss'
) -> pd.DataFrame:
    """Calculate feature importance using Mean Decrease in Accuracy (MDA)

    Args:
        model (object): Trained model with fit and predict methods.
        dataframe (pd.DataFrame): DataFrame containing features and target.
        feature_cols (list): List of feature column names.
        target_col (str): Target column name.
        sample_weight_col (str, optional): Sample weight column name. Defaults to None.
        n_splits (int, optional): Number of splits for cross-validation. Defaults to 5.
        purge (int, optional): Purge size for Purged K-Fold. Defaults to 0.
        embargo (int, optional): Embargo size for Purged K-Fold. Defaults to 0.
        scoring (str, optional): Scoring metric. That are 'neg_log_loss', 'accuracy',
                                 and 'auc' supported.
                                 Defaults to 'neg_log_loss'.
        
    Returns:
        pd.DataFrame: DataFrame containing feature importance scores with columns 
                      'feature_importances' and 'mean_score'.
    """
    X = dataframe[feature_cols].values
    y = dataframe[target_col].values
    if sample_weight_col:
        sample_weight = dataframe[sample_weight_col].values
    else:
        sample_weight = None
    
    n_features = X.shape[1]
    feature_importances = np.zeros(n_features)
    mean_scores = np.zeros(n_features)

    pkf = PurgedKFold(
        n_splits=n_splits,
        purge=purge,
        embargo=embargo
    )

    for train_index, test_index in pkf.split(X):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        if sample_weight is not None:
            sw_train = sample_weight[train_index]
            sw_test = sample_weight[test_index]
        else:
            sw_train = None
            sw_test = None

        model.fit(X_train, y_train, sample_weight=sw_train)
        if scoring == 'neg_log_loss':
            y_pred = model.predict_proba(X_test)
            baseline_score = -log_loss(y_test, y_pred, sample_weight=sw_test)
        elif scoring == 'accuracy':
            y_pred = model.predict(X_test)
            baseline_score = accuracy_score(y_test, y_pred, sample_weight=sw_test)
        elif scoring == 'auc':
            y_pred = model.predict_proba(X_test)[:, 1]
            baseline_score = roc_auc_score(y_test, y_pred, sample_weight=sw_test)
        else:
            raise ValueError("Unsupported scoring method")

        for feature_idx in range(n_features):
            X_test_permuted = X_test.copy()
            np.random.shuffle(X_test_permuted[:, feature_idx])

            if scoring == 'neg_log_loss':
                y_pred_permuted = model.predict_proba(X_test_permuted)
                permuted_score = -log_loss(y_test, y_pred_permuted, sample_weight=sw_test)
            elif scoring == 'accuracy':
                y_pred_permuted = model.predict(X_test_permuted)
                permuted_score = accuracy_score(y_test, y_pred_permuted, sample_weight=sw_test)

            feature_importances[feature_idx] += baseline_score - permuted_score
            mean_scores[feature_idx] += baseline_score

    feature_importances /= n_splits
    mean_scores /= n_splits

    result_df = pd.DataFrame({
        'feature_importances': feature_importances,
        'mean_score': mean_scores
    }, index=feature_cols)

    return result_df.sort_values(by='feature_importances', ascending=False)
    
def calculate_sfi_feature_importances(
    model: object,
    dataframe: pd.DataFrame,
    feature_cols: list,
    target_col: str,
    sample_weight_col: str = None,
    n_splits: int = 5,
    purge: int = 0,
    embargo: int = 0,
    scoring: str = 'neg_log_loss',
    show_progress: bool = False
) -> pd.DataFrame:
    """Calculate feature importance using Single Feature Importance (SFI)
    
    Args:
        model (object): Trained model with fit and predict methods.
        dataframe (pd.DataFrame): DataFrame containing features and target.
        feature_cols (list): List of feature column names.
        target_col (str): Target column name.
        sample_weight_col (str, optional): Sample weight column name. Defaults to None.
        n_splits (int, optional): Number of splits for cross-validation. Defaults to 5.
        purge (int, optional): Purge size for Purged K-Fold. Defaults to 0.
        embargo (int, optional): Embargo size for Purged K-Fold. Defaults to 0.
        scoring (str, optional): Scoring metric. That are 'neg_log_loss' and 'accuracy'
                                 supported.
                                 Defaults to 'neg_log_loss'.
            show_progress (bool, optional): Whether to show progress. Defaults to False.
                                 
    Returns:
        pd.DataFrame: DataFrame containing feature importance scores.
    """
    importances = pd.DataFrame(columns=['mean', 'std'], index=feature_cols)
    iterator = tqdm(feature_cols, desc="Calculating SFI feature importance") if show_progress else feature_cols
    for feature in iterator:
        X_feature = dataframe[[feature]].values
        y = dataframe[target_col].values
        if sample_weight_col:
            sample_weight = dataframe[sample_weight_col].values
        else:
            sample_weight = None

        scores = _cv_score(
            model,
            X_feature,
            y,
            sample_weight=sample_weight,
            n_splits=n_splits,
            purge=purge,
            embargo=embargo,
            scoring=scoring
        )

        importances.loc[feature, 'mean'] = np.mean(scores)
        importances.loc[feature, 'std'] = np.std(scores) * len(scores)**-0.5
        
    return importances.sort_values(by='mean', ascending=False)

def _get_e_vec(dot, threshold) -> tuple[np.ndarray, np.ndarray]:
    """Compute eigen vectors and reduce dimension based on threshold.
    
    Args:
        dot (np.ndarray): Dot product matrix.
        threshold (float): Threshold for eigen values.
        
    Returns:
        np.ndarray: Eigen vectors above the threshold.
    """
    eig_vals, eig_vecs = np.linalg.eig(dot)
    idx = eig_vals.argsort()[::-1]
    eig_vals, eig_vecs = eig_vals[idx], eig_vecs[:, idx]
    
    # Only positive eigen values
    eig_vals = pd.Series(eig_vals, index=[f'PC_{i+1}' for i in range(len(eig_vals))])
    eig_vecs = pd.DataFrame(eig_vecs, index=dot.index, columns=eig_vals.index)
    eig_vecs = eig_vecs.loc[:, eig_vals.index]
    
    # reduce dimension, from PCs
    cum_var = eig_vals.cumsum() / eig_vals.sum()
    dimension = cum_var.values.searchsorted(threshold)
    eig_vals, eig_vecs = eig_vals.iloc[:dimension+1], eig_vecs.iloc[:, :dimension+1]
    
    return eig_vals.values, eig_vecs.values

def orthogonal_features(
    df_X: pd.DataFrame,
    threshold: float = 0.95
) -> np.ndarray:
    """Identify orthogonal features based on correlation threshold.
    
    Args:
        df_X (pd.DataFrame): DataFrame containing feature set.
        threshold (float, optional): Correlation threshold. Defaults to 0.95.
        
    Returns:
        np.ndarray: Indices of orthogonal features.
    """
    df_z = df_X.sub(df_X.mean(), axis=1).div(df_X.std(), axis=1)
    dot = pd.DataFrame(
        np.dot(df_z.T, df_z),
        index=df_X.columns,
        columns=df_X.columns
    )
    eig_vals, eig_vecs = _get_e_vec(dot, threshold)
    df_p = np.dot(df_z, eig_vecs)
    
    return df_p

def stationary_test(
    series: pd.Series,
    alpha: float = 0.05
) -> bool:
    """Perform ADF and KPSS tests to check stationarity of a time series.
    
    Args:
        series (pd.Series): Time series data.
        alpha (float, optional): Significance level. Defaults to 0.05.
        
    Returns:
        bool: True if series is stationary, False otherwise.
    """
    adf_result = adfuller(series.dropna())
    
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=InterpolationWarning)
        kpss_result = kpss(series.dropna(), nlags="auto", regression="c")
    
    adf_stationary = adf_result[1] < alpha
    kpss_stationary = kpss_result[1] >= alpha
    
    return adf_stationary and kpss_stationary
