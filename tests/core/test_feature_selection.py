import pytest
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

from provenquant.core.feature_selection import (
  cv_score,
  feature_importance_mda,
  feature_importance_sfi,
  orthogonal_features,
  _get_e_vec
)


class TestCvScore:
  """Test suite for cv_score function"""
  
  @pytest.fixture
  def sample_data(self):
    """Create sample data for testing"""
    np.random.seed(42)
    X = np.random.randn(100, 5)
    y = np.random.randint(0, 2, 100)
    sample_weight = np.random.rand(100)
    return X, y, sample_weight
  
  def test_cv_score_neg_log_loss(self, sample_data):
    """Test cv_score with neg_log_loss scoring"""
    X, y, _ = sample_data
    model = LogisticRegression(random_state=42)
    scores = cv_score(model, X, y, n_splits=3, scoring='neg_log_loss')
    
    assert len(scores) == 3
    assert all(isinstance(s, float) for s in scores)
    assert all(s < 0 for s in scores)  # neg_log_loss should be negative
  
  def test_cv_score_accuracy(self, sample_data):
    """Test cv_score with accuracy scoring"""
    X, y, _ = sample_data
    model = LogisticRegression(random_state=42)
    scores = cv_score(model, X, y, n_splits=3, scoring='accuracy')
    
    assert len(scores) == 3
    assert all(isinstance(s, float) for s in scores)
    assert all(0 <= s <= 1 for s in scores)  # accuracy should be between 0 and 1
  
  def test_cv_score_with_sample_weight(self, sample_data):
    """Test cv_score with sample weights"""
    X, y, sample_weight = sample_data
    model = LogisticRegression(random_state=42)
    scores = cv_score(model, X, y, sample_weight=sample_weight, 
             n_splits=3, scoring='accuracy')
    
    assert len(scores) == 3
    assert all(isinstance(s, float) for s in scores)
  
  def test_cv_score_with_purge_embargo(self, sample_data):
    """Test cv_score with purge and embargo parameters"""
    X, y, _ = sample_data
    model = LogisticRegression(random_state=42)
    scores = cv_score(model, X, y, n_splits=3, purge=2, 
             embargo=2, scoring='accuracy')
    
    assert len(scores) == 3
  
  def test_cv_score_invalid_scoring(self, sample_data):
    """Test cv_score with invalid scoring method"""
    X, y, _ = sample_data
    model = LogisticRegression(random_state=42)
    
    with pytest.raises(ValueError, match="Unsupported scoring method"):
      cv_score(model, X, y, n_splits=3, scoring='invalid_metric')


class TestFeatureImportanceMDA:
  """Test suite for feature_importance_mda function"""
  
  @pytest.fixture
  def sample_dataframe(self):
    """Create sample DataFrame for testing"""
    np.random.seed(42)
    df = pd.DataFrame({
      'feature1': np.random.randn(100),
      'feature2': np.random.randn(100),
      'feature3': np.random.randn(100),
      'target': np.random.randint(0, 2, 100),
      'weight': np.random.rand(100)
    })
    return df
  
  def test_mda_basic(self, sample_dataframe):
    """Test basic MDA feature importance"""
    model = LogisticRegression(random_state=42)
    feature_cols = ['feature1', 'feature2', 'feature3']
    result = feature_importance_mda(
      model, sample_dataframe, feature_cols, 'target', n_splits=3
    )
    
    assert isinstance(result, pd.DataFrame)
    assert list(result.index) == feature_cols
    assert 'feature_importances' in result.columns
    assert 'mean_score' in result.columns
    assert len(result) == 3
  
  def test_mda_with_sample_weight(self, sample_dataframe):
    """Test MDA with sample weights"""
    model = LogisticRegression(random_state=42)
    feature_cols = ['feature1', 'feature2', 'feature3']
    result = feature_importance_mda(
      model, sample_dataframe, feature_cols, 'target',
      sample_weight_col='weight', n_splits=3
    )
    
    assert isinstance(result, pd.DataFrame)
    assert len(result) == 3
  
  def test_mda_accuracy_scoring(self, sample_dataframe):
    """Test MDA with accuracy scoring"""
    model = LogisticRegression(random_state=42)
    feature_cols = ['feature1', 'feature2', 'feature3']
    result = feature_importance_mda(
      model, sample_dataframe, feature_cols, 'target',
      n_splits=3, scoring='accuracy'
    )
    
    assert isinstance(result, pd.DataFrame)
    assert all(result['mean_score'] >= 0)
    assert all(result['mean_score'] <= 1)
  
  def test_mda_invalid_scoring(self, sample_dataframe):
    """Test MDA with invalid scoring method"""
    model = LogisticRegression(random_state=42)
    feature_cols = ['feature1', 'feature2', 'feature3']
    
    with pytest.raises(ValueError, match="Unsupported scoring method"):
      feature_importance_mda(
        model, sample_dataframe, feature_cols, 'target',
        n_splits=3, scoring='invalid'
      )


class TestFeatureImportanceSFI:
  """Test suite for feature_importance_sfi function"""
  
  @pytest.fixture
  def sample_dataframe(self):
    """Create sample DataFrame for testing"""
    np.random.seed(42)
    df = pd.DataFrame({
      'feature1': np.random.randn(100),
      'feature2': np.random.randn(100),
      'feature3': np.random.randn(100),
      'target': np.random.randint(0, 2, 100),
      'weight': np.random.rand(100)
    })
    return df
  
  def test_sfi_basic(self, sample_dataframe):
    """Test basic SFI feature importance"""
    model = LogisticRegression(random_state=42)
    feature_cols = ['feature1', 'feature2', 'feature3']
    result = feature_importance_sfi(
      model, sample_dataframe, feature_cols, 'target', n_splits=3
    )
    
    assert isinstance(result, pd.DataFrame)
    assert list(result.index) == feature_cols
    assert 'mean' in result.columns
    assert 'std' in result.columns
    assert len(result) == 3
  
  def test_sfi_with_sample_weight(self, sample_dataframe):
    """Test SFI with sample weights"""
    model = LogisticRegression(random_state=42)
    feature_cols = ['feature1', 'feature2', 'feature3']
    result = feature_importance_sfi(
      model, sample_dataframe, feature_cols, 'target',
      sample_weight_col='weight', n_splits=3
    )
    
    assert isinstance(result, pd.DataFrame)
    assert len(result) == 3
  
  def test_sfi_accuracy_scoring(self, sample_dataframe):
    """Test SFI with accuracy scoring"""
    model = RandomForestClassifier(random_state=42, n_estimators=10)
    feature_cols = ['feature1', 'feature2']
    result = feature_importance_sfi(
      model, sample_dataframe, feature_cols, 'target',
      n_splits=3, scoring='accuracy'
    )
    
    assert isinstance(result, pd.DataFrame)
    assert all(result['mean'] >= -1)
    assert all(result['mean'] <= 1)


class TestOrthogonalFeatures:
  """Test suite for orthogonal_features function"""
  
  def test_orthogonal_basic(self):
    """Test basic orthogonal features"""
    np.random.seed(42)
    df = pd.DataFrame({
      'f1': np.random.randn(100),
      'f2': np.random.randn(100),
      'f3': np.random.randn(100),
    })
    
    result = orthogonal_features(df, threshold=0.95)
    assert isinstance(result, np.ndarray)
    assert result.shape[0] == 100
  
  def test_orthogonal_with_correlated_features(self):
    """Test orthogonal features with highly correlated features"""
    np.random.seed(42)
    f1 = np.random.randn(100)
    df = pd.DataFrame({
      'f1': f1,
      'f2': f1 + np.random.randn(100) * 0.1,  # highly correlated
      'f3': np.random.randn(100),
    })
    
    result = orthogonal_features(df, threshold=0.95)
    assert isinstance(result, np.ndarray)
  
  def test_orthogonal_different_threshold(self):
    """Test orthogonal features with different threshold"""
    np.random.seed(42)
    df = pd.DataFrame({
      'f1': np.random.randn(100),
      'f2': np.random.randn(100),
      'f3': np.random.randn(100),
    })
    
    result_low = orthogonal_features(df, threshold=0.8)
    result_high = orthogonal_features(df, threshold=0.99)
    
    assert result_low.shape[1] <= result_high.shape[1]


class TestGetEVec:
  """Test suite for get_e_vec function"""
  
  def test_get_e_vec_basic(self):
    """Test basic eigenvalue decomposition"""
    np.random.seed(42)
    df = pd.DataFrame(np.random.randn(50, 5))
    df_z = df.sub(df.mean(), axis=1).div(df.std(), axis=1)
    dot = pd.DataFrame(
      np.dot(df_z.T, df_z),
      index=df.columns,
      columns=df.columns
    )
    
    eig_vals, eig_vecs = _get_e_vec(dot, threshold=0.95)
    
    assert isinstance(eig_vals, np.ndarray)
    assert isinstance(eig_vecs, np.ndarray)
    assert len(eig_vals) <= 5
    assert eig_vecs.shape[0] == 5
  
  def test_get_e_vec_threshold_effect(self):
    """Test effect of different thresholds"""
    np.random.seed(42)
    df = pd.DataFrame(np.random.randn(50, 5))
    df_z = df.sub(df.mean(), axis=1).div(df.std(), axis=1)
    dot = pd.DataFrame(
      np.dot(df_z.T, df_z),
      index=df.columns,
      columns=df.columns
    )
    
    eig_vals_low, eig_vecs_low = _get_e_vec(dot, threshold=0.8)
    eig_vals_high, eig_vecs_high = _get_e_vec(dot, threshold=0.99)
    
    assert len(eig_vals_low) <= len(eig_vals_high)