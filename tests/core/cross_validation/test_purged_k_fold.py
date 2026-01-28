import numpy as np
import pytest
from provenquant.core.cross_validation.purged_k_fold import PurgedKFold

class TestPurgedKFold:
  def test_init_default_parameters(self):
    pkf = PurgedKFold()
    assert pkf.n_splits == 5
    assert pkf.purge == 0
    assert pkf.embargo == 0

  def test_init_custom_parameters(self):
    pkf = PurgedKFold(n_splits=3, purge=5, embargo=2)
    assert pkf.n_splits == 3
    assert pkf.purge == 5
    assert pkf.embargo == 2

  def test_split_returns_correct_number_of_folds(self):
    X = np.arange(100)
    pkf = PurgedKFold(n_splits=5)
    folds = list(pkf.split(X))
    assert len(folds) == 5

  def test_split_train_test_sizes(self):
    X = np.arange(100)
    pkf = PurgedKFold(n_splits=5, purge=0, embargo=0)
    for train_idx, test_idx in pkf.split(X):
      assert len(train_idx) + len(test_idx) == 100

  def test_split_no_overlap_between_train_test(self):
    X = np.arange(100)
    pkf = PurgedKFold(n_splits=5, purge=0, embargo=0)
    for train_idx, test_idx in pkf.split(X):
      assert len(np.intersect1d(train_idx, test_idx)) == 0

  def test_split_with_purge(self):
    X = np.arange(100)
    pkf = PurgedKFold(n_splits=5, purge=5, embargo=0)
    for train_idx, test_idx in pkf.split(X):
      assert len(np.intersect1d(train_idx, test_idx)) == 0

  def test_split_with_embargo(self):
    X = np.arange(100)
    pkf = PurgedKFold(n_splits=5, purge=0, embargo=3)
    for train_idx, test_idx in pkf.split(X):
      assert len(np.intersect1d(train_idx, test_idx)) == 0

  def test_split_small_dataset(self):
    X = np.arange(10)
    pkf = PurgedKFold(n_splits=5)
    folds = list(pkf.split(X))
    assert len(folds) == 5
    for train_idx, test_idx in folds:
      assert len(test_idx) > 0

  def test_split_single_fold(self):
    X = np.arange(50)
    pkf = PurgedKFold(n_splits=1)
    train_idx, test_idx = list(pkf.split(X))[0]
    assert len(train_idx) == 0
    assert len(test_idx) == 50