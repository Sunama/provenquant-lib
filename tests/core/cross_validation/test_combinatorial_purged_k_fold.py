import numpy as np
import pytest
from provenquant.core.cross_validation.combinatorial_purged_k_fold import CombinatorialPurgedKFold

class TestCombinatorialPurgedKFold:
  def test_initialization(self):
    cpkf = CombinatorialPurgedKFold(n_splits_train=12, n_splits_test=3, purge=0, embargo=0)
    assert cpkf.n_splits_train == 12
    assert cpkf.n_splits_test == 3
    assert cpkf.purge == 0
    assert cpkf.embargo == 0

  def test_split_basic(self):
    X = np.arange(30)
    cpkf = CombinatorialPurgedKFold(n_splits_test=3, purge=0, embargo=0)
    splits = list(cpkf.split(X))
    
    assert len(splits) == 3
    for train_idx, test_idx in splits:
      assert len(train_idx) + len(test_idx) == len(X)
      assert len(np.intersect1d(train_idx, test_idx)) == 0

  def test_split_no_leakage(self):
    X = np.arange(100)
    cpkf = CombinatorialPurgedKFold(n_splits_test=4, purge=5, embargo=3)
    
    for train_idx, test_idx in cpkf.split(X):
      assert len(np.intersect1d(train_idx, test_idx)) == 0

  def test_split_with_purge(self):
    X = np.arange(100)
    cpkf = CombinatorialPurgedKFold(n_splits_test=4, purge=5, embargo=0)
    
    for train_idx, test_idx in cpkf.split(X):
      test_start, test_stop = test_idx.min(), test_idx.max()
      purged_range = np.arange(test_start - 5, test_stop)
      assert len(np.intersect1d(train_idx, purged_range)) == 0

  def test_split_with_embargo(self):
    X = np.arange(100)
    cpkf = CombinatorialPurgedKFold(n_splits_test=4, purge=0, embargo=5)
    
    for train_idx, test_idx in cpkf.split(X):
      test_stop = test_idx.max() + 1
      embargo_range = np.arange(test_stop, test_stop + 5)
      assert len(np.intersect1d(train_idx, embargo_range)) == 0

  def test_split_small_dataset(self):
    X = np.arange(10)
    cpkf = CombinatorialPurgedKFold(n_splits_test=2, purge=0, embargo=0)
    splits = list(cpkf.split(X))
    
    assert len(splits) == 2
    for train_idx, test_idx in splits:
      assert len(test_idx) > 0

  def test_split_uneven_division(self):
    X = np.arange(25)
    cpkf = CombinatorialPurgedKFold(n_splits_test=3, purge=0, embargo=0)
    splits = list(cpkf.split(X))
    
    assert len(splits) == 3
    total_samples = sum(len(test_idx) for _, test_idx in splits)
    assert total_samples == 25

  def test_split_generator(self):
    X = np.arange(50)
    cpkf = CombinatorialPurgedKFold(n_splits_test=5, purge=0, embargo=0)
    splits_gen = cpkf.split(X)
    
    assert hasattr(splits_gen, '__iter__')
    first_split = next(splits_gen)
    assert isinstance(first_split, tuple)
    assert len(first_split) == 2