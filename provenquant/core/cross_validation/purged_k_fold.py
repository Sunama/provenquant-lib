import numpy as np

class PurgedKFold:
    def __init__(self, n_splits=5, purge=0, embargo=0):
        self.n_splits = n_splits
        self.purge = purge
        self.embargo = embargo
        
    def split(self, X):
        n_samples = len(X)
        fold_sizes = (n_samples // self.n_splits) * np.ones(self.n_splits, dtype=int)
        fold_sizes[:n_samples % self.n_splits] += 1
        current = 0
        
        for fold_size in fold_sizes:
            start, stop = current, current + fold_size
            test_indices = np.arange(start, stop)
            post_gap = max(self.purge, self.embargo)
            train_indices = np.concatenate([
                np.arange(0, max(0, start - self.purge)),
                np.arange(min(n_samples, stop + post_gap), n_samples)
            ])
            yield train_indices, test_indices
            current = stop