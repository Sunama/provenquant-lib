import numpy as np

class CombinatorialPurgedKFold:
    def __init__(
        self,
        n_splits_train: int=12,
        n_splits_test: int=3,
        purge: int=0,
        embargo: int=0
    ):
        """Combinatorial Purged K-Fold Cross-Validation

        Args:
            n_splits_test (int, optional): How many folds to split the data into for testing.
            n_splits_train (int, optional): How many folds to split the data into for training.
            purge (int, optional): Number of samples to purge around the test set.
            embargo (int, optional): Number of samples to embargo after the test set.
        """
        
        self.n_splits_test = n_splits_test
        self.n_splits_train = n_splits_train
        self.purge = purge
        self.embargo = embargo

    def split(self, X):
        n_samples = len(X)
        fold_sizes_test = (n_samples // self.n_splits_test) * np.ones(self.n_splits_test, dtype=int)
        fold_sizes_test[:n_samples % self.n_splits_test] += 1
        current_test = 0
        
        test_folds = []
        for fold_size in fold_sizes_test:
            start, stop = current_test, current_test + fold_size
            test_folds.append((start, stop))
            current_test = stop
        
        for i, (test_start, test_stop) in enumerate(test_folds):
            test_indices = np.arange(test_start, test_stop)
            
            train_indices = []
            for j, (other_start, other_stop) in enumerate(test_folds):
                if i != j:
                    train_indices.extend(np.arange(other_start, other_stop))
            
            train_indices = np.array(train_indices)
            # Apply purging around the test block and embargo just after it to avoid leakage
            train_indices = train_indices[
                (train_indices < test_start - self.purge) |
                (train_indices >= test_stop + self.purge)
            ]

            if self.embargo > 0:
                train_indices = train_indices[
                    (train_indices < test_start) |
                    (train_indices >= test_stop + self.embargo)
                ]
            
            yield train_indices, test_indices