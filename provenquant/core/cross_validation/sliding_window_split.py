import numpy as np

class SlidingWindowSplit:
    def __init__(
        self,
        window_size: int,
        step_size: int
    ):
        """Sliding Window Cross-Validation

        Args:
            window_size (int): Size of the training window.
            step_size (int): Step size to move the window.
        """
        self.window_size = window_size
        self.step_size = step_size

    def split(self, X):
        n_samples = len(X)
        start = 0
        
        while start + self.window_size <= n_samples:
            train_indices = np.arange(start, start + self.window_size)
            test_indices = np.arange(start + self.window_size, min(start + self.window_size + self.step_size, n_samples))
            
            if len(test_indices) == 0:
                break
            
            yield train_indices, test_indices
            start += self.step_size