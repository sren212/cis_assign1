import numpy as np


class Covariance:
    """A light wrapper that behaves like a NumPy covariance matrix."""

    def __init__(self, matrix):
        self.matrix = np.asarray(matrix, dtype=float)

    @classmethod
    def eye(cls, size, scale=1.0):
        return cls(np.eye(size) * scale)

    def __array__(self, dtype=None):
        return np.asarray(self.matrix, dtype=dtype)

    def __repr__(self):
        return f"Covariance({self.matrix!r})"
