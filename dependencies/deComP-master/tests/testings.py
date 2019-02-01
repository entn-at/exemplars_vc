import numpy as np

try:
    import cupy as cp

    def allclose(x, y, *args, **kwargs):
        if isinstance(x, cp.ndarray):
            x = x.get()
        if isinstance(y, cp.ndarray):
            y = y.get()
        return np.allclose(x, y, *args, **kwargs)

except ImportError:
    allclose = np.allclose
