from ..utils.cp_compat import get_array_module
import numpy as np
try:
    import cupy
except ImportError:
    pass


def inv(x):
    """ Batch version of np.linalg.inv
    """
    if get_array_module(x) is np:
        return np.linalg.inv(x)

    # cupy case
    if x.ndim == 2:
        return cupy.linalg.inv(x)
    elif x.ndim == 3:
        inv = cupy.ndarray(x.shape, x.dtype)
        for i in len(inv):
            inv[i] = cupy.linalg.inv(x[i])
        return inv
    elif x.ndim == 4:
        inv = cupy.ndarray(x.shape, x.dtype)
        for i in range(inv.shape[0]):
            for j in range(inv.shape[1]):
                inv[i, j] = cupy.linalg.inv(x[i, j])
        return inv
    elif x.ndim == 5:
        inv = cupy.ndarray(x.shape, x.dtype)
        for i in range(inv.shape[0]):
            for j in range(inv.shape[1]):
                for k in range(inv.shape[2]):
                    inv[i, j, k] = cupy.linalg.inv(x[i, j, k])
        return inv
    else:
        raise ValueError('linalg_inv only support x.ndim == 5. Given',
                         x.ndim)
