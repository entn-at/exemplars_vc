import numpy as np
from .exceptions import DtypeMismatchError


def float_type(dtype):
    """ Convert to equivalent float type """
    if dtype.kind == 'f':
        return dtype
    if dtype == np.complex64:
        return np.float32
    if dtype == np.complex128:
        return np.float64

    raise DtypeMismatchError('Invalid dtype is given: ' + str(dtype))
