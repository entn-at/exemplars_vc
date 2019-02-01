
def l2(U, xp, axis=-1):
    """
    Nomrlaize U along axis, so that the standard deviation is less than 1.
    """
    if U.dtype.kind == 'c':
        Unorm = xp.sum(xp.real(xp.conj(U) * U), axis=axis, keepdims=True)
    else:
        Unorm = xp.sum(U * U, axis=axis, keepdims=True)
    return U / xp.sqrt(xp.maximum(Unorm, 1.0))


def l2_strict(U, xp, axis=-1):
    """
    Nomrlaize U along axis, so that the standard deviation is 1.
    """
    if U.dtype.kind == 'c':
        Unorm = xp.sum(xp.real(xp.conj(U) * U), axis=axis, keepdims=True)
    else:
        Unorm = xp.sum(U * U, axis=axis, keepdims=True)
    return U / xp.sqrt(Unorm)
