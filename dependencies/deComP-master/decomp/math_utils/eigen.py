""" Collection of eigen value estimator """


def spectral_radius_svd(X, xp):
    """ Largest magnitude of eigen values by SVD """
    return xp.linalg.svd(X)[1][0]


def spectral_radius_Gershgorin(X, xp, keepdims=False):
    """
    An upper bound for the largest eigen velue by
    Gershgorin circle theorem.
    https://en.wikipedia.org/wiki/Gershgorin_circle_theorem

    Here, we assume X is symmetric.

    X should be a matrix or batch of matrices, shape [..., n, n].
    The return shape is [..., 1]
    """
    return xp.max(xp.sum(xp.abs(X), axis=-2), axis=-1, keepdims=True)
