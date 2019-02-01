import numpy as np
from .utils.cp_compat import get_array_module
from .utils.data import MinibatchData, NoneIterator, AsyncMinibatchData
from .utils.data import minibatch_index
from .utils import assertion, normalize
from . import lasso


_JITTER = 1.0e-15


def solve(y, D, alpha, x=None, tol=1.0e-3,
          minibatch=None, maxiter=1000, method='block_cd',
          lasso_method='cd', lasso_iter=10, lasso_tol=1.0e-5,
          mask=None, random_seed=None):
    """
    Learn Dictionary with lasso regularization.

    argmin_{x, D} {|y - xD|^2 - alpha |x|}
    s.t. |D_j|^2 <= 1

    with
    y: [n_samples, n_channels]
    x: [n_samples, n_features]
    D: [n_features, n_channels]

    Parameters
    ----------
    y: array-like.
        Shape: [n_samples, ch]
    D: array-like.
        Initial dictionary, shape [ch, n_component]
    alpha: a positive float
        Regularization parameter
    x: array-like
        An initial estimate of x

    tol: a float.
        Criterion

    method: string
        One of ['parallel_cd', 'block_cd']

    mask: an array-like of Boolean (or integer, float)
        The missing point should be zero. One for otherwise.

    Notes
    -----
    'block_cd':
        Mairal, J., Bach FRANCISBACH, F., Ponce JEANPONCE, J., & Sapiro, G. (n.d.)
        Online Dictionary Learning for Sparse Coding.

    'parallel_cd':
        Parallelized version of 'block_cd'.
    """
    # Check all the class are numpy or cupy
    xp = get_array_module(D)
    if x is None:
        x = xp.ones((y.shape[0], D.shape[0]), dtype=D.dtype)

    rng = np.random.RandomState(random_seed)
    if x is None:
        x = xp.zeros(y.shape[:-1] + (D.shape[0], ), dtype=y.dtype)

    assertion.assert_dtypes(y=y, D=D, x=x)
    assertion.assert_dtypes(mask=mask, dtypes='f')
    assertion.assert_shapes('x', x, 'D', D, axes=1)
    assertion.assert_shapes('y', y, 'D', D, axes=[-1])
    assertion.assert_shapes('y', y, 'mask', mask)

    # batch methods
    if minibatch is None:
        raise NotImplementedError('Only online methods are implemented. '
                                  'minibatch is required.')

    if xp is np:
        # check all the array type is np
        get_array_module(y, D, x, mask)
        y = MinibatchData(y, minibatch)
        x = MinibatchData(x, minibatch)
        if mask is None:
            mask = NoneIterator()
        else:
            mask = MinibatchData(mask, minibatch)
        rng = np.random.RandomState(random_seed)
    else:
        # minibatch methods
        def get_dataset(a, needs_update=True):
            if a is None:
                return NoneIterator()
            if get_array_module(a) is not np:
                return MinibatchData(a, minibatch)
            return AsyncMinibatchData(a, minibatch, needs_update=needs_update)

        x = get_dataset(x, needs_update=True)
        y = get_dataset(y, needs_update=False)
        mask = get_dataset(mask, needs_update=False)
        rng = xp.random.RandomState(random_seed)

    if method == 'block_cd':
        if mask is None or isinstance(mask, NoneIterator):
            return solve_cd(
                y, D, alpha, x, tol, minibatch, maxiter,
                lasso_method, lasso_iter, lasso_tol, rng, xp)
        else:
            return solve_cd_mask(
                y, D, alpha, x, tol, minibatch, maxiter,
                lasso_method, lasso_iter, lasso_tol, rng, xp, mask)
    else:
        raise NotImplementedError('Method %s is not yet '
                                  ' implemented'.format(method))


def solve_cd(y, D, alpha, x, tol, minibatch, maxiter,
             lasso_method, lasso_iter, lasso_tol, rng, xp):
    """
    Mairal, J., Bach FRANCISBACH, F., Ponce JEANPONCE, J., & Sapiro, G. (n.d.)
    Online Dictionary Learning for Sparse Coding.
    """
    index = xp.arange(len(y.array))

    A = xp.zeros((D.shape[0], D.shape[0]), dtype=y.dtype)
    B = xp.zeros((D.shape[0], D.shape[1]), dtype=y.dtype)

    # Normalize first
    D = normalize.l2_strict(D, axis=-1, xp=xp)

    # iteration loop
    count = 0
    for it in range(1, maxiter):
        rng.shuffle(index)
        y.shuffle(index)
        x.shuffle(index)
        try:
            for y_minibatch, x_minibatch in zip(y, x):
                # lasso
                it2, x_minibatch_new = lasso.solve_fastpath(
                            y_minibatch, D, alpha, x=x_minibatch, tol=lasso_tol,
                            maxiter=lasso_iter, method=lasso_method, xp=xp)
                x_minibatch[...] = x_minibatch_new

                # equation (11)
                theta_plus1 = count * minibatch + 1.0
                beta = (theta_plus1 - minibatch) / theta_plus1

                # Dictionary update
                xT = x_minibatch.T
                if y.dtype.kind == 'c':
                    xT = xp.conj(xT)

                A = beta * A + xp.dot(xT, x_minibatch)
                B = beta * B + xp.dot(xT, y_minibatch)

                D_new = D.copy()
                for k in range(D_new.shape[0]):
                    uk = (B[k] - xp.dot(A[k], D_new)) / (A[k, k] + _JITTER)\
                         + D_new[k]
                    # normalize
                    D_new[k] = normalize.l2(uk, xp, axis=-1)

                if xp.max(xp.abs(D - D_new)) < tol:
                    return it, D_new, x.array
                D = D_new
                count += 1

        except KeyboardInterrupt:
            return it, D, x.array
    return maxiter, D, x.array


def solve_cd_mask(y, D, alpha, x, tol, minibatch, maxiter,
                  lasso_method, lasso_iter, lasso_tol, rng, xp, mask):
    """
    Mairal, J., Bach FRANCISBACH, F., Ponce JEANPONCE, J., & Sapiro, G. (n.d.)
    Online Dictionary Learning for Sparse Coding.
    """
    index = xp.arange(len(y.array))

    A = xp.zeros((D.shape[0], y.shape[-1], D.shape[0]), dtype=y.dtype)
    B = xp.zeros((D.shape[0], D.shape[1]), dtype=y.dtype)

    # Normalize first
    D = normalize.l2_strict(D, axis=-1, xp=xp)

    # iteration loop
    count = 0
    for it in range(1, maxiter):
        rng.shuffle(index)
        y.shuffle(index)
        x.shuffle(index)
        mask.shuffle(index)
        try:
            for y_minibatch, x_minibatch, mask_minibatch in zip(y, x, mask):
                # lasso
                it2, x_minibatch_new = lasso.solve_fastpath(
                            y_minibatch, D, alpha, x=x_minibatch, tol=lasso_tol,
                            maxiter=lasso_iter, method=lasso_method,
                            mask=mask_minibatch, xp=xp)
                x_minibatch[...] = x_minibatch_new

                # equation (11)
                theta_plus1 = count * minibatch + 1.0
                beta = (theta_plus1 - minibatch) / theta_plus1

                # Dictionary update
                xT = x_minibatch.T
                if y.dtype.kind == 'c':
                    xT = xp.conj(xT)

                A = beta * A + xp.tensordot(
                        xT, xp.expand_dims(x_minibatch, -2) *
                            xp.expand_dims(mask_minibatch, -1),
                        axes=1)  # [n_features, minibatch, n_features]
                B = beta * B + xp.dot(xT, y_minibatch * mask_minibatch)

                D_new = D.copy()
                for k in range(D_new.shape[0]):
                    AkD = xp.einsum('jk,kj->j', A[k], D)
                    Akk = xp.sum(A[k, :, k] + _JITTER)
                    uk = (B[k] - AkD) / Akk + D_new[k]
                    # normalize
                    D_new[k] = normalize.l2(uk, xp, axis=-1)

                if xp.max(xp.abs(D - D_new)) < tol:
                    return it, D_new, x.array
                D = D_new
                count += 1

        except KeyboardInterrupt:
            return it, D, x.array
    return maxiter, D, x.array
