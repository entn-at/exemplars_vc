import numpy as np
from chainer.utils import conv, conv_nd
from .utils.cp_compat import get_array_module
from .utils.data import minibatch_index
from .utils import (assertion, normalize, dtype)
from . import lasso
from .math_utils import eigen

_JITTER = 1.0e-15


def solve(y, D, alpha, stride=1, padding='SAME', x=None, tol=1.0e-4,
          minibatch=None, size_of_minibatch=None, maxiter=1000,
          lasso_method='acc_ista', lasso_iter=10, lasso_tol=1.0e-5,
          mask=None, random_seed=None):
    """
    Learn Template with lasso regularization.

    argmin_{x, D} {|y - xD|^2 - alpha |x|}
    s.t. |D_j|^2 <= 1

    with
    y: [..., n_channels]
    x: [..., n_features]
    D: [n_features, n_channels]

    Parameters
    ----------
    y: array-like.
        Shape: [..., ch]
    D: array-like.
        Initial dictionary, shape [n_template, template_size]
    alpha: a positive float
        Regularization parameter
    x: array-like
        An initial estimate of x

    tol: a float.
        Criterion

    minibatch: integer or a tuple of integers.

    mask: an array-like of Boolean (or integer, float)
        The missing point should be zero. One for otherwise.
    """
    # Check all the class are numpy or cupy
    xp = get_array_module(y, D, x)
    rng = np.random.RandomState(random_seed)

    if x is None:
        coef_size = _coef_size(D.shape[-1], y.shape[-1], stride=stride,
                               padding=padding)
        if y.ndim == 2:
            x = xp.zeros((y.shape[0], D.shape[0], coef_size), dtype=y.dtype)
        else:
            x = xp.zeros((D.shape[0], coef_size), dtype=y.dtype)

    assertion.assert_dtypes(y=y, D=D, x=x)
    assertion.assert_dtypes(mask=mask, dtypes='f')
    assertion.assert_shapes('y', y, 'mask', mask)

    if minibatch is not None and size_of_minibatch is None:
        raise ValueError('size_of_minibatch should be specified with '
                         'minibatch calculation.')

    return solve_fastpath(y, D, alpha, x, stride, padding,
                          tol, minibatch, size_of_minibatch, maxiter,
                          lasso_method, lasso_iter, lasso_tol, rng, xp,
                          mask=mask)


def solve_fastpath(y, D, alpha, x, stride, padding, tol,
                   minibatch, size_of_minibatch, maxiter,
                   lasso_method, lasso_iter, lasso_tol, rng, xp,
                   mask=None):
    """
    Fast path for dictionary learning without any default value setting nor
    shape/dtype assertions.
    """
    if mask is not None:
        raise NotImplementedError('Template matching with mask is not '
                                  'yet implemented.')
    if y.ndim == 1:
        y_2d = xp.expand_dims(y, 0)
        x = xp.expand_dims(x, 0)
    else:
        y_2d = y

    if minibatch is None:
        it, D, x = solve_batch(
            y_2d, D, alpha, x, stride, padding, tol, maxiter,
            lasso_method, lasso_iter, lasso_tol, xp)
    else:
        it, D, x = solve_minibatch(
            y_2d, D, alpha, x, stride, padding, tol,
            minibatch, size_of_minibatch, maxiter,
            lasso_method, lasso_iter, lasso_tol, rng, xp)

    if y.ndim == 1:
        x = xp.squeeze(x, 0)
    return it, D, x


def _coef_size(template_size, size, stride=1, padding='VALID'):
    """
    Returns the size of coefficient given template D and data size
    """
    if padding == 'VALID':
        pad = size - template_size
    else:  # 'SAME'
        pad = size - 1

    return int(np.floor((template_size + 2 * pad - size) / stride + 1))


def _temp2mat(D, size, stride, padding, xp):
    """ Convert template to the equivalent matrix.    """
    if xp == np:
        if padding == 'VALID':
            dmat = conv_nd.im2col_nd_cpu(
                        xp.expand_dims(D, 0), (size, ), stride=(stride, ),
                        pad=(size - D.shape[1], ))
        else:  # 'SAME'
            dmat = conv_nd.im2col_nd_cpu(
                        xp.expand_dims(D, 0), (size, ), stride=(stride, ),
                        pad=(size - 1, ))
    else:
        raise NotImplementedError('_temp2mat is not yet implemented for gpu.')

    return xp.moveaxis(xp.squeeze(dmat, 0), 1, -1)[:, ::-1]


def _coef2mat(x_orig, size, template_size, stride, padding, xp):
    """ Convert coefficient vector to equivalent matrix """
    if x_orig.ndim == 2:
        x = xp.expand_dims(x_orig, 0)
    else:
        x = x_orig

    if stride > 1:
        pad_size = (x.shape[2] - 1) * stride + 1
        pad_start = 0
        if padding == 'VALID':
            coef_size = size - pad_size
            if coef_size > template_size - 1:
                pad_start = coef_size - template_size + 1
        else:
            coef_size = pad_size - size + 1
            if coef_size < template_size:
                pad_start = template_size - coef_size
        x_pad = xp.zeros((x.shape[0], x.shape[1], pad_start + pad_size),
                         dtype=x.dtype)
        x_pad[:, :, slice(pad_start, pad_size + pad_start, stride)] = x
        x = x_pad

    if xp == np:
        if padding == 'VALID':
            xmat = conv_nd.im2col_nd_cpu(
                        x, (size, ), stride=(1, ), pad=(size - x.shape[2], ))
        else:  # 'SAME'
            xmat = conv_nd.im2col_nd_cpu(x, (size, ), stride=(1, ), pad=(0, ))
    else:
        raise NotImplementedError('_temp2mat is not yet implemented for gpu.')

    if xmat.shape[-1] > template_size:
        xmat = xmat[..., slice(xmat.shape[-1] - template_size, None)]

    if x_orig.ndim == 2:
        xmat = xp.squeeze(xmat, 0)
        return xp.moveaxis(xmat, -2, -1)[:, ::-1]
    else:
        return xp.moveaxis(xmat, -2, -1)[:, :, ::-1]


def predict(x, D, size, stride=1, padding='SAME'):
    """ Predict the latent value from x and D    """
    xp = get_array_module(x, D)
    dmat = _temp2mat(D, size, stride=stride, padding=padding, xp=xp)
    return xp.tensordot(x, dmat, 2)


def solve_batch(y, D, alpha, x, stride, padding, tol, maxiter,
                lasso_method, lasso_iter, lasso_tol, xp):
    """
    Template learning with direct method.
    This function repeats the following two linear problems,
    + y = x A
        solve x from y and A. A is a transformed matrix of D.
    + y = X d
        solve d from y and X. X is a transformed matrix of x.
    """
    # Normalize first
    D = normalize.l2_strict(D, axis=-1, xp=xp)
    size = y.shape[-1]
    template_size = D.shape[1]

    # iteration loop
    for it in range(1, maxiter):
        try:
            # solve x
            d_mat = _temp2mat(D, size, stride, padding, xp)
            it2, x_flat = lasso.solve_fastpath(
                y, d_mat.reshape(-1, d_mat.shape[-1]),  alpha,
                x=x.reshape(y.shape[0], -1), tol=lasso_tol,
                maxiter=lasso_iter, method=lasso_method, mask=None, xp=xp)
            x = x_flat.reshape(*x.shape)
            # solve D
            X = _coef2mat(x, size, template_size, stride, padding, xp)
            X = X.reshape(X.shape[0], -1, size)
            # iterative solution of XD = y
            Xt = xp.moveaxis(X, -2, -1)
            if X.dtype.kind == 'c':
                Xt = xp.conj(Xt)
            XXt = xp.tensordot(X, Xt, ((0, -1), (0, -2)))
            yX = xp.tensordot(y, X, ((0, -1), (0, -1)))

            L = eigen.spectral_radius_Gershgorin(XXt, xp) + _JITTER
            #
            D_flat = D.flatten()
            D_flat_new = D_flat + (yX - xp.dot(XXt, D_flat)) / L
            D_new = normalize.l2(xp.reshape(D_flat_new, D.shape), xp)

            if xp.max(xp.abs(D - D_new)) < tol:
                return it, D_new, x

            D = D_new
        except KeyboardInterrupt:
            return it, D, x
    return maxiter, D, x


class Minibatcher(object):
    def __init__(self, array, size_of_minibatch, xp):
        """
        A class to make a sequence of minibatches
        array:
            shape [batch, n_sequence]
        """
        self.array = array
        self.size_of_minibatch = size_of_minibatch
        self.xp = xp

    def __getitem__(self, index):
        """ index: 2d integers """
        if self.array.ndim == 2:
            return self.xp.stack(
                [self.array[i0, slice(i1, i1 + self.size_of_minibatch)]
                 for i0, i1 in zip(*index)], axis=0)
        if self.array.ndim == 3:
            return self.xp.stack(
                [self.array[i0, :, slice(i1, i1 + self.size_of_minibatch)]
                 for i0, i1 in zip(*index)], axis=0)

    def __setitem__(self, index, values):
        if self.array.ndim == 2:
            for (i0, i1), val in zip(index, values):
                self.array[i0, slice(i1, i1 + self.size_of_minibatch)] = val
        if self.array.ndim == 3:
            for (i0, i1), val in zip(zip(*index), values):
                self.array[i0, :, slice(i1, i1 + self.size_of_minibatch)] = val

    @property
    def shape(self):
        return self.array.shape

    @property
    def dtype(self):
        return self.array.dtype


def solve_minibatch(y, D, alpha, x, stride, padding, tol,
                    minibatch, size_of_minibatch, maxiter,
                    lasso_method, lasso_iter, lasso_tol, rng, xp):
    """
    Template learning with direct method.
    This function repeats the following two linear problems,
    + y = x A
        solve x from y and A. A is a transformed matrix of D.
    + y = X d
        solve d from y and X. X is a transformed matrix of x.
    """
    # Normalize first
    D = normalize.l2_strict(D, axis=-1, xp=xp)
    template_size = D.shape[1]
    coef_size = _coef_size(template_size, size_of_minibatch,
                           stride=stride, padding=padding)
    y = Minibatcher(y, size_of_minibatch, xp)
    x = Minibatcher(x, coef_size, xp)

    # \sum yX
    yX_sum = xp.zeros(D.size, dtype=y.dtype)
    # \sum XXt
    XXt_sum = xp.zeros((D.size, D.size), dtype=y.dtype)
    # iteration loop
    for it in range(1, maxiter):
        try:
            indexes = minibatch_index((y.shape[0],
                                       y.shape[-1] - size_of_minibatch),
                                      minibatch, rng)
            y_minibatch = y[indexes]
            x_minibatch = x[indexes]
            # solve x
            d_mat = _temp2mat(D, size_of_minibatch, stride, padding, xp)
            it2, x_flat = lasso.solve_fastpath(
                y_minibatch, d_mat.reshape(-1, d_mat.shape[-1]),  alpha,
                x=x_minibatch.reshape(minibatch, -1), tol=lasso_tol,
                maxiter=lasso_iter, method=lasso_method, mask=None, xp=xp)

            x_minibatch = x_flat.reshape(x_minibatch.shape)
            x[indexes] = x_minibatch

            # solve D
            X = _coef2mat(x_minibatch, size_of_minibatch, template_size,
                          stride, padding, xp)
            X = X.reshape(X.shape[0], -1, size_of_minibatch)
            # iterative solution of XD = y
            Xt = xp.moveaxis(X, -2, -1)
            if X.dtype.kind == 'c':
                Xt = xp.conj(Xt)
            XXt = xp.tensordot(X, Xt, ((0, -1), (0, -2)))
            yX = xp.tensordot(y_minibatch, X, ((0, -1), (0, -1)))
            yX_sum += yX / it
            XXt_sum += XXt / it

            L = eigen.spectral_radius_Gershgorin(XXt_sum, xp) + _JITTER
            #
            D_flat = D.flatten()
            D_flat_new = D_flat + (yX_sum - xp.dot(XXt_sum, D_flat)) / L
            D_new = normalize.l2(xp.reshape(D_flat_new, D.shape), xp)

            if xp.max(xp.abs(D - D_new)) < tol:
                return it, D_new, x.array

            D = D_new
        except KeyboardInterrupt:
            return it, D, x.array
    return maxiter, D, x.array
