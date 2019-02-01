from ..utils.cp_compat import get_array_module


_JITTER = 1.0e-15


def get_likelihood(likelihood):
    if likelihood in ['l2', 'gaussian']:
        return Gaussian()
    if likelihood in ['kl', 'poisson']:
        return Poisson()
    if isinstance(likelihood, Likelihood):
        return likelihood
    raise NotImplementedError('Likelihood {} is not implemented for nmf')


class Likelihood(object):
    """
    Base class for nmf likelihood.

    All the methods should be compatible with np.ndarray and cp.ndarray.
    e.g.
        bad: return np.dot(x, f)
        good: return x.dot(f)
    """
    def __init__(self):
        pass

    def grad_x(self, y, x, d, mask):
        """
        Gradient of x with fixed y, x, d.

        Parameters
        ----------
        y: np or cp ndarray. shape [n_samples, n_features]
        x: np or cp ndarray. shape [n_samples, n_latent]
        d: np or cp ndarray. shape [n_latent, n_features]
        mask: np or cp ndarray. shape [n_samples, n_features]

        Returns
        -------
        grad_x_pos: np or cp ndarray.
        grad_x_neg: np or cp ndarray.

        See also
        --------
        update_x
        """
        raise NotImplementedError

    def grad_d(self, y, x, d, mask):
        """
        Gradient of x with fixed y, x, d.

        Parameters
        ----------
        y: np or cp ndarray. shape [n_samples, n_features]
        x: np or cp ndarray. shape [n_samples, n_latent]
        d: np or cp ndarray. shape [n_latent, n_features]
        mask: np or cp ndarray. shape [n_samples, n_features]

        Returns
        -------
        grad_d_pos: np or cp ndarray.
        grad_d_neg: np or cp ndarray.

        See also
        --------
        update_d
        """
        raise NotImplementedError

    def logp(self, y, x, d, mask):
        """ evaluate log likelihood """
        raise NotImplementedError

    def update_x(self, y, x, d, mask):
        """
        Multiplicative update rule for x.
        Returns new x
        """
        xp = get_array_module(y)
        grad_pos, grad_neg = self.grad_x(y, x, d, mask)
        return x * xp.maximum(grad_pos, 0.0) / xp.maximum(grad_neg, _JITTER)

    def update_d(self, y, x, d, mask):
        """
        Multiplicative update rule for d.
        Returns new d
        """
        xp = get_array_module(y)
        grad_pos, grad_neg = self.grad_d(y, x, d, mask)
        return d * xp.maximum(grad_pos, 0.0) / xp.maximum(grad_neg, _JITTER)


class Gaussian(Likelihood):
    def __init__(self, scale=1.0):
        """
        Gaussian Likelihood.
        This likelihood is sometimes called `square loss`.

        scale: float. (optional).
            The scale parameter for the likelihood.
            This does not affect the update rule.
        """
        self.scale = scale

    def grad_x(self, y, x, d, mask):
        if mask is None:
            f = x.dot(d)
            return y.dot(d.T), f.dot(d.T)
        else:
            f = x.dot(d) * mask
            y = y * mask
            return y.dot(d.T), f.dot(d.T)

    def grad_d(self, y, x, d, mask):
        """ update d with l2 loss """
        if mask is None:
            f = x.dot(d)
            return x.T.dot(y), x.T.dot(f)
        else:
            f = x.dot(d) * mask
            y = y * mask
            return x.T.dot(y), x.T.dot(f)

    def logp(self, y, x, d, mask):
        """ evaluate log likelihood """
        xp = get_array_module(y)
        loss = xp.square((y - x.dot(d)) / self.scale)
        logp = -0.5 * loss - xp.log(self.scale) - xp.pi * 0.5
        if mask is None:
            return xp.sum(logp)
        else:
            return xp.sum(logp * mask)


class Poisson(Likelihood):
    """
    Poisson likelihood.
    This likelihood is sometimes called `KL loss`.
    """
    def grad_x(self, y, x, d, mask):
        if mask is None:
            f = x.dot(d) + _JITTER
            return (y / f).dot(d.T), d.T.sum(axis=0, keepdims=True)
        else:
            f = x.dot(d) + _JITTER
            y = y * mask
            return (y / f).dot(d.T), mask.dot(d.T)

    def grad_d(self, y, x, d, mask):
        """ update d with KL loss """
        if mask is None:
            f = x.dot(d) + _JITTER
            return x.T.dot(y / f), x.T.sum(axis=1, keepdims=True)
        else:
            f = x.dot(d) + _JITTER
            y = y * mask
            return x.T.dot(y / f), x.T.dot(mask)

    def logp(self, y, x, d, mask):
        """ evaluate log likelihood """
        xp = get_array_module(y)
        loss = xp.square((y - x.dot(d)) / self.scale)
        logp = -0.5 * loss - xp.log(self.scale) - xp.pi * 0.5
        if mask is None:
            return xp.sum(logp)
        else:
            return xp.sum(logp * mask)
