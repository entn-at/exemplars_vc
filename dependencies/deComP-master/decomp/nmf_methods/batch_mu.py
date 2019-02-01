from ..utils import normalize
from .grads import get_likelihood


_JITTER = 1.0e-15


def solve(y, D, x, tol, maxiter, likelihood, mask, xp):
    """
    updator_x, updator_d: callable.
        Custom updators.
    """
    lik = get_likelihood(likelihood)

    # main iteration loop
    for it in range(1, maxiter):
        # update x
        x = lik.update_x(y, x, D, mask)
        # update D
        D_new = lik.update_d(y, x, D, mask)
        D_new = normalize.l2_strict(D_new, axis=-1, xp=xp)
        if xp.max(xp.abs(D - D_new)) < tol:
            return it, D_new, x
        D = D_new

    return maxiter, D, x
