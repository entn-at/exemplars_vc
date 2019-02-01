from ..utils.cp_compat import get_array_module
from ..utils import assertion, normalize
from .grads import get_likelihood


_JITTER = 1.0e-15


def solve(y, D, x, tol, minibatch, maxiter, method,
          likelihood, mask, rng, xp,
          grad_x=None, grad_d=None, forget_rate=0.5):
    """
    Implementations of
    Serizel, R., Essid, S., & Richard, G. (2016).
    MINI-BATCH STOCHASTIC APPROACHES FOR ACCELERATED MULTIPLICATIVE UPDATES IN
    NONNEGATIVE MATRIX FACTORISATION WITH BETA-DIVERGENCE, 13-16.
    """
    lik = get_likelihood(likelihood)

    if method == 'asg-mu':
        return solve_asg_mu(y, D, x, tol, minibatch, maxiter,
                            mask, rng, xp, lik)
    elif method == 'gsg-mu':
        return solve_asg_mu(y, D, x, tol, minibatch, maxiter,
                            mask, rng, xp, lik)
    elif method == 'asag-mu':
        return solve_asag_mu(y, D, x, tol, minibatch, maxiter,
                             mask, rng, xp, lik, forget_rate)
    elif method == 'gsag-mu':
        return solve_gsag_mu(y, D, x, tol, minibatch, maxiter,
                             mask, rng, xp, lik, forget_rate)
    raise NotImplementedError('NMF with {} algorithm is not yet '
                              'implemented.'.format(method))


def solve_asg_mu(y, D, x, tol, minibatch, maxiter, mask, rng, xp, lik):
    """ Algorithm 5 in the paper """
    index = xp.arange(len(y.array))
    for it in range(1, maxiter):
        rng.shuffle(index)
        y.shuffle(index)
        x.shuffle(index)
        mask.shuffle(index)

        for y_minibatch, x_minibatch, mask_minibatch in zip(y, x, mask):
            grad_x_pos, grad_x_neg = lik.grad_x(
                            y_minibatch, x_minibatch, D, mask_minibatch)
            x_minibatch[:] = x_minibatch * \
                xp.maximum(grad_x_pos, 0.0) / xp.maximum(grad_x_neg, _JITTER)

            # update D
            grad_D_pos, grad_D_neg = lik.grad_d(y_minibatch, x_minibatch, D,
                                                mask_minibatch)
            D_new = D * xp.maximum(grad_D_pos, 0.0) / xp.maximum(grad_D_neg,
                                                                 _JITTER)
            D_new = normalize.l2_strict(D_new, axis=-1, xp=xp)
            if xp.max(xp.abs(D - D_new)) < tol:
                return it, D, x.array
            D = D_new

    return maxiter, D, x.array


def solve_gsg_mu(y, D, x, tol, minibatch, maxiter, mask, rng, xp, lik):
    """ Algorithm 6 in the paper """
    index = xp.arange(len(y.array))
    for it in range(1, maxiter):
        rng.shuffle(index)
        y.shuffle(index)
        x.shuffle(index)
        mask.shuffle(index)

        for y_minibatch, x_minibatch, mask_minibatch in zip(y, x, mask):
            grad_x_pos, grad_x_neg = lik.grad_x(
                            y_minibatch, x_minibatch, D, mask_minibatch)
            x_minibatch[:] = x_minibatch * \
                xp.maximum(grad_x_pos, 0.0) / xp.maximum(grad_x_neg, _JITTER)

        # update D
        grad_D_pos, grad_D_neg = lik.grad_d(y_minibatch, x_minibatch, D,
                                            mask_minibatch)
        D_new = D * xp.maximum(grad_D_pos, 0.0) / xp.maximum(grad_D_neg,
                                                             _JITTER)
        D_new = normalize.l2_strict(D_new, axis=-1, xp=xp)
        if xp.max(xp.abs(D - D_new)) < tol:
            return it, D, x.array
        D = D_new

    return maxiter, D, x.array


def solve_asag_mu(y, D, x, tol, minibatch, maxiter,
                  mask, rng, xp, lik, forget_rate):
    """ Algorithm 7 in the paper """
    def accumurate_grad(grad_sum, grad):
        return (1.0 - forget_rate) * grad_sum + forget_rate * grad

    index = xp.arange(len(y.array))
    for it in range(1, maxiter):
        rng.shuffle(index)
        y.shuffle(index)
        x.shuffle(index)
        mask.shuffle(index)

        grad_D_pos_sum = xp.zeros_like(D)
        grad_D_neg_sum = xp.zeros_like(D)

        for y_minibatch, x_minibatch, mask_minibatch in zip(y, x, mask):
            grad_x_pos, grad_x_neg = lik.grad_x(
                            y_minibatch, x_minibatch, D, mask_minibatch)
            x_minibatch[:] = x_minibatch * \
                xp.maximum(grad_x_pos, 0.0) / xp.maximum(grad_x_neg, _JITTER)

            # update D
            grad_D_pos, grad_D_neg = lik.grad_d(y_minibatch, x_minibatch, D,
                                                mask_minibatch)
            grad_D_pos_sum = accumurate_grad(grad_D_pos_sum, grad_D_pos)
            grad_D_neg_sum = accumurate_grad(grad_D_neg_sum, grad_D_neg)

            D_new = D * xp.maximum(grad_D_pos_sum, 0.0) \
                        / xp.maximum(grad_D_neg_sum, _JITTER)
            D_new = normalize.l2_strict(D_new, axis=-1, xp=xp)
            if xp.max(xp.abs(D - D_new)) < tol:
                return it, D, x.array
            D = D_new

    return maxiter, D, x.array


def solve_gsag_mu(y, D, x, tol, minibatch, maxiter,
                  mask, rng, xp, lik, forget_rate):
    """ Algorithm 7 in the paper """
    def accumurate_grad(grad_sum, grad):
        return (1.0 - forget_rate) * grad_sum + forget_rate * grad

    index = xp.arange(len(y.array))
    for it in range(1, maxiter):
        rng.shuffle(index)
        y.shuffle(index)
        x.shuffle(index)
        mask.shuffle(index)

        grad_D_pos_sum = xp.zeros_like(D)
        grad_D_neg_sum = xp.zeros_like(D)

        for y_minibatch, x_minibatch, mask_minibatch in zip(y, x, mask):
            grad_x_pos, grad_x_neg = lik.grad_x(
                            y_minibatch, x_minibatch, D, mask_minibatch)
            x_minibatch[:] = x_minibatch * \
                xp.maximum(grad_x_pos, 0.0) / xp.maximum(grad_x_neg, _JITTER)

            # update D
            grad_D_pos, grad_D_neg = lik.grad_d(y_minibatch, x_minibatch, D,
                                                mask_minibatch)
            grad_D_pos_sum = accumurate_grad(grad_D_pos_sum, grad_D_pos)
            grad_D_neg_sum = accumurate_grad(grad_D_neg_sum, grad_D_neg)

        D_new = D * xp.maximum(grad_D_pos_sum, 0.0)\
                        / xp.maximum(grad_D_neg_sum, _JITTER)
        D_new = normalize.l2_strict(D_new, axis=-1, xp=xp)
        if xp.max(xp.abs(D - D_new)) < tol:
            return it, D, x.array
        D = D_new

    return maxiter, D, x.array
