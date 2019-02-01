import numpy as np
from ..utils import assertion, normalize
from ..utils.cp_compat import get_array_module
from .grads import get_likelihood


_JITTER = 1.0e-15


def solve(y, D, x, tol, minibatch, maxiter, method,
          likelihood, mask, rng, xp, alpha=1.0, beta=0.5):
    """
    Kasai, H. (2017).
    Stochastic variance reduced multiplicative update for
    nonnegative matrix factorization. Retrieved from
    https://arxiv.org/pdf/1710.10781.pdf
    """
    # mini-batch methods
    lik = get_likelihood(likelihood)

    if method == 'svrmu':
        iter_minibatch = 1
    elif method == 'svrmu-acc':
        F, K = D.shape
        N = x.shape[0]
        iter_minibatch = int(np.maximum(beta * F * (3 * K + 2 * N)
                                        / (3 * F * N + 2 * K), 1.0))
    else:
        raise NotImplementedError('NMF with {} algorithm is not yet '
                                  'implemented.'.format(method))
    # with iter_minibatch
    return solve_svrmu(y, D, x, tol, minibatch, maxiter, iter_minibatch,
                       mask, rng, xp, lik, alpha)


def solve_svrmu(y, D, x, tol, minibatch, maxiter, iter_minibatch,
                mask, rng, xp, lik, alpha):
    """ Algorithm 1 in the paper
    y and x should be MinibatchData or SequentialMinibatchData
    """
    index = xp.arange(len(y.array))
    rng.shuffle(index)
    y.shuffle(index)
    x.shuffle(index)
    mask.shuffle(index)
    minibatch_num = y.n_loop
    grad_D_pos_prev = xp.zeros((minibatch_num, ) + D.shape, dtype=D.dtype)
    grad_D_neg_prev = xp.zeros((minibatch_num, ) + D.shape, dtype=D.dtype)

    for it in range(1, maxiter):
        # compute full gradient
        grad_D_pos_full = xp.zeros_like(D)
        grad_D_neg_full = xp.zeros_like(D)
        for y_minibatch, x_minibatch, mask_minibatch in zip(y, x, mask):
            grad_D_pos, grad_D_neg = lik.grad_d(y_minibatch, x_minibatch, D,
                                                mask_minibatch)
            grad_D_pos_full += grad_D_pos
            grad_D_neg_full += grad_D_neg
        grad_D_pos_full /= minibatch_num
        grad_D_neg_full /= minibatch_num

        # update D minibatch
        for k, (y_minibatch, x_minibatch, mask_minibatch) in enumerate(
                                                            zip(y, x, mask)):
            for _ in range(iter_minibatch):
                grad_x_pos, grad_x_neg = lik.grad_x(
                            y_minibatch, x_minibatch, D, mask_minibatch)
                x_minibatch[:] = x_minibatch * xp.maximum(
                            grad_x_pos, 0.0) / xp.maximum(grad_x_neg, _JITTER)

            # update D
            grad_D_pos, grad_D_neg = lik.grad_d(y_minibatch, x_minibatch, D,
                                                mask_minibatch)

            P = grad_D_pos + grad_D_neg_prev[k] + grad_D_pos_full
            Q = grad_D_neg + grad_D_pos_prev[k] + grad_D_neg_full

            D_new = D * ((1.0 - alpha) + alpha * P / xp.maximum(Q, _JITTER))
            D_new = normalize.l2_strict(xp.maximum(D_new, 0.0), axis=-1, xp=xp)

            if xp.max(xp.abs(D - D_new)) < tol:
                return it, D, x.array
            D = D_new

            grad_D_pos_prev[k] = grad_D_pos
            grad_D_neg_prev[k] = grad_D_neg

    return maxiter, D, x.array
