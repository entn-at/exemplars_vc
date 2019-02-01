from . import lasso


def solve(y, A, alpha, x=None, tol=1.0e-3, method='ista_pos', maxiter=1000,
          mask=None, **kwargs):
    return lasso.solve(y, A, alpha, x, tol, method + '_pos', maxiter,
                       mask, **kwargs)
