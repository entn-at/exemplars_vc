import numpy as np
import pytest
from decomp.utils.cp_compat import numpy_or_cupy as xp
from decomp import nmf


class NMFData(object):
    """
    Test from
    Kasai, H. (2017). Stochastic variance reduced multiplicative
    update for nonnegative matrix factorization. Retrieved from
    https://arxiv.org/pdf/1710.10781.pdf
    """
    def __init__(self, F=500, N=2000, K=10, b=200):
        # This is too small problem so I increase the size
        self.minibatch = b

        self.rng = np.random.RandomState(0)

        self.xtrue = np.maximum(self.randn(N, K), 0.0)
        self.Dtrue = np.maximum(self.randn(K, F), 0.0)
        self.y = np.dot(self.xtrue, self.Dtrue)
        self.D = self.Dtrue + 0.3 * np.maximum(
                self.randn(*self.Dtrue.shape), 0.)
        self.maxiter = 100
        v = self.rng.uniform(0.45, 1.0, size=self.y.size).reshape(self.y.shape)
        self.mask = np.rint(v).astype(np.float32)

    def randn(self, *shape):
        return self.rng.randn(*shape).astype(np.float32)

    def test(self, method):
        it, D, x = nmf.solve(xp.array(self.y), xp.array(self.D), tol=1.0e-10,
                             method=method, maxiter=self.maxiter)

    def test_minibatch(self, method, all_xp=False):
        D = xp.array(self.D, dtype=np.float32)
        y = xp.array(self.y, dtype=np.float32) if all_xp else self.y
        it, D, x = nmf.solve(y, D, tol=1.0e-10,
                             method=method, maxiter=self.maxiter,
                             minibatch=self.minibatch)

    def test_mask(self, method):
        it, D, x = nmf.solve(xp.array(self.y), xp.array(self.D), tol=1.0e-10,
                             method=method, maxiter=self.maxiter,
                             mask=xp.array(self.mask))

    def test_minibatch_mask(self, method, all_xp=False):
        D = xp.array(self.D)
        y = xp.array(self.y) if all_xp else self.y
        mask = xp.array(self.mask) if all_xp else self.mask
        it, D, x = nmf.solve(y, D, tol=1.0e-10,
                             method=method, maxiter=self.maxiter,
                             mask=mask, minibatch=self.minibatch)


@pytest.mark.parametrize("method", nmf.BATCH_METHODS)
def test_nmf(benchmark, method):
    nmf_data = NMFData(500, 2000, 10, 200)
    it_list = benchmark(nmf_data.test, method)


@pytest.mark.parametrize("method", nmf.BATCH_METHODS)
def test_nmf_mask(benchmark, method):
    nmf_data = NMFData(500, 2000, 10, 200)
    it_list = benchmark(nmf_data.test_mask, method)


@pytest.mark.parametrize("method", nmf.MINIBATCH_METHODS)
def test_nmf_minibatch(benchmark, method):
    nmf_data = NMFData(500, 20000, 10, 1000)
    it_list = benchmark(nmf_data.test_minibatch, method, all_xp=True)


@pytest.mark.parametrize("method", nmf.MINIBATCH_METHODS)
def test_nmf_minibatch_mask(benchmark, method):
    nmf_data = NMFData(5000, 2000, 10, 1000)
    it_list = benchmark(nmf_data.test_minibatch_mask, method, all_xp=True)


@pytest.mark.parametrize("method", nmf.MINIBATCH_METHODS)
def test_nmf_minibatch_lazy(benchmark, method):
    nmf_data = NMFData(5000, 20000, 10, 1000)
    it_list = benchmark(nmf_data.test_minibatch, method, all_xp=False)


@pytest.mark.parametrize("method", nmf.MINIBATCH_METHODS)
def test_nmf_minibatch_mask_lazy(benchmark, method):
    nmf_data = NMFData(5000, 20000, 10, 1000)
    it_list = benchmark(nmf_data.test_minibatch_mask, method, all_xp=False)
