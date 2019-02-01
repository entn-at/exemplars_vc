import numpy as np
import unittest
import pytest
from decomp.utils.cp_compat import numpy_or_cupy as xp
from decomp.utils.cp_compat import has_cupy
from decomp.utils import normalize
import decomp.nmf as nmf
from .testings import allclose


def get(x):
    return x.get() if hasattr(x, 'get') else x


class TestNMF(unittest.TestCase):
    """ Test for Multiplicative Batch algorithm with L2 likilihood. """
    def randn(self, *shape):
        return self.rng.randn(*shape)

    def get_y(self, x, d):
        ytrue = xp.dot(x, d)
        if self.likelihood == 'l2':
            return ytrue + self.randn(*ytrue.shape) * 0.1
        elif self.likelihood == 'kl':  # KL
            return ytrue + xp.abs(self.randn(*ytrue.shape)) * 0.1
        raise NotImplementedError('Likelihood {} is not implemented'.format(
                                                            self.likelihood))

    def error(self, x, D, mask):
        x = get(x)
        D = get(D)
        y = get(self.y)
        mask = get(mask)

        mask = np.ones(self.y.shape, self.y.dtype) if mask is None else mask
        D = normalize.l2_strict(D, axis=-1, xp=np)
        if self.likelihood == 'l2':
            loss = np.sum(np.square(y - np.dot(x, D)) * mask)
            return 0.5 * loss
        elif self.likelihood == 'kl':  # KL
            f = np.maximum(np.dot(x, D), 1.0e-15)
            return np.sum((- y * np.log(f) + f) * mask)
        raise NotImplementedError('Likelihood {} is not implemented'.format(
                                                            self.likelihood))

    def assert_minimum(self, x, D, tol, n=100, mask=None):
        loss = self.error(x, D, mask)
        for _ in range(n):
            x_new = get(x + self.randn(*x.shape) * tol)
            D_new = get(D + self.randn(*D.shape) * tol)
            assert loss < self.error(np.maximum(x_new, 0.0),
                                     np.maximum(D_new, 0.0), mask) + 1.0e-15

    @property
    def maxiter(self):
        return 3000


class TestFullbatchNMF(TestNMF):
    def setUp(self):
        self.rng = xp.random.RandomState(0)
        self.Dtrue = xp.maximum(self.randn(3, 20), 0.0)
        self.xtrue = xp.maximum(self.randn(101, 3), 0.0)

        self.y = self.get_y(self.xtrue, self.Dtrue)
        self.D = xp.maximum(self.Dtrue + self.randn(*self.Dtrue.shape) * 0.3,
                            0.1)
        self.mask = xp.rint(
            self.rng.uniform(0.3, 1, size=self.y.size)).reshape(self.y.shape)

    def run_fullbatch(self, mask):
        D = self.D.copy()
        it, D, x = nmf.solve(self.y, D, x=None, tol=1.0e-6,
                             minibatch=None, maxiter=self.maxiter,
                             method=self.method,
                             likelihood=self.likelihood, mask=mask,
                             random_seed=0)
        assert it < self.maxiter - 1
        self.assert_minimum(x, D, tol=1.0e-5, n=100, mask=mask)
        assert not allclose(x, xp.zeros_like(x), atol=1.0e-5)


class TestFullbatch_L2(TestFullbatchNMF):
    @property
    def method(self):
        return 'mu'

    @property
    def likelihood(self):
        return 'l2'

    def test_run(self):
        self.run_fullbatch(mask=None)

    def test_run_mask(self):
        self.run_fullbatch(mask=self.mask)


class TestFullbatch_KL(TestFullbatch_L2):
    @property
    def likelihood(self):
        return 'kl'


class TestMinibatchNMF(TestNMF):
    def setUp(self):
        self.rng = xp.random.RandomState(0)
        self.Dtrue = xp.maximum(self.randn(3, 20), 0.0)
        self.xtrue = xp.maximum(self.randn(1001, 3), 0.0)

        self.y = self.get_y(self.xtrue, self.Dtrue)
        self.D = xp.maximum(self.Dtrue + self.randn(*self.Dtrue.shape) * 0.3,
                            0.1)
        self.mask = xp.rint(
            self.rng.uniform(0.375, 1, size=self.y.size)).reshape(self.y.shape)

    def run_minibatch(self, mask):
        self._run_minibatch(self.y, mask)

    def run_minibatch_lazy_transfer(self, mask):
        self._run_minibatch(self.y.get(), mask)

    def _run_minibatch(self, y, mask):
        D = self.D.copy()
        start_iter = 31
        it, D, x = nmf.solve(
                     self.y, D, x=None, tol=1.0e-6,
                     minibatch=30, maxiter=start_iter,
                     method=self.method,
                     likelihood=self.likelihood, mask=mask,
                     random_seed=0)
        start_error = self.error(x, D, mask=mask)
        assert not allclose(x, xp.zeros_like(x), atol=1.0e-5)

        # make sure the iteration reduces the error
        errors = []
        for i in range(10):
            print('trial ', i)
            it, D, x = nmf.solve(
                         self.y, D, x=x, tol=1.0e-6,
                         minibatch=30, maxiter=self.maxiter,
                         method=self.method,
                         likelihood=self.likelihood, mask=mask,
                         random_seed=0)
            # just make sure the loss is decreasing
            error = self.error(x, D, mask=mask)
            assert not allclose(x, xp.zeros_like(x), atol=1.0e-5)
            assert error < start_error
            errors.append(error)
        # make sure errors are decreasing in average
        errors = np.array(errors)
        assert np.mean(errors[:4]) > np.mean(errors[-4:])

    @property
    def maxiter(self):
        return 31


class Test_SVRMU_L2(TestMinibatchNMF):
    @property
    def method(self):
        return 'svrmu'

    @property
    def likelihood(self):
        return 'l2'

    def test_run(self):
        self.run_minibatch(mask=None)

    def test_run_mask(self):
        self.run_minibatch(mask=self.mask)

    @pytest.mark.skipif(not has_cupy, reason='Needs cupy installed.')
    def test_run_lazy_transfer(self):
        self.run_minibatch_lazy_transfer(mask=None)

    @pytest.mark.skipif(not has_cupy, reason='Needs cupy installed.')
    def test_run_lazy_transfer_mask(self):
        self.run_minibatch_lazy_transfer(mask=self.mask.get())


class Test_SVRMU_KL(Test_SVRMU_L2):
    @property
    def likelihood(self):
        return 'kl'


class Test_SVRMU_ACC_L2(Test_SVRMU_L2):
    @property
    def method(self):
        return 'svrmu-acc'

    @property
    def likelihood(self):
        return 'l2'


class Test_SVRMU_ACC_KL(Test_SVRMU_ACC_L2):
    @property
    def likelihood(self):
        return 'kl'


class Test_ASG_MU_L2(Test_SVRMU_L2):
    @property
    def method(self):
        return 'asg-mu'

    @property
    def likelihood(self):
        return 'l2'


class Test_ASG_MU_KL(Test_ASG_MU_L2):
    @property
    def likelihood(self):
        return 'kl'


class Test_GSG_MU_L2(Test_ASG_MU_L2):
    @property
    def method(self):
        return 'gsg-mu'

    @property
    def likelihood(self):
        return 'l2'


class Test_GSG_MU_KL(Test_GSG_MU_L2):
    @property
    def likelihood(self):
        return 'kl'


class Test_ASAG_MU_L2(Test_ASG_MU_L2):
    @property
    def method(self):
        return 'asag-mu'

    @property
    def likelihood(self):
        return 'l2'


class Test_ASAG_MU_KL(Test_ASAG_MU_L2):
    @property
    def likelihood(self):
        return 'kl'


class Test_GSAG_MU_L2(Test_ASG_MU_L2):
    @property
    def method(self):
        return 'gsag-mu'

    @property
    def likelihood(self):
        return 'l2'


class Test_GSAG_MU_KL(Test_GSAG_MU_L2):
    @property
    def likelihood(self):
        return 'kl'


if __name__ == '__main__':
    unittest.main()
