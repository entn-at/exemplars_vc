import unittest
import numpy as np
from decomp.utils.cp_compat import numpy_or_cupy as xp
from decomp import lasso
from decomp.utils.exceptions import ShapeMismatchError, DtypeMismatchError
from decomp.utils.dtype import float_type

from .testings import allclose


class TestUtils(unittest.TestCase):
    def setUp(self):
        pass

    def test_soft_threshold(self):
        # test with scalar
        actual = lasso.soft_threshold_float(0.1, 1.0, xp=np)
        expected = 0.0
        self.assertTrue(allclose(actual, expected))

        # test with 1d-array
        actual = lasso.soft_threshold_float(xp.array([0.1, -2.0, 1.4]),
                                            1.0, xp=xp)
        expected = [0.0, -1.0, 0.4]
        self.assertTrue(allclose(actual, expected))

        # test with 2d-array
        actual = lasso.soft_threshold_float(xp.array([[0.1, -2.0, 1.4],
                                                      [1.1, 3.0, -1.4]]),
                                            1.0, xp=xp)
        expected = [[0.0, -1.0, 0.4],
                    [0.1, 2.0, -0.4]]
        self.assertTrue(allclose(actual, expected))

    def test_soft_threshold_complex(self):
        # test with 1d-array
        # real
        x = xp.array([0.1, -2.0, 1.4])
        actual = lasso.soft_threshold_complex(x + 0.0j, 1.0, xp)
        expected = lasso.soft_threshold_complex(x, 1.0, xp)
        self.assertTrue(allclose(actual, expected))
        # imag
        actual = lasso.soft_threshold_complex(x * 1.0j, 1.0, xp)
        expected = lasso.soft_threshold_complex(x, 1.0, xp) * 1.0j
        self.assertTrue(allclose(actual, expected))

        # 45deg
        actual = lasso.soft_threshold_complex(x + x * 1.0j, 1.0, xp)
        expected = lasso.soft_threshold_complex(x * np.sqrt(2.0), 1.0, xp)\
            * (np.sqrt(0.5) + np.sqrt(0.5) * 1.0j)
        self.assertTrue(allclose(actual, expected))
        # -45deg
        actual = lasso.soft_threshold_complex(x - x * 1.0j, 1.0, xp)
        expected = lasso.soft_threshold_float(x * np.sqrt(2.0), 1.0, xp)\
            * (np.sqrt(0.5) - np.sqrt(0.5) * 1.0j)
        self.assertTrue(allclose(actual, expected))


class TestLassoError(unittest.TestCase):
    """ Unit tests for """
    def test_error_in_shape(self):
        # with vector input
        # wrong A shape
        with self.assertRaises(ShapeMismatchError):
            lasso.solve(np.random.randn(5), np.random.randn(3, 4), alpha=1.0)
        # wrong x shape
        with self.assertRaises(ShapeMismatchError):
            lasso.solve(np.random.randn(5), np.random.randn(3, 5),
                        x=np.random.randn(4), alpha=1.0)
        # wrong mask shape
        with self.assertRaises(ShapeMismatchError):
            lasso.solve(np.random.randn(5), np.random.randn(3, 5),
                        x=np.random.randn(3), alpha=1.0,
                        mask=np.random.uniform(0, 1, 4))

        # with matrix input
        # wrong A shape
        with self.assertRaises(ShapeMismatchError):
            lasso.solve(np.random.randn(2, 5), np.random.randn(3, 4),
                        alpha=1.0)
        # wrong x shape
        with self.assertRaises(ShapeMismatchError):
            lasso.solve(np.random.randn(2, 5), np.random.randn(3, 5),
                        x=np.random.randn(1, 3), alpha=1.0)
        # wrong mask shape
        with self.assertRaises(ShapeMismatchError):
            lasso.solve(np.random.randn(2, 5), np.random.randn(3, 5),
                        x=np.random.randn(2, 3), alpha=1.0,
                        mask=np.random.uniform(0, 1, 8).reshape(2, 4))

        # with tensor input
        # wrong A shape
        with self.assertRaises(ShapeMismatchError):
            lasso.solve(np.random.randn(2, 4, 5),
                        np.random.randn(3, 4), alpha=1.0)
        # wrong x shape
        with self.assertRaises(ShapeMismatchError):
            lasso.solve(np.random.randn(2, 4, 5), np.random.randn(3, 5),
                        x=np.random.randn(2, 3, 3), alpha=1.0)
        # wrong mask shape
        with self.assertRaises(ShapeMismatchError):
            lasso.solve(np.random.randn(2, 4, 5), np.random.randn(3, 5),
                        x=np.random.randn(2, 4, 3), alpha=1.0,
                        mask=np.random.uniform(0, 1, 24).reshape(2, 4, 3))

    def test_error_in_dtype(self):
        # with vector input
        # mismatch dtype
        with self.assertRaises(DtypeMismatchError):
            lasso.solve(np.random.randn(5).astype(float),
                        np.random.randn(3, 4).astype(complex), alpha=1.0)
        with self.assertRaises(DtypeMismatchError):
            lasso.solve(np.random.randn(5).astype(float),
                        np.random.randn(3, 4).astype(float), alpha=1.0,
                        mask=np.random.randn(3, 4).astype(int))
        with self.assertRaises(DtypeMismatchError):
            lasso.solve(np.random.randn(5).astype(float),
                        np.random.randn(3, 4).astype(float), alpha=1.0,
                        mask=np.random.randn(3, 4).astype(complex))
        # float64 and float32 are also incompatible
        with self.assertRaises(DtypeMismatchError):
            lasso.solve(np.random.randn(5).astype(np.float32),
                        np.random.randn(3, 4).astype(np.float64), alpha=1.0)


class TestCase(unittest.TestCase):
    def error(self, x, alpha, mask):
        if mask is None:
            mask = xp.ones(self.y.shape, dtype=float_type(x.dtype))
        alpha = alpha * xp.sum(mask, axis=-1, keepdims=True)
        loss = xp.sum(0.5 / alpha * xp.square(xp.abs(
                self.y - xp.tensordot(x, self.A, axes=1))) * mask)
        return loss + xp.sum(xp.abs(x))

    def assert_minimum(self, x, alpha, tol, n=100, mask=None, message=None):
        loss = self.error(x, alpha, mask)
        for _ in range(n):
            dx = self.randn(*x.shape) * tol
            assert loss <= self.error(x + dx, alpha, mask) * (1.0 + self.tol), message

    def message(self, alpha, method):
        return '{0:s}, alpha: {1:f}'.format(method, alpha)

    @property
    def tol(self):
        return 1.0e-6

    @property
    def tol_mask(self):
        return 1.0e-6

    def uniform(self, low=0, high=1, size=None):
        return xp.array(self.rng.uniform(low, high, size=size))


class TestLasso(TestCase):
    def randn(self, *shape):
        return xp.array(self.rng.randn(*shape))

    def setUp(self):
        self.rng = np.random.RandomState(0)
        self.A = self.randn(5, 10)
        self.x_true = self.randn(5) * xp.rint(self.uniform(size=5))
        self.y = xp.dot(self.x_true,
                        self.A) + self.randn(10) * 0.1
        self.mask = xp.rint(self.uniform(0.4, 1, size=10))

    def _test(self, alpha, method):
        it, x = lasso.solve(self.y, self.A, alpha=alpha, tol=self.tol,
                            method=method, maxiter=1000)
        assert it < 1000 - 1, self.message(alpha, method)
        self.assert_minimum(x, alpha, tol=self.tol * 10.0,
                            message=self.message(alpha, method))
        # x should not be all zero
        assert not allclose(x, xp.zeros_like(x)), self.message(alpha, method)

    def _test_mask(self, alpha, method):
        it, x = lasso.solve(self.y, self.A, alpha=alpha, tol=self.tol_mask,
                            method=method, maxiter=1000, mask=self.mask)
        assert it < 1000 - 1, self.message(alpha, method)
        self.assert_minimum(x, alpha, tol=self.tol_mask * 10.0, mask=self.mask,
                            message=self.message(alpha, method))
        # x should not be all zero
        assert not allclose(x, xp.zeros_like(x)), self.message(alpha, method)
        # all 1 mask should be equiavalent to no-mask
        it, x_no_mask = lasso.solve(self.y, self.A, alpha=alpha, tol=self.tol,
                                    method=method, maxiter=1000)
        it, x = lasso.solve(self.y, self.A, alpha=alpha, tol=self.tol,
                            method=method, maxiter=1000,
                            mask=xp.ones(self.mask.shape,
                                         dtype=float_type(self.y.dtype)))
        assert allclose(x, x_no_mask, atol=self.tol), self.message(
                                                                alpha, method)

    def test(self):
        for alpha in [0.1, 1.0]:
            for method in lasso.AVAILABLE_METHODS:
                self._test(alpha, method)

    def test_mask(self):
        for alpha in [0.1, 1.0]:
            for method in lasso.AVAILABLE_METHODS:
                self._test_mask(alpha, method)

    def _test_mask_1d(self, alpha, method):
        # make sure 1d mask and tiled mask gives the same result.
        mask_1d = xp.rint(self.uniform(0.4, 1.0, size=self.y.shape[-1]))
        mask = xp.ones(self.y.shape, dtype=float_type(self.y.dtype)) * mask_1d

        it, x = lasso.solve(self.y, self.A, alpha=alpha, tol=self.tol,
                            method=method, maxiter=1000, mask=mask_1d)
        assert it < 1000 - 1, self.message(alpha, method)
        self.assert_minimum(x, alpha, tol=self.tol*10.0, mask=mask,
                            message=self.message(alpha, method))
        # x should not be all zero
        assert not allclose(x, xp.zeros_like(x)), self.message(alpha, method)

        it, x_tiled = lasso.solve(self.y, self.A, alpha=alpha, tol=self.tol,
                                  method=method, maxiter=1000, mask=mask)
        assert allclose(x, x_tiled, atol=self.tol*10,
                        rtol=1.0e-3), self.message(alpha, method)

    def test_mask1d(self):
        for alpha in [0.01, 0.1]:
            for method in lasso.AVAILABLE_METHODS:
                self._test_mask_1d(alpha, method)


class TestLassoMatrix(TestLasso):
    def setUp(self):
        self.rng = np.random.RandomState(0)
        self.A = self.randn(5, 10)
        x_true = self.randn(55) * xp.rint(self.uniform(size=55))
        self.x_true = x_true.reshape(11, 5)
        self.y = xp.dot(self.x_true,
                        self.A) + self.randn(11, 10) * 0.1
        v = self.uniform(0.4, 1.0, size=110).reshape(11, 10)
        self.mask = xp.rint(v)


class TestLassoTensor(TestLasso):
    def setUp(self):
        self.rng = np.random.RandomState(0)
        self.A = self.randn(5, 10)
        x_true = self.randn(660) * xp.rint(self.uniform(size=660))
        self.x_true = x_true.reshape(12, 11, 5)
        self.y = xp.dot(self.x_true,
                        self.A) + self.randn(12, 11, 10) * 0.1
        v = self.uniform(0.4, 1, size=1320).reshape(12, 11, 10)
        self.mask = xp.rint(v)


class TestLasso_complex(TestLasso):
    def randn(self, *shape):
        return xp.array(self.rng.randn(*shape) + self.rng.randn(*shape) * 1.0j)


class TestLasso_complexMatrix(TestLassoMatrix):
    def randn(self, *shape):
        return xp.array(self.rng.randn(*shape) + self.rng.randn(*shape) * 1.0j)


class TestLasso_complexTensor(TestLassoTensor):
    def randn(self, *shape):
        return xp.array(self.rng.randn(*shape) + self.rng.randn(*shape) * 1.0j)

    @property  # TODO here this does not well converge
    def tol_mask(self):
        return 1.0e-4


class TestLassoMatrix_float32(TestLassoMatrix):
    def randn(self, *shape):
        return xp.array(self.rng.randn(*shape).astype(np.float32))

    def setUp(self):
        super(TestLassoMatrix_float32, self).setUp()
        self.A = self.A.astype(np.float32)
        self.x_true = self.x_true.astype(np.float32)
        self.y = self.y.astype(np.float32)
        self.mask = self.mask.astype(np.float32)


class TestNnls(TestLasso):
    def setUp(self):
        self.rng = np.random.RandomState(0)
        self.A = self.randn(5, 10)
        self.x_true = np.maximum(self.randn(5), 0.0)
        self.y = xp.dot(self.x_true,
                        self.A) + self.randn(10) * 0.1
        self.mask = xp.rint(self.uniform(0.4, 1, size=10))

    def error(self, x, alpha, mask):
        x = xp.maximum(x, 0.0)
        if mask is None:
            mask = xp.ones(self.y.shape, dtype=float_type(x.dtype))
        alpha = alpha * xp.sum(mask, axis=-1, keepdims=True)
        loss = xp.sum(0.5 / alpha * xp.square(xp.abs(
                self.y - xp.tensordot(x, self.A, axes=1))) * mask)
        return loss + xp.sum(xp.abs(x))

    def test(self):
        for alpha in [0.01, 0.1]:
            for method in lasso.AVAILABLE_NNLS_METHODS:
                self._test(alpha, method)

    def test_mask(self):
        for alpha in [0.01, 0.1]:
            for method in lasso.AVAILABLE_NNLS_METHODS:
                self._test_mask(alpha, method)

    def test_mask1d(self):
        for alpha in [0.001, 0.01]:
            for method in lasso.AVAILABLE_NNLS_METHODS:
                self._test_mask_1d(alpha, method)


class TestLasso_equivalence(TestCase):
    """
    All the methods should get the global minimum.
    """
    def randn(self, *shape):
        return xp.array(self.rng.randn(*shape))

    @property
    def alpha(self):
        return 0.1

    def setUp(self):
        self.rng = np.random.RandomState(0)
        self.A = self.randn(5, 10)
        x_true = self.randn(55) * xp.rint(self.uniform(size=55))
        self.x_true = x_true.reshape(11, 5)
        self.y = xp.dot(self.x_true,
                        self.A) + self.randn(11, 10) * 0.1
        v = self.uniform(0.4, 1.0, size=110).reshape(11, 10)
        self.mask = xp.rint(v)

        self.it, self.x = lasso.solve(
                    self.y, self.A, alpha=self.alpha, tol=1.0e-6,
                    method='ista', maxiter=1000)
        self.mask_it, self.mask_x = lasso.solve(
                    self.y, self.A, alpha=self.alpha, tol=1.0e-6,
                    method='ista', maxiter=1000, mask=self.mask)
        self.methods = list(lasso.AVAILABLE_METHODS)
        self.methods.remove('ista')

    def test_compare(self):
        for method in self.methods:
            it, x = lasso.solve(self.y, self.A, alpha=self.alpha, tol=1.0e-6,
                                method=method, maxiter=1000)
            assert it < 1000 - 1, self.message(self.alpha, method)

            if method != 'fista':
                assert allclose(x - self.x, 0.0, atol=1.0e-4),\
                        self.message(self.alpha, method)
            # x should not be all zero
            assert not allclose(x, xp.zeros_like(x)), self.message(self.alpha,
                                                                   method)

    def test_compare_mask(self):
        for method in self.methods:
            it, x = lasso.solve(self.y, self.A, alpha=self.alpha, tol=1.0e-6,
                                method=method, maxiter=1000, mask=self.mask)
            assert it < 1000 - 1, self.message(self.alpha, method)

            if method != 'fista':
                assert allclose(x - self.mask_x, 0.0, atol=1.0e-4),\
                        self.message(self.alpha, method)
            # x should not be all zero
            assert not allclose(x, xp.zeros_like(x)), self.message(self.alpha,
                                                                   method)


class TestLasso_equivalence_complex(TestLasso_equivalence):
    """
    All the methods should get the global minimum.
    """
    def randn(self, *shape):
        return xp.array(self.rng.randn(*shape) + self.rng.randn(*shape) * 1.0j)


class TestLasso_equivalence_illcondition(TestLasso_equivalence):
    """
    With an ill-conditioned problem with regularization.
    """
    @property
    def alpha(self):
        return 0.5

    def setUp(self):
        self.rng = np.random.RandomState(0)
        self.A = self.randn(8, 5)
        x_true = self.randn(10 * 8) * xp.rint(self.uniform(size=10 * 8))
        self.x_true = x_true.reshape(10, 8)
        self.y = xp.dot(self.x_true,
                        self.A) + self.randn(10, 5) * 0.1
        v = self.uniform(0.4, 1.0, size=10 * 5).reshape(10, 5)
        self.mask = xp.rint(v)

        self.it, self.x = lasso.solve(
                    self.y, self.A, alpha=self.alpha, tol=1.0e-6,
                    method='ista', maxiter=1000)
        self.mask_it, self.mask_x = lasso.solve(
                    self.y, self.A, alpha=self.alpha, tol=1.0e-6,
                    method='ista', maxiter=1000, mask=self.mask)
        self.methods = list(lasso.AVAILABLE_METHODS)
        self.methods.remove('ista')


class TestLasso_equivalence_illcondition_complex(TestLasso_equivalence):
    """
    All the methods should get the global minimum.
    """
    def randn(self, *shape):
        return xp.array(self.rng.randn(*shape) + self.rng.randn(*shape) * 1.0j)

    @property
    def alpha(self):
        return 1.0


class TestLasso_bad_condition(TestCase):
    """
    The solution must be found even with various alpha
    """
    def randn(self, *shape):
        return xp.array(self.rng.randn(*shape))

    def setUp(self):
        self.rng = np.random.RandomState(0)
        # The design matrix is highly correlated
        self.A = self.randn(9, 10) + self.randn(10) * 0.3
        x_true = self.randn(99) * xp.rint(self.uniform(size=99))
        self.x_true = x_true.reshape(11, 9)
        self.y = xp.dot(self.x_true,
                        self.A) + self.randn(11, 10) * 0.1
        v = self.uniform(0.4, 1.0, size=110).reshape(11, 10)
        self.mask = xp.rint(v)
        self.methods = list(lasso.AVAILABLE_METHODS)

    def test(self):
        alphas = np.exp(np.linspace(np.log(0.1), np.log(10.0), 3))
        for method in self.methods:
            for alpha in alphas:
                it, x = lasso.solve(self.y, self.A, alpha=alpha, tol=1.0e-6,
                                    method=method, maxiter=3000)
                assert it < 3000 - 1, self.message(alpha, method)
                self.assert_minimum(x, alpha, tol=1.0e-5,
                                    message=self.message(alpha, method))

    def test_mask(self):
        alphas = np.exp(np.linspace(np.log(0.1), np.log(10.0), 3))
        for method in self.methods:
            for alpha in alphas:
                it, x = lasso.solve(self.y, self.A, alpha=alpha, tol=1.0e-6,
                                    method=method, maxiter=3000,
                                    mask=self.mask)
                assert it < 3000 - 1, self.message(alpha, method)
                self.assert_minimum(x, alpha, tol=1.0e-5, mask=self.mask,
                                    message=self.message(alpha, method))


class TestLasso_bad_condition2(TestCase):
    def setUp(self):
        self.rng = np.random.RandomState(0)
        # The design matrix is highly correlated
        self.A = self.randn(5, 10) + self.randn(10) * 0.3
        x_true = self.randn(55) * xp.rint(self.uniform(size=55))
        self.x_true = x_true.reshape(11, 5)
        self.y = xp.dot(self.x_true,
                        self.A) + self.randn(11, 10) * 0.1
        v = self.uniform(0.4, 1.0, size=110).reshape(11, 10)
        self.mask = xp.rint(v)
        self.methods = list(lasso.AVAILABLE_METHODS)


if __name__ == '__main__':
    unittest.main()
