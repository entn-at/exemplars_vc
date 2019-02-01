import numpy as np
import unittest
from decomp.utils.cp_compat import numpy_or_cupy as xp
from decomp import template_matching as template
import decomp.utils.normalize as normalize

from .testings import allclose


class TestUtils(unittest.TestCase):
    def test_valid(self):
        # stride 1
        # n_template: 2, template_size: 3
        D = xp.array([[1, 2, 3], [4, 5, 6]], dtype=np.float32)
        x = xp.array([[1, 2, 3, 4, 5], [10, 20, 30, 40, 50]], dtype=np.float32)
        size, stride, padding = 7, 1, 'VALID'
        dmat = template._temp2mat(D, size, stride, padding, xp=xp)
        assert dmat.shape[1] == template._coef_size(D.shape[-1], size, stride,
                                                    padding)
        expected = xp.array([[[1, 2, 3, 0, 0, 0, 0],
                              [0, 1, 2, 3, 0, 0, 0],
                              [0, 0, 1, 2, 3, 0, 0],
                              [0, 0, 0, 1, 2, 3, 0],
                              [0, 0, 0, 0, 1, 2, 3]],
                             [[4, 5, 6, 0, 0, 0, 0],
                              [0, 4, 5, 6, 0, 0, 0],
                              [0, 0, 4, 5, 6, 0, 0],
                              [0, 0, 0, 4, 5, 6, 0],
                              [0, 0, 0, 0, 4, 5, 6]]], dtype=xp.float32)
        assert allclose(dmat, expected)

        xmat = template._coef2mat(x, size, D.shape[-1], stride, padding,
                                  xp=xp)
        expected = xp.array([[[1, 2, 3, 4, 5, 0, 0],
                              [0, 1, 2, 3, 4, 5, 0],
                              [0, 0, 1, 2, 3, 4, 5]],
                             [[10, 20, 30, 40, 50, 0, 0],
                              [0, 10, 20, 30, 40, 50, 0],
                              [0, 0, 10, 20, 30, 40, 50]]], dtype=xp.float32)
        assert allclose(xmat, expected)
        assert allclose(xp.tensordot(x, dmat, 2), xp.tensordot(D, xmat, 2))

        # stride 2
        size, stride, padding = 7, 2, 'VALID'
        dmat = template._temp2mat(D, size, stride, padding, xp=xp)
        assert dmat.shape[1] == template._coef_size(D.shape[-1], size, stride,
                                                    padding)

        x = xp.array([[1, 2, 3], [10, 20, 30]], dtype=np.float32)
        xmat = template._coef2mat(x, size, D.shape[-1], stride, padding, xp=xp)
        expected = xp.array([[[1, 0, 2, 0, 3, 0, 0],
                              [0, 1, 0, 2, 0, 3, 0],
                              [0, 0, 1, 0, 2, 0, 3]],
                             [[10, 0, 20, 0, 30, 0, 0],
                              [0, 10, 0, 20, 0, 30, 0],
                              [0, 0, 10, 0, 20, 0, 30]]], dtype=xp.float32)
        assert allclose(xmat, expected)
        assert allclose(xp.tensordot(x, dmat, 2), xp.tensordot(D, xmat, 2))

        # stride 2
        size, stride, padding = 6, 2, 'VALID'
        dmat = template._temp2mat(D, size, stride, padding, xp=xp)
        assert dmat.shape[1] == template._coef_size(D.shape[-1], size, stride,
                                                    padding)
        expected = xp.array([[[0, 1, 2, 3, 0, 0],
                              [0, 0, 0, 1, 2, 3]],
                             [[0, 4, 5, 6, 0, 0],
                              [0, 0, 0, 4, 5, 6]]], dtype=xp.float32)
        assert allclose(dmat, expected)

        x = xp.array([[1, 2], [10, 20]], dtype=np.float32)
        xmat = template._coef2mat(x, size, D.shape[-1], stride, padding,
                                  xp=xp)
        expected = xp.array([[[0, 1, 0, 2, 0, 0],
                              [0, 0, 1, 0, 2, 0],
                              [0, 0, 0, 1, 0, 2]],
                             [[0, 10, 0, 20, 0, 0],
                              [0, 0, 10, 0, 20, 0],
                              [0, 0, 0, 10, 0, 20]]], dtype=xp.float32)
        assert allclose(xmat, expected)
        assert allclose(xp.tensordot(x, dmat, 2), xp.tensordot(D, xmat, 2))

    def test_same(self):
        # stride 1
        # n_template: 2, template_size: 3
        D = xp.array([[1, 2, 3], [4, 5, 6]], dtype=np.float32)
        x = xp.array([[1, 2, 3, 4, 5, 6, 7, 8, 9],
                      [10, 20, 30, 40, 50, 60, 70, 80, 90]], dtype=np.float32)
        size, stride, padding = 7, 1, 'SAME'
        dmat = template._temp2mat(D, size, stride, padding, xp=xp)
        assert dmat.shape[1] == template._coef_size(D.shape[-1], size, stride,
                                                    padding)

        expected = xp.array([[[3, 0, 0, 0, 0, 0, 0],
                              [2, 3, 0, 0, 0, 0, 0],
                              [1, 2, 3, 0, 0, 0, 0],
                              [0, 1, 2, 3, 0, 0, 0],
                              [0, 0, 1, 2, 3, 0, 0],
                              [0, 0, 0, 1, 2, 3, 0],
                              [0, 0, 0, 0, 1, 2, 3],
                              [0, 0, 0, 0, 0, 1, 2],
                              [0, 0, 0, 0, 0, 0, 1]],
                             [[6, 0, 0, 0, 0, 0, 0],
                              [5, 6, 0, 0, 0, 0, 0],
                              [4, 5, 6, 0, 0, 0, 0],
                              [0, 4, 5, 6, 0, 0, 0],
                              [0, 0, 4, 5, 6, 0, 0],
                              [0, 0, 0, 4, 5, 6, 0],
                              [0, 0, 0, 0, 4, 5, 6],
                              [0, 0, 0, 0, 0, 4, 5],
                              [0, 0, 0, 0, 0, 0, 4]]], dtype=xp.float32)
        assert allclose(dmat, expected)

        xmat = template._coef2mat(x, size, D.shape[-1], stride, padding,
                                  xp=xp)
        expected = xp.array([[[3, 4, 5, 6, 7, 8, 9],
                              [2, 3, 4, 5, 6, 7, 8],
                              [1, 2, 3, 4, 5, 6, 7]],
                             [[30, 40, 50, 60, 70, 80, 90],
                              [20, 30, 40, 50, 60, 70, 80],
                              [10, 20, 30, 40, 50, 60, 70]]], dtype=xp.float32)
        assert allclose(xmat, expected)
        assert allclose(xp.tensordot(x, dmat, 2),
                        xp.tensordot(D, xmat, 2))

        # stride 2
        size, stride, padding = 7, 2, 'SAME'
        dmat = template._temp2mat(D, size, stride, padding, xp=xp)
        assert dmat.shape[1] == template._coef_size(D.shape[-1], size, stride,
                                                    padding)
        expected = xp.array([[[3, 0, 0, 0, 0, 0, 0],
                              [1, 2, 3, 0, 0, 0, 0],
                              [0, 0, 1, 2, 3, 0, 0],
                              [0, 0, 0, 0, 1, 2, 3],
                              [0, 0, 0, 0, 0, 0, 1]],
                             [[6, 0, 0, 0, 0, 0, 0],
                              [4, 5, 6, 0, 0, 0, 0],
                              [0, 0, 4, 5, 6, 0, 0],
                              [0, 0, 0, 0, 4, 5, 6],
                              [0, 0, 0, 0, 0, 0, 4]]], dtype=xp.float32)
        assert allclose(dmat, expected)

        x = xp.array([[1, 2, 3, 4, 5],
                      [10, 20, 30, 40, 50]], dtype=np.float32)
        xmat = template._coef2mat(x, size, D.shape[-1], stride, padding,
                                  xp=xp)
        expected = xp.array([[[2, 0, 3, 0, 4, 0, 5],
                              [0, 2, 0, 3, 0, 4, 0],
                              [1, 0, 2, 0, 3, 0, 4]],
                             [[20, 0, 30, 0, 40, 0, 50],
                              [0, 20, 0, 30, 0, 40, 0],
                              [10, 0, 20, 0, 30, 0, 40]]], dtype=xp.float32)
        assert allclose(xmat, expected)
        assert allclose(xp.tensordot(x, dmat, 2), xp.tensordot(D, xmat, 2))

        # stride 2
        size, stride, padding = 6, 2, 'SAME'
        dmat = template._temp2mat(D, size, stride, padding, xp=xp)
        print(template._coef_size(D.shape[-1], size, stride,
                                                    padding))
        assert dmat.shape[1] == template._coef_size(D.shape[-1], size, stride,
                                                    padding)
        expected = xp.array([[[2, 3, 0, 0, 0, 0],
                              [0, 1, 2, 3, 0, 0],
                              [0, 0, 0, 1, 2, 3],
                              [0, 0, 0, 0, 0, 1]],
                             [[5, 6, 0, 0, 0, 0],
                              [0, 4, 5, 6, 0, 0],
                              [0, 0, 0, 4, 5, 6],
                              [0, 0, 0, 0, 0, 4]]], dtype=xp.float32)
        assert allclose(dmat, expected)

        x = xp.array([[1, 2, 3, 4],
                      [10, 20, 30, 40]], dtype=np.float32)
        xmat = template._coef2mat(x, size, D.shape[-1], stride, padding, xp=xp)
        expected = xp.array([[[2, 0, 3, 0, 4, 0],
                              [0, 2, 0, 3, 0, 4],
                              [1, 0, 2, 0, 3, 0]],
                             [[20, 0, 30, 0, 40, 0],
                              [0, 20, 0, 30, 0, 40],
                              [10, 0, 20, 0, 30, 0]]], dtype=xp.float32)
        #assert allclose(xmat, expected)
        assert allclose(xp.tensordot(x, dmat, 2), xp.tensordot(D, xmat, 2))

    def test_dot(self):
        for padding in ['VALID', 'SAME']:
            self.one_test_dot(99, 3, 29, 1, padding)
            self.one_test_dot(300, 4, 29, 2, padding)
            self.one_test_dot(301, 4, 29, 2, padding)
            self.one_test_dot(301, 3, 29, 2, padding)
            self.one_test_dot(301, 3, 28, 2, padding)
            self.one_test_dot(100, 1, 9, 3, padding)
            self.one_test_dot(101, 1, 9, 3, padding)
            self.one_test_dot(102, 1, 9, 3, padding)

    def one_test_dot(self, size, n_template, template_size, stride, padding):
        rng = xp.random.RandomState(0)
        d = rng.randn(n_template, template_size)

        dmat = template._temp2mat(d, size, stride, padding, xp=xp)
        assert dmat.shape[1] == template._coef_size(d.shape[-1], size, stride,
                                                    padding)
        x = rng.randn(dmat.shape[0], dmat.shape[1])
        xmat = template._coef2mat(x, size, d.shape[-1], stride, padding, xp=xp)
        print(xmat.shape)
        assert allclose(xp.tensordot(x, dmat, 2),
                        xp.tensordot(d, xmat, 2))
        # matrix dot expression
        fit = xp.tensordot(x, dmat, 2)
        assert allclose(xp.dot(x.reshape(-1),
                               dmat.reshape(-1, dmat.shape[-1])), fit)
        assert allclose(xp.dot(d.reshape(-1),
                               xmat.reshape(-1, xmat.shape[-1])), fit)
        # with batch x
        batch_size = 4
        x = rng.randn(batch_size, dmat.shape[0], dmat.shape[1])
        xmat = template._coef2mat(x, size, d.shape[-1], stride, padding, xp=xp)

        assert allclose(xp.tensordot(x, dmat, 2),
                        xp.tensordot(d, xmat, ((0, 1), (-3, -2))))


class TestCase(unittest.TestCase):
    def error(self, x, D, alpha, mask):
        mask = xp.ones(self.y.shape, dtype=float) if mask is None else mask
        alpha = alpha * xp.sum(mask, axis=-1, keepdims=True)
        D = normalize.l2(D, axis=-1, xp=xp)

        f = template.predict(x, D, self.y.shape[-1], stride=1)
        loss = xp.sum(0.5 / alpha * xp.square(xp.abs(self.y - f) * mask))
        return loss + xp.sum(xp.abs(x))

    def assert_minimum(self, x, D, alpha, tol, n=3, mask=None, message=None):
        loss = self.error(x, D, alpha, mask)
        for _ in range(n):
            dx = self.randn(*x.shape) * tol
            assert loss < self.error(x + dx, D, alpha, mask), message

    def message(self, alpha, method):
        return '{0:s}, alpha: {1:f}'.format(method, alpha)


class TestTemplateMatching(TestCase):
    def randn(self, *shape):
        return self.rng.randn(*shape)

    def setUp(self):
        self.rng = xp.random.RandomState(0)
        self.Dtrue = self.randn(3, 10) + self.randn(10) * 0.5

        size = 100
        # stride 1
        actual = template._temp2mat(self.Dtrue, size, stride=1, padding='SAME',
                                    xp=xp)
        xtrue = self.randn(actual.shape[0], actual.shape[1])
        self.xtrue = xtrue * xp.rint(
            self.rng.uniform(0.49, 1, size=xtrue.size).reshape(xtrue.shape))
        self.y = template.predict(self.xtrue, self.Dtrue, size, stride=1)
        self.y += self.randn(*self.y.shape) * 0.1

        self.D = self.Dtrue + self.randn(*self.Dtrue.shape) * 1.0

    def _test_run(self):
        D = self.D.copy()
        alpha = 0.1
        maxiter = 1000
        it, D, x = template.solve(self.y, D, alpha, x=None, tol=1.0e-4,
                                  minibatch=None, maxiter=maxiter,
                                  lasso_method='acc_ista', lasso_iter=1000)
        self.assertTrue(it < maxiter - 1)
        self.assert_minimum(x, D, alpha, tol=1.0e-4, n=6)
        self.assertFalse(allclose(x, xp.zeros_like(x)))

    def test_run_minibatch(self):
        D = self.D.copy()
        alpha = 0.1
        maxiter = 1000
        it, D, x = template.solve(self.y, D, alpha, x=None, tol=1.0e-4,
                                  minibatch=3, size_of_minibatch=30,
                                  maxiter=maxiter,
                                  lasso_method='acc_ista', lasso_iter=1000)
        self.assertTrue(it < maxiter - 1)
        self.assert_minimum(x, D, alpha, tol=1.0e-4, n=6)
        self.assertFalse(allclose(x, xp.zeros_like(x)))

'''


class TestTemplateMatching_batch(TestTemplateMatching):
    """ Template matching with batch calculation"""
    def setUp(self):
        self.rng = xp.random.RandomState(0)
        self.Dtrue = self.randn(3, 10) + self.randn(10) * 0.5

        size = 100
        batch = 3
        # stride 1
        actual = template._temp2mat(self.Dtrue, size, stride=1, xp=xp)
        xtrue = self.randn(batch*actual.shape[0]).reshape(
                batch, self.Dtrue.shape[0], -1)
        self.xtrue = xtrue * xp.rint(
                self.rng.uniform(0.49, 1, size=xtrue.size).reshape(xtrue.shape))
        self.y = template.predict(self.xtrue, self.Dtrue, size, stride=1)
        self.y += self.randn(*self.y.shape) * 0.1

        self.D = self.Dtrue + self.randn(*self.Dtrue.shape) * 1.0
'''
