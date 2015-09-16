import unittest
import copy
import sys
import logging

from nose.tools import assert_true
import numpy as np
from numpy.testing import assert_array_almost_equal
from sklearn import mixture
from sklearn.datasets import make_blobs

# set up logging to file - see previous section for more details
logging.basicConfig(level=logging.DEBUG,
                    format='%(asctime)s %(name)-12s %(levelname)-8s %(message)s',
                    datefmt='%m-%d %H:%M')
# define a Handler which writes INFO messages or higher to the sys.stderr
console = logging.StreamHandler()
console.setLevel(logging.INFO)
# set a format which is simpler for console use
formatter = logging.Formatter('%(name)-12s: %(levelname)-8s %(message)s')
# tell the handler to use this format
console.setFormatter(formatter)
# add the handler to the root logger
logging.getLogger('').addHandler(console)


def test_dummy():
    # This is a dummy test
    var1 = 1
    var2 = 2
    result = var1 + var2
    assert_true(result == 3)
    logging.info('DummyTest: Result should be three: ' + str(result))
    logging.info('DummyTest: OK')


def test_default_create_pgmm():
    model = mixture.PGMM()
    assert_true(model.n_components == 1, msg='Wrong n_components')
    assert_true(model.n_pc == 1, msg='Wrong n_pc')
    assert_true(model.covariance_type == 'UUR', msg='Wrong covariance_type')
    assert_true(model.tol == 1e-3, msg='Wrong tol')
    assert_true(model.min_covar == 1e-7, msg='Wrong min_covar')
    assert_true(model.random_state == None, msg='Wrong random_state')
    assert_true(model.n_iter == 100, msg='Wrong n_iter')
    assert_true(model.n_init == 1, msg='Wrong n_init')
    assert_true(model.params == 'wmpn', msg='Wrong params')
    assert_true(model.init_params == 'wmpn', msg='Wrong init_params')
    assert_true(model.verbose == 0, msg='Wrong verbose')
    assert_true(model.weights_.shape == (1,), msg='Wrong weights_')
    assert_true(~model.converged_, msg='Wrong converged_')
    logging.info('DefaultCreationTest: OK')

def test_custom_create_pgmm():
    model = mixture.PGMM(n_components=2,
                          n_pc=2,
                          covariance_type='UUU',
                          random_state=1,
                          tol=1e-6,
                          min_covar=1e-6,
                          n_iter=10,
                          n_init=2,
                          params='w',
                          init_params='m',
                          verbose=2)
    assert_true(model.n_components == 2, msg='Wrong n_components')
    assert_true(model.n_pc == 2, msg='Wrong n_pc')
    assert_true(model.covariance_type == 'UUU', msg='Wrong covariance_type')
    assert_true(model.tol == 1e-6, msg='Wrong tol')
    assert_true(model.min_covar == 1e-6, msg='Wrong min_covar')
    assert_true(model.random_state == 1, msg='Wrong random_state')
    assert_true(model.n_iter == 10, msg='Wrong n_iter')
    assert_true(model.n_init == 2, msg='Wrong n_init')
    assert_true(model.params == 'w', msg='Wrong params')
    assert_true(model.init_params == 'm', msg='Wrong init_params')
    assert_true(model.verbose == 2, msg='Wrong verbose')
    assert_true(model.weights_.shape == (2,), msg='Wrong weights_')
    assert_true(~model.converged_, msg='Wrong converged_')
    logging.info('CustomCreationTest: OK')


def test_simple_fit_mean():
    n_samples = 10000
    n_features = 5
    centers = np.array([[10, 5, 1, -5, -10]])
    X, y = make_blobs(n_features=n_features,
                      n_samples=n_samples,
                      centers=centers)
    model = mixture.PGMM()
    model.fit(X)
    assert_array_almost_equal(model.means_, centers, decimal=1)
    logging.info('TestSimpleFitMeans: OK')