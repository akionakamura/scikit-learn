import unittest
import copy
import sys
import logging

from nose.tools import assert_true
import numpy as np
from numpy.testing import (assert_array_equal, assert_array_almost_equal,
                           assert_raises)
from scipy import stats
from sklearn import mixture
from sklearn.datasets.samples_generator import make_spd_matrix
from sklearn.utils.testing import assert_greater
from sklearn.utils.testing import assert_raise_message
from sklearn.metrics.cluster import adjusted_rand_score
from sklearn.externals.six.moves import cStringIO as StringIO

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


def test_default_create_mppca():
    model = mixture.MPPCA()
    assert_true(model.n_components == 1, msg='Wrong n_components')
    assert_true(model.n_pc == 1, msg='Wrong n_pc')
    assert_true(model.tol == 1e-3, msg='Wrong tol')
    assert_true(model.min_covar == 1e-3, msg='Wrong min_covar')
    assert_true(model.random_state == None, msg='Wrong random_state')
    assert_true(model.n_iter == 100, msg='Wrong n_iter')
    assert_true(model.n_init == 1, msg='Wrong n_init')
    assert_true(model.params == 'wmpn', msg='Wrong params')
    assert_true(model.init_params == 'wmpn', msg='Wrong init_params')
    assert_true(model.verbose == 0, msg='Wrong verbose')
    assert_true(model.weights_.shape == (1,), msg='Wrong weights_')
    assert_true(~model.converged_, msg='Wrong converged_')
    logging.info('DefaultCreationTest: OK')


def test_custom_create_mppca():
    model = mixture.MPPCA(n_components=2, n_pc=2,
                          random_state=1, tol=1e-6, min_covar=1e-6,
                          n_iter=10, n_init=2, params='w', init_params='m',
                          verbose=2)
    assert_true(model.n_components == 2, msg='Wrong n_components')
    assert_true(model.n_pc == 2, msg='Wrong n_pc')
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

def test_set_get_covars():
    n_components=2
    principal_subspace = np.tile(np.array([[1, 2], [3, 4], [5, 6]]), (n_components, 1, 1))
    noise = np.tile(2.0, n_components)
    expected_result = np.dot(principal_subspace[0], principal_subspace[0].T) + noise[0] * np.eye(3)
    model = mixture.MPPCA(n_components=n_components)
    model._set_covars(principal_subspace, noise)
    result = model._get_covars()
    for comp in range(n_components):
        assert_true(np.array_equal(result[comp], expected_result), msg='Wrong covar')
    logging.info('CovarSetAndGetTest: OK')