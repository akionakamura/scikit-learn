import unittest
import copy
import sys
import logging

from nose.tools import assert_true
import numpy as np
from numpy.testing import assert_array_almost_equal
from numpy.testing import assert_equal
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


def test_simple_fit_mean_rrr():
    n_samples = 10000
    n_features = 5
    centers = np.array([[10, 5, 1, -5, -10]])
    X, y = make_blobs(n_features=n_features,
                      n_samples=n_samples,
                      centers=centers)
    model = mixture.PGMM(covariance_type='RRR')
    model.fit(X)
    assert_array_almost_equal(model.means_, centers, decimal=1)
    logging.info('TestSimpleFitMeansRRR: OK')


def test_simple_fit_mean_rru():
    n_samples = 10000
    n_features = 5
    centers = np.array([[10, 5, 1, -5, -10]])
    X, y = make_blobs(n_features=n_features,
                      n_samples=n_samples,
                      centers=centers)
    model = mixture.PGMM(covariance_type='RRU')
    model.fit(X)
    assert_array_almost_equal(model.means_, centers, decimal=1)
    logging.info('TestSimpleFitMeansRRU: OK')


def test_simple_fit_mean_rur():
    n_samples = 10000
    n_features = 5
    centers = np.array([[10, 5, 1, -5, -10]])
    X, y = make_blobs(n_features=n_features,
                      n_samples=n_samples,
                      centers=centers)
    model = mixture.PGMM(covariance_type='RUR')
    model.fit(X)
    assert_array_almost_equal(model.means_, centers, decimal=1)
    logging.info('TestSimpleFitMeansRUR: OK')


def test_simple_fit_mean_ruu():
    n_samples = 10000
    n_features = 5
    centers = np.array([[10, 5, 1, -5, -10]])
    X, y = make_blobs(n_features=n_features,
                      n_samples=n_samples,
                      centers=centers)
    model = mixture.PGMM(covariance_type='RUU')
    model.fit(X)
    assert_array_almost_equal(model.means_, centers, decimal=1)
    logging.info('TestSimpleFitMeansRUU: OK')


def test_simple_fit_mean_urr():
    n_samples = 10000
    n_features = 5
    centers = np.array([[10, 5, 1, -5, -10]])
    X, y = make_blobs(n_features=n_features,
                      n_samples=n_samples,
                      centers=centers)
    model = mixture.PGMM(covariance_type='URR')
    model.fit(X)
    assert_array_almost_equal(model.means_, centers, decimal=1)
    logging.info('TestSimpleFitMeansURR: OK')


def test_simple_fit_mean_uru():
    n_samples = 10000
    n_features = 5
    centers = np.array([[10, 5, 1, -5, -10]])
    X, y = make_blobs(n_features=n_features,
                      n_samples=n_samples,
                      centers=centers)
    model = mixture.PGMM(covariance_type='URU')
    model.fit(X)
    assert_array_almost_equal(model.means_, centers, decimal=1)
    logging.info('TestSimpleFitMeansURU: OK')


def test_simple_fit_mean_uur():
    n_samples = 10000
    n_features = 5
    centers = np.array([[10, 5, 1, -5, -10]])
    X, y = make_blobs(n_features=n_features,
                      n_samples=n_samples,
                      centers=centers)
    model = mixture.PGMM(covariance_type='UUR')
    model.fit(X)
    assert_array_almost_equal(model.means_, centers, decimal=1)
    logging.info('TestSimpleFitMeansUUR: OK')


def test_simple_fit_mean_uuu():
    n_samples = 10000
    n_features = 5
    centers = np.array([[10, 5, 1, -5, -10]])
    X, y = make_blobs(n_features=n_features,
                      n_samples=n_samples,
                      centers=centers)
    model = mixture.PGMM(covariance_type='UUU')
    model.fit(X)
    assert_array_almost_equal(model.means_, centers, decimal=1)
    logging.info('TestSimpleFitMeansUUU: OK')


def test_fit_rrr():
    n_samples1 = 10000
    n_features = 5
    centers1 = np.array([[10, 5, 1, -5, -10],
                        [-10, -5, -1, 5, 10]])
    cluster_std1 = np.array([[1.0, 2.0, 3.0, 4.0, 5.0],
                            [5.0, 4.0, 3.0, 2.0, 1.0]])

    X1, y1 = make_blobs(n_features=n_features,
                      n_samples=n_samples1,
                      centers=centers1,
                      cluster_std=cluster_std1)

    n_samples2 = 5000
    centers2 = np.array([[10, 5, 1, -5, -10]])
    cluster_std2 = np.array([[1.0, 2.0, 3.0, 4.0, 5.0]])
    X2, y2 = make_blobs(n_features=n_features,
                      n_samples=n_samples2,
                      centers=centers2,
                      cluster_std=cluster_std2)
    X = np.vstack((X1, X2))

    model = mixture.PGMM(covariance_type='RRR', n_components=2, n_pc=3)
    model.fit(X)
    assert_array_almost_equal(np.sum(model.means_, 0), np.sum(centers1, 0), decimal=0)
    assert_array_almost_equal(np.sort(model.weights_), np.array([0.333, 0.666]), decimal=1)
    assert_equal(model.means_.shape, np.array([2, n_features]))
    assert_equal(model.weights_.shape, np.array([2]))
    assert_equal(model.noise_.shape, np.array([1]))
    assert_equal(model.principal_subspace_.shape, np.array([n_features, 3]))
    assert_equal(model.covars_.shape, np.array([2, n_features, n_features]))
    logging.info('TestFitRRR: OK')


def test_fit_rru():
    n_samples1 = 10000
    n_features = 5
    centers1 = np.array([[10, 5, 1, -5, -10],
                        [-10, -5, -1, 5, 10]])
    cluster_std1 = np.array([[1.0, 2.0, 3.0, 4.0, 5.0],
                            [5.0, 4.0, 3.0, 2.0, 1.0]])

    X1, y1 = make_blobs(n_features=n_features,
                      n_samples=n_samples1,
                      centers=centers1,
                      cluster_std=cluster_std1)

    n_samples2 = 5000
    centers2 = np.array([[10, 5, 1, -5, -10]])
    cluster_std2 = np.array([[1.0, 2.0, 3.0, 4.0, 5.0]])
    X2, y2 = make_blobs(n_features=n_features,
                      n_samples=n_samples2,
                      centers=centers2,
                      cluster_std=cluster_std2)
    X = np.vstack((X1, X2))

    model = mixture.PGMM(covariance_type='RRU', n_components=2, n_pc=3)
    model.fit(X)
    assert_array_almost_equal(np.sum(model.means_, 0), np.sum(centers1, 0), decimal=0)
    assert_array_almost_equal(np.sort(model.weights_), np.array([0.333, 0.666]), decimal=1)
    assert_equal(model.means_.shape, np.array([2, n_features]))
    assert_equal(model.weights_.shape, np.array([2]))
    assert_equal(model.noise_.shape, np.array([n_features]))
    assert_equal(model.principal_subspace_.shape, np.array([n_features, 3]))
    assert_equal(model.covars_.shape, np.array([2, n_features, n_features]))
    logging.info('TestFitRRU: OK')


def test_fit_rur():
    n_samples1 = 10000
    n_features = 5
    centers1 = np.array([[10, 5, 1, -5, -10],
                        [-10, -5, -1, 5, 10]])
    cluster_std1 = np.array([[1.0, 2.0, 3.0, 4.0, 5.0],
                            [5.0, 4.0, 3.0, 2.0, 1.0]])

    X1, y1 = make_blobs(n_features=n_features,
                      n_samples=n_samples1,
                      centers=centers1,
                      cluster_std=cluster_std1)

    n_samples2 = 5000
    centers2 = np.array([[10, 5, 1, -5, -10]])
    cluster_std2 = np.array([[1.0, 2.0, 3.0, 4.0, 5.0]])
    X2, y2 = make_blobs(n_features=n_features,
                      n_samples=n_samples2,
                      centers=centers2,
                      cluster_std=cluster_std2)
    X = np.vstack((X1, X2))

    model = mixture.PGMM(covariance_type='RUR', n_components=2, n_pc=3)
    model.fit(X)
    assert_array_almost_equal(np.sum(model.means_, 0), np.sum(centers1, 0), decimal=0)
    assert_array_almost_equal(np.sort(model.weights_), np.array([0.333, 0.666]), decimal=1)
    assert_equal(model.means_.shape, np.array([2, n_features]))
    assert_equal(model.weights_.shape, np.array([2]))
    assert_equal(model.noise_.shape, np.array([2]))
    assert_equal(model.principal_subspace_.shape, np.array([n_features, 3]))
    assert_equal(model.covars_.shape, np.array([2, n_features, n_features]))
    logging.info('TestFitRUR: OK')


def test_fit_ruu():
    n_samples1 = 10000
    n_features = 5
    centers1 = np.array([[10, 5, 1, -5, -10],
                        [-10, -5, -1, 5, 10]])
    cluster_std1 = np.array([[1.0, 2.0, 3.0, 4.0, 5.0],
                            [5.0, 4.0, 3.0, 2.0, 1.0]])

    X1, y1 = make_blobs(n_features=n_features,
                      n_samples=n_samples1,
                      centers=centers1,
                      cluster_std=cluster_std1)

    n_samples2 = 5000
    centers2 = np.array([[10, 5, 1, -5, -10]])
    cluster_std2 = np.array([[1.0, 2.0, 3.0, 4.0, 5.0]])
    X2, y2 = make_blobs(n_features=n_features,
                      n_samples=n_samples2,
                      centers=centers2,
                      cluster_std=cluster_std2)
    X = np.vstack((X1, X2))

    model = mixture.PGMM(covariance_type='RUU', n_components=2, n_pc=3)
    model.fit(X)
    assert_array_almost_equal(np.sum(model.means_, 0), np.sum(centers1, 0), decimal=0)
    assert_array_almost_equal(np.sort(model.weights_), np.array([0.333, 0.666]), decimal=1)
    assert_equal(model.means_.shape, np.array([2, n_features]))
    assert_equal(model.weights_.shape, np.array([2]))
    assert_equal(model.noise_.shape, np.array([2, n_features]))
    assert_equal(model.principal_subspace_.shape, np.array([n_features, 3]))
    assert_equal(model.covars_.shape, np.array([2, n_features, n_features]))
    logging.info('TestFitRUU: OK')


def test_fit_urr():
    n_samples1 = 10000
    n_features = 5
    centers1 = np.array([[10, 5, 1, -5, -10],
                        [-10, -5, -1, 5, 10]])
    cluster_std1 = np.array([[1.0, 2.0, 3.0, 4.0, 5.0],
                            [5.0, 4.0, 3.0, 2.0, 1.0]])

    X1, y1 = make_blobs(n_features=n_features,
                      n_samples=n_samples1,
                      centers=centers1,
                      cluster_std=cluster_std1)

    n_samples2 = 5000
    centers2 = np.array([[10, 5, 1, -5, -10]])
    cluster_std2 = np.array([[1.0, 2.0, 3.0, 4.0, 5.0]])
    X2, y2 = make_blobs(n_features=n_features,
                      n_samples=n_samples2,
                      centers=centers2,
                      cluster_std=cluster_std2)
    X = np.vstack((X1, X2))

    model = mixture.PGMM(covariance_type='URR', n_components=2, n_pc=3)
    model.fit(X)
    assert_array_almost_equal(np.sum(model.means_, 0), np.sum(centers1, 0), decimal=0)
    assert_array_almost_equal(np.sort(model.weights_), np.array([0.333, 0.666]), decimal=1)
    assert_equal(model.means_.shape, np.array([2, n_features]))
    assert_equal(model.weights_.shape, np.array([2]))
    assert_equal(model.noise_.shape, np.array([1]))
    assert_equal(model.principal_subspace_.shape, np.array([2, n_features, 3]))
    assert_equal(model.covars_.shape, np.array([2, n_features, n_features]))
    logging.info('TestFitURR: OK')


def test_fit_uru():
    n_samples1 = 10000
    n_features = 5
    centers1 = np.array([[10, 5, 1, -5, -10],
                        [-10, -5, -1, 5, 10]])
    cluster_std1 = np.array([[1.0, 2.0, 3.0, 4.0, 5.0],
                            [5.0, 4.0, 3.0, 2.0, 1.0]])

    X1, y1 = make_blobs(n_features=n_features,
                      n_samples=n_samples1,
                      centers=centers1,
                      cluster_std=cluster_std1)

    n_samples2 = 5000
    centers2 = np.array([[10, 5, 1, -5, -10]])
    cluster_std2 = np.array([[1.0, 2.0, 3.0, 4.0, 5.0]])
    X2, y2 = make_blobs(n_features=n_features,
                      n_samples=n_samples2,
                      centers=centers2,
                      cluster_std=cluster_std2)
    X = np.vstack((X1, X2))

    model = mixture.PGMM(covariance_type='URU', n_components=2, n_pc=3)
    model.fit(X)
    assert_array_almost_equal(np.sum(model.means_, 0), np.sum(centers1, 0), decimal=0)
    assert_array_almost_equal(np.sort(model.weights_), np.array([0.333, 0.666]), decimal=1)
    assert_equal(model.means_.shape, np.array([2, n_features]))
    assert_equal(model.weights_.shape, np.array([2]))
    assert_equal(model.noise_.shape, np.array([n_features]))
    assert_equal(model.principal_subspace_.shape, np.array([2, n_features, 3]))
    assert_equal(model.covars_.shape, np.array([2, n_features, n_features]))
    logging.info('TestFitURU: OK')


def test_fit_uur():
    n_samples1 = 10000
    n_features = 5
    centers1 = np.array([[10, 5, 1, -5, -10],
                        [-10, -5, -1, 5, 10]])
    cluster_std1 = np.array([[1.0, 2.0, 3.0, 4.0, 5.0],
                            [5.0, 4.0, 3.0, 2.0, 1.0]])

    X1, y1 = make_blobs(n_features=n_features,
                      n_samples=n_samples1,
                      centers=centers1,
                      cluster_std=cluster_std1)

    n_samples2 = 5000
    centers2 = np.array([[10, 5, 1, -5, -10]])
    cluster_std2 = np.array([[1.0, 2.0, 3.0, 4.0, 5.0]])
    X2, y2 = make_blobs(n_features=n_features,
                      n_samples=n_samples2,
                      centers=centers2,
                      cluster_std=cluster_std2)
    X = np.vstack((X1, X2))

    model = mixture.PGMM(covariance_type='UUR', n_components=2, n_pc=3)
    model.fit(X)
    assert_array_almost_equal(np.sum(model.means_, 0), np.sum(centers1, 0), decimal=0)
    assert_array_almost_equal(np.sort(model.weights_), np.array([0.333, 0.666]), decimal=1)
    assert_equal(model.means_.shape, np.array([2, n_features]))
    assert_equal(model.weights_.shape, np.array([2]))
    assert_equal(model.noise_.shape, np.array([2]))
    assert_equal(model.principal_subspace_.shape, np.array([2, n_features, 3]))
    assert_equal(model.covars_.shape, np.array([2, n_features, n_features]))
    logging.info('TestFitUUR: OK')


def test_fit_uuu():
    n_samples1 = 10000
    n_features = 5
    centers1 = np.array([[10, 5, 1, -5, -10],
                        [-10, -5, -1, 5, 10]])
    cluster_std1 = np.array([[1.0, 2.0, 3.0, 4.0, 5.0],
                            [5.0, 4.0, 3.0, 2.0, 1.0]])

    X1, y1 = make_blobs(n_features=n_features,
                      n_samples=n_samples1,
                      centers=centers1,
                      cluster_std=cluster_std1)

    n_samples2 = 5000
    centers2 = np.array([[10, 5, 1, -5, -10]])
    cluster_std2 = np.array([[1.0, 2.0, 3.0, 4.0, 5.0]])
    X2, y2 = make_blobs(n_features=n_features,
                      n_samples=n_samples2,
                      centers=centers2,
                      cluster_std=cluster_std2)
    X = np.vstack((X1, X2))

    model = mixture.PGMM(covariance_type='UUU', n_components=2, n_pc=3)
    model.fit(X)
    assert_array_almost_equal(np.sum(model.means_, 0), np.sum(centers1, 0), decimal=0)
    assert_array_almost_equal(np.sort(model.weights_), np.array([0.333, 0.666]), decimal=1)
    assert_equal(model.means_.shape, np.array([2, n_features]))
    assert_equal(model.weights_.shape, np.array([2]))
    assert_equal(model.noise_.shape, np.array([2, n_features]))
    assert_equal(model.principal_subspace_.shape, np.array([2, n_features, 3]))
    assert_equal(model.covars_.shape, np.array([2, n_features, n_features]))
    logging.info('TestFitUUU: OK')
