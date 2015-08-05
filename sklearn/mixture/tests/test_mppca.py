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

def test_create_mppca():
    model = mixture.MPPCA(n_components=1, n_pc=1,
                 random_state=None, tol=1e-3, min_covar=1e-3,
                 n_iter=100, n_init=1, params='wmpn', init_params='wmpn',
                 verbose=0)
    assert_true(2 == 3)
    logging.info(model.n_pc)
