import unittest
import copy
import sys

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