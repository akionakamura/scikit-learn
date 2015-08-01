"""
Mixture of Probabilistic Principal Component Analysis

"""

# Author: Thiago Akio Nakamura <akionakas@gmail.com>

import warnings
import numpy as np
from scipy import linalg
from time import time

from ..base import BaseEstimator
from ..utils import check_random_state, check_array
from ..utils.extmath import logsumexp
from ..utils.validation import check_is_fitted
from .. import cluster

from sklearn.externals.six.moves import zip