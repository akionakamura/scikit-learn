"""
The :mod:`sklearn.mixture` module implements mixture modeling algorithms.
"""

from .gmm import sample_gaussian, log_multivariate_normal_density
from .gmm import GMM, distribute_covar_matrix_to_match_covariance_type
from .gmm import _validate_covars
from .dpgmm import DPGMM, VBGMM
from .mppca import MPPCA
from .pgmm import PGMM

__all__ = ['DPGMM',
           'GMM',
           'VBGMM',
           'MPPCA',
           'PGMM',
           '_validate_covars',
           'distribute_covar_matrix_to_match_covariance_type',
           'log_multivariate_normal_density',
           'sample_gaussian']
