"""
Mixture of Probabilistic Principal Component Analysis

"""

# Author: Thiago Akio Nakamura <akionakas@gmail.com>

import warnings
import numpy as np
from scipy import linalg
from time import time
from sklearn.decomposition import PCA

from ..base import BaseEstimator
from ..utils import check_random_state, check_array
from ..utils.extmath import logsumexp
from ..utils.validation import check_is_fitted
from .. import cluster

EPS = np.finfo(float).eps


def log_multivariate_normal_density(X, means, covars):
    """Compute the log probability under a multivariate Gaussian distribution.

    Parameters
    ----------
    X : array_like, shape (n_samples, n_features)
        List of n_features-dimensional data points.  Each row corresponds to a
        single data point.

    means : array_like, shape (n_components, n_features)
        List of n_features-dimensional mean vectors for n_components Gaussians.
        Each row corresponds to a single mean vector.

    covars : array_like
        List of n_components covariance parameters for each Gaussian. The shape
        depends on `covariance_type`:
            (n_components, n_features)      if 'spherical',
            (n_features, n_features)    if 'tied',
            (n_components, n_features)    if 'diag',
            (n_components, n_features, n_features) if 'full'

    covariance_type : string
        Type of the covariance parameters.  Must be one of
        'spherical', 'tied', 'diag', 'full'.  Defaults to 'diag'.

    Returns
    -------
    lpr : array_like, shape (n_samples, n_components)
        Array containing the log probabilities of each data point in
        X under each of the n_components multivariate Gaussian distributions.
    """
    return _log_multivariate_normal_density_mppca(X, means, covars)


def sample_gaussian(mean, covar, n_samples=1,
                    random_state=None):
    """Generate random samples from a Gaussian distribution.

    Parameters
    ----------
    mean : array_like, shape (n_features,)
        Mean of the distribution.

    covar : array_like, optional
        Covariance of the distribution.

    n_samples : int, optional
        Number of samples to generate. Defaults to 1.

    Returns
    -------
    X : array, shape (n_features, n_samples)
        Randomly generated sample
    """
    rng = check_random_state(random_state)
    n_dim = len(mean)
    rand = rng.randn(n_dim, n_samples)
    if n_samples == 1:
        rand.shape = (n_dim,)

    s, U = linalg.eigh(covar)
    s.clip(0, out=s)        # get rid of tiny negatives
    np.sqrt(s, out=s)
    U *= s
    rand = np.dot(U, rand)

    return (rand.T + mean).T


class MPPCA(BaseEstimator):
    """Mixture of Probabilistic Principal Component Analysis

    Representation of a mixture of Probabilistic PCA distribution.
    This class allows for easy evaluation of, sampling from, and
    maximum-likelihood estimation of the parameters of a MPPCA distribution.

    Read more in the :ref:`User Guide <mppca>`.

    Parameters
    ----------
    n_components : int, optional
        Number of mixture components. Defaults to 1.

    n_pc : int, optional
        Number of principal components on each mixture component.
        Defaults to 1.

    random_state: RandomState or an int seed (None by default)
        A random number generator instance

    min_covar : float, optional
        Floor on the diagonal of the covariance matrix to prevent
        overfitting.  Defaults to 1e-3.

    tol : float, optional
        Convergence threshold. EM iterations will stop when average
        gain in log-likelihood is below this threshold.  Defaults to 1e-3.

    n_iter : int, optional
        Number of EM iterations to perform.

    n_init : int, optional
        Number of initializations to perform. the best results is kept

    params : string, optional
        Controls which parameters are updated in the training
        process.  Can contain any combination of 'w' for weights,
        'm' for means, 'p' for principal subspace and 'n' for noise.
        Defaults to 'wmpn'.

    init_params : string, optional
        Controls which parameters are updated in the initialization
        process.  Can contain any combination of 'w' for weights,
        'm' for means, 'p' for principal subspace and 'n' for noise.
        Defaults to 'wmpn'.

    verbose : int, default: 0
        Enable verbose output. If 1 then it always prints the current
        initialization and iteration step. If greater than 1 then
        it prints additionally the change and time needed for each step.

    Attributes
    ----------
    weights_ : array, shape (`n_components`,)
        This attribute stores the mixing weights for each mixture component.

    means_ : array, shape (`n_components`, `n_features`)
        Mean parameters for each mixture component.

    noise_ : array, shape (`n_components`,)
        This attribute stores the isotropic noise for each mixture component.

    principal_subspace_ : array, shape (n_components, n_features, n_pc)
        The principal subspace matrix for each mixture component.

    covars_ : array, shape (n_components, n_features, n_pc)
        Covariance parameters for each mixture component,
        defined by [noise_ * I + (principal_subspace * principal_subspace.T)] for each component

    converged_ : bool
        True when convergence was reached in fit(), False otherwise.


    Examples
    --------



    """
    # TODO(akio) create Example

    def __init__(self, n_components=1, n_pc=1,
                 random_state=None, tol=1e-3, min_covar=1e-3,
                 n_iter=100, n_init=1, params='wmpn', init_params='wmpn',
                 verbose=0):
        self.n_components = n_components
        self.n_pc = n_pc
        self.tol = tol
        self.min_covar = min_covar
        self.random_state = random_state
        self.n_iter = n_iter
        self.n_init = n_init
        self.params = params
        self.init_params = init_params
        self.verbose = verbose

        if n_init < 1:
            raise ValueError('MPPCA estimation requires at least one run')

        self.weights_ = np.ones(self.n_components) / self.n_components

        # flag to indicate exit status of fit() method: converged (True) or
        # n_iter reached (False)
        self.converged_ = False

    def _get_covars(self):
        """Covariance parameters for each mixture component."""
        return self.covars_

    def _set_covars(self, principal_subspace, noise):
        """Provide values for covariance"""
        n_features = principal_subspace.shape[1]
        covars = np.zeros((self.n_components, n_features, n_features))
        for comp in range(self.n_components):
            covars[comp] = principal_subspace[comp].dot(principal_subspace[comp].T) + noise[comp] * np.eye(n_features)
        _validate_covars(covars, self.n_components)
        self.covars_ = covars

    def score_samples(self, X):
        """Return the per-sample likelihood of the data under the model.

        Compute the log probability of X under the model and
        return the posterior distribution (responsibilities) of each
        mixture component for each element of X.

        Parameters
        ----------
        X: array_like, shape (n_samples, n_features)
            List of n_features-dimensional data points. Each row
            corresponds to a single data point.

        Returns
        -------
        logprob : array_like, shape (n_samples,)
            Log probabilities of each data point in X.

        responsibilities : array_like, shape (n_samples, n_components)
            Posterior probabilities of each mixture component for each
            observation
        """
        check_is_fitted(self, 'means_')

        X = check_array(X)
        if X.ndim == 1:
            X = X[:, np.newaxis]
        if X.size == 0:
            return np.array([]), np.empty((0, self.n_components))
        if X.shape[1] != self.means_.shape[1]:
            raise ValueError('The shape of X  is not compatible with self')

        lpr = (log_multivariate_normal_density(X, self.means_, self.covars_) +
               np.log(self.weights_))
        logprob = logsumexp(lpr, axis=1)
        responsibilities = np.exp(lpr - logprob[:, np.newaxis])
        return logprob, responsibilities

    def sum_score(self, X, y=None):
        """Compute the sum log probability under the model.

        Parameters
        ----------
        X : array_like, shape (n_samples, n_features)
            List of n_features-dimensional data points.  Each row
            corresponds to a single data point.

        Returns
        -------
        logprob : float
            Sum of the Log probabilities of every data point in X
        """
        logprob, _ = self.score_samples(X)
        return logprob.sum()

    def score(self, X, y=None):
        """Compute the log probability under the model.

        Parameters
        ----------
        X : array_like, shape (n_samples, n_features)
            List of n_features-dimensional data points.  Each row
            corresponds to a single data point.

        Returns
        -------
        logprob : array_like, shape (n_samples,)
            Log probabilities of each data point in X
        """
        logprob, _ = self.score_samples(X)
        return logprob

    def predict(self, X):
        """Predict label for data.

        Parameters
        ----------
        X : array-like, shape = [n_samples, n_features]

        Returns
        -------
        C : array, shape = (n_samples,) component memberships
        """
        logprob, responsibilities = self.score_samples(X)
        return responsibilities.argmax(axis=1)

    def predict_proba(self, X):
        """Predict posterior probability of data under each PPCA
        in the model.

        Parameters
        ----------
        X : array-like, shape = [n_samples, n_features]

        Returns
        -------
        responsibilities : array-like, shape = (n_samples, n_components)
            Returns the probability of the sample for each Gaussian
            (state) in the model.
        """
        logprob, responsibilities = self.score_samples(X)
        return responsibilities

    def sample(self, n_samples=1, random_state=None):
        """Generate random samples from the model.

        Parameters
        ----------
        n_samples : int, optional
            Number of samples to generate. Defaults to 1.

        Returns
        -------
        X : array_like, shape (n_samples, n_features)
            List of samples
        """
        check_is_fitted(self, 'means_')

        if random_state is None:
            random_state = self.random_state
        random_state = check_random_state(random_state)
        weight_cdf = np.cumsum(self.weights_)

        X = np.empty((n_samples, self.means_.shape[1]))
        rand = random_state.rand(n_samples)
        # decide which component to use for each sample
        comps = weight_cdf.searchsorted(rand)
        # for each component, generate all needed samples
        for comp in range(self.n_components):
            # occurrences of current component in X
            comp_in_X = (comp == comps)
            # number of those occurrences
            num_comp_in_X = comp_in_X.sum()
            if num_comp_in_X > 0:
                X[comp_in_X] = sample_gaussian(
                    self.means_[comp], self.covars_[comp],
                    num_comp_in_X, random_state=random_state).T
        return X

    def fit_predict(self, X, y=None):
        """Fit and then predict labels for data.

        Warning: due to the final maximization step in the EM algorithm,
        with low iterations the prediction may not be 100% accurate

        Parameters
        ----------
        X : array-like, shape = [n_samples, n_features]

        Returns
        -------
        C : array, shape = (n_samples,) component memberships
        """
        return self._fit(X, y).argmax(axis=1)

    def _fit(self, X, y=None, do_prediction=False):
        """Estimate model parameters with the EM algorithm.

        A initialization step is performed before entering the
        expectation-maximization (EM) algorithm. If you want to avoid
        this step, set the keyword argument init_params to the empty
        string '' when creating the MPPCA object. Likewise, if you would
        like just to do an initialization, set n_iter=0.

        Parameters
        ----------
        X : array_like, shape (n, n_features)
            List of n_features-dimensional data points.  Each row
            corresponds to a single data point.

        Returns
        -------
        responsibilities : array, shape (n_samples, n_components)
            Posterior probabilities of each mixture component for each
            observation.
        """

        # initialization step
        X = check_array(X, dtype=np.float64)
        if X.shape[0] < self.n_components:
            raise ValueError(
                'MPPCA estimation with %s components, but got only %s samples' %
                (self.n_components, X.shape[0]))

        max_log_prob = -np.infty

        if self.verbose > 0:
            print('Expectation-maximization algorithm started.')

        for init in range(self.n_init):
            if self.verbose > 0:
                print('Initialization ' + str(init + 1))
                start_init_time = time()

            if 'm' in self.init_params or not hasattr(self, 'means_'):
                self.means_ = cluster.KMeans(
                    n_clusters=self.n_components,
                    random_state=self.random_state).fit(X).cluster_centers_
                if self.verbose > 1:
                    print('\tMeans have been initialized.')

            if 'w' in self.init_params or not hasattr(self, 'weights_'):
                self.weights_ = np.tile(1.0 / self.n_components,
                                        self.n_components)
                if self.verbose > 1:
                    print('\tWeights have been initialized.')

            if 'n' in self.init_params or not hasattr(self, 'noise_'):
                self.noise_ = np.tile(1.0, self.n_components)
                if self.verbose > 1:
                    print('\tNoise value have been initialized.')

            if 'p' in self.init_params or not hasattr(self, 'principal_subspace_'):
                pca = PCA(n_components=self.n_pc)
                pca.fit(X)
                self.principal_subspace_ = np.tile(pca.components_.T, (self.n_components, 1, 1))
                self._set_covars(self.principal_subspace_, self.noise_)
                if self.verbose > 1:
                    print('\tPrincipal sub-space have been initialized.')

            # EM algorithms
            current_log_likelihood = None
            # reset self.converged_ to False
            self.converged_ = False

            # this line should be removed when 'thresh' is removed in v0.18
            tol = self.tol

            for i in range(self.n_iter):
                if self.verbose > 0:
                    print('\tEM iteration ' + str(i + 1))
                    start_iter_time = time()
                prev_log_likelihood = current_log_likelihood
                # Expectation step
                log_likelihoods, responsibilities = self.score_samples(X)
                current_log_likelihood = log_likelihoods.mean()

                # Check for convergence.
                # (should compare to self.tol when deprecated 'thresh' is
                # removed in v0.18)
                if prev_log_likelihood is not None:
                    change = abs(current_log_likelihood - prev_log_likelihood)
                    if self.verbose > 1:
                        print('\t\tChange: ' + str(change))
                    if change < tol:
                        self.converged_ = True
                        if self.verbose > 0:
                            print('\t\tEM algorithm converged.')
                        break

                # Maximization step
                self._do_mstep(X, responsibilities, self.params,
                               self.min_covar)
                if self.verbose > 1:
                    print('\t\tEM iteration ' + str(i + 1) + ' took {0:.5f}s'.format(
                        time() - start_iter_time))

            # if the results are better, keep it
            if self.n_iter:
                if current_log_likelihood > max_log_prob:
                    max_log_prob = current_log_likelihood
                    best_params = {'weights': self.weights_,
                                   'means': self.means_,
                                   'principal_subspace': self.principal_subspace_,
                                   'noise': self.noise_}
                    if self.verbose > 1:
                        print('\tBetter parameters were found.')

            if self.verbose > 1:
                print('\tInitialization ' + str(init + 1) + ' took {0:.5f}s'.format(
                    time() - start_init_time))

        # check the existence of an init param that was not subject to
        # likelihood computation issue.
        if np.isneginf(max_log_prob) and self.n_iter:
            raise RuntimeError(
                "EM algorithm was never able to compute a valid likelihood " +
                "given initial parameters. Try different init parameters " +
                "(or increasing n_init) or check for degenerate data.")

        if self.n_iter:
            self.principal_subspace_ = best_params['principal_subspace']
            self.noise_ = best_params['noise']
            self._set_covars(self.principal_subspace_, self.noise_)
            self.means_ = best_params['means']
            self.weights_ = best_params['weights']
        else:  # self.n_iter == 0 occurs when using GMM within HMM
            # Need to make sure that there are responsibilities to output
            # Output zeros because it was just a quick initialization
            responsibilities = np.zeros((X.shape[0], self.n_components))

        return responsibilities

    def fit(self, X, y=None):
        """Estimate model parameters with the EM algorithm.

        A initialization step is performed before entering the
        expectation-maximization (EM) algorithm. If you want to avoid
        this step, set the keyword argument init_params to the empty
        string '' when creating the GMM object. Likewise, if you would
        like just to do an initialization, set n_iter=0.

        Parameters
        ----------
        X : array_like, shape (n, n_features)
            List of n_features-dimensional data points.  Each row
            corresponds to a single data point.

        Returns
        -------
        self
        """
        self._fit(X, y)
        return self

    def _do_mstep(self, X, responsibilities, params, min_covar=0):
        """ Perform the Mstep of the EM algorithm and return the class weights
        """
        weights = responsibilities.sum(axis=0)
        weighted_X_sum = np.dot(responsibilities.T, X)
        inverse_weights = 1.0 / (weights[:, np.newaxis] + 10 * EPS)

        if 'w' in params:
            self.weights_ = (weights / (weights.sum() + 10 * EPS) + EPS)
        if 'm' in params:
            self.means_ = weighted_X_sum * inverse_weights
        new_subspace, new_noise = _covar_mstep_mppca(
                self, X, responsibilities, weighted_X_sum, inverse_weights,
                min_covar)
        if 'p' in params:
            self.principal_subspace_ = new_subspace
        if 'n' in params:
            self.noise_ = new_noise

    def _n_parameters(self):
        """Return the number of free parameters in the model."""
        ndim = self.means_.shape[1]
        cov_params = self.n_components * (ndim*self.n_pc - self.n_pc*(self.n_pc - 1)/2) + self.n_components
        mean_params = ndim * self.n_components
        return int(cov_params + mean_params + self.n_components - 1)

    def bic(self, X):
        """Bayesian information criterion for the current model fit
        and the proposed data

        Parameters
        ----------
        X : array of shape(n_samples, n_dimensions)

        Returns
        -------
        bic: float (the lower the better)
        """
        return (-2 * self.score(X).sum() +
                self._n_parameters() * np.log(X.shape[0]))

    def aic(self, X):
        """Akaike information criterion for the current model fit
        and the proposed data

        Parameters
        ----------
        X : array of shape(n_samples, n_dimensions)

        Returns
        -------
        aic: float (the lower the better)
        """
        return - 2 * self.score(X).sum() + 2 * self._n_parameters()


#########################################################################
# some helper routines
#########################################################################

def _log_multivariate_normal_density_mppca(X, means, covars, min_covar=1.e-7):
    """Log probability for full covariance matrices."""
    n_samples, n_dim = X.shape
    nmix = len(means)
    log_prob = np.empty((n_samples, nmix))
    for c, (mu, cv) in enumerate(zip(means, covars)):
        try:
            cv_chol = linalg.cholesky(cv, lower=True)
        except linalg.LinAlgError:
            # The model is most probably stuck in a component with too
            # few observations, we need to reinitialize this components
            try:
                cv_chol = linalg.cholesky(cv + min_covar * np.eye(n_dim),
                                          lower=True)
            except linalg.LinAlgError:
                raise ValueError("'covars' must be symmetric, "
                                 "positive-definite")

        cv_log_det = 2 * np.sum(np.log(np.diagonal(cv_chol)))
        cv_sol = linalg.solve_triangular(cv_chol, (X - mu).T, lower=True).T
        log_prob[:, c] = - .5 * (np.sum(cv_sol ** 2, axis=1) +
                                 n_dim * np.log(2 * np.pi) + cv_log_det)

    return log_prob


def _validate_covars(covars, n_components):
    """Do basic checks on matrix covariance sizes and values"""
    from scipy import linalg
    if len(covars.shape) != 3:
        raise ValueError("covars must have shape "
                         "(n_components, n_dim, n_dim)")
    elif covars.shape[1] != covars.shape[2]:
        raise ValueError("covars must have shape "
                         "(n_components, n_dim, n_dim)")
    for n, cv in enumerate(covars):
        if (not np.allclose(cv, cv.T)
                or np.any(linalg.eigvalsh(cv) <= 0)):
            raise ValueError("component %d of 'full' covars must be "
                             "symmetric, positive-definite" % n)


def _covar_mstep_mppca(gmm, X, responsibilities, weighted_X_sum, norm,
                      min_covar):
    n_features = X.shape[1]
    n_data = X.shape[0]
    Minv = np.empty((gmm.n_components, gmm.n_pc, gmm.n_pc))
    S = np.empty((gmm.n_components, n_features, n_features))
    W = np.empty((gmm.n_components, n_features, gmm.n_pc))
    noises = np.empty(gmm.n_components)
    for c in range(gmm.n_components):
        post = responsibilities[:, c]
        mu = gmm.means_[c]
        diff = X - mu
        Minv[c] = np.linalg.inv(gmm.noise_[c] * np.eye(gmm.n_pc) + np.dot(gmm.principal_subspace_[c].T, gmm.principal_subspace_[c]))
        with np.errstate(under='ignore'):
            # Underflow Errors in doing post * X.T are  not important
            S[c] = np.dot(post * diff.T, diff) / (n_data*gmm.weights_[c] + 10 * EPS)

        W[c] = S[c].dot(gmm.principal_subspace_[c])\
            .dot(np.linalg.inv(gmm.noise_[c] * np.eye(gmm.n_pc)
                               + Minv[c].dot(gmm.principal_subspace_[c].T).dot(S[c]).dot(gmm.principal_subspace_[c])))
        noises[c] = np.trace(S[c] - np.dot(S[c], np.dot(gmm.principal_subspace_[c], np.dot(Minv[c], W[c].T))))/n_features
    return W, noises