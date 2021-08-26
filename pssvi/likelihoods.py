import objax
import jax.numpy as np
from jax import grad, jacrev, vmap
from jax.scipy.special import erf, gammaln, logsumexp
from jax.scipy.linalg import cholesky
from .cubature import (
    gauss_hermite,
    variational_expectation_cubature,
    moment_match_cubature,
    statistical_linear_regression_cubature,
    log_density_cubature,
    predict_cubature
)
from .utils import (
    solve,
    transpose,
    softplus,
    softplus_inv,
    sigmoid,
    sigmoid_diff,
    pep_constant,
    mvn_logpdf,
    mvn_logpdf_and_derivs
)
import math

LOG2PI = math.log(2 * math.pi)

__all__ = ["Gaussian"]

class Likelihood(objax.Module):
    """
    The likelihood model class, p(yₙ|fₙ). Each likelihood implements its own methods used during inference:
        Moment matching is used for EP
        Variational expectation is used for VI
        Statistical linearisation is used for PL
        Analytical linearisation is used for EKS
        Log-likelihood gradients are used for Laplace
    If no custom parameter update method is provided, cubature is used (Gauss-Hermite by default).
    The requirement for all inference methods to work is the implementation of the following methods:
        evaluate_likelihood(), which simply evaluates the likelihood given the latent function
        evaluate_log_likelihood()
        conditional_moments(), which return E[y|f] and Cov[y|f]
    """

    def __call__(self, y, f):
        return self.evaluate_likelihood(y, f)

    def evaluate_likelihood(self, y, f):
        raise NotImplementedError

    def evaluate_log_likelihood(self, y, f):
        raise NotImplementedError

    def conditional_moments(self, f):
        raise NotImplementedError

    def log_likelihood_gradients_(self, y, f):
        """
        Evaluate the Jacobian and Hessian of the log-likelihood
        """
        log_lik = self.evaluate_log_likelihood(y, f)
        f = np.squeeze(f)
        J = jacrev(self.evaluate_log_likelihood, argnums=1)
        H = jacrev(J, argnums=1)
        return log_lik, J(y, f), H(y, f)

    def log_likelihood_gradients(self, y, f):
        """
        Most likelihoods factorise across data points. For multi-latent models, a custom method must be implemented.
        """
        # align shapes and compute mask
        y = y.reshape(-1, 1)
        f = f.reshape(-1, 1)
        mask = np.isnan(y)
        y = np.where(mask, f, y)

        # compute gradients of the log likelihood
        log_lik, J, H = vmap(self.log_likelihood_gradients_)(y, f)

        # apply mask
        mask = np.squeeze(mask)
        log_lik = np.where(mask, 0., log_lik)
        J = np.where(mask, np.nan, J)
        H = np.where(mask, np.nan, H)

        return log_lik, J, np.diag(H)

    def variational_expectation_(self, y, m, v, cubature=None):
        """
        If no custom variational expectation method is provided, we use cubature.
        """
        return variational_expectation_cubature(self, y, m, v, cubature)

    def variational_expectation(self, y, m, v, cubature=None):
        """
        Most likelihoods factorise across data points. For multi-latent models, a custom method must be implemented.
        """

        # align shapes and compute mask
        y = y.reshape(-1, 1, 1)
        m = m.reshape(-1, 1, 1)
        v = np.diag(v).reshape(-1, 1, 1)
        mask = np.isnan(y)
        y = np.where(mask, m, y)

        # compute variational expectations and their derivatives
        var_exp, dE_dm, d2E_dm2 = vmap(self.variational_expectation_, (0, 0, 0, None))(y, m, v, cubature)

        # apply mask
        var_exp = np.where(np.squeeze(mask), 0., np.squeeze(var_exp))
        dE_dm = np.where(mask, np.nan, dE_dm)
        d2E_dm2 = np.where(mask, np.nan, d2E_dm2)

        return var_exp, np.squeeze(dE_dm, axis=2), np.diag(np.squeeze(d2E_dm2, axis=(1, 2)))

    def log_density(self, y, mean, cov, cubature=None):
        """
        """
        return log_density_cubature(self, y, mean, cov, cubature)

    def moment_match_(self, y, cav_mean, cav_cov, power=1.0, cubature=None):
        """
        If no custom moment matching method is provided, we use cubature.
        """
        return moment_match_cubature(self, y, cav_mean, cav_cov, power, cubature)

    def moment_match(self, y, m, v, power=1.0, cubature=None):
        """
        """
        # align shapes and compute mask
        y = y.reshape(-1, 1)
        m = m.reshape(-1, 1)
        mask = np.isnan(y)
        y = np.where(mask, m, y)

        lZ, dlZ, d2lZ = self.moment_match_(y, m, v, power, cubature)

        return lZ, dlZ, d2lZ

    def statistical_linear_regression_(self, m, v, cubature=None):
        """
        If no custom SLR method is provided, we use cubature.
        """
        return statistical_linear_regression_cubature(self, m, v, cubature)

    def statistical_linear_regression(self, m, v, cubature=None):
        """
        Most likelihoods factorise across data points. For multi-latent models, a custom method must be implemented.
        TODO: multi-dim case
        """

        # align shapes and compute mask
        m = m.reshape(-1, 1, 1)
        v = np.diag(v).reshape(-1, 1, 1)

        # compute SLR
        mu, omega, d_mu, d2_mu = vmap(self.statistical_linear_regression_, (0, 0, None))(m, v, cubature)
        return (
            np.squeeze(mu, axis=2),
            np.diag(np.squeeze(omega, axis=(1, 2))),
            np.diag(np.squeeze(d_mu, axis=(1, 2))),
            np.diag(np.squeeze(d2_mu, axis=(1, 2))),
        )

    def observation_model(self, f, sigma):
        """
        The implicit observation model is:
            h(fₙ,rₙ) = E[yₙ|fₙ] + √Cov[yₙ|fₙ] σₙ
        """
        conditional_expectation, conditional_covariance = self.conditional_moments(f)
        obs_model = conditional_expectation + cholesky(conditional_covariance) @ sigma
        return np.squeeze(obs_model)

    def jac_obs(self, f, sigma):
        return np.squeeze(jacrev(self.observation_model, argnums=0)(f, sigma))

    def jac_obs_sigma(self, f, sigma):
        return np.squeeze(jacrev(self.observation_model, argnums=1)(f, sigma))

    def analytical_linearisation(self, m, sigma=None):
        """
        Compute the Jacobian of the state space observation model w.r.t. the
        function fₙ and the noise term σₙ.
        The implicit observation model is:
            h(fₙ,rₙ) = E[yₙ|fₙ] + √Cov[yₙ|fₙ] σₙ
        The Jacobians are evaluated at the means, fₙ=m, σₙ=0, to be used during
        Extended Kalman smoothing.
        """
        sigma = np.array([[0.0]]) if sigma is None else sigma

        m = m.reshape(-1, 1, 1)
        sigma = sigma.reshape(-1, 1, 1)

        Jf, Jsigma = vmap(jacrev(self.observation_model, argnums=(0, 1)))(m, sigma)

        Hf = vmap(jacrev(self.jac_obs, argnums=0))(m, sigma)
        Hsigma = vmap(jacrev(self.jac_obs_sigma, argnums=1))(m, sigma)

        return (
            np.diag(np.squeeze(Jf, axis=(1, 2))),
            np.diag(np.squeeze(Hf, axis=(1, 2))),
            np.diag(np.squeeze(Jsigma, axis=(1, 2))),
            np.diag(np.squeeze(Hsigma, axis=(1, 2))),
        )

    def predict(self, mean_f, var_f, cubature=None):
        """
        predict in data space given predictive mean and var of the latent function
        TODO: multi-latent case
        """
        if mean_f.shape[0] > 1:
            return vmap(predict_cubature, [None, 0, 0, None])(
                self,
                mean_f.reshape(-1, 1, 1),
                var_f.reshape(-1, 1, 1),
                cubature
            )
        else:
            return predict_cubature(self, mean_f, var_f, cubature)


class Gaussian(Likelihood):
    """
    The Gaussian likelihood:
        p(yₙ|fₙ) = 𝓝(yₙ|fₙ,σ²)
    """
    def __init__(self,
                 variance=0.1,
                 fix_variance=False):
        """
        :param variance: The observation noise variance, σ²
        """
        if fix_variance:
            self.transformed_variance = objax.StateVar(np.array(softplus_inv(variance)))
        else:
            self.transformed_variance = objax.TrainVar(np.array(softplus_inv(variance)))
        super().__init__()
        self.name = 'Gaussian'
        self.link_fn = lambda f: f

    @property
    def variance(self):
        return softplus(self.transformed_variance.value)

    def evaluate_likelihood(self, y, f):
        """
        Evaluate the Gaussian function 𝓝(yₙ|fₙ,σ²).
        Can be used to evaluate Q cubature points.
        :param y: observed data yₙ [scalar]
        :param f: mean, i.e. the latent function value fₙ [Q, 1]
        :return:
            𝓝(yₙ|fₙ,σ²), where σ² is the observation noise [Q, 1]
        """
        return (2 * np.pi * self.variance) ** -0.5 * np.exp(-0.5 * (y - f) ** 2 / self.variance)

    def evaluate_log_likelihood(self, y, f):
        """
        Evaluate the log-Gaussian function log𝓝(yₙ|fₙ,σ²).
        Can be used to evaluate Q cubature points.
        :param y: observed data yₙ [scalar]
        :param f: mean, i.e. the latent function value fₙ [Q, 1]
        :return:
            log𝓝(yₙ|fₙ,σ²), where σ² is the observation noise [Q, 1]
        """
        return np.squeeze(-0.5 * np.log(2 * np.pi * self.variance) - 0.5 * (y - f) ** 2 / self.variance)

    def conditional_moments(self, f):
        """
        The first two conditional moments of a Gaussian are the mean and variance:
            E[y|f] = f
            Var[y|f] = σ²
        """
        return f, np.array([[self.variance]])

    def variational_expectation_(self, y, post_mean, post_cov, cubature=None):
        """
        Computes the "variational expectation", i.e. the
        expected log-likelihood, and its derivatives w.r.t. the posterior mean
            E[log p(yₙ|fₙ)] = ∫ log p(yₙ|fₙ) 𝓝(fₙ|mₙ,vₙ) dfₙ
        :param y: observed data (yₙ) [scalar]
        :param post_mean: posterior mean (mₙ) [scalar]
        :param post_cov: posterior variance (vₙ) [scalar]
        :param cubature: the function to compute sigma points and weights to use during cubature
        :return:
            exp_log_lik: the expected log likelihood, E[log p(yₙ|fₙ)]  [scalar]
            dE_dm: derivative of E[log p(yₙ|fₙ)] w.r.t. mₙ  [scalar]
            d2E_dm2: 2nd derivative of E[log p(yₙ|fₙ)] w.r.t. mₙ  [scalar]
        """
        # TODO: multi-dim case
        # Compute expected log likelihood:
        # E[log p(yₙ|fₙ)] = ∫ log p(yₙ|fₙ) 𝓝(fₙ|mₙ,vₙ) dfₙ
        exp_log_lik = (
            -0.5 * np.log(2 * np.pi)
            - 0.5 * np.log(self.variance)
            - 0.5 * ((y - post_mean) ** 2 + post_cov) / self.variance
        )
        # Compute first derivative:
        dE_dm = (y - post_mean) / self.variance
        # Compute second derivative:
        d2E_dm2 = -1 / self.variance
        return exp_log_lik, dE_dm, d2E_dm2.reshape(-1, 1)

    def moment_match_(self, y, cav_mean, cav_cov, power=1.0, cubature=None):
        """
        Closed form Gaussian moment matching.
        Calculates the log partition function of the EP tilted distribution:
            logZₙ = log ∫ 𝓝(yₙ|fₙ,σ²) 𝓝(fₙ|mₙ,vₙ) dfₙ = E[𝓝(yₙ|fₙ,σ²)]
        and its derivatives w.r.t. mₙ, which are required for moment matching.
        :param y: observed data (yₙ)
        :param cav_mean: cavity mean (mₙ)
        :param cav_cov: cavity covariance (vₙ)
        :param power: EP power [scalar]
        :param cubature: not used
        :return:
            lZ: the log partition function, logZₙ [scalar]
            dlZ: first derivative of logZₙ w.r.t. mₙ (if derivatives=True) [scalar]
            d2lZ: second derivative of logZₙ w.r.t. mₙ (if derivatives=True) [scalar]
        """
        lik_cov = self.variance * np.eye(cav_cov.shape[0])
        # log partition function, lZ:
        # logZₙ = log ∫ 𝓝(yₙ|fₙ,σ²) 𝓝(fₙ|mₙ,vₙ) dfₙ
        #       = log 𝓝(yₙ|mₙ,σ²+vₙ)
        lZ, dlZ, d2lZ = mvn_logpdf_and_derivs(
            y,
            cav_mean,
            lik_cov / power + cav_cov
        )
        constant = pep_constant(lik_cov, power)
        lZ += constant
        return lZ, dlZ, d2lZ

    def log_density(self, y, mean, cov, cubature=None):
        """
        logZₙ = log ∫ 𝓝(yₙ|fₙ,σ²) 𝓝(fₙ|mₙ,vₙ) dfₙ = E[𝓝(yₙ|fₙ,σ²)]
        :param y: observed data (yₙ)
        :param mean: cavity mean (mₙ)
        :param cov: cavity variance (vₙ)
        :param cubature: not used
        :return:
            lZ: the log density, logZₙ [scalar]
        """
        # logZₙ = log ∫ 𝓝(yₙ|fₙ,σ²) 𝓝(fₙ|mₙ,vₙ) dfₙ = log 𝓝(yₙ|mₙ,σ²+vₙ)
        lZ = mvn_logpdf(
            y,
            mean,
            self.variance * np.eye(cov.shape[0]) + cov
        )
        return lZ

    def predict(self, mean_f, var_f, cubature=None):
        return mean_f, var_f + self.variance
