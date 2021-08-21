import objax
import jax.numpy as np
from jax import vmap
from jax.lax import scan
from jax.ops import index, index_update
from jax.scipy.linalg import cho_factor, cho_solve

from .utils import transpose, gaussian_expected_log_lik, compute_cavity



class GaussianDistribution(objax.Module):
    """
    A small class defined to handle the fact that we often need access to both the mean / cov parameterisation
    of a Gaussian and its natural parameterisation.
    Important note: for simplicity we let nat2 = inv(cov) rather than nat2 = -0.5inv(cov). The latter is the proper
    natural parameter, but for Gaussian distributions we need not worry about the -0.5 (it cancels out anyway).
    """

    def __init__(self, mean, covariance):
        self.mean_ = objax.StateVar(mean)
        self.covariance_ = objax.StateVar(covariance)
        nat1, nat2 = self.reparametrise(mean, covariance)
        self.nat1_, self.nat2_ = objax.StateVar(nat1), objax.StateVar(nat2)

    def __call__(self):
        return self.mean, self.covariance

    @property
    def mean(self):
        return self.mean_.value

    @property
    def covariance(self):
        return self.covariance_.value

    @property
    def nat1(self):
        return self.nat1_.value

    @property
    def nat2(self):
        return self.nat2_.value

    @staticmethod
    def reparametrise(param1, param2):
        chol = cho_factor(param2)
        reparam1 = cho_solve(chol, param1)
        reparam2 = cho_solve(chol, np.tile(np.eye(param2.shape[1]), [param2.shape[0], 1, 1]))
        return reparam1, reparam2

    def update_mean_cov(self, mean, covariance):
        self.mean_.value = mean
        self.covariance_.value = covariance
        self.nat1_.value, self.nat2_.value = self.reparametrise(mean, covariance)

    def update_nat_params(self, nat1, nat2):
        self.nat1_.value = nat1
        self.nat2_.value = nat2
        self.mean_.value, self.covariance_.value = self.reparametrise(nat1, nat2)


class BaseModel(objax.Module):
    """
    The parent model class: initialises all the common model features and implements shared methods
    TODO: move as much of the generic functionality as possible from this class to the inference class.
    """
    def __init__(self,
                 kernel,
                 likelihood,
                 X,
                 Y,
                 func_dim=1):
        super().__init__()
        if X.ndim < 2:
            X = X[:, None]
        if Y.ndim < 2:
            Y = Y[:, None]
        self.X = np.array(X)
        self.Y = np.array(Y)
        self.kernel = kernel
        self.likelihood = likelihood
        self.num_data = self.X.shape[0]  # number of data
        self.func_dim = func_dim  # number of latent dimensions
        self.obs_dim = Y.shape[1]  # dimensionality of the observations, Y
        if isinstance(self.kernel, Independent):
            pseudo_lik_size = self.func_dim  # the multi-latent case
        else:
            pseudo_lik_size = self.obs_dim
        self.pseudo_likelihood = GaussianDistribution(
            mean=np.zeros([self.num_data, pseudo_lik_size, 1]),
            covariance=1e2 * np.tile(np.eye(pseudo_lik_size), [self.num_data, 1, 1])
        )
        self.posterior_mean = objax.StateVar(np.zeros([self.num_data, self.func_dim, 1]))
        self.posterior_variance = objax.StateVar(np.tile(np.eye(self.func_dim), [self.num_data, 1, 1]))
        self.ind = np.arange(self.num_data)
        self.num_neighbours = np.ones(self.num_data)
        self.mask_y = np.isnan(self.Y).reshape(Y.shape[0], Y.shape[1])
        if self.func_dim != self.obs_dim:
            self.mask_pseudo_y = np.tile(self.mask_y, [1, pseudo_lik_size])  # multi-latent case
        else:
            self.mask_pseudo_y = self.mask_y

    def __call__(self, X=None):
        if X is None:
            self.update_posterior()
        else:
            return self.predict(X)

    def prior_sample(self, num_samps=1, X=None, seed=0):
        raise NotImplementedError

    def update_posterior(self):
        raise NotImplementedError

    def compute_log_lik(self, pseudo_y=None, pseudo_var=None):
        """ Compute the log likelihood of the pseudo model, i.e. the log normaliser of the approximate posterior """
        raise NotImplementedError

    def predict(self, X, R=None):
        raise NotImplementedError

    def predict_y(self, X, R=None, cubature=None):
        """
        predict y at new test locations X
        TODO: check non-Gaussian likelihoods
        """
        mean_f, var_f = self.predict(X, R)
        mean_f, var_f = mean_f.reshape(mean_f.shape[0], -1, 1), var_f.reshape(var_f.shape[0], -1, 1)
        mean_y, var_y = vmap(self.likelihood.predict, (0, 0, None))(mean_f, var_f, cubature)
        return np.squeeze(mean_y), np.squeeze(var_y)

    def negative_log_predictive_density(self, X, Y, R=None, cubature=None):
        predict_mean, predict_var = self.predict(X, R)
        if Y.ndim < 2:
            Y = Y.reshape(-1, 1)
        if (predict_mean.ndim > 1) and (predict_mean.shape[1] != Y.shape[1]):  # multi-latent case
            pred_mean, pred_var = predict_mean[..., None], predict_var[..., None] * np.eye(predict_var.shape[1])
        else:
            pred_mean, pred_var = predict_mean.reshape(-1, 1, 1), predict_var.reshape(-1, 1, 1)
        log_density = vmap(self.likelihood.log_density, (0, 0, 0, None))(
            Y.reshape(-1, 1),
            pred_mean,
            pred_var,
            cubature
        )
        return -np.nanmean(log_density)

    def group_natural_params(self, nat1, nat2, batch_ind=None):
        if (batch_ind is not None) and (batch_ind.shape[0] != self.num_data):
            nat1 = index_update(self.pseudo_likelihood.nat1, index[batch_ind], nat1)
            nat2 = index_update(self.pseudo_likelihood.nat2, index[batch_ind], nat2)
        return nat1, nat2

    def conditional_posterior_to_data(self, batch_ind=None, post_mean=None, post_cov=None):
        if batch_ind is None:
            batch_ind = np.arange(self.num_data)
        ind = self.ind[batch_ind]
        if post_mean is None:
            post_mean = self.posterior_mean.value[ind]
        if post_cov is None:
            post_cov = self.posterior_variance.value[ind]
        return post_mean, post_cov

    def conditional_data_to_posterior(self, mean_f, cov_f):
        return mean_f, cov_f

    def expected_density_pseudo(self):
        expected_density = vmap(gaussian_expected_log_lik)(  # parallel operation
            self.pseudo_likelihood.mean,
            self.posterior_mean.value,
            self.posterior_variance.value,
            self.pseudo_likelihood.covariance,
            self.mask_pseudo_y
        )
        return np.sum(expected_density)

    def compute_full_pseudo_lik(self):
        return self.pseudo_likelihood()

    def compute_full_pseudo_nat(self, batch_ind):
        return self.pseudo_likelihood.nat1[batch_ind], self.pseudo_likelihood.nat2[batch_ind]

    def cavity_distribution(self, batch_ind=None, power=1.):
        """ Compute the power EP cavity for the given data points """
        if batch_ind is None:
            batch_ind = np.arange(self.num_data)

        nat1lik_full, nat2lik_full = self.compute_full_pseudo_nat(batch_ind)

        # then compute the cavity
        cavity_mean, cavity_cov = vmap(compute_cavity, [0, 0, 0, 0, None])(
            self.posterior_mean.value[batch_ind],
            self.posterior_variance.value[batch_ind],
            nat1lik_full,
            nat2lik_full,
            power
        )
        return cavity_mean, cavity_cov