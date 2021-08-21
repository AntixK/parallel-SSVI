import objax
from .base import BaseModel
from jax import vmap
import jax.numpy as np
from jax.lib import xla_bridge
from jax.lax import scan
from .utils import temporal_conditional, transpose, diag, gaussian_expected_log_lik, inv_vmap
from .kalman import kalman_filter, rauch_tung_striebel_smoother
from .utils import input_admin, process_noise_covariance

class MarkovGP(BaseModel):
    """
    The stochastic differential equation (SDE) form of a Gaussian process (GP) model.
    Implements methods for inference and learning using state space methods, i.e. Kalman filtering and smoothing.
    Constructs a linear time-invariant (LTI) stochastic differential equation (SDE) of the following form:
        dx(t)/dt = F x(t) + L w(t)
              yₙ ~ p(yₙ | f(t_n)=H x(t_n))
    where w(t) is a white noise process and where the state x(t) is Gaussian distributed with initial
    state distribution x(t)~𝓝(0,Pinf).
    """
    def __init__(self,
                 kernel,
                 likelihood,
                 X,
                 Y,
                 R=None,
                 parallel=None):
        if parallel is None:  # if using a GPU, then run the parallel filter
            parallel = xla_bridge.get_backend().platform == 'gpu'
        (X, Y, self.R, self.dt) = input_admin(X, Y, R)
        H = kernel.measurement_model()
        func_dim = H.shape[0]  # number of latent dimensions
        super().__init__(kernel, likelihood, X, Y, func_dim=func_dim)
        self.state_dim = self.kernel.stationary_covariance().shape[0]
        self.minf = np.zeros([self.state_dim, 1])  # stationary state mean
        self.spatio_temporal = np.any(~np.isnan(self.R))
        self.parallel = parallel
        if (self.func_dim != self.obs_dim) and self.spatio_temporal:
            self.mask_pseudo_y = None  # sparse spatio-temporal case, no mask required

    @staticmethod
    def filter(*args, **kwargs):
        return kalman_filter(*args, **kwargs)

    @staticmethod
    def smoother(*args, **kwargs):
        return rauch_tung_striebel_smoother(*args, **kwargs)

    @staticmethod
    def temporal_conditional(*args, **kwargs):
        return temporal_conditional(*args, **kwargs)

    def compute_full_pseudo_nat(self, batch_ind):
        if self.spatio_temporal:  # spatio-temporal case
            B, C = self.kernel.spatial_conditional(self.X[batch_ind], self.R[batch_ind])
            nat1lik_full = transpose(B) @ self.pseudo_likelihood.nat1[batch_ind]
            nat2lik_full = transpose(B) @ self.pseudo_likelihood.nat2[batch_ind] @ B
            return nat1lik_full, nat2lik_full
        else:  # temporal case
            return self.pseudo_likelihood.nat1[batch_ind], self.pseudo_likelihood.nat2[batch_ind]

    def compute_full_pseudo_lik(self):
        # TODO: running this 3 times per training loop is wasteful - store in memory?
        if self.spatio_temporal:  # spatio-temporal case
            B, C = self.kernel.spatial_conditional(self.X, self.R)
            # TODO: more efficient way to do this?
            nat1lik_full = transpose(B) @ self.pseudo_likelihood.nat1
            nat2lik_full = transpose(B) @ self.pseudo_likelihood.nat2 @ B
            pseudo_var_full = inv_vmap(nat2lik_full + 1e-12 * np.eye(nat2lik_full.shape[1]))  # <---------- bottleneck
            pseudo_y_full = pseudo_var_full @ nat1lik_full
            return pseudo_y_full, pseudo_var_full
        else:  # temporal case
            return self.pseudo_likelihood.mean, self.pseudo_likelihood.covariance

    def update_posterior(self):
        """
        Compute the posterior via filtering and smoothing
        """
        pseudo_y, pseudo_var = self.compute_full_pseudo_lik()
        log_lik, (filter_mean, filter_cov) = self.filter(self.dt,
                                                         self.kernel,
                                                         pseudo_y,
                                                         pseudo_var,
                                                         mask=self.mask_pseudo_y,
                                                         parallel=self.parallel)
        dt = np.concatenate([self.dt[1:], np.array([0.0])], axis=0)
        smoother_mean, smoother_cov, _ = self.smoother(dt,
                                                       self.kernel,
                                                       filter_mean,
                                                       filter_cov,
                                                       parallel=self.parallel)
        self.posterior_mean.value, self.posterior_variance.value = smoother_mean, smoother_cov

    def compute_kl(self):
        """
        KL[q()|p()]
        """
        pseudo_y, pseudo_var = self.compute_full_pseudo_lik()
        log_lik_pseudo = self.compute_log_lik(pseudo_y, pseudo_var)

        expected_density_pseudo = vmap(gaussian_expected_log_lik)(  # parallel operation
            pseudo_y,
            self.posterior_mean.value,
            self.posterior_variance.value,
            pseudo_var,
            self.mask_pseudo_y
        )

        kl = np.sum(expected_density_pseudo) - log_lik_pseudo  # KL[approx_post || prior]
        return kl

    def compute_log_lik(self, pseudo_y=None, pseudo_var=None):
        """
        int p(f) N(pseudo_y | f, pseudo_var) df
        """
        if pseudo_y is None:
            pseudo_y, pseudo_var = self.compute_full_pseudo_lik()

        log_lik_pseudo, (_, _) = self.filter(
            self.dt,
            self.kernel,
            pseudo_y,
            pseudo_var,
            mask=self.mask_pseudo_y,
            parallel=self.parallel
        )
        return log_lik_pseudo

    def conditional_posterior_to_data(self, batch_ind=None, post_mean=None, post_cov=None):
        """
        compute
        q(f) = int p(f | u) q(u) du = N(f | B post_mean, B post_cov B' + C)
        where
        q(u) = N(u | post_mean, post_cov)
        p(f | u) = N(f | Bu, C)
        """
        if batch_ind is None:
            batch_ind = np.arange(self.num_data)
        if post_mean is None:
            post_mean = self.posterior_mean.value[batch_ind]
        if post_cov is None:
            post_cov = self.posterior_variance.value[batch_ind]

        if self.spatio_temporal:
            B, C = self.kernel.spatial_conditional(self.X[batch_ind], self.R[batch_ind])
            mean_f = B @ post_mean
            cov_f = B @ post_cov @ transpose(B) + C
            return mean_f, cov_f
        else:
            return post_mean, post_cov

    def predict(self, X=None, R=None, pseudo_lik_params=None):
        """
        predict at new test locations X
        """
        if X is None:
            X = self.X
        elif len(X.shape) < 2:
            X = X[:, None]
        if R is None:
            R = X[:, 1:]
        X = X[:, :1]  # take only the temporal component

        if pseudo_lik_params is None:
            pseudo_y, pseudo_var = self.compute_full_pseudo_lik()
        else:
            pseudo_y, pseudo_var = pseudo_lik_params  # this deals with the posterior sampling case
        _, (filter_mean, filter_cov) = self.filter(self.dt,
                                                   self.kernel,
                                                   pseudo_y,
                                                   pseudo_var,
                                                   mask=self.mask_pseudo_y,  # mask has no effect here (loglik not used)
                                                   parallel=self.parallel)
        dt = np.concatenate([self.dt[1:], np.array([0.0])], axis=0)
        smoother_mean, smoother_cov, gain = self.smoother(dt,
                                                          self.kernel,
                                                          filter_mean,
                                                          filter_cov,
                                                          return_full=True,
                                                          parallel=self.parallel)

        # add dummy states at either edge
        inf = 1e10 * np.ones_like(self.X[0, :1])
        X_aug = np.block([[-inf], [self.X[:, :1]], [inf]])

        # predict the state distribution at the test time steps:
        state_mean, state_cov = self.temporal_conditional(X_aug, X, smoother_mean, smoother_cov, gain, self.kernel)
        # extract function values from the state:
        H = self.kernel.measurement_model()
        if self.spatio_temporal:
            # TODO: if R is fixed, only compute B, C once
            B, C = self.kernel.spatial_conditional(X, R)
            W = B @ H
            test_mean = W @ state_mean
            test_var = W @ state_cov @ transpose(W) + C
        else:
            test_mean, test_var = H @ state_mean, H @ state_cov @ transpose(H)

        if np.squeeze(test_var).ndim > 2:  # deal with spatio-temporal case (discard spatial covariance)
            test_var = diag(np.squeeze(test_var))
        return np.squeeze(test_mean), np.squeeze(test_var)

    def filter_energy(self):
        pseudo_y, pseudo_var = self.compute_full_pseudo_lik()
        _, (filter_mean, filter_cov) = self.filter(self.dt,
                                                   self.kernel,
                                                   pseudo_y,
                                                   pseudo_var,
                                                   mask=self.mask_pseudo_y,  # mask has no effect here (loglik not used)
                                                   parallel=self.parallel,
                                                   return_predict=True)
        H = self.kernel.measurement_model()
        mean = H @ filter_mean
        var = H @ filter_cov @ transpose(H)
        filter_energy = -np.sum(vmap(self.likelihood.log_density)(self.Y, mean, var))
        return filter_energy

    def prior_sample(self, num_samps=1, X=None, seed=0):
        """
        Sample from the model prior f~N(0,K) multiple times using a nested loop.
        :param num_samps: the number of samples to draw [scalar]
        :param X: the input locations at which to sample (defaults to training inputs) [N, 1]
        :param seed: the random seed for sampling
        :return:
            f_samples: the prior samples [num_samps, N, func_dim]
        """
        if X is None:
            dt = self.dt
        else:
            dt = np.concatenate([np.array([0.0]), np.diff(np.sort(X))])
        sd = self.state_dim
        H = self.kernel.measurement_model()
        Pinf = self.kernel.stationary_covariance()
        As = vmap(self.kernel.state_transition)(dt)
        Qs = vmap(process_noise_covariance, [0, None])(As, Pinf)
        jitter = 1e-8 * np.eye(sd)
        f0 = np.zeros([dt.shape[0], self.func_dim, 1])

        def draw_full_sample(carry_, _):
            f_sample_i, i = carry_
            gen0 = objax.random.Generator(seed - 1 - i)
            m0 = np.linalg.cholesky(Pinf) @ objax.random.normal(shape=(sd, 1), generator=gen0)

            def sample_one_time_step(carry, inputs):
                m, k = carry
                A, Q = inputs
                chol_Q = np.linalg.cholesky(Q + jitter)  # <--- can be a bit unstable
                gen = objax.random.Generator(seed + i * k + k)
                q_samp = chol_Q @ objax.random.normal(shape=(sd, 1), generator=gen)
                m = A @ m + q_samp
                f = H @ m
                return (m, k+1), f

            (_, _), f_sample = scan(f=sample_one_time_step,
                                    init=(m0, 0),
                                    xs=(As, Qs))

            return (f_sample, i+1), f_sample

        (_, _), f_samples = scan(f=draw_full_sample,
                                 init=(f0, 0),
                                 xs=np.zeros(num_samps))

        return f_samples

    def posterior_sample(self, X=None, num_samps=1, seed=0):
        """
        Sample from the posterior at the test locations.
        Posterior sampling works by smoothing samples from the prior using the approximate Gaussian likelihood
        model given by the pseudo-likelihood, 𝓝(f|μ*,σ²*), computed during training.
         - draw samples (f*) from the prior
         - add Gaussian noise to the prior samples using auxillary model p(y*|f*) = 𝓝(y*|f*,σ²*)
         - smooth the samples by computing the posterior p(f*|y*)
         - posterior samples = prior samples + smoothed samples + posterior mean
                             = f* + E[p(f*|y*)] + E[p(f|y)]
        See Arnaud Doucet's note "A Note on Efficient Conditional Simulation of Gaussian Distributions" for details.
        :param X: the sampling input locations [N, 1]
        :param num_samps: the number of samples to draw [scalar]
        :param seed: the random seed for sampling
        :return:
            the posterior samples [N_test, num_samps]
        """
        if X is None:
            train_ind = np.arange(self.num_data)
            test_ind = train_ind
        else:
            if X.ndim < 2:
                X = X[:, None]
            X = np.concatenate([self.X, X])
            X, ind = np.unique(X, return_inverse=True)
            train_ind, test_ind = ind[:self.num_data], ind[self.num_data:]
        post_mean, _ = self.predict(X)
        prior_samp = self.prior_sample(X=X, num_samps=num_samps, seed=seed)  # sample at training locations
        lik_chol = np.tile(np.linalg.cholesky(self.pseudo_likelihood.covariance), [num_samps, 1, 1, 1])
        gen = objax.random.Generator(seed)
        prior_samp_train = prior_samp[:, train_ind]
        prior_samp_y = prior_samp_train + lik_chol @ objax.random.normal(shape=prior_samp_train.shape, generator=gen)

        def smooth_prior_sample(i, prior_samp_y_i):
            smoothed_sample, _ = self.predict(X, pseudo_lik_params=(prior_samp_y_i, self.pseudo_likelihood.covariance))
            return i+1, smoothed_sample

        _, smoothed_samples = scan(f=smooth_prior_sample,
                                   init=0,
                                   xs=prior_samp_y)

        return (prior_samp[..., 0, 0] - smoothed_samples + post_mean[None])[:, test_ind]