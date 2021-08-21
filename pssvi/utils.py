import math
from jax import vmap
import jax.numpy as np
import numpy as nnp
from jax.scipy.linalg import cho_factor, cho_solve

LOG2PI = math.log(2 * math.pi)
INV2PI = (2 * math.pi) ** -1

def process_noise_covariance(A, Pinf):
    Q = Pinf - A @ Pinf @ transpose(A)
    return Q

def diag(P):
    """
    a broadcastable version of np.diag, for when P is size [N, D, D]
    """
    return np.diagonal(P, axis1=1, axis2=2)

def transpose(P):
    return np.swapaxes(P, -1, -2)

def inv(P):
    """
    Compute the inverse of a PSD matrix using the Cholesky factorisation
    """
    L = cho_factor(P)
    return cho_solve(L, np.eye(P.shape[-1]))

def inv_vmap(P):
    """
    Compute the inverse of a PSD matrix using the Cholesky factorisation
    """
    L = cho_factor(P)
    return cho_solve(L, np.tile(np.eye(P.shape[-1]), [P.shape[0], 1, 1]))

def solve(P, Q):
    """
    Compute P^-1 Q, where P is a PSD matrix, using the Cholesky factorisation
    """
    L = cho_factor(P)
    return cho_solve(L, Q)

def input_admin(t, y, r):
    """
    Order the inputs.
    :param t: training inputs [N, 1]
    :param y: observations at the training inputs [N, 1]
    :param r: training spatial inputs
    :return:
        t_train: training inputs [N, 1]
        y_train: training observations [N, R]
        r_train: training spatial inputs [N, R]
        dt_train: training step sizes, Δtₙ = tₙ - tₙ₋₁ [N, 1]
    """
    assert t.shape[0] == y.shape[0]
    if t.ndim < 2:
        t = nnp.expand_dims(t, 1)  # make 2-D
    if y.ndim < 2:
        y = nnp.expand_dims(y, 1)  # make 2-D
    if r is None:
        if t.shape[1] > 1:
            r = t[:, 1:]
            t = t[:, :1]
        else:
            r = nnp.nan * t  # np.empty((1,) + x.shape[1:]) * np.nan
    if r.ndim < 2:
        r = nnp.expand_dims(r, 1)  # make 2-D
    ind = nnp.argsort(t[:, 0], axis=0)
    t_train = t[ind, ...]
    y_train = y[ind, ...]
    r_train = r[ind, ...]
    dt_train = nnp.concatenate([np.array([0.0]), nnp.diff(t_train[:, 0])])
    return (np.array(t_train, dtype=np.float64), np.array(y_train, dtype=np.float64),
            np.array(r_train, dtype=np.float64), np.array(dt_train, dtype=np.float64))

def mvn_logpdf(x, mean, cov, mask=None):
    """
    evaluate a multivariate Gaussian (log) pdf
    """
    if mask is not None:
        # build a mask for computing the log likelihood of a partially observed multivariate Gaussian
        maskv = mask.reshape(-1, 1)
        mean = np.where(maskv, x, mean)
        cov_masked = np.where(maskv + maskv.T, 0., cov)  # ensure masked entries are independent
        cov = np.where(np.diag(mask), INV2PI, cov_masked)  # ensure masked entries return log like of 0

    n = mean.shape[0]
    cho, low = cho_factor(cov)
    log_det = 2 * np.sum(np.log(np.abs(np.diag(cho))))
    diff = x - mean
    scaled_diff = cho_solve((cho, low), diff)
    distance = diff.T @ scaled_diff
    return np.squeeze(-0.5 * (distance + n * LOG2PI + log_det))

def gaussian_expected_log_lik(Y, q_mu, q_covar, noise, mask=None):
    """
    :param Y: N x 1
    :param q_mu: N x 1
    :param q_covar: N x N
    :param noise: N x N
    :param mask: N x 1
    :return:
        E[log 𝓝(yₙ|fₙ,σ²)] = ∫ log 𝓝(yₙ|fₙ,σ²) 𝓝(fₙ|mₙ,vₙ) dfₙ
    """

    if mask is not None:
        # build a mask for computing the log likelihood of a partially observed multivariate Gaussian
        maskv = mask.reshape(-1, 1)
        q_mu = np.where(maskv, Y, q_mu)
        noise = np.where(maskv + maskv.T, 0., noise)  # ensure masked entries are independent
        noise = np.where(np.diag(mask), INV2PI, noise)  # ensure masked entries return log like of 0
        q_covar = np.where(maskv + maskv.T, 0., q_covar)  # ensure masked entries are independent
        q_covar = np.where(np.diag(mask), 1e-20, q_covar)  # ensure masked entries return trace term of 0

    ml = mvn_logpdf(Y, q_mu, noise)
    trace_term = -0.5 * np.trace(solve(noise, q_covar))
    return ml + trace_term


def compute_cavity(post_mean, post_cov, site_nat1, site_nat2, power, jitter=1e-8):
    """
    remove local likelihood approximation  from the posterior to obtain the marginal cavity distribution
    """
    post_nat2 = inv(post_cov + jitter * np.eye(post_cov.shape[1]))
    cav_cov = inv(post_nat2 - power * site_nat2)  # cavity covariance
    cav_mean = cav_cov @ (post_nat2 @ post_mean - power * site_nat1)  # cavity mean
    return cav_mean, cav_cov


def temporal_conditional(X, X_test, mean, cov, gain, kernel):
    """
    predict from time X to time X_test give state mean and covariance at X
    """
    Pinf = kernel.stationary_covariance()[None, ...]
    minf = np.zeros([1, Pinf.shape[1], 1])
    mean_aug = np.concatenate([minf, mean, minf])
    cov_aug = np.concatenate([Pinf, cov, Pinf])
    gain = np.concatenate([np.zeros_like(gain[:1]), gain])

    # figure out which two training states each test point is located between
    ind_test = np.searchsorted(X.reshape(-1, ), X_test.reshape(-1, )) - 1

    # project from training states to test locations
    test_mean, test_cov = predict_from_state(X_test, ind_test, X, mean_aug, cov_aug, gain, kernel)

    return test_mean, test_cov


def compute_conditional_statistics(x_test, x, kernel, ind):
    """
    This version uses cho_factor and cho_solve - much more efficient when using JAX
    Predicts marginal states at new time points. (new time points should be sorted)
    Calculates the conditional density:
             p(xₙ|u₋, u₊) = 𝓝(Pₙ @ [u₋, u₊], Tₙ)
    :param x_test: time points to generate observations for [N]
    :param x: inducing state input locations [M]
    :param kernel: prior object providing access to state transition functions
    :param ind: an array containing the index of the inducing state to the left of every input [N]
    :return: parameters for the conditional mean and covariance
            P: [N, D, 2*D]
            T: [N, D, D]
    """
    dt_fwd = x_test[..., 0] - x[ind, 0]
    dt_back = x[ind + 1, 0] - x_test[..., 0]
    A_fwd = kernel.state_transition(dt_fwd)
    A_back = kernel.state_transition(dt_back)
    Pinf = kernel.stationary_covariance()
    Q_fwd = Pinf - A_fwd @ Pinf @ A_fwd.T
    Q_back = Pinf - A_back @ Pinf @ A_back.T
    A_back_Q_fwd = A_back @ Q_fwd
    Q_mp = Q_back + A_back @ A_back_Q_fwd.T

    jitter = 1e-8 * np.eye(Q_mp.shape[0])
    chol_Q_mp = cho_factor(Q_mp + jitter)
    Q_mp_inv_A_back = cho_solve(chol_Q_mp, A_back)  # V = Q₋₊⁻¹ Aₜ₊

    # The conditional_covariance T = Q₋ₜ - Q₋ₜAₜ₊ᵀQ₋₊⁻¹Aₜ₊Q₋ₜ == Q₋ₜ - Q₋ₜᵀAₜ₊ᵀL⁻ᵀL⁻¹Aₜ₊Q₋ₜ
    T = Q_fwd - A_back_Q_fwd.T @ Q_mp_inv_A_back @ Q_fwd
    # W = Q₋ₜAₜ₊ᵀQ₋₊⁻¹
    W = Q_fwd @ Q_mp_inv_A_back.T
    P = np.concatenate([A_fwd - W @ A_back @ A_fwd, W], axis=-1)
    return P, T


def predict_from_state_(x_test, ind, x, post_mean, post_cov, gain, kernel):
    """
    predict the state distribution at time t by projecting from the neighbouring inducing states
    """
    P, T = compute_conditional_statistics(x_test, x, kernel, ind)
    # joint posterior (i.e. smoothed) mean and covariance of the states [u_, u+] at time t:
    mean_joint = np.block([[post_mean[ind]],
                           [post_mean[ind + 1]]])
    cross_cov = gain[ind] @ post_cov[ind + 1]
    cov_joint = np.block([[post_cov[ind], cross_cov],
                          [cross_cov.T, post_cov[ind + 1]]])
    return P @ mean_joint, P @ cov_joint @ P.T + T


def predict_from_state(x_test, ind, x, post_mean, post_cov, gain, kernel):
    """
    wrapper function to vectorise predict_at_t_()
    """
    predict_from_state_func = vmap(
        predict_from_state_, (0, 0, None, None, None, None, None)
    )
    return predict_from_state_func(x_test, ind, x, post_mean, post_cov, gain, kernel)