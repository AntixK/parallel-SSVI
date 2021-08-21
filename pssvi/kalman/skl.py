import math
import jax.numpy as np
from jax import vmap
from jax.lax import scan
from typing import Callable, Sequence, Any

from ..utils import transpose, mvn_logpdf, solve

LOG2PI = math.log(2 * math.pi)
INV2PI = (2 * math.pi) ** -1



def _sequential_kf(As, Qs, H, ys, noise_covs, m0, P0, masks, return_predict=False):

    def body(carry, inputs):
        y, A, Q, obs_cov, mask = inputs
        m, P, ell = carry
        m_ = A @ m
        P_ = A @ P @ A.T + Q

        obs_mean = H @ m_
        HP = H @ P_
        S = HP @ H.T + obs_cov

        ell_n = mvn_logpdf(y, obs_mean, S, mask)
        ell = ell + ell_n

        K = solve(S, HP).T
        m = m_ + K @ (y - obs_mean)
        P = P_ - K @ HP
        if return_predict:
            return (m, P, ell), (m_, P_)
        else:
            return (m, P, ell), (m, P)

    (_, _, loglik), (fms, fPs) = scan(f=body,
                                      init=(m0, P0, 0.),
                                      xs=(ys, As, Qs, noise_covs, masks))
    return loglik, fms, fPs


def kalman_filter(dt, kernel, y, noise_cov, mask=None, parallel=False, return_predict=False):
    """
    Run the Kalman filter to get p(fₙ|y₁,...,yₙ).
    Assumes a heteroscedastic Gaussian observation model, i.e. var is vector valued
    :param dt: step sizes [N, 1]
    :param kernel: an instantiation of the kernel class, used to determine the state space model
    :param y: observations [N, D, 1]
    :param noise_cov: observation noise covariances [N, D, D]
    :param mask: boolean mask for the observations (to indicate missing data locations) [N, D, 1]
    :param parallel: flag to switch between parallel and sequential implementation of Kalman filter
    :param return_predict: flag whether to return predicted state, rather than updated state
    :return:
        ell: the log-marginal likelihood log p(y), for hyperparameter optimisation (learning) [scalar]
        means: intermediate filtering means [N, state_dim, 1]
        covs: intermediate filtering covariances [N, state_dim, state_dim]
    """
    if mask is None:
        mask = np.zeros_like(y, dtype=bool)
    Pinf = kernel.stationary_covariance()
    minf = np.zeros([Pinf.shape[0], 1])

    As = vmap(kernel.state_transition)(dt)
    Qs = vmap(process_noise_covariance, [0, None])(As, Pinf)
    H = kernel.measurement_model()

    if parallel:
        raise NotImplementedError
    else:
        ell, means, covs = _sequential_kf(As, Qs, H, y, noise_cov, minf, Pinf, mask, return_predict=return_predict)
    return ell, (means, covs)


def _sequential_rts(fms, fPs, As, Qs, H, return_full):

    def body(carry, inputs):
        fm, fP, A, Q = inputs
        sm, sP = carry

        pm = A @ fm
        AfP = A @ fP
        pP = AfP @ A.T + Q

        C = solve(pP, AfP).T

        sm = fm + C @ (sm - pm)
        sP = fP + C @ (sP - pP) @ C.T
        if return_full:
            return (sm, sP), (sm, sP, C)
        else:
            return (sm, sP), (H @ sm, H @ sP @ H.T, C)

    _, (sms, sPs, gains) = scan(f=body,
                                init=(fms[-1], fPs[-1]),
                                xs=(fms, fPs, As, Qs),
                                reverse=True)
    return sms, sPs, gains


def process_noise_covariance(A, Pinf):
    Q = Pinf - A @ Pinf @ transpose(A)
    return Q


def rauch_tung_striebel_smoother(dt, kernel, filter_mean, filter_cov, return_full=False, parallel=False):
    """
    Run the RTS smoother to get p(fₙ|y₁,...,y_N),
    :param dt: step sizes [N, 1]
    :param kernel: an instantiation of the kernel class, used to determine the state space model
    :param filter_mean: the intermediate distribution means computed during filtering [N, state_dim, 1]
    :param filter_cov: the intermediate distribution covariances computed during filtering [N, state_dim, state_dim]
    :param return_full: a flag determining whether to return the full state distribution or just the function(s)
    :param parallel: flag to switch between parallel and sequential implementation of smoother
    :return:
        smoothed_mean: the posterior marginal means [N, obs_dim]
        smoothed_var: the posterior marginal variances [N, obs_dim]
    """
    Pinf = kernel.stationary_covariance()

    As = vmap(kernel.state_transition)(dt)
    Qs = vmap(process_noise_covariance, [0, None])(As, Pinf)
    H = kernel.measurement_model()

    if parallel:
        raise NotImplementedError
    else:
        means, covs, gains = _sequential_rts(filter_mean, filter_cov, As, Qs, H, return_full)
    return means, covs, gains
