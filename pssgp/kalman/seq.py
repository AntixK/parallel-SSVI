import math
from ..lgssm import LGSSM
import jax.numpy as np
from jax import vmap
from jax.lax import scan, cond
from jax.numpy import concatenate
from jax.scipy.linalg import cho_factor, cho_solve

LOG2PI = math.log(2 * math.pi)
INV2PI = (2 * math.pi) ** -1

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

def _sequential_kalman_filter(lgssm: LGSSM,
                              observations,
                              return_loglikelihood: bool = False,
                              return_predicted:bool=False):
    P0, F, Q, H, R = lgssm
    m0 = np.zeros(P0.shape[0])

    def body(carry, inp):
        ell, m, P, *_ = carry
        y, F, Q = inp

        mp = F @ m
        Pp = F @ (P @ F.T) + Q
        Pp = 0.5 * (Pp + Pp.T)

        def update(m, P, ell):
            S = H @ (P @ H.T) + R
            yp = H @ m
            chol = cho_factor(S)
            ell_t = mvn_logpdf(y, yp, S)

            Kt = cho_solve(chol, H @ P)

            m += Kt.T @ (y - yp)
            P -= Kt.T @ S @ Kt
            ell += ell_t
            return ell, m,P

        nan_y = ~y.isnan()
        nan_res = (ell, mp, Pp)
        ell, m,P = cond(nan_y, lambda:update(mp, Pp, ell), lambda: nan_res)
        P = 0.5 * (P + P.T)
        return ell, m, P, mp,Pp

    ells, fms,fPs, mps, Pps = scan(
                                f = body,
                                xs =(observations, F, Q),
                                init=(0., m0,P0, m0, P0)
    )
    return_vals = (fms, fPs) + ((ells[-1],) if return_loglikelihood else ()) + (
                    (mps, Pps) if return_predicted else ())
    return return_vals


def _sequential_rts_smoother(lgssm: LGSSM,
                    ms: np.ndarray,
                    Ps: np.ndarray,
                    mps: np.ndarray,
                    Pps: np.ndarray):
    _, F, Q,*_ = lgssm

    def body(carry, inputs):
        F, Q, m, P, mp, Pp = inputs
        sm, sP = carry

        L = cho_factor(Pp)
        Ct = cho_solve(L, F @ P)

        sm = m + Ct @ (sm - mp).T
        sP = P + Ct @ (sP - Pp).T @ Ct

        return sm, 0.5 * (sP + sP.T)

    (sms, sPs) = scan(f=body,
                      init=(F[1:], Q[1:], ms[:-1], Ps[:-1], mps[1:], Pps[1:]),
                      xs=(ms[-1], Ps[-1]),
                      reverse=True)
    sms = concatenate([sms, ms[-1].expand_dims(0)], axis=0)
    sPs = concatenate([sPs, Ps[-1].expand_dims(0)], axis=0)
    return sms, sPs


def seq_kalman_filter_smmother(lgssm: LGSSM,
                               observations):
    fms, fPs, mps, Pps = _sequential_kalman_filter(lgssm, observations, return_predicted=True)
    return _sequential_rts_smoother(lgssm, fms, fPs, mps, Pps)