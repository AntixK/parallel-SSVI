from ..lgssm import LGSSM
import jax.numpy as np
from jax import vmap
from jax.lax import scan
from jax.numpy import concatenate
from jax.scipy.linalg import cho_factor, cho_solve


def _sequential_rt_smoother(lgssm: LGSSM,
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
