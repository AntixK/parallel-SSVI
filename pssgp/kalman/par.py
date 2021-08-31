import math
from ..lgssm import LGSSM
import jax.numpy as np
from jax import vmap
from jax.lax import scan, cond
from jax.numpy import concatenate
from jax.scipy.linalg import cho_factor, cho_solve

LOG2PI = math.log(2 * math.pi)
INV2PI = (2 * math.pi) ** -1

def first_filtering_element(m0, P0, F, Q, H, R, y):
    """
    Equation 11 in "GPR in Logarithmic Time"
    """

    def _res_nan():
        A = np.zeros_like(F)
        b = m0
        C = P0
        eta = np.zeros_like(m0)
        J = np.zeros_like(F)

        return A, b, C, J, eta

    def _res_not_nan():
        S1 = H @ (P0 @ H.T) + R
        S1_chol = cho_factor(S1)
        K1t = cho_solve(S1_chol, H @ P0)

        A = np.zeros_like(F)
        b = m0 - K1t.T @ (y - H @ m0)
        C = P0 - (K1t.T @ S1) @ K1t

        S = H @ (Q @ H.T) + R
        chol = cho_factor(S)

        HF = H @ F
        eta = HF.T @ np.squeeze(cho_solve(chol, np.expand_dims(y, 1)))
        J = HF.T @ cho_solve(chol, HF)
        return A, b, C, J, eta

    res = cond(np.isnan(y), _res_nan, _res_not_nan)
    return res

def _generic_filtering_element_nan(F, Q):
    A = F
    b = np.zeros(F.shape[:2], dtype=F.dtype)
    C= Q
    eta = np.zeros(F.shape[:2], dtype=F.dtype)
    J = np.zeros_like(F)

    return A, b, C, J, eta

def _generic_filtering_element(F, Q, H, R, y):
    """
    Equation 10 in "GPR in Logarithmic Time"
    """

    S = H @ (Q @ H.T) + R

    chol = cho_factor(S)
    Kt = cho_solve(chol, H @ Q)
    A = F - (Kt.T @ H) @ F

    b= Kt.T @ y
    C = Q - (Kt.T @ H) @ Q

    HF = H @ F
    eta = HF.T @ np.squeeze(cho_solve(chol, np.expand_dims(y, 1)))
    J = HF.T @ cho_solve(chol, HF)

    return A, b, C, J, eta

def make_associative_scan_filtering_elements(m0, P0, Fs,  Qs, H, R, observations):
    init_res = first_filtering_element(m0, P0, Fs[0], Qs[0], H, R observations[0])
    

