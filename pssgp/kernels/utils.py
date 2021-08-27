from functools import partial
import torch
from ..lgssm import LGSSM
from ..cssgp import ContinuousSSGP
from gpytorch.lazy import KroneckerProductLazyTensor, DiagLazyTensor,NonLazyTensor


@torch.jit.script
def _torch_balance_ss(F: torch.Tensor,
                      iter: int) -> torch.Tensor:
    dim = F.shape[0]
    dtype = F.dtype
    d = torch.ones((dim,))
    for k in range(iter):
        for i in range(dim):
            tmp = torch.clone(F[:, i])
            tmp[i] = 0.
            c = torch.linalg.norm(tmp, 2)
            tmp2 = torch.clone(F[i, :])
            tmp2[i] = 0.

            r = torch.linalg.norm(tmp2, 2)
            f = torch.sqrt(r / c)
            d[i] *= f
            F[:, i] *= f
            F[i, :] /= f
    return d


def balance_ss(G: torch.Tensor,
               L: torch.Tensor,
               H: torch.Tensor,
               Q: torch.Tensor,
               n_iter:int = 10):
    """

    :param G:
    :param L:
    :param H:
    :param Q:
    :param n_iter:
    :return:
    """
    d = partial(_torch_balance_ss, iter=n_iter)(G)

    Gt= G * d[None, :] / d[:, None]
    Lt = L / d[:, None]
    Ht = H * d[None, :]

    tmp3 = Lt.abs().max()
    Lt = Lt / tmp3
    Qt= (tmp3 ** 2) * Q

    tmp4 = Ht.abs().max()
    Ht = Ht / tmp4
    Qt = (tmp4 ** 2) * Qt

    return Gt, Lt, Ht, Qt


def solve_lyap_vec(G: torch.Tensor,
                   Q: torch.Tensor,
                   L: torch.Tensor):
    """
    Find the solution Pinf to the vectorized Layapunov equation
            G Pinf + Pinf G' + L Q L' = 0

    :param G: [torch.Tensor]
    :param Q: [torch.Tensor]
    :param L: [torch.Tensor]
    :return: [torch.Tensor] Solution to the Lyapunov equation
    """
    dim = G.shape[0]

    op1 = NonLazyTensor(G)
    op2 = DiagLazyTensor(torch.ones(dim))

    F1 = KroneckerProductLazyTensor(op2, op1).evaluate()
    F2 = KroneckerProductLazyTensor(op1, op2).evaluate()

    Fs = F1 + F2

    Qs = L @ (Q @ L.t())
    Pinf = torch.linalg.solve(Fs, Qs.view(-1, 1)).view(dim, dim)
    Pinf = -0.5 * (Pinf + Pinf.t())
    return Pinf


def _cssgp_to_lgssm(cssgp: ContinuousSSGP,
                    R: torch.Tensor,
                    ts: torch.Tensor,
                    t0: torch.Tensor) -> LGSSM:
    """
    Method to convert a given Continuous State Space GP model
    to a discrete linear Gaussian State Space model
    :param cssgp: [ContinuousSSGP] Continuous SSGP model
    :param R: [torch.Tensor] Observation covariance
    :param ts: [torch.Tensor] time
    :param t0: [torch.Tensor] Initial time
    :return: LSGGM model
    """
    Pinf, G, L, H, Q = cssgp
    N = G.shape[0]

    ts = torch.vstack([t0, ts])
    dts = (ts[1:] - t0[:-1]).view(-1, 1, 1)
    F = torch.matrix_exp(dts * G[None, :, :])

    z = torch.zeros_like(G)
    Phi = torch.cat(
        [torch.cat([G, L @ (Q @ L.t())], dim=1),
         torch.cat([z, -G.t()], dim=1)],
        dim=0)

    AB = torch.matrix_exp(dts * Phi[None, :, :])
    AB = AB @ torch.cat([z, torch.eye(N)], dim=0)
    Qs = AB[:, :N, :] @ F.transpose(0, 1)

    return LGSSM(Pinf, F, Qs, H, R)