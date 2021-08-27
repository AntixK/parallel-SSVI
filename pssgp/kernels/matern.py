import math
import torch
import numpy as np
from ..lgssm import LGSSM
from scipy.special import binom
from ..cssgp import ContinuousSSGP
from .utils import solve_lyap_vec, balance_ss, _cssgp_to_lgssm
from gpytorch.kernels import Kernel, ScaleKernel, MaternKernel

__all__ = ["MaternKernel"]


class MaternKernel(Kernel):

    has_lengthscale = True

    def __init__(self,
                 nu: float=2.5,
                 lengthscale_prior: float = None,
                 outputscale_prior: float = None,
                 **kwargs) -> None:
        """
        A thin wrapper around GPytorch's Matern kernel
        that also provides the corresponding state space
        model.
        :param nu: Smoothness parameter
        :param kwargs: Miscellaneous parameters
                       Refer https://docs.gpytorch.ai/en/latest/kernels.html#maternkernel
        """
        self.kernel = ScaleKernel(MaternKernel(nu, **kwargs),
                                  outputscale_prior=outputscale_prior,
                                  **kwargs)
        self.nu = nu

    def forward(self, x1, x2, diag=False, last_dim_is_batch=False, **params):
        return self.kernel(x1, x2, diag, **params)

    def _get_cssgp(self) -> ContinuousSSGP:
        """
        Computes the equivalent continuous state space
        formulation of a GP with Matern covariance.
        :return: Continuous SSGP model
        """
        G, L, H, Q = get_matern_sde(self.outputscale, self.lengthscale, self.nu)
        if self.nu < 1:
            Pinf = torch.diag([self.outputscale.sum()])

        elif self.nu < 2:
            l = math.sqrt(3) / self.lengthscale.sum()
            Pinf = torch.diag(torch.tensor([self.outputscale.sum(),
                                            l ** 2 * self.outputscale.sum()]))

        else:
            Gb, Lb, Hb, Qb = balance_ss(G, L, H, Q)
            Pinf = solve_lyap_vec(Gb, Lb, Qb)
        return ContinuousSSGP(Pinf, Gb, Lb, Hb, Qb)

    def LGSSM(self,
              R: torch.Tensor,
              ts:torch.Tensor,
              t0: float = 0.0) -> LGSSM:
        cssgp = self._get_cssgp()
        t0 = torch.Tensor(t0).view(1,1)
        return _cssgp_to_lgssm(cssgp, R, ts, t0)


def get_matern_sde(variance: torch.Tensor,
                   lengthscale: torch.Tensor,
                   nu: float):
    d = int(nu + 0.5)
    l = math.sqrt(2*d - 1) / lengthscale
    G = _get_transition_matrix(l, nu)
    ones = torch.ones((1,))
    L = torch.diag(ones, diagonal=-d + 1)[: , 0:1]
    H = torch.eye(d)[: , 0]
    Q = _get_brownian_cov(variance, l, nu)
    return G, L, H, Q


def _get_brownian_cov(variance: torch.Tensor,
                      l: torch.Tensor,
                      nu: float):
    d = int(nu + 0.5)
    q = (2 * l) ** (2 * d - 1) * variance * math.factorial(d - 1) ** 2 / math.factorial(2 * d - 2)
    return q * torch.eye(1)


def _get_transition_matrix(l: torch.Tensor,
                           nu: float):
    d = int(nu + 0.5)
    G = torch.diag(torch.ones((d-1,)), diagonal=1)
    binomial_coeffs = torch.from_numpy(binom(d, np.arange(0, d, dtype=int)))

    lambda_powers = l ** np.arange(d, 0, -1, dtype=np.float32)
    update_indices = [[d - 1, k] for k in range(d)]
    G.scatter_add_(0, update_indices, -1*lambda_powers * binomial_coeffs)
    return G





