import torch
from gpytorch.likelihoods.likelihood import Likelihood
from gpytorch.models import ExactGP


class StateSpaceGP(ExactGP):

    def __init__(self,
                 train_x: torch.Tensor,
                 train_y: torch.Tensor,
                 likelihood: Likelihood,
                 parallel: bool = False) -> None:
        super(StateSpaceGP, self).__init__(train_x, train_y, likelihood)

        


