import math
import torch
import gpytorch
import numpy as np
import random
from matplotlib import pyplot as plt
from pssgp.kernels import MyMaternKernel

from unittest import TestCase

# We will use the simplest form of GP model, exact inference
class ExactGPModel(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood, use_gpy):
        super(ExactGPModel, self).__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ConstantMean()

        if use_gpy:
            self.covar_module = gpytorch.kernels.ScaleKernel(
                                    gpytorch.kernels.MaternKernel(nu=1.5))
        else:
            self.covar_module = MyMaternKernel(nu=1.5)

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)

def run(model,likelihood, train_x, train_y):
    # Find optimal model hyperparameters
    model.train()
    likelihood.train()

    # Use the adam optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=0.1)  # Includes GaussianLikelihood parameters

    # "Loss" for GPs - the marginal log likelihood
    mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)

    for i in range(50):
        # Zero gradients from previous iteration
        optimizer.zero_grad()
        # Output from model
        output = model(train_x)
        # Calc loss and backprop gradients
        loss = -mll(output, train_y)
        loss.backward()
        # print('Iter %d/%d - Loss: %.3f   lengthscale: %.3f   noise: %.3f' % (
        #     i + 1, 50, loss.item(),
        #     model.covar_module.base_kernel.lengthscale.item(),
        #     model.likelihood.noise.item()
        # ))
        optimizer.step()

    # Get into evaluation (predictive posterior) mode
    model.eval()
    likelihood.eval()

    # Test points are regularly spaced along [0,1]
    # Make predictions by feeding model through likelihood
    with torch.no_grad(), gpytorch.settings.fast_pred_var():
        test_x = torch.linspace(0, 1, 51)
        observed_pred = likelihood(model(test_x))

    return observed_pred


class TestCompatitibilityWithGpyTorch(TestCase):

    def setUp(self) -> None:
        torch.manual_seed(0)
        np.random.seed(0)
        random.seed(0)
        torch.use_deterministic_algorithms(True)
        torch.backends.cudnn.benchmark = True

        # Training data is 100 points in [0,1] inclusive regularly spaced
        self.train_x = torch.linspace(0, 1, 100)
        # True function is sin(2*pi*x) with Gaussian noise
        self.train_y = torch.sin(self.train_x * (2 * math.pi)) + \
                  torch.randn(self.train_x.size()) * math.sqrt(0.04)

        # self.likelihood = gpytorch.likelihoods.GaussianLikelihood()

    def test_result(self):
        likelihood1 = gpytorch.likelihoods.GaussianLikelihood()
        gpymodel = ExactGPModel(self.train_x,
                                     self.train_y,
                                     likelihood1,
                                     use_gpy=True)

        likelihood2 = gpytorch.likelihoods.GaussianLikelihood()
        mymodel = ExactGPModel(self.train_x,
                                self.train_y,
                                likelihood2,
                                use_gpy=False)

        result1 = run(gpymodel, likelihood1, train_x=self.train_x, train_y=self.train_y)
        result2 = run(mymodel, likelihood2, train_x=self.train_x, train_y=self.train_y)

        assert torch.allclose(result1.loc, result2.loc)

