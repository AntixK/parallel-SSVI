import objax
from jax import vmap
import jax.numpy as np
from jax.scipy.linalg import cho_factor, cho_solve, block_diag
from .utils import scaled_squared_euclid_dist, softplus, softplus_inv, rotation_matrix
from warnings import warn

__all__ = ["Matern12","Matern32"]

class Kernel(objax.Module):
    """
    """

    def __call__(self, X, X2):
        return self.K(X, X2)

    def K(self, X, X2):
        raise NotImplementedError('kernel function not implemented')

    def measurement_model(self):
        raise NotImplementedError

    def inducing_precision(self):
        return None, None

    def kernel_to_state_space(self, R=None):
        raise NotImplementedError

    def spatial_conditional(self, R=None, predict=False):
        """
        """
        return None, None


class StationaryKernel(Kernel):
    """
    """

    def __init__(self,
                 variance=1.0,
                 lengthscale=1.0,
                 fix_variance=False,
                 fix_lengthscale=False):
        # check whether the parameters are to be optimised
        if fix_lengthscale:
            self.transformed_lengthscale = objax.StateVar(softplus_inv(np.array(lengthscale)))
        else:
            self.transformed_lengthscale = objax.TrainVar(softplus_inv(np.array(lengthscale)))
        if fix_variance:
            self.transformed_variance = objax.StateVar(softplus_inv(np.array(variance)))
        else:
            self.transformed_variance = objax.TrainVar(softplus_inv(np.array(variance)))

    @property
    def variance(self):
        return softplus(self.transformed_variance.value)

    @property
    def lengthscale(self):
        return softplus(self.transformed_lengthscale.value)

    def K(self, X, X2):
        r2 = scaled_squared_euclid_dist(X, X2, self.lengthscale)
        return self.K_r2(r2)

    def K_r2(self, r2):
        # Clipping around the (single) float precision which is ~1e-45.
        r = np.sqrt(np.maximum(r2, 1e-36))
        return self.K_r(r)

    @staticmethod
    def K_r(r):
        raise NotImplementedError('kernel not implemented')

    def kernel_to_state_space(self, R=None):
        raise NotImplementedError

    def measurement_model(self):
        raise NotImplementedError

    def state_transition(self, dt):
        raise NotImplementedError

    def stationary_covariance(self):
        raise NotImplementedError


class Matern12(StationaryKernel):
    """
    The Matern 1/2 kernel. Functions drawn from a GP with this kernel are not
    differentiable anywhere. The kernel equation is

    k(r) = σ² exp{-r}

    where:
    r  is the Euclidean distance between the input points, scaled by the lengthscales parameter ℓ.
    σ² is the variance parameter
    """

    @property
    def state_dim(self):
        return 1

    def K_r(self, r):
        return self.variance * np.exp(-r)

    def kernel_to_state_space(self, R=None):
        F = np.array([[-1.0 / self.lengthscale]])
        L = np.array([[1.0]])
        Qc = np.array([[2.0 * self.variance / self.lengthscale]])
        H = np.array([[1.0]])
        Pinf = np.array([[self.variance]])
        return F, L, Qc, H, Pinf

    def stationary_covariance(self):
        Pinf = np.array([[self.variance]])
        return Pinf

    def measurement_model(self):
        H = np.array([[1.0]])
        return H

    def state_transition(self, dt):
        """
        Calculation of the discrete-time state transition matrix A = expm(FΔt) for the exponential prior.
        :param dt: step size(s), Δtₙ = tₙ - tₙ₋₁ [scalar]
        :return: state transition matrix A [1, 1]
        """
        A = np.broadcast_to(np.exp(-dt / self.lengthscale), [1, 1])
        return A


class Matern32(StationaryKernel):
    """
    The Matern 3/2 kernel. Functions drawn from a GP with this kernel are once
    differentiable. The kernel equation is

    k(r) = σ² (1 + √3r) exp{-√3 r}

    where:
    r  is the Euclidean distance between the input points, scaled by the lengthscales parameter ℓ,
    σ² is the variance parameter.
    """

    @property
    def state_dim(self):
        return 2

    def K_r(self, r):
        sqrt3 = np.sqrt(3.0)
        return self.variance * (1.0 + sqrt3 * r) * np.exp(-sqrt3 * r)

    def kernel_to_state_space(self, R=None):
        lam = 3.0 ** 0.5 / self.lengthscale
        F = np.array([[0.0,       1.0],
                      [-lam ** 2, -2 * lam]])
        L = np.array([[0],
                      [1]])
        Qc = np.array([[12.0 * 3.0 ** 0.5 / self.lengthscale ** 3.0 * self.variance]])
        H = np.array([[1.0, 0.0]])
        Pinf = np.array([[self.variance, 0.0],
                         [0.0, 3.0 * self.variance / self.lengthscale ** 2.0]])
        return F, L, Qc, H, Pinf

    def stationary_covariance(self):
        Pinf = np.array([[self.variance, 0.0],
                         [0.0, 3.0 * self.variance / self.lengthscale ** 2.0]])
        return Pinf

    def measurement_model(self):
        H = np.array([[1.0, 0.0]])
        return H

    def state_transition(self, dt):
        """
        Calculation of the discrete-time state transition matrix A = expm(FΔt) for the Matern-3/2 prior.
        :param dt: step size(s), Δtₙ = tₙ - tₙ₋₁ [scalar]
        :return: state transition matrix A [2, 2]
        """
        lam = np.sqrt(3.0) / self.lengthscale
        A = np.exp(-dt * lam) * (dt * np.array([[lam, 1.0], [-lam**2.0, -lam]]) + np.eye(2))
        return A
