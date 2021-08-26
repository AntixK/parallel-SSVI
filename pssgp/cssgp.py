from collections import namedtuple

"""
The continuous state space characterization of a temporal GP.
Given a temporal GP with a covariance kernel K, the continuous 
state space formulation is given as

dx/dt = Gx(t) + L w(t)
y_k = H x(t_k) + e_k
q(t) ~ N(0, Q)

With the initial state x_0 ~ N(0, P0).

P0 is the steady state covariance.
w(t) is the white noise process with spectral density Q

e_k is the observation noise with covariance R. 
We shall model that directly in the LGSSM model.
"""
ContinuousSSGP = namedtuple("ContinuousSSGP", ["P0", "G", "L", "H", "Q"])