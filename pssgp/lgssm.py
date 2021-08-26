from collections import namedtuple

"""
The Linear Gaussian State Space Model (LGSSM) is the discrete-time 
equivalent of the continuous state space formulation of a GP.
The LGSSM is formulated as

x_k = F_k-1 x_k-1 + q_k-1
y_k = H x_k + e_k
q_k-1 ~ N(0, Q_k-1)

P0 is the steady state covariance.
w(t) is the white noise process with spectral density Q

e_k is the observation noise with covariance R. 
"""

LGSSM = namedtuple("LGSSM", ["P0", "F", "Q", "H", "R"])