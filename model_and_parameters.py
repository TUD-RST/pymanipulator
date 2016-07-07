# -*- coding: utf8 -*-

from numpy import r_, sin, cos
import numpy as np
import scipy.integrate

"""
This module contains some general parameters and some functions to simulate
the manipulator model
"""


# manipulator model inertia parameter
kappa = 0.9

# sliding surface
beta = 2.5
mu = 8
z4_park = 1.4

# simulation
dt = .001
z4_offset = 1e-6

# numerical procedure parameter:
eps = 1e-5  # smaller absolute values are considered "0"


def zero_crossing_simulation(rhs_fnc, zcf, z0, t_values):
    """
    scipy.odeint does not provide a zero crossing function
    naive (and slow) approach

    rhs: rhs function
    zcf: the function whose zerocrossing shall be detected
         takes the state (shape =(n,m) returns shape=n
    z0: initial state
    t_values: time values (up to which the zc event is suspected)
    """

    res = scipy.integrate.odeint(rhs_fnc, z0, t_values)

    zerocc = zcf(res)  # zero crossing candidate

    test_idx = 2  # ignore numerical noise at the beginning
    try:
        # find the first index where zcc has a different sign
        cond = np.sign(zerocc[test_idx:]) != np.sign(zerocc[test_idx])
        idx0 = np.where( cond )[0][0]  # first index of True-entry
    except IndexError:
        raise ValueError("There was no zero crossing")

    idx0 += test_idx

    if zerocc[idx0] == 0:
        idx0 += 1

    t_values = t_values[:idx0] * 1  # *1 creates a copy (prevent referencing)

    return t_values, res[:idx0, :]

# note: the following functions need some information about the sliding surface
# which is only available at runtime. Therfore a suitable container object
# will be passed via partial evaluation of this function (functools partial)


def rhs(z, t, phi_container=None):
    """
    returns dz/dt.
    z is the state of the reduced dynamics on the sliding surface: z=(z3,z4)
    """

    z3, z4 = z
    z2 = phi_container.phi(z4)

    z3d_value = z4 - (1 + kappa * cos(z3)) * z2
    z4d_value = - kappa * z2 * sin(z3) * (z4 - kappa * cos(z3) * z2)

    return r_[z3d_value, z4d_value]


def z4d(z, phi_container=None):
    """
    drift term of the reduced dynamics
    """
    z3, z4 = z.T
    z2 = phi_container.phi(z4)

    z4d_value = - kappa * z2 * sin(z3) * (z4 - kappa * cos(z3) * z2)
    return z4d_value


def z4_drift(z2, z3, z4):
    """
    z4-drift (of the dynamics, which is not reduced to the sliding surface)
    """
    # this is needed for maneuver C
    return -kappa * z2 * sin(z3) * (z4 - kappa * cos(z3) * z2)


def z2_fnc(z, phi_container=None):
    """
    returns z2 as a function of z4, assuming sliding mode (-> reduced dynamics)
    """
    z3, z4 = z.T
    z2 = phi_container.phi(z4)

    return z2
