# -*- coding: utf8 -*-

import numpy as np
from numpy import r_, pi
from scipy.interpolate import interp1d
import functools


import sliding_surface
from model_and_parameters import mu, beta, z4_park, dt
from model_and_parameters import rhs, z2_fnc, z4d, z4_drift
from model_and_parameters import zero_crossing_simulation


"""
This module serves for calculating the "characteristic line", i.e. the
numerical relation between the initial values of maneuver C and the achieved
z1 displacement.

It also provides functions to use this relation for the execution of a desired
maneuverC
"""


# Note: This Code overlaps with _do_maneuverC in the maneuver_library
# However, that code duplication is not so easy to solve
# This module should stay independent  (for creating the "characteristic line")

def maneuverC(z3_star):
    """
    performs maneuverC for a given z3_star
    assumes to be in parking regime with z4 = z4_park
    """

    # Determine the sign of the drift right next to the pariking position
    drift_sign = np.sign(z4_drift(z2=-1e-3, z3=z3_star, z4=z4_park))

    if drift_sign < 0:
        C.phi = C.phi_down
    else:
        C.phi = C.phi_up

    z0 = r_[z3_star, z4_park + 1.0e-6 * drift_sign]

    tt = np.arange(0, 20, dt)
    tt_, res = zero_crossing_simulation(rhs, z2_fnc, z0, tt)

    z3, z4 = res.T
    z2 = C.phi(z4)
    assert dt == (tt[1] - tt[0])
    z1 = np.cumsum(z2) * dt

    #print u"∆z_1/pi = ", z1[-1]/pi, "   :  ", drift_sign

    return tt_, z1, z2, z3, z4


def create_characteristic_line():

    # just single branch

    # complete interval (0, 2pi)
    z3_values = np.linspace(.01, 1.99, 503) * pi

    # remove values near to multiples of pi
    def f_func(x):  # filter func
        return np.abs( .5 - (x / pi) % 1) < .475

    z3_values = np.array(filter(f_func, z3_values))

    dz1 = []
    z3_cross = []

    for z3_val in z3_values:
        tt, z1, z2, z3, z4 = maneuverC(z3_val)
        dz1.append(z1[-1])
        z3_cross.append(z3[-1])

    dz1 = np.array(dz1)
    z3_cross = np.array(z3_cross)

    result = np.vstack((z3_values, dz1, z3_cross)).T

    np.save(fname, result)
    print "file written:", fname


def test_characteristic_line():
    result = np.load(fname)
    z3_values, dz1, z3_cross = result.T

    pl.figure(1)
    pl.plot(z3_values / pi, dz1 / pi, 'b.')
    pl.title('dz1(z3_star)')
    pl.figure(2)
    pl.plot(z3_values / pi, z3_cross / pi, 'r.-')
    pl.title('z_3_cross(z3_star)')


def get_characteristic_line_fncs():
    """
    returns two interpolation functions (for each monoton rising part)
    """

    result = np.load(fname)
    z3_values, dz1, z3_cross = result.T

    # find the gap in z3:
    dz3 = np.diff(z3_values)
    i_gap = np.argmax(dz3) + 1  # index after the gap
    # assert that the gap is somewhere in the middle:
    assert .4 < i_gap * 1.0 / len(z3_values) < 0.6

    z3_fnc1 = interp1d(dz1[:i_gap], z3_values[:i_gap])
    z3_fnc2 = interp1d(dz1[i_gap:], z3_values[i_gap:])

    return z3_fnc1, z3_fnc2  # z3_starC in dependency of dz1


def test_characteristic_line_fncs():
    """
    creates some figures for testing and illustration purposes
    """

    z3_fnc1, z3_fnc2 = get_characteristic_line_fncs()

    dz1_values1 = np.linspace( min(z3_fnc1.x), max(z3_fnc1.x), 1000 )
    dz1_values2 = np.linspace( min(z3_fnc2.x), max(z3_fnc2.x), 1000 )

    pl.figure(1)
    pl.plot(z3_fnc1(dz1_values1) / pi, dz1_values1 / pi, 'r-')
    pl.plot(z3_fnc2(dz1_values2) / pi, dz1_values2 / pi, 'm-')


def get_man_C_exec_parameters(z3, z4, dz1):
    """
    takes relevant state components (z3, z4) and desired Delta_z1

    returns:
        z3-start value for the maneuver C
        dz1_rest value (nonzero when a second instance is necessary)
    """

    # transform dz1 to reachable range:
    dz1_reach = dz1 % (2 * pi) - 2 * pi  # \in [-2*pi, 0)

    z3_fnc1, z3_fnc2 = get_characteristic_line_fncs()

    if dz1_reach < -pi:
        # a second instance of this maneuver will be necessary
        dz1_rest = dz1_reach + pi
        dz1_reach = -pi
    else:
        dz1_rest = 0

    # two candidates for the z3-start-value
    z3_1 = z3_fnc1(r_[dz1_reach])
    z3_2 = z3_fnc2(r_[dz1_reach])

    cond1 = (0 < r_[z3_1, z3_2]).all()
    cond2 = (r_[z3_1, z3_2] < ( 2 * pi)).all()
    assert cond1
    assert cond2

    assert z4 > 0
    # the case z4 < 0 will be handled outside by using symmetry properties

    if z3_1 < z3 < z3_2:
        res = z3_2
    else:
        res = z3_1

    return res, dz1_rest


def main():

    print "create and test the characteristic line for maneuver C"

    # uncommenting the following line recalculates the whole characteristic
    # line (takes some time)
    # create_characteristic_line()

    test_characteristic_line()
    test_characteristic_line_fncs()
    pl.show()


# File where the data for the characteristic line is stored
fname = 'man_c_raw_result.npy'

# maxvalue (to respect maximum absolute value of z2) will practically
# not be reached
mv = 20


class Container(object):
    pass

# make the phi-Functions rw-accessible from everywhere
C = Container()
C.phi = None

C.phi_down = sliding_surface.SlidingSurface(z4_star=0, z4_cross=z4_park,
                                            beta=beta, mu1=mu, mu2=mu,
                                            res_sign=-1, singleEndBranch=True,
                                            minval1=0.2, maxval1=mv,
                                            maxval2=mv).fnc

C.phi_up = sliding_surface.SlidingSurface(z4_star=z4_park * 2, z4_cross=z4_park,
                                          beta=beta, mu1=mu, mu2=mu,
                                          res_sign=-1, singleEndBranch=True,
                                          minval1=0.2, maxval1=mv,
                                          maxval2=mv).fnc

# partially evaluate the functions such that they have runtime access
# tho the correct sliding surface
rhs = functools.partial(rhs, phi_container=C)
z2_fnc = functools.partial(z2_fnc, phi_container=C)
z4d = functools.partial(z4d, phi_container=C)

if __name__ == "__main__":
    import pylab as pl
    main()