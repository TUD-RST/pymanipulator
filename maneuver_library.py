# -*- coding: utf8 -*-

import copy
import numpy as np
from numpy import r_, pi
import functools

import sliding_surface as ssf_lib
from model_and_parameters import rhs, z4d, z2_fnc
from model_and_parameters import beta, mu, dt, z4_park, z4_offset, eps
from model_and_parameters import zero_crossing_simulation
import man_C_aux

"""
This module contains functions to perform all individual maneuvers
"""

# TODO: these parameters should live in a config file


###################
# General code
###################


class Container(object):
    """
    Empty Class to store arbitrary objects as attributes
    """

    def __init__(self, **kwargs):
        # add all key-word args
        self.add(**kwargs)

    def add(self, **kwargs):
        # we do not want to overwrite something
        assert not (set(kwargs.keys()).intersection(self.__dict__.keys()))
        self.__dict__.update(kwargs)


# make the phi-Functions rw-accessible from everywhere
C = Container()
C.phi = None


class ManeuverResult(Container):
    """
    Container class to store all maneuver-related information
    """

    def __init__(self, **kwargs):
        self.done = True
        self.pr_start = []
        self.pr_end = []
        self.maneuver_name = "?"
        self.label_borders = [0]
        self.label_chain = []
        self.phi_list = []
        self.debug_flag = False

        # add all key-word args
        self.add(**kwargs)

    def deep_copy(self):
        """
        creates a new instance which has the same attributes with the same
        values, but the attributes are copies of the attributes of the original
        instance
        """
        res = type(self)()
        empty = type(self)()  # for comparison

        for k, v in self.__dict__.items():
            if k in empty.__dict__:
                continue
            if isinstance(v, np.ndarray):
                v = v * 1  # copy
            res.__dict__.update({k: v})
        # manually add some attributes
        res.maneuver_name = self.maneuver_name
        res.label_borders = self.label_borders
        res.label_chain = self.label_chain

        res.phi_list = copy.copy(self.phi_list)

        return res

    def save(self, fname):
        """
        save container via numpy.savez
        """

        np.savez(fname, tt=self.tt, zz=self.zz, acc=self.acc,
                 pr_start=self.pr_start, pr_end=self.pr_end,
                 label_chain=self.label_chain,
                 label_borders=self.label_borders,
                 phi_list=self.phi_list)
        print "file written:", fname

    def load(self, fname):
        """
        load container from numpy.savez-format
        """

        res = np.load(fname)
        self.tt = res['tt']
        self.zz = res['zz']
        self.acc = res['acc']

        self.pr_start = res['pr_start']
        self.pr_end = res['pr_end']
        self.label_chain = res['label_chain']
        self.label_borders = res['label_borders']
        self.phi_list = res['phi_list']

        self.z1, self.z2, self.z3, self.z4 = self.zz.T

    def set_name(self, maneuver_name):
        self.maneuver_name = maneuver_name
        assert len(self.label_chain) == 0
        self.label_chain.append(self.maneuver_name)

    def get_phi_labels(self):
        return [d['label'] for d in self.phi_list]

    def phi_dict(self):
        """
        return a dictionary containing information about the sliding surface
        of this maneuver
        """
        phi = getattr(self, 'phi', None)
        phi1 = getattr(self, 'phi1', None)
        phi2 = getattr(self, 'phi2', None)

        d = {'phi': phi, 'phi1': phi1, 'phi2': phi2}

        # functions can not be serialized -> store evaluated functions
        new_d = {}
        z4 = np.linspace(-5, 5, 1000)
        for name, func in d.items():
            if not func is None:
                z2 = func(z4)
                new_d[name] = (z2, z4)
            else:
                new_d[name] = None
        new_d['label'] = self.maneuver_name

        return new_d


def join_containers(C1, C2):
    """
    takes the content of two ManeuverResult-container objects and returns a
    joined container
    """
    res = ManeuverResult()
    res.zz = np.vstack((C1.zz, C2.zz[1:, :]))
    res.tt = np.hstack((C1.tt, C2.tt[1:] + C1.tt[-1]))
    res.acc = np.hstack([C1.acc, C2.acc[1:]])

    res.label_chain = C1.label_chain + C2.label_chain
    C2_borders_transformed = [b + C1.tt[-1] for b in C2.label_borders]
    res.label_borders = C1.label_borders + C2_borders_transformed

    # store sliding surface information
    if len(C1.phi_list) > 0:
        res.phi_list = copy.copy(C1.phi_list)
    else:
        res.phi_list.append( C1.phi_dict() )

    if len(C2.phi_list) > 0:
        res.phi_list += C2.phi_list
    else:
        res.phi_list.append( C2.phi_dict() )

    # quick access to state components
    res.z1, res.z2, res.z3, res.z4 = res.zz.T

    return res


def join_containers_with_pr(C1, C2, *containers, **kwargs):
    """
    connect the states of the two containers by a parking regime such that
    the resulting trajectory is consistent
    """

    containers = [copy.copy(c) for c in containers ]
#    assert C1.z4[-1] == C2.z4[0]
    assert np.allclose(C1.z4[-1], C2.z4[0])
    assert np.abs(C1.z2[-1]) < 2e-2
    assert np.abs(C2.z2[0]) < 2e-2

    z4 = C1.z4[-1]

    # incorporate the z3_change by preceding parking regimes
    z3_offset = kwargs.get('z3_offset', 0)

    # apply z3 these changes to second container (already done for first one)

    # C2_orig = C2  # for debugging
    C2 = C2.deep_copy()
    C2.z3 += z3_offset
    C2.zz[:, 2] += z3_offset

    # make z1 consistent
    C2.zz[:, 0] += C1.z1[-1] - C2.z1[0]

    dz3 = C2.z3[0] - C1.z3[-1]

    z3_offset_new1 = 0
    if np.abs(dz3) > 2 * pi:
        # this would mean more than one rotation -> not necessary

        z3_offset_new1 = int(dz3 / (2 * pi)) * 2 * pi

        C2.z3 -= z3_offset_new1
        C2.zz[:, 2] -= z3_offset_new1
        dz3 = C2.z3[0] - C1.z3[-1]

        assert np.abs(dz3) < 2 * pi

    # how long do we have to wait to reach a position which is
    # equivalent to C.z3[0]?

    z3_offset_new2 = 0

    if not np.sign(dz3) == np.sign(z4):
        # we need to calculate a new dz3 (with the appropriate sign)
        # -> new target for z3
        z3_offset_new2 = 2 * pi * np.sign(z4)
        C2.z3 += z3_offset_new2
        C2.zz[:, 2] += z3_offset_new2
        dz3 = C2.z3[0] - C1.z3[-1]

        assert np.abs(dz3) < 2 * pi

    steps = int(dz3 / z4 / dt)

    assert steps > 0

    tt = np.arange(steps + 1) * dt
    # tt = [0, 1, ...] but first index will be omitted by join_containers
    zz = np.zeros((steps + 1, 4))
    zz[:, :] = C1.zz[-1, :]  # use broadcasting here
    zz[:, 2] = C1.z3[-1] + z4 * tt

    assert np.abs( zz[-1, 2] - C2.z3[0] ) < np.abs(z4 * dt)

    Cpr = ManeuverResult(tt=tt, zz=zz, acc=tt * 0)
    Cpr.set_name("P")

    tmp = join_containers(C1, Cpr)

    res = join_containers(tmp, C2)

    # save the start and end-value of parking regime
    res.pr_start.extend(C1.pr_start)
    res.pr_start.append(C1.tt[-1] + dt)
    res.pr_start.extend(C2.pr_start)

    res.pr_end.extend(C1.pr_end)
    res.pr_end.append(tmp.tt[-1])
    res.pr_end.extend(C2.pr_end)

    z3_offset += z3_offset_new1 + z3_offset_new2

    if len(containers) > 0:
        return join_containers_with_pr(res, containers[0],
                                       *containers[1:], z3_offset=z3_offset)
    else:
        return res


###################
# Simulation related code
###################


_rhs = functools.partial(rhs, phi_container=C)
z4d = functools.partial(z4d, phi_container=C)
z2_fnc = functools.partial(z2_fnc, phi_container=C)


def _neg_rhs(*args):
    return -_rhs(*args)


########################
# Maneuver specific code
########################

############################################################################

###################
# Maneuver D
###################


def maneuverD(z_end):
    """
    performs maneuver D (1 or 2) by performing maneuver A in backward time
    with appropriately changed conditions:
    """

    # Adapt z3 for maneuver A

    z_start = np.array(z_end) * 1
    z_start[2] *= -1
    z_start[2] %= 2 * pi

    resA = maneuverA(z_start)

    resD = ManeuverResult(phi=resA.phi)

    # manually add the endpoint (=start point for maneuver A),
    # because maneuverA started slightly off for numerical reasons
    resD.add( tt=np.hstack( (resA.tt, resA.tt[-1] + dt) ) )
    resD.add( zz=np.vstack( (resA.zz[::-1, :], z_start) ) )

    resD.zz[:, 2] *= -1
    resD.zz[:, 2] %= 2 * pi  # TODO!!: might cause discontinuities (? )

    resD.add(z2=resD.zz[:, 1], z3=resD.zz[:, 2], z4=resD.zz[:, -1])

    # special treatment of z1:
    # determine where it has to start, such that it ends at the intended value
    z1_raw = np.cumsum(resD.z2) * dt
    Z1 = z_end[0] + z1_raw - z1_raw[-1]
    resD.zz[:, 0] = Z1  # overwrite it on two places
    resD.add(z1=Z1)
    resD.add(acc=-resA.acc[::-1], z4_park=resA.z4_park)

    resD.set_name(resA.maneuver_name.replace("A", "D"))
    return resD


def maneuverD1(z_end):
    """
    Warning: This function is deprecated and only has reference purpose.
    Use maneuverD instead

    performs maneuver D1 in backward time.

    The following quantities are determined:

    z3_star
    z4_park
    Delta_z1_D
    acc_D (the input trajectory for the last maneuver)
    """

    z1_cross, z2_cross, z3_cross, z4_cross = z_end

    if pi / 2 < z3_cross < pi:
        z2_sign = 1
    elif pi < z3_cross < 1.5 * pi:
        z2_sign = -1
    else:
        raise ValueError("unexpected z3_value")

    z4_sign = z2_sign  # first and third quadrant

    #TODO: more consistency-checks (asserts)

    assert z4_cross == 0

    # set the function for the switching line
    C.phi = ssf_lib.SlidingSurface(z4_star=0,
                                   z4_cross=z4_park * z4_sign, beta=beta,
                                   mu1=mu, mu2=mu, res_sign=z2_sign)

    # initial value for the reduced dynamics
    z_cross = [z3_cross, z4_cross + z4_offset * z4_sign]
    tt = np.arange(0, 20, dt)
    TT, ZZ = zero_crossing_simulation(_neg_rhs, z2_fnc, z_cross, tt)

    Z3, Z4 = ZZ[::-1, :].T  # reverse time direction
    z3_star = Z3[0]
    Z2 = C.phi(Z4)

    ZZtmp = ZZ * 1
    ZZtmp[:, 0] *= 1

    #TODO!!: Verstehen, warum hier komische Werte rauskommen
    acc = -C.phi.deriv_fnc(Z4) * z4d(ZZtmp) / 40

    # cope with numerical errors
    # insert the equlibrium manually
    assert abs(Z2[-1]) < 2e-2
    assert abs(Z4[-1]) < eps

    Z3 = np.hstack( (Z3, [Z3[-1]]) )
    Z4 = np.hstack( (Z4, [0]) )
    Z2 = np.hstack( (Z2, [0]) )
    acc = np.hstack( (acc[::-1], [0]) )
    ZZ = np.vstack( (ZZ, [Z3[-1], Z4[-1]]) )

    TT = np.hstack( (TT, TT[-1] + dt) )

    Z1 = np.cumsum(Z2) * dt
    Z1 += z1_cross - Z1[-1]
    assert Z1[-1] == z1_cross

    Delta_z1 = Z1[-1] - Z1[0]

    # overall state:
    zz = np.vstack((Z1, Z2, Z3, Z4)).T

    res = ManeuverResult(z3_star=z3_star, Delta_z1=Delta_z1, acc=acc)
    res.add(z4_park=z4_park * z4_sign)

    res.add(phi=C.phi)
    res.add(tt=TT, zz=zz, z1=Z1, z2=Z2, z3=Z3, z4=Z4)

    res.add(delta_z1_D=Z1[-1] - Z1[0])

    return res


###################
# Maneuver A
###################

def maneuverA(z_start):
    """
    general function for maneuverA. depending on the initial state it
    decides which submaneuver is to perform
    """

    z1_star, z2_star, z3_star, z4_star = z_start

    assert z4_star == z2_star == 0

    if pi / 2 < z3_star < 3 * pi / 2:
        return _maneuverA1(z_start)
    elif z3_star < pi / 2 or z3_star > 3 * pi / 2:
        return _maneuverA2(z_start)
    else:
        err = "invalid z3_star/pi: %f for maneuverA" % (z3_star / pi)
        raise ValueError(err)


def _maneuverA1(z_start):
    """
    performing maneuverA1 in forward time

    """

    z1_star, z2_star, z3_star, z4_star = z_start

    assert z4_star == z2_star == 0

    if pi / 2 < z3_star < pi:
        z2_sign = -1
    elif pi < z3_star < 1.5 * pi:
        z2_sign = 1
    else:
        raise ValueError( "invalid z3_star/pi: %f for A1" % (z3_star / pi) )

    z4_sign = z2_sign  # first and third quadrant

    C.phi = ssf_lib.SlidingSurface(z4_star=0,
                                   z4_cross=z4_park * z4_sign, beta=beta,
                                   mu1=mu, mu2=mu, res_sign=z2_sign)

    Z_star = [z3_star, z4_star + z4_offset * z4_sign]
    tt = np.arange(0, 20, dt)
    TT, ZZ = zero_crossing_simulation(_rhs, z2_fnc, Z_star, tt)

    Z3, Z4 = ZZ.T
    Z2 = C.phi(Z4)

    Z1 = np.cumsum(Z2) * dt + z1_star

    acc = C.phi.deriv_fnc(Z4) * z4d(ZZ)

    res = ManeuverResult(acc=acc)
    res.set_name("A1")
    res.add(phi=C.phi)
    res.add(z4_park=z4_park * z4_sign)

    # overall state:
    zz = np.vstack((Z1, Z2, Z3, Z4)).T

    res.add(tt=TT, zz=zz, z1=Z1, z2=Z2, z3=Z3, z4=Z4)

    return res


def _maneuverA2(z_start):
    """
    performing maneuver A2 in forward time
    """

    z1_star, z2_star, z3_star, z4_star = z_start

    if 0 < z3_star < pi/2:
        z4_park1 = z4_park
        z4_park2 = -z4_park
        z2_sign = -1
        z4_sign = 1
    elif 3*pi/2 < z3_star < 2*pi:
        z4_park1 = -z4_park
        z4_park2 = z4_park
        z2_sign = 1
        z4_sign = -1
    else:
        raise ValueError( "invalid z3_star/pi: %f for A2" % (z3_star / pi) )

    # first part of the maneuver
    C.phi = ssf_lib.SlidingSurface(z4_star=0, z4_cross=z4_park1,
                                   beta=beta, mu1=mu, mu2=mu, res_sign=z2_sign)

    z0 = r_[z3_star, z4_star + z4_offset * z4_sign]
    tt = np.arange(0, 20, dt)
    TT, ZZ = zero_crossing_simulation(_rhs, z4d, z0, tt)
    Z3, Z4 = ZZ.T
    Z2 = C.phi(Z4)
    Z1 = np.cumsum(Z2) * dt
    zz = np.vstack((Z1, Z2, Z3, Z4)).T

    ACC = C.phi.deriv_fnc(Z4) * z4d(ZZ)

    res1 = ManeuverResult(tt=TT, zz=zz, z1=Z1, z2=Z2, z3=Z3, z4=Z4)
    res1.add(acc=ACC, phi=C.phi)

    # second part of the maneuver:
    min_z2_val = z2_sign * res1.z2[-1]  # min. abs val of first branch
    # the sign must be included
    C.phi = ssf_lib.SlidingSurface(z4_star=0, z4_cross=z4_park2,
                                   beta=beta, mu1=mu, mu2=mu, res_sign=z2_sign,
                                   minval1=min_z2_val)

    z0 = r_[res1.z3[-1], res1.z4[-1]]
    TT, ZZ = zero_crossing_simulation(_rhs, z4d, z0, tt)
    Z3, Z4 = ZZ.T
    Z2 = C.phi(Z4)
    Z1 = np.cumsum(Z2) * dt + res1.z1[-1]
    zz = np.vstack((Z1, Z2, Z3, Z4)).T

    ACC = C.phi.deriv_fnc(Z4) * z4d(ZZ)

    res2 = ManeuverResult(tt=TT, zz=zz, z1=Z1, z2=Z2, z3=Z3, z4=Z4)
    res2.add(acc=ACC, phi=C.phi)

    res = join_containers(res1, res2)
    res.add(phi1=res1.phi, phi2=res2.phi)
    res.add(z4_park=z4_park2)
    res.set_name("A2")

    return res

############################################################################
# Maneuver B
############################################################################

# Special values for z3_B_star such that z3 at the end is the same
# (for both the first and third quadrant)
z3_B_start_Q1 = 0.70765 * pi
z3_B_start_Q3 = (2 - 0.70765) * pi


def _do_maneuverB(z_start):
    """
    perform maneuver B at a given initial state
    """

    z1, z2, z3, z4 = z_start

    assert z2 == 0
    assert abs(abs(z4) - z4_park) < eps

    # correct numerical variation
    z4 = np.sign(z4) * z4_park

    z4_park1 = z4

    if z4_park1 > 0:
        # breaking @ Q1, accelerating @ Q3
        z3_B_star = z3_B_start_Q1
    else:
        # breaking @ Q3, accelerating @ Q1
        z3_B_star = z3_B_start_Q3

    resD = maneuverD([z1, z2, z3_B_star, 0])
    resA = maneuverA(resD.zz[-1, :])

    res = join_containers(resD, resA)
    res.add(phiA=resA.phi, phiD=resD.phi)

    return res


def maneuverB(resA, resD):
    """
    this function decides whether or not maneuver B is neccessary and
    performs it if so.
    """

    if resA.z4_park == resD.z4_park:

        # insert some pseudo data, such that the data structure is not empty
        arr = np.array([0])
        res = ManeuverResult(acc=arr * 1.0)
        res.add(tt=arr * 1.0, zz=resA.zz[-1:, :] * 1.0)
        res.add(z4_park=resA.zz[-1, 3] * 1.0)
        res.done = False

    else:
        # actually perform maneuver B
        res = _do_maneuverB(resA.zz[-1,:])

    res.label_borders = [0]
    res.label_chain = []
    res.set_name("B")
    res = _maneuverB_phi_handling(res)
    return res


def _maneuverB_phi_handling(res):
    """
    combine the sliding surface data from B.D1 and B.A1 to one data set
    """
    assert res.maneuver_name == "B"
    PL = res.phi_list
    assert len(PL) == 2
    assert PL[0]['label'] == 'D1'
    assert PL[1]['label'] == 'A1'

    # unpack the tuples
    phi1_z2, phi1_z4 = PL[0]['phi']
    phi2_z2, phi2_z4 = PL[1]['phi']

    # z4 is independent (-> linspace)
    # z2 is the function value
    # z2_new should be the max w.r.t. absolute value

    z2_new = absmax(phi1_z2, phi2_z2)

    d_new = {'label': 'B', 'phi': (z2_new, phi1_z4), 'phi1': None, 'phi2': None}

    res.phi_list = [d_new]

    return res


############################################################################
# Maneuver C
############################################################################

def maneuverC(resB, resD):
    """
    general function of maneuverC:

    takes the results of maneuversB and D and performs C such that the desired
    preconditions for D are fullfilled
    """

    z_cross_B = resB.zz[-1, :]
    z_cross = resD.zz[-1, :]
    Delta_z1_D = resD.z1[-1] - resD.z1[0]

    z1_now = z_cross_B[0]
    z1_des = z_cross[0]

    # Determine the maneuver parameters
    Delta_z1_C = z1_des - Delta_z1_D - z1_now  # z1-displacement for maneuverC

    z1, z2, z3, z4 = z_cross_B
    z3_star_C, dz1_rest = \
            man_C_aux.get_man_C_exec_parameters(z3, z4, Delta_z1_C)

    # now perform the maneuver (first instance)
    resC1 = _do_maneuverC(z3_star_C, z4)
    resC1.set_name("C1")

    if not dz1_rest == 0:
        # second instance is necessary
        z3_new = resC1.z3[-1]
        z4_new = resC1.z4[-1]

        z3_star_C_new, dz1_rest_new = \
        man_C_aux.get_man_C_exec_parameters(z3_new, z4_new, dz1_rest)

        resC2 = _do_maneuverC(z3_star_C_new, z4_new)
        resC2.set_name("C2")

        res = join_containers_with_pr(resC1, resC2)

    else:
        res = resC1

    return res


def _do_maneuverC(z3_starC, z4):
    """
    executes maneuverC
    """

    assert abs(abs(z4) - z4_park) < eps

    # local (including sign):

    z4_park1 = z4

    z4_sign = np.sign(z4)
    if z4 > 0:
        z2_sign = -1
    else:
        z2_sign = 1

    # Determine the sign of the drift right next to the pariking position
    drift_sign = np.sign( man_C_aux.z4_drift( z2=1e-3 * z2_sign,
                                              z3=z3_starC, z4=z4) )

    # maximum value (to respect constraints on abs(z2))
    mv = 20  # actually this will never be reached

    # up and down are meant w.r.t absolute value here:

    phi_down = ssf_lib.SlidingSurface(z4_star=0, z4_cross=z4_park1,
                                      beta=beta, mu1=mu, mu2=mu,
                                      res_sign=z2_sign, singleEndBranch=True,
                                      maxval1=mv, maxval2=mv)

    phi_up = ssf_lib.SlidingSurface(z4_star=z4_park1 * 2,
                                    z4_cross=z4_park1,
                                    beta=beta, mu1=mu, mu2=mu, res_sign=z2_sign,
                                    singleEndBranch=True, maxval1=mv,
                                    maxval2=mv)

    if drift_sign < 0:
        C.phi = phi_down
    else:
        C.phi = phi_up

    z0 = r_[z3_starC, z4_park1 + 1.0e-6*drift_sign]
    tt = np.arange(0, 20, dt)
    TT, ZZ = zero_crossing_simulation(_rhs, z2_fnc, z0, tt)

    Z3, Z4 = ZZ.T
    Z2 = C.phi(Z4)
    Z1 = np.cumsum(Z2) * dt

    # overall state:
    zz = np.vstack((Z1, Z2, Z3, Z4)).T

    ret = ManeuverResult(tt=TT, zz=zz, z1=Z1, z2=Z2, z3=Z3, z4=Z4)
    ret.add(acc=Z1 * 0, phi1=C.phi, phi2=None)

    return ret


def absmax(a, b):
    """
    Helper-Function. Takes two equally shaped array and returns an array of the
    same shape, whose entries are those with the respective maximum value
    """
    assert a.shape == b.shape and a.ndim == 1

    c = np.c_[a, b]
    j = np.argmax(np.abs(c), axis=1)
    i = np.arange(c.shape[0])
    res = c[i, j]
    assert res.shape == a.shape
    return res
