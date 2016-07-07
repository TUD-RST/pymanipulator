# -*- coding: utf8 -*-

from numpy import pi
import maneuver_library as ml


"""
    Perform a complete equilibrium transition and save the result
"""

z4_park = 1.4


start_eq = [0, 0, 3 * pi / 4, 0]
end_eq = [pi * 0.8, 0, 3 * pi / 4, 0]


resA = ml.maneuverA(start_eq)
resD = ml.maneuverD(end_eq)
resB = ml.maneuverB(resA, resD)

resAB = ml.join_containers_with_pr(resA, resB)
resAB.debug_flag = True
resC = ml.maneuverC(resAB, resD)
res = ml.join_containers_with_pr(resAB, resC, resD)


res.save("data/man_all.npz")
print "done"
 