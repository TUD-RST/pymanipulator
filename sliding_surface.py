# -*- coding: utf8 -*-

from numpy import r_, vectorize
import numpy as np


"""
Collections of objects which define sliding surfaces
"""


class AbstractSlidingSurface(object):

    def __init__(self, *args, **kwargs):
        raise TypeError("This class is not intend")

    # these two methods will be overridden in subclasses
    def fnc_nv(self):
        pass

    def deriv_fnc_nv(self):
        pass

    def pack(self):
        """
        makes serialization via pickle module possible by deleting the
        vectorized functions which are not serializable
        """
        del self.fnc        # =vectorize(self.fnc_nv1)
        del self.deriv_fnc  # =vectorize(self.deriv_fnc_nv)

    def unpack(self):
        """
        creates the vectorized functions
        intention: call this method after loading this object from serialization
        """
        self.fnc = vectorize(self.fnc_nv)
        self.deriv_fnc = vectorize(self.deriv_fnc_nv)

    def __call__(self, *args, **kwargs):
        """
        redirect calls of the object to the self.fnc method
        (which is the mathematical function of the sliding surface)
        """
        return self.fnc(*args, **kwargs)

    def plot(self):

        import pylab as pl
        y = np.linspace(-3, 3, 1000)
        x = self(y)
        pl.plot(x, y)
        pl.show()


class SlidingSurface(AbstractSlidingSurface):
    """
    Objects of this class represent a function that defines a sliding surface
    in the z2-z4-plane. This function might consist out of two branches
    and additionally respects some limits on the function value

    see self.fnc_nv for details (and formulas)
    """

    def __init__(self, **kwargs):
        # if singleBranch == True, just consider first branch of the function
        self.singleBranch = False

        # if singleEndBranch == True, just consider second branch of the func
        self.singleEndBranch = False
        self.verbose = False

        # maximum and minimum z2-values of the sliding surface
        self.maxval1 = 100.
        self.maxval2 = 100.
        self.minval1 = 0.
        self.minval2 = 0.

        # sign of the resulting function value (-1 means -> left )
        self.res_sign = 1
        self.update(**kwargs)

        self.unpack()

    def update(self, **kwargs):
        """
        calculates some auxiliary quantities and stores them in the internal
        dict
        """
        self.__dict__.update(kwargs)
        assert not (self.singleBranch and self.singleEndBranch)
        big = 1e5

        # L_mid: "middle point of the sliding surface"
        if self.singleBranch:
            self.L_mid = big
        elif self.singleEndBranch:
            self.L_mid = -big
        else:
            self.L_mid = self.mu2 * 1.0 / (self.mu2 + self.mu1)

    def Q1(self, L, mu1):
        """
        calculates auxiliary function values for the sliding surface
        by using mu1 (first branch)
        """
        # must return a scalar value
        ex = 1. / self.beta
        res = ( mu1 * self.beta * L ) ** ex
        return res

    def Q2(self, L, mu2):
        """
        calculates auxiliary function values for the sliding surface
        by using mu2 (second branch)
        """
        # must return a scalar value
        ex = 1. / self.beta
        res = (mu2 * self.beta * (1 - L)) ** ex
        return res

    def fnc_nv(self, z4):
        """
        function which defines the sliding surface in the sense of
        z2 = fnc(z4)
        """
        s = self
        # calculate a normalized argument \in [0, 1]
        L = (z4 - s.z4_star) / (s.z4_cross - s.z4_star)

        # dbg:
        if s.verbose:
            print "L=", L

        # constant values if agrument is outside of domain
        if L < 0 and not self.singleEndBranch:
            return s.minval1 * s.res_sign
        if L > 1 and not self.singleBranch:
            return s.minval2 * s.res_sign

        # determine on which branch we are
        if L < s.L_mid:
            res = s.Q1(L, s.mu1)
            res = np.clip(res, s.minval1, s.maxval1)

        else:
            res = s.Q2(L, s.mu2)
            res = np.clip(res, s.minval2, s.maxval2)

        return res * s.res_sign

    def deriv_fnc_nv(self, z4):
        """
        explicit implementation of the derivative of the sliding surface
        function:
            z2' (z4)
        """
        s = self
        L = (z4 - s.z4_star) / (s.z4_cross - s.z4_star)
        dL_dz4 = 1 / (s.z4_cross - s.z4_star)

        ex_alt = 1. / self.beta
        ex = ex_alt - 1


        fncval = self.fnc_nv(z4) * s.res_sign  # compensate the sign

        #if L <= 0 or  L >= 1:
        if fncval == s.minval1 or fncval == 0.:
            # constant value -> no slope
            return 0.

        if L < self.L_mid:
            # simple but not efficient:
            if fncval == self.maxval1 or fncval == self.minval1:
                return 0  # constant value -> no slope
            res = (self.mu1 * self.beta) ** ex_alt * ex_alt * L ** ex
        else:
            if fncval == self.maxval2 or fncval == self.minval2:
                return 0
            res = -(self.mu2 * self.beta) ** ex_alt * ex_alt * (1 - L) **ex

        return dL_dz4 * res * s.res_sign


def main():
    """
    main function of the module
    serves to show some realizations of the sliding surface
    """
    import pylab as pl

    z4 = r_[-3.:3:10000j]

    # SlidingSurface objects are named em ("EquationMaker") for historical
    # reasons
    em0 = SlidingSurface(z4_star=-1.6, z4_cross=-.6, mu2=5.0, beta=2.5,
                         vz=-1., res_sign=1., minval2=0.7, singleEndBranch=True)

    direction = 1
    z4_park = 1.4
    z4_cross=z4_park * direction
    z4_star_gf = -1.5 * z4_park * direction
    mu = 15

    em2 = SlidingSurface(z4_star=z4_star_gf, z4_cross=z4_cross,
                         beta=2.5, mu1=mu, mu2=mu, res_sign=direction)

    mu = 72
    em3 = SlidingSurface(z4_star=z4_star_gf, z4_cross=z4_cross,
                         beta=2.5, mu1=mu, mu2=mu, minval1=6.)

    em4 = SlidingSurface(z4_star=z4_star_gf, z4_cross=z4_cross,
                         beta=2.7, mu1=mu, mu2=mu, res_sign=direction,
                         minval1=5.)

    def plot_em(em):
        x2 = em(z4)
        pl.plot(x2, z4)

    plot_em(em0)
    plot_em(em2)
    plot_em(em3)
    plot_em(em4)

    pl.show()
    return


if __name__ == '__main__':
    main()
