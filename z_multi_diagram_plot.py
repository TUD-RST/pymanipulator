# -*- coding: utf-8 -*-


import maneuver_library as ml
import numpy as np
from numpy import pi, exp, r_
import pylab as pl

try:
    from image_script_tracking import savefig2
    # function that adds infos about the creating script and revision
    # to meta info block of the created figure
except ImportError:
    # replacement without special features
    def savefig2(fname):
        pl.savefig(fname)


"""
This script is used to generate the final big plot of the paper
"""

man_all = ml.ManeuverResult()
man_all.load("data/man_all.npz")


def state_from_container(C, t0, t1):
    """
    returns all state-components, corresponding to the time interval [t0,t1]
    """
    i0 = int(t0/dt)
    i1 = int(t1/dt)+1 # +1 because we want this instant to be included

    return C.tt[i0:i1], C.zz[i0:i1, :]


def plot_CC_frames(tt, x1, x3, T, dt):
    """
    Plots several cartesian coordinate frames in one picture
    (with different gray values)
    """
    d_idx = int(T / dt)
    index_values = r_[np.arange(0, len(tt), d_idx), -1]

    # light gray -> black
    alpha = np.linspace(0.8, 0, len(index_values))


    for k,i in enumerate(index_values):
        CCframe(x1[i], x3[i], xy =0, alpha = alpha[k])


    pl.tick_params(\
    axis='x',          # changes apply to the x-axis
    which='both',      # both major and minor ticks are affected
    top = 'off', bottom='off', labelbottom='off')     # ticks along the bottom edge are off

    pl.tick_params(\
    axis='y',          # changes apply to the x-axis
    which='both',      # both major and minor ticks are affected
    left='off', right = "off", labelleft='off')

    label_annotation(tt)


def apply_zoom(xy_offset = None):
    if xy_offset is None:
        dx, dy = 0, 0
    else:
        dx, dy = xy_offset

    pl.axis([-2.1+dx, 2.1+dx, -1.8+dy, 1.8+dy])


def CCframe(x1, x3, xy = 0, alpha = 1.0):
    """
    creates a cartesian coordinate frame
    """

    # Postitions of the joints
    alpha = str(alpha)
    J0= 0+0j + xy
    J1= 1*exp(1j*x1) + xy
    J2= J1+0.8*exp(1j*(x1+x3))
    pl.plot(r_[J0,].real, r_[J0,].imag, 'ks', color = alpha, ms = 8)
    pl.plot(r_[J0, J1].real, r_[J0, J1].imag, 'k-', color = alpha, lw=3)
    pl.plot(r_[J2, J1].real, r_[J2, J1].imag, 'ko-', color = alpha)
    pl.xticks= []
    pl.yticks= []
    pl.axis('equal')

    xmin = np.min(r_[J0, J1, J2].real)
    xmax = np.max(r_[J0, J1, J2].real)

    ymin = np.min(r_[J0, J1, J2].imag)
    ymax = np.max(r_[J0, J1, J2].imag)


def label_annotation(tt):
    """
    plots the name of a maneuver to the canvas according to the current
    time instant given by tt[1]
    """
    t_akt = tt[1]
    label_idx  = max(np.where(man_all.label_borders <= t_akt)[0])
    label = man_all.label_chain[label_idx]
    pl.annotate(label, xy=(0.07, .8), color = "black",
                                    xycoords = "axes fraction",
                                    xytext=None, arrowprops=None)


def round_arrow(mxy, cxy, r, angle, length = .05, double=False, color=None):
    """
    mxy: middle point of the arc (not the center of the corresponding circle)
    cxy: (absolute) direction of center (not the center itself)
    r: radius
    angle: spread_angle in degree
    double: (defaul: False) arrow in both directions
    """
    v  = cxy-mxy
    v = r*v/np.linalg.norm(v)
    C = mxy + v

    # rotations via complex calculations
    C = C[0] + 1j*C[1]
    v = v[0] + 1j*v[1]

    angle *= np.pi/180
    phi = np.linspace(-angle/2., angle/2., 50)

    P = C - v*exp(1j*phi)

    if color is None:
        color = '0.0'
    pl.plot(P.real, P.imag, '-', color=color)
    dP = np.diff(P)[-1]

    dP /= np.abs(dP) # normed vector

    dP *= length

    pl.quiver([P.real[-1]],[P.imag[-1]],
              [dP.real],[dP.imag], color = color,
              angles='xy', headwidth = 4, pivot = 'tail', scale=1)

    if double:
        dP0 = -np.diff(P)[0]
        dP0*= length/np.abs(dP0) # normed vector
        pl.quiver([P.real[0]],[P.imag[0]],
                  [dP0.real],[dP0.imag], color = color,
                  angles='xy', headwidth = 4, pivot = 'tail', scale=1)

##################################################
# Global Part (Script)
##################################################

pl.rcParams['text.usetex'] = True
pl.rcParams['figure.subplot.bottom'] = .02
pl.rcParams['figure.subplot.left'] = .02
pl.rcParams['figure.subplot.top'] = .98
pl.rcParams['figure.subplot.right'] = .98

pl.rcParams['font.family'] = 'serif'


save_plots = True

mm = 1.0 / 25.4  # mm to inch
scale = 3
fs = [80 * mm * scale, 180 * mm * scale]  # figsize


# rule of thumb every 0.4 seconds one image:
T = 0.4
dt = man_all.tt[1] - man_all.tt[0]


fig = pl.figure(1,figsize=fs)
MB = man_all.label_borders  # Maneuver Borders

axis_z2z4 = [-2.7, 2.7, -2, 2]

# Maneuver A
tt, state = state_from_container(man_all, MB[0], MB[1])
z1, z2, z3, z4  = state.T

rows = 9
cols = 3

axz2 = ([-3, 3], [0, 0], '0.75')
axz4 = ([0, 0], [-3, 3], '0.75')

# keyword args for annotation
annkwargs = dict(color=".55", xycoords="axes fraction",
                 xytext=None, arrowprops=None)

# middle column color:
mcc = ".5"

# this if-statement serves to include and exclude single rows for image
# debuging and adaption
if 1:
    k = 0
    pl.subplot(rows, cols, k*cols+1)
    plot_CC_frames(tt, z1, z3, T, dt)
    round_arrow(np.r_[1, -1], np.r_[0,0], 3, -20)
    apply_zoom(xy_offset = [0,-.3])

    # middle column
    pl.subplot(rows, cols, k*cols+2)
    pl.plot(*axz2)
    pl.plot(*axz4)
    pl.plot(z2, z4, 'k')
    pl.plot(z2[0:1], z4[0:1], 'ko')
    pl.plot(z2[-1:], z4[-1:], 'ko')
    round_arrow(np.r_[-1.4, -.7], np.r_[-1,-.7], .4, 170, color = mcc)
    pl.annotate(s = "$\mathrm z_2$", xy=(0.9, .4), **annkwargs)
    pl.annotate(s = "$\mathrm z_4$", xy=(0.43, .86), **annkwargs)
    pl.annotate(s = "$\mathrm z_4^{\mathrm{pA}}$", xy=(0.54, .11), **annkwargs)
    pl.axis(axis_z2z4)

    # right column
    pl.subplot(rows, cols, k*cols+3)
    pl.plot(tt,z1/pi, 'k-', label=r"\z_1/\pi")
    pl.plot(tt, z3/pi, 'k--',  label=r"\z_3/\pi")
    pl.grid(True)



    # parking regime (A-P-B)
    k = 1
    pl.subplot(rows, cols, k*cols+1)

    tt, state = state_from_container(man_all, MB[1], MB[2])
    z1, z2, z3, z4  = state.T

    plot_CC_frames(tt, z1, z3, T, dt)
    round_arrow(np.r_[.1, -2.1], np.r_[0,0], 2, -40)
    apply_zoom(xy_offset=[0,-.9])

    # middle column
    pl.subplot(rows, cols, k*cols+2)
    pl.plot(*axz2)
    pl.plot(*axz4)
    pl.plot(z2[0:1], z4[0:1], 'ko')
    pl.annotate(s = "$\mathrm z_2$", xy=(0.9, .4), **annkwargs)
    pl.annotate(s = "$\mathrm z_4$", xy=(0.43, .86), **annkwargs)
    pl.annotate(s = "$\mathrm z_4^{\mathrm{pA}}$", xy=(0.54, .11), **annkwargs)
    pl.axis(axis_z2z4)


    pl.subplot(rows, cols, k*cols+3)
    pl.plot(tt,z1/pi, 'k-', label=r"\z_1/\pi")
    pl.plot(tt, z3/pi, 'k--',  label=r"\z_3/\pi")
    pl.grid(True)



    # Maneuver B


    tt, state = state_from_container(man_all, MB[2], MB[3])
    z1, z2, z3, z4  = state.T

    k = 2
    pl.subplot(rows, cols, k*cols+1)
    plot_CC_frames(tt, z1, z3, T, dt)
    round_arrow(np.r_[-1.2, -.8], np.r_[0,0], 2, -30)
    round_arrow(np.r_[-1.2, -.8], np.r_[0,0], 2, 30)
    apply_zoom(xy_offset = [0,-.2])

    # middle column
    pl.subplot(rows, cols,k*cols+2)
    pl.plot(*axz2)
    pl.plot(*axz4)
    pl.plot(z2, z4, 'k')
    pl.plot(z2[0:1], z4[0:1], 'ko')
    pl.plot(z2[-1:], z4[-1:], 'ko')
    round_arrow(np.r_[-1.4, -.7], np.r_[-1,-.7], .4, -170, color = mcc)
    pl.annotate(s = "$\mathrm z_2$", xy=(0.9, .4), **annkwargs)
    pl.annotate(s = "$\mathrm z_4$", xy=(0.4, .86), **annkwargs)
    pl.annotate(s = "$\mathrm z_4^{\mathrm{pA}}$", xy=(0.54, .11), **annkwargs)
    pl.annotate(s = "$\mathrm z_4^{\mathrm{pD}}$", xy=(0.54, .68), **annkwargs)
    pl.axis(axis_z2z4)
    pl.subplot(rows, cols,k*cols+3)
    pl.plot(tt, z1/pi, 'k-', label=r"\z_1/\pi")
    pl.plot(tt, z3/pi, 'k--',  label=r"\z_3/\pi")
    pl.grid(True)


#
#    pl.show()
#    raise SystemExit


    # parking regime (B-P-C)

    pr2_start = man_all.pr_start[1]
    pr2_end = man_all.pr_end[1]

    k = 3



    tt, state = state_from_container(man_all, MB[3], MB[4])
    z1, z2, z3, z4  = state.T

    pl.subplot(rows, cols, k*cols+1)
    plot_CC_frames(tt, z1, z3, T, dt)
    round_arrow(np.r_[-.65, -1.9], np.r_[0,-1], 2, 30)
    apply_zoom(xy_offset=[0,-1.1])

    pl.subplot(rows, cols, k*cols+2)

    # middle column
    pl.plot(*axz2)
    pl.plot(*axz4)
    pl.plot(z2[0:1], z4[0:1], 'ko')
    pl.annotate(s = "$\mathrm z_2$", xy=(0.9, .4), **annkwargs)
    pl.annotate(s = "$\mathrm z_4$", xy=(0.4, .86), **annkwargs)
    pl.annotate(s = "$\mathrm z_4^{\mathrm{pD}}$", xy=(0.54, .78), **annkwargs)
    pl.axis(axis_z2z4)


    pl.subplot(rows, cols, k*cols+3)
    pl.plot(tt,z1/pi, 'k-', label=r"\z_1/\pi")
    pl.plot(tt, z3/pi, 'k--',  label=r"\z_3/\pi")
    pl.grid(True)



# Maneuver C1

    pr3_start = man_all.pr_start[2]
    pr3_end = man_all.pr_end[2]

    k = 4

    tt, state = state_from_container(man_all, MB[4], MB[5])
    z1, z2, z3, z4  = state.T

    pl.subplot(rows, cols, k*cols+1)
    plot_CC_frames(tt, z1, z3, T, dt)
    round_arrow(np.r_[-1.5, -.2], np.r_[-.5,0], 1.3, -150)
#    apply_zoom()
    pl.axis([-3.3, 3.3, -1.65, 1.8])

    pl.subplot(rows, cols, k*cols+2)
    # middle column
    pl.plot(*axz2)
    pl.plot(*axz4)
    pl.plot(z2, z4, 'k')
    pl.plot(z2[0:1], z4[0:1], 'ko')
    pl.plot(z2[-1:], z4[-1:], 'ko')
    pl.annotate(s = "$\mathrm z_2$", xy=(0.9, .4), **annkwargs)
    pl.annotate(s = "$\mathrm z_4$", xy=(0.435, .91), **annkwargs)
    pl.annotate(s = "$\mathrm z_4^{\mathrm{pD}}$", xy=(0.54, .78), **annkwargs)
    round_arrow(np.r_[-1.5, 1.2], np.r_[-1.2,2.2], 4, 25, double=True,
                                                                color = mcc)

    pl.axis(axis_z2z4)

    pl.subplot(rows, cols, k*cols+3)
    pl.plot(tt,z1/pi, 'k-', label=r"\z_1/\pi")
    pl.plot(tt, z3/pi, 'k--',  label=r"\z_3/\pi")
    pl.grid(True)


# parking regime (C1 P C2)

    k = 5

    tt, state = state_from_container(man_all, MB[5], MB[6])
    z1, z2, z3, z4  = state.T

    pl.subplot(rows, cols, k*cols+1)
    plot_CC_frames(tt, z1, z3, T, dt)
    round_arrow(np.r_[-.3, 2.1], np.r_[0,1], 1.7, 50)
    apply_zoom(xy_offset=[0,.9])

    # middle column
    pl.subplot(rows, cols, k*cols+2)
    pl.plot(*axz2)
    pl.plot(*axz4)
    pl.annotate(s = "$\mathrm z_2$", xy=(0.9, .4), **annkwargs)
    pl.annotate(s = "$\mathrm z_4$", xy=(0.435, .91), **annkwargs)
    pl.annotate(s = "$\mathrm z_4^{\mathrm{pD}}$", xy=(0.54, .78), **annkwargs)
    pl.plot(z2, z4, 'k')
    pl.plot(z2[0:1], z4[0:1], 'ko')
    pl.plot(z2[-1:], z4[-1:], 'ko')

    pl.axis(axis_z2z4)

    pl.subplot(rows, cols, k*cols+3)
    pl.plot(tt,z1/pi, 'k-', label=r"\z_1/\pi")
    pl.plot(tt, z3/pi, 'k--',  label=r"\z_3/\pi")

    # apply semi-manual zoom
    axis_z1z3 = pl.axis()
    pl.axis(axis_z1z3[:2]+tuple(r_[axis_z1z3[2:]]*1.2) )
    pl.grid(True)


    # Maneuver (C2)
    k = 6

    tt, state = state_from_container(man_all, MB[6], MB[7])
    z1, z2, z3, z4  = state.T

    pl.subplot(rows, cols, k*cols+1)
    plot_CC_frames(tt, z1, z3, T, dt)
    round_arrow(np.r_[.1, 1.5], np.r_[0,0], 1.7, -40)
    #round_arrow(np.r_[-1.5, -1], np.r_[-1.5,-0], .4, 270) # 2nd joint
    apply_zoom(xy_offset=[0,.3])

    # middle column
    pl.subplot(rows, cols, k*cols+2)
    pl.plot(*axz2)
    pl.plot(*axz4)
    pl.annotate(s = "$\mathrm z_2$", xy=(0.9, .4), **annkwargs)
    pl.annotate(s = "$\mathrm z_4$", xy=(0.435, .91), **annkwargs)
    pl.annotate(s = "$\mathrm z_4^{\mathrm{pD}}$", xy=(0.54, .78), **annkwargs)
    pl.plot(z2, z4, 'k')
    pl.plot(z2[0:1], z4[0:1], 'ko')
    pl.plot(z2[-1:], z4[-1:], 'ko')
    round_arrow(np.r_[-.6, 1.2], np.r_[-.5,2.2], 4, 6, double=True, color = mcc)

    pl.axis(axis_z2z4)

    pl.subplot(rows, cols, k*cols+3)
    pl.plot(tt,z1/pi, 'k-', label=r"\z_1/\pi")
    pl.plot(tt, z3/pi, 'k--',  label=r"\z_3/\pi")
    pl.grid(True)

    # parking regime (C2 P D)

    k = 7

    tt, state = state_from_container(man_all, MB[7], MB[8])
    z1, z2, z3, z4  = state.T

    pl.subplot(rows, cols, k*cols+1)
    plot_CC_frames(tt, z1, z3, T, dt)
    round_arrow(np.r_[1.2, 1.8], np.r_[.48,.79], 1.7, 70)
    apply_zoom(xy_offset=[.3,.9])

    pl.subplot(rows, cols, k*cols+2)
    # middle column
    pl.subplot(rows, cols, k*cols+2)
    pl.plot(*axz2)
    pl.plot(*axz4)
    pl.annotate(s = "$\mathrm z_2$", xy=(0.9, .4), **annkwargs)
    pl.annotate(s = "$\mathrm z_4$", xy=(0.435, .91), **annkwargs)
    pl.annotate(s = "$\mathrm z_4^{\mathrm{pD}}$", xy=(0.54, .78), **annkwargs)
    pl.plot(z2, z4, 'k')
    pl.plot(z2[0:1], z4[0:1], 'ko')

    pl.axis(axis_z2z4)

    pl.subplot(rows, cols, k*cols+3)
    pl.plot(tt,z1/pi, 'k-', label=r"\z_1/\pi")
    pl.plot(tt, z3/pi, 'k--',  label=r"\z_3/\pi")
    pl.grid(True)


    # Maneuver D
    k = 8

    tt, state = state_from_container(man_all, MB[8], man_all.tt[-1])
    z1, z2, z3, z4  = state.T

    pl.subplot(rows, cols, k*cols+1)
    plot_CC_frames(tt, z1, z3, T, dt)
    round_arrow(np.r_[.0, 1.4], np.r_[0,0], 1.7, 50)
    apply_zoom(xy_offset=[0,.3])


    # middle column
    pl.subplot(rows, cols, k*cols+2)
    pl.plot(*axz2)
    pl.plot(*axz4)
    pl.annotate(s = "$\mathrm z_2$", xy=(0.9, .4), **annkwargs)
    pl.annotate(s = "$\mathrm z_4$", xy=(0.54, .91), **annkwargs)
    pl.annotate(s = "$\mathrm z_4^{\mathrm{pD}}$", xy=(0.41, .8), **annkwargs)
    pl.plot(z2, z4, 'k')
    pl.plot(z2[0:1], z4[0:1], 'ko')
    pl.plot(z2[-1:], z4[-1:], 'ko')

    round_arrow(np.r_[1.2, .8], np.r_[-1,.45], .4, -170, color = mcc)

    pl.axis(axis_z2z4)

    pl.subplot(rows, cols, k*cols+3)
    pl.plot(tt,z1/pi, 'k-', label=r"\z_1/\pi")
    pl.plot(tt, z3/pi, 'k--',  label=r"\z_3/\pi")
    pl.grid(True)


if save_plots:
    pl.figure(1)
    savefig2('images/total_plot.pdf')
    #pl.figure(2)

pl.show()

