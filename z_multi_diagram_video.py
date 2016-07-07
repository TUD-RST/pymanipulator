# -*- coding: utf-8 -*-


import maneuver_library as ml
import numpy as np
from numpy import cos, pi, exp, r_, c_
from model_and_parameters import kappa

# IPython shell for interactive debugging:
#from IPython import embed as IPS

import pylab as pl


"""
Module for generating the frames of an integrated animation
(cartesian coordinates, time evolution, z2-z4-plane)


function succession: main-> video -> videoframe -> (frame[s]_*)
"""

def frame_CC(ax, x1, x3, alpha = 1.0):
    """
    creates one frame to plot the manipulator in cartesian coordinates
    """

    # positions of the joints
    alpha = str(alpha)
    J0= 0+0j
    J1= J0 + 1*exp(1j*x1)
    J2= J1+0.8*exp(1j*(x1+x3))
    ax.plot(r_[J0,].real, r_[J0,].imag, 'ks', color = alpha, ms = 15)
    ax.plot(r_[J0, J1].real, r_[J0, J1].imag, 'k-', color = alpha, lw=3)
    ax.plot(r_[J2, J1].real, r_[J2, J1].imag, 'ko-', color = alpha)
    ax.xticks= []
    ax.yticks= []
    ax.axis('equal')

    # zoom such that everything is visible

    xmin = np.min(r_[J0, J1, J2].real)
    xmax = np.max(r_[J0, J1, J2].real)

    ymin = np.min(r_[J0, J1, J2].imag)
    ymax = np.max(r_[J0, J1, J2].imag)


def frames_time_evolution(ax1, ax2, ax3, idx):
    """
    time evolution of some quantities
    """

    idx_0 = idx - 100
    idx_end = idx + 100

    i0 = max(0, idx_0)

    t0 = idx_0*dt
    t_end = idx_end*dt

    tt = man_all.tt[i0:idx]
    z1 = man_all.z1[i0:idx]
    z2 = man_all.z2[i0:idx]
    z3 = man_all.z3[i0:idx]
    z4 = man_all.z4[i0:idx]
    x4 = man_all.x4[i0:idx]

    ax1.plot(tt, z1/pi, 'b-')
    ax1.plot(tt, z3/pi, 'g-')
    ax1.hlines(np.arange(-6,17), -10, 100, 'k', alpha=0.3)
    ax1.axis([t0, t_end, ax1._ymin/pi, ax1._ymax/pi])

    ax1.text(-0.2, 0.35, r"$z_1/\pi$,",  color="blue", transform = ax1.transAxes,
             rotation=90, fontsize=14)
    ax1.text(-0.2, 0.55, r"$z_3/\pi$",  color="green", transform = ax1.transAxes,
             rotation=90, fontsize=14)

    ax2.plot(tt, z2, 'b-')
    ax2.plot(tt, x4, 'g-')
    ax2.text(-0.2, 0.39, r"$x_2$,",  color="blue", transform = ax2.transAxes,
             rotation=90, fontsize=14)
    ax2.text(-0.2, 0.64, r"$x_4$",  color="green", transform = ax2.transAxes,
             rotation=90, fontsize=14)
    ax2.axis([t0, t_end, ax2._ymin, ax2._ymax])

    ax3.plot(tt, z4, 'b-')
    ax3.set_xlabel("$t$", fontsize=14)
    ax3.text(-0.2, 0.5, r"$z_4$",  color="blue", transform = ax3.transAxes,
             rotation=90, fontsize=14)
    ax3.axis([t0, t_end, ax3._ymin, ax3._ymax])

    xticks = ax1.get_xticks()
    ax1.set_xticklabels([""]*len(xticks))
    ax2.set_xticklabels([""]*len(xticks))

    # !!data set specific
    ax2.set_yticks(np.arange(-2, 4))
    ax3.set_yticks(np.arange(-1, 3))


def frame_z2z4(ax, z2, z4, phi_dict):

    # flat working copy (original dict remains unchanged by pop):
    phi_dict = dict(phi_dict)
    phi_label = phi_dict.pop('label')
    items = phi_dict.items()
    keys, values = zip(*items)

    if not phi_label == "P":
        assert len(values) == 3 and values.count(None) == 2

        phi_tup = sorted(values)[-1]# ignore the 2 None's
        phi_z2, phi_z4 = phi_tup

        ax.plot(phi_z2, phi_z4, 'k--')
    ax.plot(z2, z4, 'ko')
    ax.set_xlabel("$z_2$", fontsize=13)
    ax.set_ylabel("$z_4$", fontsize=13)

    # apply zoom information
    ax.axis([ax._xmin, ax._xmax, ax._ymin, ax._ymax])
    print [ax._xmin, ax._xmax, ax._ymin, ax._ymax]


def videoframe(fname, t_akt, x1, x2, x3, x4, z4, idx, alpha = 0):
    frame_CC(axes.ax0, x1, x3, alpha)

    axes.ax0.text(1.92, -1.2, 't=%4.2f' %t_akt)

#    axes.ax0.annotate('$t=%4.2f$' %t_akt, xy=(1.9, -1.2),
#                                    xytext=None, arrowprops=None, fontsize=12)

    label_idx  = max(np.where(ma.label_borders <= t_akt)[0])
    label = ma.label_chain[label_idx]
    axes.ax0.text(1.64, -1.5, "maneuver: "+label)
#    axes.ax0.annotate(label, xy=(1.9, -1.5),
#                                    xytext=None, arrowprops=None)

    axes.ax0.axis([-2.5, 2.5]*2)

    # ----------------
    z2 = x2
    phi_dict = man_all.phi_list[label_idx]
    frame_z2z4(axes.ax6, z2, z4, phi_dict)
    # ----------------

    # time evolution of some quantities:
    frames_time_evolution(axes.ax1, axes.ax2,axes.ax3, idx)


    pl.savefig(fname, quality=95)

    for ax in axes.axlist:
        pass
        ax.cla()


def video(fps, timezoom=1, K=25):
    """
    creates images for video generation
    K:  number of constant pics at beginning and end
    """

    t, state = state_from_container(man_all)

    x1, x2, x3, z4 = state.T

    x4 = z4 - (1 + kappa * cos(x3)) * x2

    t_end = t[-1]
    L = len(t)

    # determine indices
    Npic = t_end*fps*timezoom
    Npic_total = int(Npic+2*K)  # include beginning and end

    for pic_idx in range(0, Npic_total):

        i=int((pic_idx-K)*1./Npic * L)
        print "i=", i
        if i < 0:
            i = 0
        if i >= L:
            i = L-1


        fname = 'video/vid_%04i.jpg' % (pic_idx)
        videoframe(fname, t[i], x1[i], x2[i], x3[i], x4[i], z4[i], i)
        print "%i / %i" % (pic_idx, Npic_total-1)

        #break


def main():
    """
    main function of that module
    """

    man_all.x4 = man_all.z4 - ( 1+ kappa*cos(man_all.z3) )*man_all.z2


    pl.rcParams['text.usetex'] = True
    pl.rcParams['text.usetex'] = True
    pl.rc('text.latex', preamble = r'\usepackage{color}')

    pl.rcParams['font.size'] = 10
    pl.rcParams['figure.subplot.bottom'] = 0.07
    pl.rcParams['figure.subplot.left'] = 0.07
    pl.rcParams['figure.subplot.top'] = 0.985
    pl.rcParams['figure.subplot.right'] = 0.7
    pl.rcParams['figure.subplot.hspace'] = 0.3
    pl.rcParams['legend.numpoints'] = 2

    dpi = 100
    w = 1024/dpi
    h = 768/dpi
    fig = pl.figure(1,figsize=tuple(r_[w, h]), dpi=dpi)
    axes.ax0 = pl.gca()


    # plt.plot([0,1,2], [0,0,1])
    # left, bottom, w, h

    x1 = 0.79
    w1 = .2
    h1 = .14
    h2 = .22

    bott0 = pl.rcParams['figure.subplot.bottom']
    bott = bott0 + .165
    dy = h1 + 0.01

    axes.ax1 = pl.axes([x1, bott+dy*3, w1, h1+1*dy])
    axes.ax2 = pl.axes([x1, bott+dy*2, w1, h1])
    axes.ax3 = pl.axes([x1, bott+dy*1, w1, h1])
    # axes.ax4 = pl.axes([x1, bott+dy*2, w1, h1])
    # axes.ax5 = pl.axes([x1, bott+dy*1, w1, h1])
    axes.ax6 = pl.axes([x1, bott0+dy*0, w1, h2])

    axes.axlist = [axes.ax0, axes.ax1, axes.ax2, axes.ax3, axes.ax6]

    # store the zoom information (this should be valid for all frames)
    axes.ax1._ymin = np.min(c_[man_all.z1, man_all.z3]) - .3
    axes.ax1._ymax = np.max(c_[man_all.z1, man_all.z3]) + .3

    axes.ax2._ymin = np.min(c_[man_all.z2, man_all.x4]) - .3
    axes.ax2._ymax = np.max(c_[man_all.z2, man_all.x4]) + .3

    axes.ax3._ymin = np.min(man_all.z4)-.3
    axes.ax3._ymax = np.max(man_all.z4)+.3

    axes.ax6._xmin = np.min(man_all.z2)-.1
    axes.ax6._xmax = np.max(man_all.z2)+.1
    axes.ax6._ymin = np.min(man_all.z4)-.5
    axes.ax6._ymax = np.max(man_all.z4)+.1

    video(fps = 25, K = 20)

    # now e.g. use mencoder "mf://*.jpg" -mf fps=25 -o animation_kk.avi -ovc lavc -lavcopts vcodec=mpeg2video:vbitrate=3800:vhq:keyint=250

    pl.show()


def state_from_container(C, t0=0, t1=None):
    """
    helper-function
    returns all state-components, corresponding to the time interval [t0,t1]
    """
    i0 = int(t0/dt)
    if t1 == None:
        i1 = None  # None-Index means "until the end"
    else:
        i1 = int(t1/dt)
    return C.tt[i0:i1], C.zz[i0:i1, :]

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

if __name__ == "__main__":

    man_all = ml.ManeuverResult()  # global object to store the maneuver data
    man_all.load("data/man_all.npz")
    axes = ml.Container()  # to globally store the various axes-objects
    mm = 1./25.4  # mm to inch

    dt = man_all.tt[1]-man_all.tt[0]

    ma = man_all
    pr1_start = ma.pr_start[0]
    pr1_end = ma.pr_end[0]

    main()
else:
    raise SystemError("This module is not intended to be imported")