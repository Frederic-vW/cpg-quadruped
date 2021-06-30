#!/usr/bin/python3
# -*- coding: utf-8 -*-
# Central Pattern Generator model for teaching
# FvW 06/2020

import os
import sys
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import matplotlib.animation as animation
from matplotlib.animation import FuncAnimation, PillowWriter
from scipy.interpolate import interp1d
from scipy.signal import morlet2, cwt
#from scipy.signal import correlate #, medfilt
#from scipy.ndimage import median_filter
#import cv2

fs_ = 20


class Quad(object):
    """
    Central Pattern Generator for Quadruped locomotion
    based on the Morris-Lecar model

    References
    ----------

    .. [1] Buono L and Golubitsky M, "Models of central pattern generators for
           quadruped locomotion I. Primary Gaits". J. Math. Biol. 42,
           291--326 (2001)

    .. [2] Buono L and Golubitsky M, "Models of central pattern generators for
           quadruped locomotion II. Secondary Gaits". J. Math. Biol. 42,
           327--346 (2001)
    """

    def __init__(self, mode, T=100, dt=0.01):
        print("\n[+] Quadruped locomotion (Morris-Lecar):")
        self.mode = mode
        self.T = T
        self.dt = dt
        self.n_t = int(T/dt)
        print(f"[+] n = {self.n_t:d} time steps")
        self.parameters()
        #self.run()
        #self.plot()
        #self.animate()

    def connectivity(self):
        """
        indices here: 0..7
        original paper: 1..8
        Returns:
            C: integer array (N,2)
               C[i,0]: neighbour `(i-2) mod N` of neuron `i`
               C[i,1]: neighbour `(i+(-1)^i) mod N` of neuron `i`

        explicitly:
        i | (i-2)%N | (i+eps_i)%N | (i+4)%N
        0      6           1          4
        1      7           0          5
        2      0           3          6
        3      1           2          7
        4      2           5          0
        5      3           4          1
        6      4           7          2
        7      5           6          3
        """
        N = 8
        C = np.zeros((8,2),dtype=np.int)
        for i in range(N):
            k = (i-2)%N
            ei = (-1)**i
            l = (i + ei)%N
            #m = (i + 4)%N
            #print(f"i = {i:d}, {k:d} > {i:d}, {l:d} > {i:d}")
            #print(f"i = {i:d}, {k:d}, {l:d}, {m:d}")
            C[i,0] = k
            C[i,1] = l
        #print(C)
        return C

    def parameters(self):
        """
        set parameters
        """
        # Morris-Lecar parameters
        self.gCa = 3
        self.gK = 1.8
        self.gL = 0.6
        self.VCa = 1.0
        self.VK = -0.8
        self.VL = -1.8
        self.I = 1.0

        # gait parameters
        # parameters: alpha, beta, gamma, delta
        # Table 12:
        # 'pace': [0.3, 0.3, -0.32, -0.32],
        # 'bound': [-0.32, -0.32, 0.3, 0.3],
        # Table 13:
        # 'pace': [0.2, 0.2, -0.2, -0.2],
        # 'bound': [-0.2, -0.2, 0.2, 0.2],

        if (self.mode == 'pronk'):
            self.alpha = 0.2
            self.beta = 0.2
            self.gamma = 0.2
            self.delta = 0.2
        elif (self.mode == 'pace'):
            self.alpha = 0.2
            self.beta = 0.2
            self.gamma = -0.2
            self.delta = -0.2
        elif (self.mode == 'bound'):
            self.alpha = -0.2
            self.beta = -0.2
            self.gamma = 0.2
            self.delta = 0.2
        elif (self.mode == 'trot'):
            self.alpha = -0.6
            self.beta = -0.6
            self.gamma = -0.6
            self.delta = -0.6
        elif (self.mode == 'jump'):
            self.alpha = 0.01
            self.beta = -0.01
            self.gamma = 0.2
            self.delta = 0.2
        elif (self.mode == 'walk'):
            self.alpha = 0.01
            self.beta = -0.01
            self.gamma = -1.2
            self.delta = -1.2
        elif (self.mode == 'canter'):
            self.alpha = 0.17
            self.beta = -0.2
            self.gamma = -0.9
            self.delta = -1
            self.gCa = 8
            self.x00 = 0.4
            self.y00 = 0.3
        elif (self.mode == 'runwalk'):
            self.alpha = -0.78
            self.beta = -0.56
            self.gamma = 0.12
            self.delta = -1.14
            self.gCa = 2
            self.x00 = -1.2147809
            self.y00 = -0.058746844
        elif (self.mode == 'doublebond'):
            self.alpha = -0.6
            self.beta = -0.77
            self.gamma = 0.3
            self.delta = 0.5
            self.gCa = 3
            self.x00 = -1
            self.y00 = 0.6
        else:
            print("ERROR: mode not defined")
            sys.exit()

        '''
        self.par = {
            'pronk': [0.2, 0.2, 0.2, 0.2],
            'pace': [0.2, 0.2, -0.2, -0.2],
            'bound': [-0.2, -0.2, 0.2, 0.2],
            'trot': [-0.6, -0.6, -0.6, -0.6],
            'jump': [0.01, -0.01, 0.2, 0.2],
            'walk': [0.01, -0.01, -1.2, -1.2],
        }
        '''
        #self.mode = ['pronk', 'pace', 'bound', 'trot', 'jump', 'walk'][5]
        # pronk: ok
        # pace: ok, but dependent on init. cond.
        # bound: ok, but dependent on init. cond.
        # trot: ok, but dependent on init. cond.
        # jump: ok
        # walk: ok

        ''' Paper 2: Secondary Gaits
        'canter': gCa=8, alpha=0.17, beta=-0.2, gamma=-0.9, delta=-1
        'running walk': gCa=2, alpha=-0.78, beta=-0.56, gamma=0.12, delta=-1.14
        'double bond': gCa=3, alpha=-0.6, beta=-0.77, gamma=0.3, delta=0.5
        '''

        '''
        mode = 'canter'
        gCa = 8
        alpha = 0.17
        beta = -0.2
        gamma = -0.9
        delta = -1
        X[0,0] = 0.4
        Y[0,0] = 0.3
        '''

        '''
        mode = 'running walk'
        gCa = 2
        alpha = -0.78
        beta = -0.56
        gamma = 0.12
        delta = -1.14
        X[0,0] = -1.2147809
        Y[0,0] = -0.058746844
        '''

        '''
        mode = 'double bond'
        gCa = 3
        alpha = -0.6
        beta = -0.77
        gamma = 0.3
        delta = 0.5
        X[0,0] = -1
        Y[0,0] = 0.6
        '''

    def run(self):
        N = 8 # number of neurons
        # auxiliary functions
        phi = 0.2
        v1 = 0.2
        v2 = 0.4
        v3 = 0.3
        v4 = 0.2
        m = lambda v: 0.5*( 1 + np.tanh((v-v1)/v2) )
        n = lambda v: 0.5*( 1 + np.tanh((v-v3)/v4) )
        tau = lambda v: np.cosh( (v-v3)/(2*v4) )

        print(f"[+] mode: {self.mode:s}")
        alpha = self.alpha
        beta = self.beta
        gamma = self.gamma
        delta = self.delta

        #n_t = 10000
        dt = self.dt
        n_t = self.n_t
        X = np.zeros((n_t,N))
        #Y = np.zeros((n_t,N))
        x = np.zeros(N)
        y = np.zeros(N)
        #x = np.random.randn(N)
        #y = np.random.randn(N)
        #X[0,:] = x
        #Y[0,:] = y
        dxdt = np.zeros(N)
        dydt = np.zeros(N)

        #''' random initial conditions
        for i in range(8):
            x[i] = np.random.randn()
            y[i] = np.random.randn()
            #X[0,i] = np.random.randn() # 0 # .31
            #Y[0,i] = np.random.randn() # -1.24 # 0.52
            #X[0,i] = -2*np.random.rand() # 0 # .31
            #Y[0,i] = -1*np.random.rand() # -1.24 # 0.52
        #'''

        # apply initial conditions if provided (secondary gaits)
        try:
            #X[0,0] = x00
            #Y[0,0] = y00
            x[0] = x00
            y[0] = y00
            x[1:] = 0.0
            y[1:] = 0.0
        except:
            pass
        X[0,:] = x
        #Y[0,:] = y

        # copy parameters to improve readability
        gCa = self.gCa
        VCa = self.VCa
        gK = self.gK
        VK = self.VK
        gL = self.gL
        VL = self.VL
        I = self.I

        # connectivity
        C = self.connectivity()
        n0 = int(5*n_t)
        for t in range(1,n0+n_t):
            for i in range(N):
                #v = X[t-1,i]
                #w = Y[t-1,i]
                v = x[i]
                w = y[i]
                # vector dxdt, dydt
                dxdt[i] = -gCa*m(v)*(v-VCa) -gL*(v-VL) -gK*w*(v-VK) + I
                dydt[i] = phi*tau(v)*(n(v)-w)
                # scalar dxdt, dydt
                #dxdt += (alpha*X[t-1,C[i,0]] + gamma*X[t-1,C[i,1]]) #+xi*x(i+4)
                #dydt += (beta*Y[t-1,C[i,0]] + delta*Y[t-1,C[i,1]]) #+eta*y(i+4)
                dxdt[i] += (alpha*x[C[i,0]] + gamma*x[C[i,1]]) #+xi*x(i+4)
                dydt[i] += (beta*y[C[i,0]] + delta*y[C[i,1]]) #+eta*y(i+4)
            x += (dxdt*dt)
            y += (dydt*dt)
            if (t >= n0):
                X[t-n0,:] = x
                #Y[t-n0,:] = Y[t-1,i] + dydt*dt

        # interpolate
        n_interp = int(n_t/10)
        t_ = np.arange(n_t)
        f_ip = interp1d(t_, X, axis=0, kind='linear')
        t_new = np.linspace(0, n_t-1, n_interp)
        X = f_ip(t_new)
        self.data = {}
        self.data['LH'] = X[:,0] # left hind
        self.data['RH'] = X[:,1] # right hind
        self.data['LF'] = X[:,2] # left fore
        self.data['RF'] = X[:,3] # right fore

    def plot(self):
        # 0: left hind leg
        # 1: right hind leg
        # 2: left fore leg
        # 3: right fore leg
        p_ann = {
            'xy' : (0.01,0.80),
            'xycoords' : 'axes fraction',
            'fontsize' : 22,
            'fontweight' : 'bold'
        }
        fig, ax = plt.subplots(4, 1, figsize=(16,8), sharex=True)
        legs = ['LF', 'RF', 'LH', 'RH']
        for i, leg in enumerate(legs):
            ax[i].plot(self.data[leg], '-k')
            ax[i].annotate(leg, **p_ann)
            ax[i].grid()
        ax[0].set_title(f"{self.mode:s}", fontsize=22)
        plt.tight_layout()
        plt.show()

    def animate0(self):
        """
        make movie
        """
        loc = {
            "LF" : (50,100),
            "RF" : (280,100),
            "LH" : (40,300),
            "RH" : (280,240),
        }
        # find local maxima in activation time courses
        peaks = {}
        for leg in self.data:
            peaks[leg] = locmax(self.data[leg])
        # load animal background image
        img = mpimg.imread('./img/quadruped.png')
        movname = f"cpg_quad_{self.mode:s}.mp4"
        im_list = []
        n_interp = len(self.data['LF'])
        print("[+] Accumulate image arrays...")
        for t in range(n_interp):
            if (t%10 == 0): print(f"\tt = {t:d}/{n_interp:d}\r", end="")
            fig1 = plt.figure() # figsize=(8,4)
            ax1 = plt.gca()
            #fig1, ax1 = plt.subplots() # figsize=(8,4)
            ax1.imshow(img)
            ax1.axis('off')
            plt.tight_layout()
            for leg in peaks:
                if t in peaks[leg]:
                    ax1.scatter(loc[leg][0], loc[leg][1], s=200, lw=0.5,
                                edgecolors='None', facecolors='r')
            fig1.canvas.draw()
            y = np.fromstring(fig1.canvas.tostring_rgb(), dtype=np.uint8, sep='')
            y = y.reshape(fig1.canvas.get_width_height()[::-1] + (3,))
            im_list.append(y.copy())
            plt.close()
            plt.cla()
            plt.clf()
        #print("\nim_list: ", len(im_list))
        print("")

        print("[+] Make animation...")
        fig, ax = plt.subplots()
        ax.axis('off')
        ims = []
        for t in range(n_interp):
            if (t%10 == 0): print(f"\tt = {t:d}/{n_interp:d}\r", end="")
            img_ = im_list[t]
            im = ax.imshow(img_)
            ims.append([im])
        print("")
        #print("ims: ", type(ims), len(ims), type(ims[0]))
        #print(ims[0])

        ani = animation.ArtistAnimation(fig, ims, interval=50, blit=True,
                                        repeat_delay=1000)
        # [1]
        print("saving animation...")
        #ani.save(movname)
        # [2]
        writer = animation.FFMpegWriter(fps=15, metadata=dict(artist='Me'), bitrate=1800)
        ani.save(movname, writer=writer)
        print("...animation saved.")
        plt.show()

    '''
    def animate1(self):
        """
        make movie: use cv2
        """
        loc = {
            "LF" : (20,100),
            "RF" : (260,100),
            "LH" : (20,330),
            "RH" : (280,250),
        }
        # find local maxima in activation time courses
        peaks = {}
        for leg in self.data:
            peaks[leg] = locmax(self.data[leg])
        # load animal background image
        img = mpimg.imread('./img/quadruped.png')
        movname = f"cpg_quad2_{self.mode:s}.mp4"
        #os.chdir("/home/frederic/data/graphs/tmp/")
        #fnames = []

        # get first image
        fig = plt.figure() # figsize=(8,4)
        plt.imshow(img)
        #plt.tight_layout()
        #plt.show()
        fig.canvas.draw()
        y = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep='')
        y = y.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        #print(type(y), y.shape, y.dtype) #, y[:10])
        nx, ny = y.shape[1], y.shape[0]
        #nt = 100

        framerate = 30
        out = cv2.VideoWriter(movname, cv2.VideoWriter_fourcc(*'mp4v'), \
                              framerate, (nx,ny))

        n_interp = len(self.data['LF'])
        for t in range(n_interp):
            if (t%10 == 0): print(f"\tt = {t:d}/{n_interp:d}\r", end="")
            #fname = "tmp_{:04d}.png".format(t)
            #fnames.append(fname)
            plt.close()
            plt.cla()
            plt.clf()
            fig = plt.figure()
            plt.imshow(img)
            for leg in peaks:
                if t in peaks[leg]:
                    #plt.annotate(leg, xy=loc[leg], xycoords='data', fontsize=28)
                    plt.plot(loc[leg][0], loc[leg][1], 'or', ms=20)
            plt.axis('off')
            fig.canvas.draw()
            y = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep='')
            y = y.reshape(fig.canvas.get_width_height()[::-1] + (3,))
            out.write(y)
            #plt.savefig(fname, dpi=90, bbox_inches="tight")
            #plt.show()
            #plt.close(); plt.cla(); plt.clf()
        out.release()
        print("")
        #cmd = videoCmd(movname, frate=25, width=128, height=128)
        #os.system(cmd)
        #for fname in fnames: os.remove(fname)
    '''

    def animate2(self):
        """
        make movie: save figures as png, system call to ffmpeg
        """
        loc = {
            "LF" : (20,100),
            "RF" : (260,100),
            "LH" : (20,330),
            "RH" : (280,250),
        }
        # find local maxima in activation time courses
        peaks = {}
        for leg in self.data:
            peaks[leg] = locmax(self.data[leg])
        # load animal background image
        img = mpimg.imread('./img/quadruped.png')
        movname = f"cpg_quad2_{self.mode:s}"
        os.chdir("/home/frederic/data/graphs/tmp/")
        fnames = []
        n_interp = len(self.data['LF'])
        for t in range(n_interp):
            if (t%10 == 0): print(f"\tt = {t:d}/{n_interp:d}\r", end="")
            fname = "tmp_{:04d}.png".format(t)
            fnames.append(fname)
            plt.figure()
            plt.imshow(img)
            for leg in peaks:
                if t in peaks[leg]:
                    plt.annotate(leg, xy=loc[leg], xycoords='data', fontsize=28)
            plt.axis('off')
            plt.savefig(fname, dpi=90, bbox_inches="tight")
            #plt.show()
            plt.close(); plt.cla(); plt.clf()
        print("")
        cmd = videoCmd(movname, frate=25, width=128, height=128)
        os.system(cmd)
        for fname in fnames: os.remove(fname)

    def animate3(self):
        """
        make movie
        """
        movname = f"cpg_quad_{self.mode:s}.mp4"
        print(f"[+] Animate data as movie: {movname:s}")
        loc = {
            "LF" : (50,100),
            "RF" : (280,100),
            "LH" : (40,300),
            "RH" : (280,240),
        }
        # find local maxima in activation time courses
        peaks = {}
        for leg in self.data:
            peaks[leg] = locmax(self.data[leg])
        # load animal background image
        img = mpimg.imread('./img/quadruped.png')

        fig, ax = plt.subplots()
        ax.imshow(img)
        ax.axis('off')
        s = 200
        lf = ax.scatter(loc['LF'][0], loc['LF'][1], s=s, lw=0.5,
                        edgecolors='None', facecolors='None')
        rf = ax.scatter(loc['RF'][0], loc['RF'][1], s=s, lw=0.5,
                        edgecolors='None', facecolors='None')
        lh = ax.scatter(loc['LH'][0], loc['LH'][1], s=s, lw=0.5,
                        edgecolors='None', facecolors='None')
        rh = ax.scatter(loc['RH'][0], loc['RH'][1], s=s, lw=0.5,
                        edgecolors='None', facecolors='None')

        def update(i):
            if (i%10 == 0): print(f"\tt = {i:d}\r", end="")
            if i in peaks['LF']:
                lf.set_facecolors('r')
            else:
                lf.set_facecolors('none')
            if i in peaks['RF']:
                rf.set_facecolors('r')
            else:
                rf.set_facecolors('none')
            if i in peaks['LH']:
                lh.set_facecolors('r')
            else:
                lh.set_facecolors('none')
            if i in peaks['RH']:
                rh.set_facecolors('r')
            else:
                rh.set_facecolors('none')

        # make animation
        n_interp = len(self.data['LF'])
        ani = FuncAnimation(fig, update, interval=50, save_count=n_interp)
        ani.save(movname)
        plt.show()
        print("")


def bifurcation_diagram():
    """
    Create a bifurcation diagram for the single-cell Morris-Lecar model
    """
    Is = np.linspace(50, 250, 100)
    nI = len(Is)
    dt = 0.05
    T = 1000
    nt = int(T/dt)
    n0 = int(nt/2)
    x_hi = np.zeros(nI)
    x_lo = np.zeros(nI)
    time = dt*np.arange(nt)
    for i, i_ext in enumerate(Is):
        print(f"i = {i:d}/{nI:d}")
        I_ext = i_ext*np.ones(int(T/dt))
        params = {
            'T': T,
            'dt': dt,
            'n0': n0,
            'sd': 0.0001,
            'I_ext': I_ext,
            'doplot': False,
        }
        X, _ = ml1(**params)
        #X = median_filter(X, size=5)
        loc_mx = locmax(X)
        loc_mn = locmax(-X)
        x_hi[i] = np.mean(X[loc_mx])
        x_lo[i] = np.mean(X[loc_mn])
        '''
        plt.figure(figsize=(15,3))
        plt.plot(time, X, '-k')
        plt.plot(time[loc_mx], X[loc_mx], 'or', ms=8, alpha=0.3)
        plt.plot(time[loc_mn], X[loc_mn], 'ob', ms=8, alpha=0.3)
        plt.tight_layout()
        plt.show()
        yn = input("...")
        if (yn == 'n'): sys.exit()
        '''

    fig = plt.figure(figsize=(10,5))
    plt.plot(Is, x_lo, '-ok', lw=2)
    plt.plot(Is, x_hi, '-ok', lw=2)
    plt.xlabel("I", fontsize=16)
    plt.ylabel("x", fontsize=16)
    plt.tight_layout()
    plt.show()


def acov(x, lmax):
    """
    Autocovariance

    Parameters
    ----------
        x : 1D array of float
            input signal
        lmax : int
            maximum time lag

    Returns
    -------
        C : 1D array of float
            autocovariance coefficients
    """
    n = len(x)
    C = np.zeros(lmax)
    for k in range(lmax):
        C[k] = np.mean((x[0:n-k]-np.mean(x[0:n-k])) * (x[k:n]-np.mean(x[k:n])))
        #C[k] = np.mean(x[0:n-k]*x[k:n])
    return C


def ccov(x, y, lmax):
    """
    Cross-covariance

    Parameters
    ----------
        x : 1D array of float
            input signal 1
        y : 1D array of float
            input signal 2
        lmax : int
            maximum time lag

    Returns
    -------
        C : 1D array of float
            autocovariance coefficients
    """
    #n = len(x)
    C = np.zeros(2*lmax+1)
    xm = x-x.mean()
    ym = y-y.mean()
    lags = np.array([k for k in range(-lmax,lmax+1)])
    for i, k in enumerate(lags):
        yk = np.roll(ym,k)
        C[i] = np.mean(xm * yk)
        #C[k] = np.mean((x[0:n-k]-np.mean(x[0:n-k])) * (x[k:n]-np.mean(x[k:n])))
        #C[k] = np.mean(x[0:n-k]*x[k:n])
    return lags, C


def locmax(x):
    """Get local maxima of 1D-array

    Args:
        x: numeric sequence
    Returns:
        m: list, 1D-indices of local maxima
    """
    dx = np.diff(x) # discrete 1st derivative
    zc = np.diff(np.sign(dx)) # zero-crossings of dx
    m = 1 + np.where(zc == -2)[0] # indices of local max.
    return m


def zcross(x, mode):
    """Get zero crossings of a vector
    mode: 'np': '-' => '+'
          'pn': '+' => '-'
          'all': all
    FvW 08-2007
    """
    zc = np.diff(np.sign(x))
    if ( mode == "pn" ):
        zc = 1+np.where(zc == -2)[0]
    elif ( mode == "np" ):
        zc = 1+np.where(zc == 2)[0]
    elif ( mode == "all" ):
        zc = 1+np.where(np.abs(zc) == 2)[0]
    else:
        zc = ([], )
    #zc = np.array(zc)
    return zc


def mod(x,N):
    """
    modulo function
    """
    if (x <= 0): x += N
    if (x > N): x -= N
    return x


def ml1(gL=2, VL=-60, gCa=4, VCa=120, gK=8, VK=-84, C=20, I_ext=None,
        V1=-1.2, V2=18, V3=2, V4=30, phi=0.04, sd=5, v0=-60, w0=0,
        T=100, dt=0.05, n0=0, doplot=False):
    """
    Single-cell Morris-Lecar dynamics

    Parameters
    ----------
    gL : float,  optional
        leak conductance
    VL : float,  optional
        reversal potential of leak current
    gCa : float,  optional
        Ca2+ conductance
    VCa : float,  optional
        reversal potential of Ca2+ current
    gK : float,  optional
        K+ conductance
    VK : float,  optional
        reversal potential of K+ current
    C : float,  optional
        membrane capacitance
    I_ext : float, list, tuple, nd.array, optional
        external current, if list, tuple or nd.array, use first value
    V1 : float, optional
        shape parameter for Ca2+ channel steady-state open probability function
    V2 : float, optional
        shape parameter for Ca2+ channel steady-state open probability function
    V3 : float, optional
        shape parameter for K+ channel steady-state open probability function
    V4 : float, optional
        shape parameter for K+ channel steady-state open probability function
    phi : float, optional
        parameter in the K+ current ODE
    sd : float, optional
        standard deviation of the noise source
    v0 : float, optional
        initial value of the voltage variable V
    w0 : float, optional
        initial value of the K+ current variable w
    T : float, optional
        total simulation time
    dt : float, optional
        sampling time interval
    v_mn : float, optional
        plotting limit: minimum voltage
    v_mx : float, optional
        plotting limit: maximum voltage
    w_mn : float, optional
        plotting limit: minimum K+ channel open fraction
    w_mx : float, optional
        plotting limit: maximum K+ channel open fraction
    doplot : boolean, optional
        construct the plot or not

    Returns
    -------
    X : 1D array of float
        voltage values
    Y : 1D array of float
        K+ channel open fraction

    Example
    -------
    >>> ...

    """
    nt = int(T/dt)
    #C1 = 1/C
    sd_sqrt_dt = sd*np.sqrt(dt)
    try:
        # in case I_ext is provided
        I = np.hstack( (I_ext[0]*np.ones(n0), I_ext) )
    except:
        # I_ext not provided, set to zero
        I = np.zeros(n0+nt)

    # initial conditions
    v = v0
    w = w0
    X = np.zeros(nt)
    X[0] = v
    Y = np.zeros(nt)
    Y[0] = w

    # steady-state functions
    m_inf = lambda v: 0.5*(1 + np.tanh((v-V1)/V2))
    w_inf = lambda v: 0.5*(1 + np.tanh((v-V3)/V4))
    lambda_w = lambda v: phi*np.cosh((v-V3)/(2*V4))

    for t in range(n0+nt):
        if (t%100 == 0): print(f"t={t:d}/{n0+nt:d}\r", end="")
        # Morris-Lecar equations
        dvdt = 1/C * (-gL*(v-VL) -gCa*m_inf(v)*(v-VCa) -gK*w*(v-VK) + I[t])
        dwdt = lambda_w(v) * (w_inf(v) - w)
        # integrate
        v += (dvdt*dt + sd_sqrt_dt*np.random.randn()) # Ito
        w += (dwdt*dt)
        if (t >= n0):
            X[t-n0] = v
            Y[t-n0] = w
    print("")

    if doplot:
        time = np.arange(nt)*dt
        fig, ax = plt.subplots(1,1,figsize=(16,4))
        ax.plot(time, X, '-k', lw=2)
        ax.set_ylim(-80,60)
        ax.set_xlabel("time [ms]", fontsize=fs_)
        ax.set_ylabel("voltage [mV]", fontsize=fs_)
        ax.set_title(f"Single cell Morris-Lecar", fontsize=fs_)
        ax.grid(True)
        plt.tight_layout()
        plt.show()

    return X, Y


def ml2(gL=2, VL=-60, gCa=4, VCa=120, gK=8, VK=-84, C=20, I_ext=None,
        V1=-1.2, V2=18, V3=2, V4=30, phi=0.04, sd=5, v0=-60, w0=0,
        T=100, dt=0.05, n0=0, coupling='mode-A', doplot=False):
    """
    Two cell Morris-Lecar model
    """
    nt = int(T/dt)
    #C1 = 1/C
    sd_sqrt_dt = sd*np.sqrt(dt)
    try:
        # in case I_ext is provided
        I = np.hstack( (I_ext[0]*np.ones(n0), I_ext) )
    except:
        # I_ext not provided, set to zero
        I = np.zeros(n0+nt)

    if coupling == 'mode-A':
        J = 0.01
    elif coupling == 'mode-B':
        J = -0.01
    else:
        J = 0

    # initial conditions
    v0, w0 = 0, 0 # neuron-1
    v1, w1 = 0, 0 # neuron-2
    X0 = np.zeros(nt)
    Y0 = np.zeros(nt)
    X1 = np.zeros(nt)
    Y1 = np.zeros(nt)

    # steady-state functions
    m_inf = lambda v: 0.5*(1 + np.tanh((v-V1)/V2))
    w_inf = lambda v: 0.5*(1 + np.tanh((v-V3)/V4))
    lambda_w = lambda v: phi*np.cosh((v-V3)/(2*V4))

    for t in range(n0+nt):
        if (t%100 == 0): print(f"t={t:d}/{n0+nt:d}\r", end="")
        # Morris-Lecar equations
        # neuron-1
        dv0dt = 1/C * (-gL*(v0-VL) -gCa*m_inf(v0)*(v0-VCa) -gK*w0*(v0-VK) \
                      + I[t]) + J*(v1-v0)
        dw0dt = lambda_w(v0) * (w_inf(v0) - w0)
        # neuron-2
        dv1dt = 1/C * (-gL*(v1-VL) -gCa*m_inf(v1)*(v1-VCa) -gK*w1*(v1-VK) \
                       + I[t]) + J*(v0-v1)
        dw1dt = lambda_w(v1) * (w_inf(v1) - w1)
        # integrate
        v0 += (dv0dt*dt + sd_sqrt_dt*np.random.randn()) # Ito
        w0 += (dw0dt*dt)
        v1 += (dv1dt*dt + sd_sqrt_dt*np.random.randn()) # Ito
        w1 += (dw1dt*dt)
        if (t >= n0):
            X0[t-n0] = v0
            X1[t-n0] = v1
            Y0[t-n0] = w0
            Y1[t-n0] = w1
    print("")

    if doplot:
        time = np.arange(nt)*dt
        fig, ax = plt.subplots(2, 1, figsize=(16,8))
        ax[0].plot(time, X0, '-k', lw=2)
        ax[0].grid(True)
        ax[0].set_xlabel("time [a.u.]", fontsize=fs_)
        ax[0].set_ylabel("voltage [a.u.]", fontsize=fs_)
        ax[0].set_title(f"Two cell Morris-Lecar", fontsize=fs_)
        ax[1].plot(time, X1, '-k', lw=2)
        ax[1].grid(True)
        ax[1].set_xlabel("time [a.u.]", fontsize=fs_)
        ax[1].set_ylabel("voltage [a.u.]", fontsize=fs_)
        plt.tight_layout()
        plt.show()

    return X0, Y0, X1, Y1


def ml8(gL=0.6, VL=-1.8, gCa=3, VCa=1, gK=1.8, VK=-0.8, C=1, I_ext=1.0,
        V1=0.2, V2=0.4, V3=0.3, V4=0.2, phi=0.2, sd=5, v0=-60, w0=0,
        T=100, dt=0.05, n0=0, mode="walk", doplot=False, doanimate=False):
    """
    Central Pattern Generator for Quadruped locomotion
    based on the Morris-Lecar model

    References
    ----------

    .. [1] Buono L and Golubitsky M, "Models of central pattern generators for
           quadruped locomotion I. Primary Gaits". J. Math. Biol. 42,
           291--326 (2001)

    .. [2] Buono L and Golubitsky M, "Models of central pattern generators for
           quadruped locomotion II. Secondary Gaits". J. Math. Biol. 42,
           327--346 (2001)
    """

    """
    indices here: 0..7
    original paper: 1..8
    Returns:
        C: integer array (N,2)
           C[i,0]: neighbour `(i-2) mod N` of neuron `i`
           C[i,1]: neighbour `(i+(-1)^i) mod N` of neuron `i`

    explicitly:
    i | (i-2)%N | (i+eps_i)%N | (i+4)%N
    0      6           1          4
    1      7           0          5
    2      0           3          6
    3      1           2          7
    4      2           5          0
    5      3           4          1
    6      4           7          2
    7      5           6          3
    """
    N = 8
    K = np.zeros((8,2),dtype=np.int)
    for i in range(N):
        k = (i-2)%N
        ei = (-1)**i
        l = (i + ei)%N
        #m = (i + 4)%N
        #print(f"i = {i:d}, {k:d} > {i:d}, {l:d} > {i:d}")
        #print(f"i = {i:d}, {k:d}, {l:d}, {m:d}")
        K[i,0] = k
        K[i,1] = l

    """
    coupling constants (gait determinants)
    """
    # Morris-Lecar parameters
    # gait parameters
    # parameters: alpha, beta, gamma, delta
    # Table 12:
    # 'pace': [0.3, 0.3, -0.32, -0.32],
    # 'bound': [-0.32, -0.32, 0.3, 0.3],
    # Table 13:
    # 'pace': [0.2, 0.2, -0.2, -0.2],
    # 'bound': [-0.2, -0.2, 0.2, 0.2],

    # [alpha, beta, gamma, delta]
    modes = {}
    modes['pronk'] = [0.2, 0.2, 0.2, 0.2]
    modes['pace'] = [0.2, 0.2, -0.2, -0.2]
    modes['bound'] = [-0.2, -0.2, 0.2, 0.2]
    modes['trot'] = [-0.6, -0.6, -0.6, -0.6]
    modes['jump'] = [0.01, -0.01, 0.2, 0.2]
    modes['walk'] = [0.01, -0.01, -1.2, -1.2]
    modes['canter'] = [0.17, -0.2, -0.9, -1]
    if mode == 'canter':
        gCa = 8
        x00 = 0.4
        y00 = 0.3
    modes['runwalk'] = [-0.78, -0.56, 0.12, -1.14]
    if mode == 'runwalk':
        gCa = 2
        x00 = -1.2147809
        y00 = -0.058746844
    modes['doublebond'] = [-0.6, -0.77, 0.3, 0.5]
    if mode == 'doublebond':
        gCa = 3
        x00 = -1
        y00 = 0.6

    if mode not in modes:
        print("ERROR: mode not defined")
        sys.exit()

    #self.mode = ['pronk', 'pace', 'bound', 'trot', 'jump', 'walk'][5]
    # pronk: ok
    # pace: ok, but dependent on init. cond.
    # bound: ok, but dependent on init. cond.
    # trot: ok, but dependent on init. cond.
    # jump: ok
    # walk: ok

    print(f"[+] mode: {mode:s}")
    alpha, beta, gamma, delta = modes[mode]

    n_t = int(T/dt)
    X = np.zeros((n_t,N))
    #Y = np.zeros((n_t,N))
    v = np.zeros(N)
    w = np.zeros(N)
    dvdt = np.zeros(N)
    dwdt = np.zeros(N)

    #''' random initial conditions
    for i in range(8):
        v[i] = np.random.randn()
        w[i] = np.random.randn()
        #X[0,i] = np.random.randn() # 0 # .31
        #Y[0,i] = np.random.randn() # -1.24 # 0.52
        #X[0,i] = -2*np.random.rand() # 0 # .31
        #Y[0,i] = -1*np.random.rand() # -1.24 # 0.52
    #'''

    # apply initial conditions if provided (secondary gaits)
    try:
        #X[0,0] = x00
        #Y[0,0] = y00
        v[0] = x00
        w[0] = y00
        v[1:] = 0.0
        w[1:] = 0.0
    except:
        pass
    X[0,:] = v
    #Y[0,:] = w

    # steady-state functions
    m_inf = lambda v: 0.5*( 1 + np.tanh((v-V1)/V2) )
    w_inf = lambda v: 0.5*( 1 + np.tanh((v-V3)/V4) )
    lambda_w = lambda v: phi*np.cosh((v-V3)/(2*V4))

    for t in range(1,n0+n_t):
        # single node dynamics
        dvdt = -gCa*m_inf(v)*(v-VCa) -gL*(v-VL) -gK*w*(v-VK) + I_ext
        dwdt = lambda_w(v) * (w_inf(v) - w)
        # changes due to connections
        dvdt += (alpha*v[K[i,0]] + gamma*v[K[:,1]]) #+xi*x(i+4)
        dwdt += (beta*w[K[:,0]] + delta*w[K[:,1]]) #+eta*y(i+4)
        # integrate
        v += (dvdt*dt)
        w += (dwdt*dt)
        '''
        for i in range(N):
            #v = X[t-1,i]
            #w = Y[t-1,i]
            v = x[i]
            w = y[i]
            # vector dxdt, dydt
            dxdt[i] = -gCa*m(v)*(v-VCa) -gL*(v-VL) -gK*w*(v-VK) + I_ext
            dydt[i] = phi*tau(v)*(n(v)-w)
            # scalar dxdt, dydt
            #dxdt += (alpha*X[t-1,K[i,0]] + gamma*X[t-1,K[i,1]]) #+xi*x(i+4)
            #dydt += (beta*Y[t-1,K[i,0]] + delta*Y[t-1,K[i,1]]) #+eta*y(i+4)
            dxdt[i] += (alpha*x[K[i,0]] + gamma*x[K[i,1]]) #+xi*x(i+4)
            dydt[i] += (beta*y[K[i,0]] + delta*y[K[i,1]]) #+eta*y(i+4)
        x += (dxdt*dt)
        y += (dydt*dt)
        '''
        if (t >= n0):
            X[t-n0,:] = v
            #Y[t-n0,:] = Y[t-1,i] + dydt*dt

    # interpolate
    downsample_fct = 5
    dt_interp = downsample_fct*dt
    n_interp = int(n_t/downsample_fct)
    t_ = np.arange(n_t)
    f_ip = interp1d(t_, X, axis=0, kind='linear')
    t_new = np.linspace(0, n_t-1, n_interp)
    X = f_ip(t_new)
    data = {}
    data['LH'] = X[:,0] # left hind
    data['RH'] = X[:,1] # right hind
    data['LF'] = X[:,2] # left fore
    data['RF'] = X[:,3] # right fore
    data['time'] = dt_interp*np.arange(n_interp)

    doanimate0 = doanimate1 = doanimate2 = doanimate3 = False

    if doplot:
        # 0: left hind leg
        # 1: right hind leg
        # 2: left fore leg
        # 3: right fore leg
        p_ann = {
            'xy' : (0.01,0.80),
            'xycoords' : 'axes fraction',
            'fontsize' : 22,
            'fontweight' : 'bold'
        }
        fig, ax = plt.subplots(4, 1, figsize=(16,8), sharex=True)
        legs = ['LF', 'RF', 'LH', 'RH']
        for i, leg in enumerate(legs):
            ax[i].plot(data[leg], '-k')
            ax[i].annotate(leg, **p_ann)
            ax[i].grid()
        ax[0].set_title(f"{mode:s}", fontsize=22)
        plt.tight_layout()
        plt.show()

    if doanimate0:
        """
        make movie
        """
        loc = {
            "LF" : (50,100),
            "RF" : (280,100),
            "LH" : (40,300),
            "RH" : (280,240),
        }
        # find local maxima in activation time courses
        peaks = {}
        for leg in data:
            peaks[leg] = locmax(data[leg])
        # load animal background image
        img = mpimg.imread('./img/quadruped.png')
        movname = f"cpg_quad_{mode:s}.mp4"
        im_list = []
        n_interp = len(data['LF'])
        print("[+] Accumulate image arrays...")
        for t in range(n_interp):
            if (t%10 == 0): print(f"\tt = {t:d}/{n_interp:d}\r", end="")
            fig1 = plt.figure() # figsize=(8,4)
            ax1 = plt.gca()
            #fig1, ax1 = plt.subplots() # figsize=(8,4)
            ax1.imshow(img)
            ax1.axis('off')
            plt.tight_layout()
            for leg in peaks:
                if t in peaks[leg]:
                    ax1.scatter(loc[leg][0], loc[leg][1], s=200, lw=0.5,
                                edgecolors='None', facecolors='r')
            fig1.canvas.draw()
            y = np.fromstring(fig1.canvas.tostring_rgb(), dtype=np.uint8, sep='')
            y = y.reshape(fig1.canvas.get_width_height()[::-1] + (3,))
            im_list.append(y.copy())
            plt.close()
            plt.cla()
            plt.clf()
        #print("\nim_list: ", len(im_list))
        print("")

        print("[+] Make animation...")
        fig, ax = plt.subplots()
        ax.axis('off')
        ims = []
        for t in range(n_interp):
            if (t%10 == 0): print(f"\tt = {t:d}/{n_interp:d}\r", end="")
            img_ = im_list[t]
            im = ax.imshow(img_)
            ims.append([im])
        print("")
        #print("ims: ", type(ims), len(ims), type(ims[0]))
        #print(ims[0])

        ani = animation.ArtistAnimation(fig, ims, interval=50, blit=True,
                                        repeat_delay=1000)
        # [1]
        print("saving animation...")
        #ani.save(movname)
        # [2]
        writer = animation.FFMpegWriter(fps=15, metadata=dict(artist='Me'),
                                        bitrate=1800)
        ani.save(movname, writer=writer)
        print("...animation saved.")
        plt.show()

    '''
    if doanimate1:
        """
        make movie: use cv2
        """
        loc = {
            "LF" : (20,100),
            "RF" : (260,100),
            "LH" : (20,330),
            "RH" : (280,250),
        }
        # find local maxima in activation time courses
        peaks = {}
        for leg in data:
            peaks[leg] = locmax(data[leg])
        # load animal background image
        img = mpimg.imread('./img/quadruped.png')
        movname = f"cpg_quad2_{mode:s}.mp4"
        #os.chdir("/home/frederic/data/graphs/tmp/")
        #fnames = []

        # get first image
        fig = plt.figure() # figsize=(8,4)
        plt.imshow(img)
        #plt.tight_layout()
        #plt.show()
        fig.canvas.draw()
        y = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep='')
        y = y.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        #print(type(y), y.shape, y.dtype) #, y[:10])
        nx, ny = y.shape[1], y.shape[0]
        #nt = 100

        framerate = 30
        out = cv2.VideoWriter(movname, cv2.VideoWriter_fourcc(*'mp4v'), \
                              framerate, (nx,ny))

        n_interp = len(data['LF'])
        for t in range(n_interp):
            if (t%10 == 0): print(f"\tt = {t:d}/{n_interp:d}\r", end="")
            #fname = "tmp_{:04d}.png".format(t)
            #fnames.append(fname)
            plt.close()
            plt.cla()
            plt.clf()
            fig = plt.figure()
            plt.imshow(img)
            for leg in peaks:
                if t in peaks[leg]:
                    #plt.annotate(leg, xy=loc[leg], xycoords='data', fontsize=28)
                    plt.plot(loc[leg][0], loc[leg][1], 'or', ms=20)
            plt.axis('off')
            fig.canvas.draw()
            y = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep='')
            y = y.reshape(fig.canvas.get_width_height()[::-1] + (3,))
            out.write(y)
            #plt.savefig(fname, dpi=90, bbox_inches="tight")
            #plt.show()
            #plt.close(); plt.cla(); plt.clf()
        out.release()
        print("")
        #cmd = videoCmd(movname, frate=25, width=128, height=128)
        #os.system(cmd)
        #for fname in fnames: os.remove(fname)
    '''

    if doanimate2:
        """
        make movie: save figures as png, system call to ffmpeg
        """
        loc = {
            "LF" : (20,100),
            "RF" : (260,100),
            "LH" : (20,330),
            "RH" : (280,250),
        }
        # find local maxima in activation time courses
        peaks = {}
        for leg in data:
            peaks[leg] = locmax(data[leg])
        # load animal background image
        img = mpimg.imread('./img/quadruped.png')
        movname = f"cpg_quad2_{mode:s}"
        os.chdir("/home/frederic/data/graphs/tmp/")
        fnames = []
        n_interp = len(data['LF'])
        for t in range(n_interp):
            if (t%10 == 0): print(f"\tt = {t:d}/{n_interp:d}\r", end="")
            fname = "tmp_{:04d}.png".format(t)
            fnames.append(fname)
            plt.figure()
            plt.imshow(img)
            for leg in peaks:
                if t in peaks[leg]:
                    plt.annotate(leg, xy=loc[leg], xycoords='data', fontsize=28)
            plt.axis('off')
            plt.savefig(fname, dpi=90, bbox_inches="tight")
            #plt.show()
            plt.close(); plt.cla(); plt.clf()
        print("")
        cmd = videoCmd(movname, frate=25, width=128, height=128)
        os.system(cmd)
        for fname in fnames: os.remove(fname)

    if doanimate:
        """
        make movie
        """
        #movname = f"cpg_quad_{mode:s}.mp4"
        movname = f"cpg_quad_{mode:s}.gif"
        print(f"[+] Animate data as movie: {movname:s}")
        loc = {
            "LF" : (50,100),
            "RF" : (280,100),
            "LH" : (40,300),
            "RH" : (280,240),
        }
        # find local maxima in activation time courses
        peaks = {}
        for leg in data:
            peaks[leg] = locmax(data[leg])
        # load animal background image
        #img = mpimg.imread('./img/quadruped.png')
        img = mpimg.imread('./img/quadruped_bw.png')

        fig, ax = plt.subplots()
        ax.imshow(img)
        ax.axis('off')
        s = 200
        lf = ax.scatter(loc['LF'][0], loc['LF'][1], s=s, lw=0.5,
                        edgecolors='None', facecolors='None')
        rf = ax.scatter(loc['RF'][0], loc['RF'][1], s=s, lw=0.5,
                        edgecolors='None', facecolors='None')
        lh = ax.scatter(loc['LH'][0], loc['LH'][1], s=s, lw=0.5,
                        edgecolors='None', facecolors='None')
        rh = ax.scatter(loc['RH'][0], loc['RH'][1], s=s, lw=0.5,
                        edgecolors='None', facecolors='None')

        def update(i):
            if (i%10 == 0): print(f"\tt = {i:d}/{n_interp:d}\r", end="")
            if i in peaks['LF']:
                lf.set_facecolors('r')
            else:
                lf.set_facecolors('none')
            if i in peaks['RF']:
                rf.set_facecolors('r')
            else:
                rf.set_facecolors('none')
            if i in peaks['LH']:
                lh.set_facecolors('r')
            else:
                lh.set_facecolors('none')
            if i in peaks['RH']:
                rh.set_facecolors('r')
            else:
                rh.set_facecolors('none')

        # make animation
        n_interp = len(data['LF'])
        ani = FuncAnimation(fig, update, interval=50, save_count=n_interp)
        #ani.save(movname)
        ani.save(movname, writer=PillowWriter(fps=24))
        plt.show()
        print("Animation created and saved.")

    return data


def phase_plane(v=np.array([]), w=np.array([]),
                gL=2, VL=-60, gCa=4, VCa=120, gK=8, VK=-84, C=20, I_ext=0,
                V1=-1.2, V2=18, V3=2, V4=30, phi=0.04, sd=5, v0=-60, w0=0,
                T=100, dt=0.05,
                v_mn=-80, v_mx=60, w_mn=0, w_mx=0.6,
                doplot=True):
    """
    Make a (V,w) phase-plane plot

    Plot the vector field defined by the Morris-Lecar equations for
    dV/dt and dw/dt (streamlines), the V-nullcline, the w-nullcline, and
    the trajectory defined by the data points.

    Parameters
    ----------
    v : 1D array, optional
        time course of voltage values; if empty, not trajectory is plotted
    w : 1D array, optional
        time course of open K+ fraction; if empty, not trajectory is plotted
    gL : float,  optional
        leak conductance
    VL : float,  optional
        reversal potential of leak current
    gCa : float,  optional
        Ca2+ conductance
    VCa : float,  optional
        reversal potential of Ca2+ current
    gK : float,  optional
        K+ conductance
    VK : float,  optional
        reversal potential of K+ current
    C : float,  optional
        membrane capacitance
    I_ext : float, list, tuple, nd.array, optional
        external current, if list, tuple or nd.array, use first value
    V1 : float, optional
        shape parameter for Ca2+ channel steady-state open probability function
    V2 : float, optional
        shape parameter for Ca2+ channel steady-state open probability function
    V3 : float, optional
        shape parameter for K+ channel steady-state open probability function
    V4 : float, optional
        shape parameter for K+ channel steady-state open probability function
    phi : float, optional
        parameter in the K+ current ODE
    sd : float, optional
        standard deviation of the noise source
    v0 : float, optional
        initial value of the voltage variable V
    w0 : float, optional
        initial value of the K+ current variable w
    T : float, optional
        total simulation time
    dt : float, optional
        sampling time interval
    v_mn : float, optional
        plotting limit: minimum voltage
    v_mx : float, optional
        plotting limit: maximum voltage
    w_mn : float, optional
        plotting limit: minimum K+ channel open fraction
    w_mx : float, optional
        plotting limit: maximum K+ channel open fraction
    doplot : boolean, optional
        construct the plot or not

    Example
    -------
    >>> ...

    """
    # plot phase-plane for first current value if I_ext is an array
    if isinstance(I_ext, (list, tuple, np.ndarray)):
        I_ext = I_ext[0]
    C_ = 1./C
    #print(f"u_mn = {u_mn:.3f}, u_mx = {u_mx:.3f}")
    #print(f"v_mn = {v_mn:.3f}, v_mx = {v_mx:.3f}")
    v_ = np.arange(v_mn, v_mx, 0.05)
    w_ = np.arange(w_mn, w_mx, 0.05)
    v_null_x = w_null_x = v_
    V, W = np.meshgrid(v_, w_)

    # background color
    #c_int = 128 # gray intensity as integer
    #c_hex = '#{:s}'.format(3*hex(c_int)[2:])
    cmap = matplotlib.cm.get_cmap('gnuplot2')

    v_null_y = (-gL*(v_-VL) -gCa*0.5*(1.+np.tanh((v_-V1)/V2))*(v_-VCa) \
                + I_ext) / (gK*(v_-VK))
    w_null_y = 0.5*(1.+np.tanh((v_-V3)/V4))
    Dv = C_ * (-gL*(V-VL) -gCa*0.5*(1.+np.tanh((V-V1)/V2))*(V-VCa) \
               -gK*W*(V-VK) + I_ext)
    Dw = phi*np.cosh((V-V3)/(2.*V4))*(0.5*(1.+np.tanh((V-V3)/V4))-W)
    #V, W = np.meshgrid(v_, w_)

    # +++ Figure +++
    fig = plt.figure(figsize=(6,6))
    #fig.patch.set_facecolor(c_hex)
    ax = plt.gca()
    #ax.patch.set_facecolor(c_hex)
    plt.plot(v_, np.zeros_like(v_), '-k', lw=1) # x-axis
    plt.plot(np.zeros_like(w_), w_, '-k', lw=1) # y-axis
    # nullcline: dV/dt = 0
    plt.plot(v_null_x, v_null_y, '-k', lw=3, label=r"$\frac{dV}{dt}=0$")
    # nullcline: dw/dt = 0
    plt.plot(w_null_x, w_null_y, '--k', lw=3, label=r"$\frac{dw}{dt}=0$")
    ps = {'x': v_, 'y': w_, 'u': Dv, 'v': Dw, 'density': 1.0, \
          'color': 'k', 'linewidth': 0.3, 'arrowsize': 0.80}
    p_ = plt.streamplot(**ps)
    # color nodes according to index j
    v_norm = (v-v_mn)/(v_mx-v_mn)
    w_norm = (w-w_mn)/(w_mx-w_mn)
    for j in range(0, len(v), 10):
        plt.plot(v[j], w[j], marker='o', color='b', ms=6, alpha=0.5)
        #cj = cmap(u)
        #plt.plot(v[i,j], w[i,j], color=cj, marker='o', ms=4)

    #plt.grid(True)
    plt.axis([v_mn, v_mx, w_mn, w_mx])
    plt.xlabel("V", fontsize=18)
    plt.ylabel("w", fontsize=18)
    ax.tick_params(axis='both', labelsize=16)
    ax.legend(loc=1, fontsize=18)
    #plt.axis('equal')
    plt.tight_layout()
    plt.show()


def fp_stability(gL=2, VL=-60, gCa=4, VCa=120, gK=8, VK=-84, C=20, I_ext=0,
                 V1=-1.2, V2=18, V3=2, V4=30, phi=0.04, v=-40, w=0.1):
    # infinity functions
    m_inf = lambda v: 0.5*(1 + np.tanh((v-V1)/V2))
    w_inf = lambda v: 0.5*(1 + np.tanh((v-V3)/V4))
    lambda_w = lambda v: phi*np.cosh((v-V3)/(2*V4))
    # fixed points
    v_null = lambda v: (-gL*(v-VL)-gCa*m_inf(v)*(v-VCa)+I_ext)/(gK*(v-VK))
    w_null = lambda v: w_inf(v)
    fp = lambda v: (-gL*(v-VL)-gCa*m_inf(v)*(v-VCa)+I_ext)/(gK*(v-VK))-w_inf(v)
    vs = np.linspace(-80,60,1000)
    ys = fp(vs)
    zs = zcross(ys,'all')
    #print("zero crossings: ", zs)
    for z in zs:
        v = vs[z]
        w = v_null(v)
        J = np.zeros((2,2))
        print(f"Fixed point: v = {v:.3f}, w = {w:.3f}")
        J[0,0] = 1/C*(-gCa*((v-VCa)/(2*(np.cosh((v-V1)/V2))**2) + m_inf(v)) -gK*w -gL)
        J[0,1] = 1/C*(-gK*(v-VK))
        J[1,0] = phi*np.sinh((v-V3)/(2*V4))*(w_inf(v)-w) \
               + lambda_w(v)/(2*(np.cosh((v-V3)/V4))**2)
        J[1,1] = -lambda_w(v)
        # eigenvalue analysis
        l0, l1 = np.linalg.eigvals(J)
        print(f"Jacobian eigenvalues: l0 = {l0:.4f}, l1 = {l1:.4f}")


def check_exports():
    f = ("/home/frederic/Projects/UNSW/teaching/T2_2020/NEUR3101/2020/Pracs/"
         "P2_CPG/task1_data4.csv")
    x = np.loadtxt(f, skiprows=1)
    print(x.shape)
    plt.figure(figsize=(16,3))
    plt.plot(x[:,0], x[:,1], '-k', lw=2)
    plt.grid()
    plt.tight_layout()
    plt.show()


def oscillator_types(mode="type1"):
    """
    Plot mean action potential frequency as a function of input current

    Parameters
    ----------
    mode : {'type1', 'type2'}, optional
        'type1' : use type1 oscillator parameters
        'type2' : use type2 oscillator parameters

    """
    print("mode: ", mode)
    T = 2000
    dt = 0.05
    nt = int(T/dt)
    n0 = int(nt/5) # initial transient to ignore

    run = True # True: run simulation, False: re-load previous results
    if run:
        sd = 0.1
        nI = 50
        nruns = 10 # 10
        if mode == "type1":
            print("[+] Type I oscillator")
            Is = np.linspace(50,180,nI)
            params = {'T' : T,
                      'dt' : dt,
                      'n0' : n0,
                      'sd' : sd,
                      'I_ext' : None,
                      'v0' : -16,
                      'w0' : 0.014915,
                      'V1' : -1.2,
                      'V2' : 18,
                      'V3' : 12,
                      'V4' : 17.4,
                      'phi' : 1/15,
                      'gCa' : 4,
                      'gK' : 8,
                      'gL' : 2,
                      'VCa' : 120,
                      'VK' : -84,
                      'VL' : -60,
                      'C' : 20,
                      'doplot' : False,
            }

        if mode == "type2":
            print("[+] Type II oscillator")
            Is = np.linspace(20,150,nI)
            params = {'T' : T,
                      'dt' : dt,
                      'n0' : n0,
                      'sd' : sd,
                      'I_ext' : None,
                      'v0' : -16,
                      'w0' : 0.014915,
                      'V1' : -1.2,
                      'V2' : 18,
                      'V3' : 2,
                      'V4' : 30,
                      'phi' : 0.04,
                      'gCa' : 4.4,
                      'gK' : 8,
                      'gL' : 2,
                      'VCa' : 120,
                      'VK' : -84,
                      'VL' : -60,
                      'C' : 20,
                      'doplot' : False,
            }

        # main frequency in voltage time course
        freq_arr = np.zeros((nI,nruns))
        # action potential detection threshold
        ap_thr = 10
        for i, I in enumerate(Is):
            print(f"I = {I:.3f} [{i:d}/{nI:d}]")
            params['I_ext'] = I*np.ones(nt)
            params['doplot'] = not True
            for j in range(nruns):
                X, _ = ml1(**params) # run ML simulation
                #X = X[n0:] # ignore initial transient
                peaks = locmax(X) # all local maxima (peaks)
                peaks = peaks[X[peaks]>ap_thr] # delete peaks below AP threshold
                #peaks = np.array([p for p in peaks if X[p] > ap_thr ])
                if ( len(peaks) > 0 ):
                    ppi = dt*np.diff(ms) # peak-to-peak intervals
                    ppi_mean = np.mean(ppi) # [ms]
                    ppi_std = np.std(ppi) # [ms]
                    freq = 1e3/ppi_mean # [Hz]
                    #print(f"ppi_mean = {ppi_mean:.1f} (\pm {ppi_std:.1f}) ms")
                    #print(f"AP frequency = {freq:.4f} Hz")
                    freq_arr[i,j] = freq
                    ''' check peak detection visually
                    time = dt*np.arange(len(X))
                    plt.figure(figsize=(12,4))
                    plt.plot(time, X, '-k')
                    plt.plot(time[peaks], X[peaks], 'ob', ms=8, alpha=0.5)
                    plt.tight_layout()
                    plt.show()
                    '''
            #yn = input("...")
            #if yn=="n": sys.exit()
        np.savez(f"./data/fs_main_{mode:s}.npy", Is=Is, freq_arr=freq_arr)
    else:
        # re-load results (do not run simulation)
        try:
            fname = f"./data/fs_main_{mode:s}.npy"
            data = np.load(fname)
            Is = data['Is']
            freq_arr = data['freq_arr']
        except:
            print(f"file {fname:s} not found, exiting.")
            #print("(run the simulation first)")
            sys.exit()

    # mean/median across runs
    #freqs = np.mean(freq_arr, axis=1)
    freqs = np.median(freq_arr, axis=1)

    # Figure
    plt.figure(figsize=(8,4))
    ax = plt.gca()
    ax.plot(Is, freqs, '-ok', lw=2)
    ax.tick_params(axis='both', labelsize=16)
    plt.xlabel(r"$I_{ext} \: [\mu A / cm^2]$", fontsize=16)
    plt.ylabel("f [Hz]", fontsize=16)
    plt.title(f"Morris-Lecar oscillator {mode:s}", fontsize=20)
    plt.tight_layout()
    plt.show()


def spectrogram(x, fs, f_min, f_max, n_freq):
    """
    Plot a spectrogram

    The function computes the continuous wavelet transform, using the Morlet
    wavelet with parameter `w` set to 6.

    Parameters
    ----------
    x : 1D array
        real-valued signal for which the spectrogram is computed
    fs: scalar
        sampling frequency
    f_min : scalar
        minimum frequency displayed
    f_max : scalar
        maximum frequency displayed
    n_freq : int
        number of frequency bins

    Example
    -------
    >>> from cpg_quad import spectrogram
    >>> import numpy as np
    >>> import matplotlib.pyplot as plt
    >>> n = 5000
    >>> t, dt = np.linspace(0, 10, n, retstep=True)
    >>> fs = 1/dt
    >>> f = np.linspace(2, 3, n)
    >>> y = np.cos(2*np.pi*f*t)
    >>> plt.figure(figsize=(8,4))
    >>> plt.plot(t, y, '-k')
    >>> plt.show()
    >>> spectrogram(y, fs, f_min=0.1, f_max=10, n_freq=50)

    """
    w = 6.
    dt = 1/fs
    time = dt*np.arange(x.shape[0])
    freq = np.linspace(f_min, f_max, int(n_freq))
    widths = w*fs / (2*freq*np.pi)
    cwtm = cwt(x, morlet2, widths, w=w)
    plt.figure(figsize=(10,4))
    plt.pcolormesh(time, freq, np.abs(cwtm), cmap='viridis', shading='gouraud')
    plt.xlabel("time")
    plt.ylabel("freq.")
    plt.colorbar()
    plt.tight_layout()
    plt.show()


def videoCmd(movname, frate=20, width=512, height=512):
    # generate command string to convert png images to mp4 using ffmpeg
    # ffmpeg -hide_banner -loglevel panic
    # ffmpeg -nostats -loglevel 0
    # ffmpeg -loglevel error [other commands]
    verbosity = "-nostats -loglevel 0"
    cmd = 'ffmpeg {:s} -r {:d} -f image2 -s {:d}x{:d} -i tmp_%04d.png -vcodec'
    cmd += ' libx264 -crf 25 -pix_fmt yuv420p'
    cmd += ' -vf "scale=trunc(iw/2)*2:trunc(ih/2)*2" {:s}.mp4'
    cmd = cmd.format(verbosity, frate, width, height, movname)
    print("[+] Video command (ffmpeg):")
    print("[+] {:s}".format(cmd))
    return cmd


def wrap_up():
    """
    Make figures for wrap-up session
    """
    os.chdir(("/home/frederic/Projects/UNSW/teaching/T2_2020/NEUR3101/2020/"
              "Pracs/P2_CPG/data/"))

    #''' Task-1
    f = "task1_data1.csv"
    #f = "task1_data2.csv"
    #f = "task1_data3.csv"
    #f = "task1_data4.csv"
    x = np.loadtxt(f, skiprows=1)
    plt.figure(figsize=(16,3))
    plt.plot(x[:,0], x[:,1], '-k', lw=2)
    plt.xlabel("time [a.u.]", fontsize=fs_)
    plt.ylabel("voltage [a.u.]", fontsize=fs_)
    plt.annotate(f, xy=(0.8,0.8,), xycoords='axes fraction', fontsize=fs_)
    plt.grid()
    plt.tight_layout()
    plt.show()
    #'''

    ''' Task-2
    f = "task2_data1.csv"
    #f = "task2_data2.csv"
    x = np.loadtxt(f, skiprows=1)
    print(x.shape)
    fig, ax = plt.subplots(2,1,figsize=(16,6))
    ax[0].plot(x[:,0], x[:,1], '-k', lw=2)
    ax[1].plot(x[:,0], x[:,2], '-k', lw=2)
    ax[1].set_xlabel("time [a.u.]", fontsize=fs_)
    ax[0].set_ylabel("voltage [a.u.]", fontsize=fs_)
    ax[1].set_ylabel("voltage [a.u.]", fontsize=fs_)
    #plt.annotate(f, xy=(0.8,0.8,), xycoords='axes fraction', fontsize=fs_)
    ax[0].grid()
    ax[1].grid()
    plt.tight_layout()
    plt.show()
    '''

    ''' Task-3
    #f = "task3_data_mode-A.csv" # Pronk
    #f = "task3_data_mode-B.csv" # Pace
    #f = "task3_data_mode-C.csv" # Bound
    #f = "task3_data_mode-D.csv" # Trot
    f = "task3_data_mode-E.csv" # Jump
    #f = "task3_data_mode-F.csv" # Walk
    X = np.loadtxt(f, skiprows=1)
    print(X.shape)

    fig, ax = plt.subplots(4,1,figsize=(16,8), sharex=True)
    ax[0].plot(X[:,0], X[:,1], '-k', lw=2, label="left front")
    ax[1].plot(X[:,0], X[:,2], '-k', lw=2, label="right front")
    ax[2].plot(X[:,0], X[:,3], '-k', lw=2, label="left rear")
    ax[3].plot(X[:,0], X[:,4], '-k', lw=2, label="right rear")
    ax[3].set_xlabel("time [a.u.]", fontsize=fs_)
    for i in range(4):
        ax[i].grid(True)
        ax[i].legend(loc=2, fontsize=fs_+4)
    plt.tight_layout()
    plt.show()
    '''


def main():
    # 'pronk', 'pace', 'bound', 'trot', 'jump', 'walk'
    #q = Quad(mode='walk', T=500, dt=0.05)
    #check_exports()
    #wrap_up()
    #ml1(T=1000, dt=0.05, sd=0.1, I0=90, I1=110)
    #ml2(T=1000, dt=0.05, sd=0.1, I0=90, I1=120, coupling='mode-B')
    #bifurcation_diagram()
    oscillator_types(mode="type2")

    '''
    task = 'Task3--'
    save_ = not True

    T, dt = 100, 0.05
    sd = 0.1

    if task == 'Task1':
        #I0 = I1 = -0.1; id=2
        #I0 = I1 = 0.01; id=1
        I0 = I1 = 0.25; id=4
        #I0 = I1 = 1.00; id=3
        X, Y = fhn1(T, dt, sd, I0, I1, doplot=True)
        plt.xcorr(X,X,maxlags=500); plt.show() # cross-correlation
        time = np.arange(X.shape[0])*dt
        fname = f"task1_data{id:d}.csv"
        if save_:
            print(f"[+] save data as: {fname:s}")
            np.savetxt(fname, np.vstack((time,X)).T, fmt='%.5f', delimiter=' ',\
                       newline='\n', header='time, voltage', footer='', \
                       comments='# ', encoding=None)

    if task == 'Task2':
        #I0 = I1 = -2.8
        #X0, Y0, X1, Y1 = fhn2(T, dt, sd, I0, I1, coupling='mode-A', doplot=True)
        I0 = I1 = 0.05
        X0, Y0, X1, Y1 = fhn2(T, dt, sd, I0, I1, coupling='mode-B', doplot=True)
        time = np.arange(X0.shape[0])*dt
        fname = "task2_data2.csv"
        if save_:
            np.savetxt(fname, np.vstack((time,X0,X1)).T, fmt='%.5f', \
                       delimiter=' ', newline='\n', \
                       header='time, V1, V2', footer='', comments='# ', \
                       encoding=None)

    if task == 'Task3':
        X = fhn8(mode='F', doplot=True)
        time = np.arange(X.shape[0])*dt
        fname = "task3_data_mode-F.csv"
        if save_:
            np.savetxt(fname, np.vstack((time,X[:,:4].T)).T, fmt='%.5f', \
                       delimiter=' ', newline='\n', \
                       header='time, V1, V2, V3, V4', footer='', comments='# ',\
                       encoding=None)
    '''


if __name__ == "__main__":
    os.system("clear")
    main()
