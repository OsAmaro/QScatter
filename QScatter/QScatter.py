#!/usr/bin/python
"""
    QScatter (quick-scattering-toolkit)
    
    References:
        [1] Blackburn et al, Phys. Rev. A 96, 022128, Scaling laws for positron production in laser–electron-beam collisions, https://journals.aps.org/pra/abstract/10.1103/PhysRevA.96.022128
        [2] N. Neitz and A. Di Piazza, Phys. Rev. Lett. 111, 054802 , Stochasticity Effects in Quantum Radiation Reaction, https://journals.aps.org/prl/abstract/10.1103/PhysRevLett.111.054802
        [3] Marija Vranic et al 2016 New J. Phys. 18 073035, Quantum radiation reaction in head-on laser-electron beam interaction, https://iopscience.iop.org/article/10.1088/1367-2630/18/7/073035
        [4] Óscar Amaro and Marija Vranic 2021 New J. Phys. 23 115001, Optimal laser focusing for positron production in laser–electron scattering, https://iopscience.iop.org/article/10.1088/1367-2630/ac2e83
        [5] Fabien Niel PhD thesis, Classical and Quantum Description of Plasma and Radiation in Strong Fields 
"""

# numpy
import numpy as np
np.random.seed(1234)
from numpy.random import default_rng
rng = default_rng()
# import functions
from scipy.special import kv, iv, erf
from scipy.integrate import quad
from numpy import log, log10, sin, cos, exp, sqrt, tan, pi, heaviside
# interpolate
from scipy import interpolate
# physical constants
from scipy.constants import c, alpha, hbar, e
from scipy.constants import electron_mass, elementary_charge, speed_of_light
from scipy.constants import c, m_e, e, epsilon_0 # physical constants
m_eV = electron_mass * speed_of_light**2 / elementary_charge; # electron mass [eV]
m = m_eV*1e-9; #[GeV] = 0.5109989461e-3
# root finding
from scipy.optimize import fsolve
from scipy import optimize
# plotting
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
# progress bar
from tqdm.notebook import tqdm
from tqdm import trange
from time import sleep
# h5py
import h5py
# os
import os
# glob
import glob
from scipy.integrate import odeint

class MathTextSciFormatter(mticker.Formatter):
    def __init__(self, fmt="%1.2e"):
        self.fmt = fmt
    def __call__(self, x, pos=None):
        s = self.fmt % x
        decimal_point = '.'
        positive_sign = '+'
        tup = s.split('e')
        significand = tup[0].rstrip(decimal_point)
        sign = tup[1][0].replace(positive_sign, '')
        exponent = tup[1][1:].lstrip('0')
        if exponent:
            exponent = '10^{%s%s}' % (sign, exponent)
        if significand and exponent:
            s =  r'%s{\times}%s' % (significand, exponent)
        else:
            s =  r'%s%s' % (significand, exponent)
        return "${}$".format(s)

def arraycenter(x):
    """
        returns centered array
    """
    return np.array([(x[i]+x[i+1])/2 for i in range(len(x)-1)])

def gauss3D(z,x,y,a0,W0,lbd):
    """
        standard Gaussian laser field 3D
    """
    zR = pi*W0**2/lbd
    return a0/sqrt(1+(z/zR)**2) * exp(-(x**2+y**2)/(W0**2*(1+(z/zR)**2)))


################################################################
lbd = 0.8e-6; #[\mu m]
w_p = 2*pi*c/(lbd) # laser frequency
ES = m_e**2*c**3/(e*hbar) # Schwinger field
Eref = m_e*c*w_p/e # normalizing field
aS = ES/Eref # normalized Schwinger field
lbdc = hbar/(m_e*c); # Compton wavelength

    
# "global variables"
adim_positron_yield = 20
gdim_electron_gamma = 50
photon_axis_min = 7 #log(energy[eV]), eg: 7 means 10MeV
photon_axis_max = 9 #log(energy[eV]), eg: 9 means 1GeV
photon_axis_dim = 101 #[]
phidim = 10 #[] length of phase vector


################################################################
# Amaro 2021 a0eff distributions (extended)
# Óscar Amaro and Marija Vranic 2021 New J. Phys. 23 115001, Tables 1 and 2
################################################################

def dNda_Short_2D(a,a0,R,W0,lbd):
    """
        Short beam (2D): L << zR
    """
    return W0/R/sqrt(pi)/a0 * (a/a0)**((W0/R)**2-1) / sqrt(log(a0/a))
dNda_Short_2D = np.vectorize(dNda_Short_2D)

def dNda_Short_3D(a, a0, R, W0, lbd, DeltaL, DeltaT):
    """
        Short beam (3D): L << zR
        Analytical distribution of particles in a0eff (dNb/da0eff):
        input:
            a - a0eff []
            Delta - transverse offset of scattering [\mu m]
    """
    zR = pi*W0**2/lbd; #[\mu m]
    W = W0*sqrt(1+(DeltaL/zR)**2)
    return np.heaviside(a0*W0/W-a,0) * (W/R)**2 / a * (a/a0 * W/W0)**((W/R)**2) * iv(0,2*DeltaT/R * (W/R) * sqrt(log(a0/a * W0/W ))) * exp(-(DeltaT/R)**2);
dNda_Short_3D = np.vectorize(dNda_Short_3D)

def dNda_Thin_2D(a,a0,L,W0,lbd):
    """
        Thin beam (2D): R << W0
    """
    zR = pi*W0**2/lbd; #[\mu m]
    az = a0/sqrt(sqrt(1+(L/4/zR)**2))
    return np.heaviside(a-az,0) * 2*a0**2/a**3/sqrt(1-(a/a0)**4) / (L/4/zR)
dNda_Thin_2D = np.vectorize(dNda_Thin_2D)

def dNda_Thin_3D(a,a0,L,W0,lbd,DeltaL=0):
    """
        Thin beam (3D): R << W0
        Analytical distribution of particles in a0eff (dNb/da0eff):
        input:
            a - a0eff []
            Delta - transverse offset of scattering [\mu m]
    """
    zR = pi*W0**2/lbd; #[\mu m]
    zm = -L/4+DeltaL
    zp = +L/4+DeltaL
    azm = a0/sqrt(1+(zm/zR)**2)
    azp = a0/sqrt(1+(zp/zR)**2)
    return 4*zR/L * a0**2/a**2 / sqrt(a0**2-a**2) * (np.heaviside(a-azp, 0)-np.sign(zm)*np.heaviside(a-azm, 0))
dNda_Thin_3D = np.vectorize(dNda_Thin_3D)

def dNda_Wide_3D(a,a0,L,W0,lbd=0.8,DeltaL=0):
    """
    Wide beam (3D): R >> W0
    Analytical distribution of particles in a0eff (dNb/da0eff):
    input:
        a - a0eff []
        Delta - transverse offset of scattering [\mu m]
    """
    zR = pi*W0**2/lbd; #[\mu m]
    zm = DeltaL - L/4
    zp = DeltaL + L/4
    azm = a0/sqrt(1+(zm/zR)**2)
    azp = a0/sqrt(1+(zp/zR)**2)
    return 2*pi*W0**2*zR/a*( \
                               (1/3/a * sqrt(a0**2-a**2) * (2+(a0/a)**2))*(np.heaviside(a-azp,0)-np.sign(zm)*np.heaviside(a-azm,0)) + \
                               (zp/zR * (1+(zp/zR)**2 /3))*np.heaviside(azp-a,0) - \
                               (zm/zR * (1+(zm/zR)**2 /3))*np.heaviside(azm-a,0) \
                              )
dNda_Wide_3D = np.vectorize(dNda_Wide_3D)


def get_adist(R, L, a0, W0, lbd, ebeam_geometry='None', Dpll=0, Dprpx=0, Dprpy=0):
    """
        [4] numerical distribution in a0eff
    """
    Nsmpl = int(1e5)
    if ebeam_geometry=='Wide':
        R = 50*W0;
    elif ebeam_geometry=='Thin':
        R = 0.01*W0;
    elif ebeam_geometry=='Short':
        L = 0.01*zR;

    xdist = R * rng.standard_normal(Nsmpl) #[\mu m] + Dprpy
    ydist = R * rng.standard_normal(Nsmpl) #[\mu m] + Dprpy
    zdist = (L/4 * rng.random(Nsmpl) - L/2) + Dpll #[\mu m]
    adist = gauss3D(zdist,xdist,ydist,a0,W0,lbd) #[]
    
    return adist


############################################
#### CRR electron energy distributions
############################################
    
def dNdg_Short(g,g0,a0,W0,R,lonrise,lbd=0.8):
    """
        dNdg particle distribution in final energy gF Short beam
    """
    zR = pi*W0**2/lbd; #[\mu m]
    Ka = 4.4199e-5 * (1/lbd) * (lonrise/50)**2 * (1/10)**2
    gmin = g0/(1+Ka*g0*a0**2)
    gmax = g0
    if g>=gmax or g<=gmin:
        return 0
    else:
        return 1 * (W0/R)**2 /2 * g0/g/(g0-g) * ((g0-g)/(Ka*g0*g*a0**2))**((W0/R)**2/2)

def dNdg_Thin(g,g0,a0,W0,L,lonrise,lbd=0.8):
    """
    dNdg particle distribution in energy gF
    """
    zR = pi*W0**2/lbd; #[\mu m]
    Ka = 4.4199e-5 * (1/lbd) * (lonrise/50)**2 * (1/10)**2
    gmin = g0/(1+Ka*g0*a0**2)
    gmax = g0/(1+Ka*g0*(a0/np.sqrt(1+(L/4/zR)**2))**2)
    if g>=gmax or g<=gmin:
        return 0
    else:
        return g0/g**2 / (g0/g-1)**1.5 * ( (1/gmin-1/g0)/(1/gmin-1/g) )**0.5 
    
def dNdg_Wide(g,g0,a0,W0,L,lonrise,lbd=0.8):
    """
    dNdg particle distribution in energy gF
    """
    zR = pi*W0**2/lbd; #[\mu m]
    Ka = 4.4199e-5 * (1/lbd) * (lonrise/50)**2 * (1/10)**2
    gmin = g0/(1+Ka*g0*a0**2)
    gmax = g0/(1+Ka*g0*(a0/np.sqrt(1+(L/4/zR)**2))**2)
    if g<=gmin:
        return 0
    elif g>gmin and g<=gmax:
        return np.sqrt(g0/gmin-g0/g)/(g0/g-1)**2.5 * g0/(6*g**2) * (2*(g0/g-1)+(g0/gmin-1))
    elif g>gmax and g<g0:
        return 1/1.15*g0/g / (g0-g) * (L*(L**2+16*zR**2))/(128*zR**3)
    else:
        return 0

############################################
#### nBW
############################################

def g(x):
    """
        g function
    """
    return ( 1 + 4.8*(1+x)*log(1+1.7*x) + 2.44*x**2 )**(-2/3)

def ppm(a0,w0,n,w):
    """
        eq (4) pair creation
    """
    return alpha * a0 * n * scR((2*a0*w0*w)/(m**2))

def scR(x):
    """
        eq (5) auxiliary functional of P±
    """
    return (0.453*kv(1/3,(4)/(3*x))**2) / (1+0.145*x**(1/4)*log(1+2.26*x)+0.330*x)

def phic(g0, a0, w0, n):
    """
        eq (7) critical phase (phic is implicit function of chic)
    """
    return sqrt( (2*pi**2*n**2/log(2))*log((2*g0*a0*w0)/(chic(g0,a0,w0,n)*m)) )

def chic(g0, a0, w0, n):
    """
        eq (8) critical chi (implicit)
    """
    def nonlin(chi):
        return chi**4 * g(chi)**2 - (72*log(2))/(pi**2*alpha**2)*((g0*w0)/(n*m))**2 * log((2*g0*a0*w0)/(m*chi))
    res = fsolve(nonlin, 1e-10)
    return res

def Omega(g0,a0,w0,n):
    """
        eq (13) radiated energy
    """
    return (sqrt(2*pi)*g0*m) * ( (2*log((2*g0*a0*w0)/(m*chic(g0,a0,w0,n))))/( 1+2*log((2*g0*a0*w0)/(m*chic(g0,a0,w0,n))) ) )**(1/2)

def chicrr(g0,a0,w0,n):
    """
        eq (14) critical chi with radiation reaction
    """
    res = chic(g0,a0,w0,n) / (1+Omega(g0,a0,w0,n)/(2*g0*m)) ;
    return res

def gf(g0,a0,w0,n):
    """
        eq (15) approximate final electron energy
    """
    Om = Omega(g0,a0,w0,n)
    return (2*g0*m-Om)/(2*g0*m+Om) * g0
    
def gphi(g0,a0,w0,n,phi):
    """
        eq (16) electron energy assuming "radiated power and χ as functions of phase are approximately Gaussian in form"
    """
    sigma = sigsq(g0,a0,w0,n)
    Om = Omega(g0,a0,w0,n)
    return gf(g0,a0,w0,n) + ( g0*Om/(2*g0*m+Om) ) * (1+erf((phi-phic(g0,a0,w0,n))/(sqrt(2)*sigma)))

def chiphi(g0,a0,w0,n,phi):
    """
        eq (17) chi in gaussian form
    """
    res = (chic(g0,a0,w0,n))/(1+(Omega(g0,a0,w0,n))/(2*g0*m)) * exp(-(phi-phic(g0,a0,w0,n))**2/(2*sigsq(g0,a0,w0,n)));
    return res

def sigsq(g0,a0,w0,n):
    """
        eq (18) sigma squared
    """
    return (pi**2*n**2)/(log(2)) * (1+2*log((2*g0*a0*w0)/(m*chic(g0,a0,w0,n))))**(-1)

def dNgdw(g0,a0,w0,n,w):
    """
        eq (20) photon spectra
    """
    chi0 = 2*g0*a0*w0/m;
    res = (sqrt(3)*pi*alpha*Fhe(g0,a0,w0,n))/(sqrt(2*log(2))) * (a0*n)/(sqrt(g0*m)) * (chicrr(g0,a0,w0,n)/chi0)/(sqrt(1+2*log(chi0/chic(g0,a0,w0,n)))) * (exp(-(2*w)/(3*chicrr(g0,a0,w0,n)*(g0*m-w))))/(sqrt(3*chicrr(g0,a0,w0,n)*(g0*m-w)+4*w));
    return res

def Fhe(g0,a0,w0,n):
    """
        eq (21) hard photon correction
    """
    arg = np.real((sqrt(2*log(2))*phic(g0,a0,w0,n)/(2*pi*n)));
    res = 0.5*(1-erf(arg));
    return res

def wc(g0, a0, w0, n):
    """
        eq (23) critical frequency
    """
    return (g0*m) * (sqrt( (2*chicrr(g0,a0,w0,n)*m)/(a0*g0*w0) ))/(1+sqrt( (2*chicrr(g0,a0,w0,n)*m)/(a0*g0*w0) ))

def Np(g0, a0, w0, n):
    """
        eq (24) positron yield
    """
    wcc = wc(g0,a0,w0,n);
    res = (3*sqrt(pi)*ppm(a0,w0,n,wcc)*chicrr(g0,a0,w0,n)/sqrt(2)) * ((g0*m-wcc)**2/(g0*m)) * dNgdw(g0,a0,w0,n,wcc);
    return res

def gp(g0, a0, w0, n):
    """
        eq (25) positron average energy
    """
    wcc = wc(g0,a0,w0,n);
    res = (wcc/(2*m))*(1+(pi**(3/2)*alpha/(3*sqrt(2*log(2))))*(n*a0**2*w0*wcc/m**2)*g(a0*w0*wcc/m**2))**(-1);
    return res

def TErber(chi):
    """
        eq (A4) Erber approximation of pair creation rate
    """
    return 0.16/chi * kv(1/3,4/(3*chi))**2

"""
Specific to reproduce plots in [1]
"""
def chicmod(g0, a0, w0, n):
    """
        critical chi mod (fig 3)
    """
    def nonlin(chi):
        return chi**4 - (72*log(2))/(pi**2*alpha**2)*((g0*w0)/(n*m))**2*log((2*g0*a0*w0)/(m*chi));
    return fsolve(nonlin, 1e-10);

def chiphimod(g0,a0,w0,n,phi):
    """
        eq (17) chi in gaussian form mod, assume Omega~0
    """
    return (chic(g0,a0,w0,n)) * exp(-(phi)**2/(2*sigsq(g0,a0,w0,n)));

def chiphimod2(g0,a0,w0,n,phi):
    """
        eq (16) and eq (6)
    """
    sigsqq = sigsq(g0,a0,w0,n);
    Omegaa = Omega(g0,a0,w0,n);
    phicc = phic(g0, a0, w0, n);
    def nonlin(chi):
        return chi**2 * g(chi) - 3*w0/(alpha*m)*( g0*Omegaa/(2*g0*m+Omegaa)*2/sqrt(2*pi*sigsqq) * exp(-(phi-phicc)**2/(2*sigsqq)) );
    return fsolve(nonlin, 1e-8);

def dNgdwmod(g0,a0,w0,n,w):
    """
        eq (20) photon spectra modified, assume chicrr->chic
    """
    chi0 = 2*g0*a0*w0/m;
    return (sqrt(3)*pi*alpha*Fhe(g0,a0,w0,n))/(sqrt(2*log(2))) * (a0*n)/(sqrt(g0*m)) * (chicrr(g0,a0,w0,n)/chi0)/(sqrt(1+2*log(chi0/chic(g0,a0,w0,n)))) * (exp(-(2*w)/(3*chic(g0,a0,w0,n)*(g0*m-w))))/(sqrt(3*chic(g0,a0,w0,n)*(g0*m-w)+4*w));

def Npmod(g0, a0, w0, n):
    """
        eq (24) positron yield without radiation reaction
    """
    wcc = wcmod(g0,a0,w0,n);
    return (3*sqrt(pi)*ppm(a0,w0,n,wcc)*chic(g0,a0,w0,n)/sqrt(2)) * ((g0*m-wcc)**2/(g0*m)) * dNgdw(g0,a0,w0,n,wcc);

def phicmod(g0, a0, w0, n):
    """
        critical phase mod (fig 3)
    """
    return sqrt( (2*pi**2*n**2/log(2))*log((2*g0*a0*w0)/(chicmod(g0,a0,w0,n)*m)) );

def wcmod(g0, a0, w0, n):
    """
        eq (23) critical frequenc, no rrr
    """
    return (g0*m) * (sqrt( (2*chic(g0,a0,w0,n)*m)/(a0*g0*w0) ))/(1+sqrt( (2*chic(g0,a0,w0,n)*m)/(a0*g0*w0) ));

############################################


#### nCS PW
#### Neil DiPiazza 2013

def get_PW_gF(g0, a0, lbd, tau0):
    """
        g0[] initial electron energy
        a0[] laser vector potential
        lbd[m] laser central wavelength
        tau0[s] pulse duration as in Vranic2014PRL
        theta[] defined as theta=0 is frontal collision

        gf[] final electron energy
    """
    lbd = 0.8e-6;
    theta = 0;
    eta = 0.375; # sin^2 temporal profile

    omega0 = 2*pi*c/lbd; #[1/s] laser frequency

    #[] CRR k factor from equation 3
    k = 1/(4*pi*epsilon_0) * (1 - cos(pi-theta))**2 * eta/3 * (e**2 * omega0**2)/(m_e * c**3) * a0**2 * tau0;

    return g0 / (1 + k * g0)
    
def get_PW_gF_gaunt(g0, a0, lbd, tau0):
    """
        g0[] initial electron energy
        a0[] laser vector potential
        lbd[m] laser central wavelength
        tau0[s] pulse duration as in Vranic2014PRL
        theta[] defined as theta=0 is frontal collision

        gf[] final electron energy, corrected with gaunt factor
    """
    lbd = 0.8e-6;
    theta = 0;
    eta = 0.375; # sin^2 temporal profile

    omega0 = 2*pi*c/lbd; #[1/s] laser frequency

    #[] CRR k factor from equation 3
    k = 1/(4*pi*epsilon_0) * (1 - cos(pi-theta))**2 * eta/3 * (e**2 * omega0**2)/(m_e * c**3) * a0**2 * tau0;

    w_p = 2*pi*c/(lbd) # laser frequency
    ES = m_e**2*c**3/(e*hbar) # Schwinger field
    Eref = m_e*c*w_p/e # normalizing field
    aS = ES/Eref # normalized Schwinger field
    chi = 2*g0*a0/aS
    
    return g0 / (1 + gaunt(chi) * k * g0)
    
############################################

#### Vranic2016NJP
#### thF

def get_PW_sigF(g0,a0,lbd,tau):
    """
        [3] Vranic2016NJP eq 17 energy spread limit LP
        "It is worth noting that the result presented in equation (17) does not depend on the laser polarisation, but solely on intensity and duration."
    """
    I22 = 1e-4 * (a0/0.855/lbd)**2;
    sigF = sqrt( 1.455e-4 * sqrt(I22) * g0**3/(1+6.12e-5*g0*I22*tau)**3 )
    return sigF

def get_PW_thF(g0,a0,lbd,tau):
    """
        [3] Vranic2016NJP eq ? energy spread limit
        g0[]
        a0[]
        lbd[\mu m]
        tau[fs]
    """
    gF = get_PW_gF(g0, a0, lbd, tau*1e-15);
    sigF = get_PW_sigF(g0, a0, lbd, tau);
    # Vranic2016NJP net beam divergence 
    thF = 0.225 * a0 * sigF/gF**2
    return thF


######################

def gaunt(chi):
    """
        gaunt factor
        
        input: chi
        output: gaunt(chi)
    """
    return (9*sqrt(3))/(8*pi) * quad( lambda u: ((2 * u**2 * kv(5/3, u))/(2 + 3 *chi*u)**2 + (
    36 * chi**2 * u**3 * kv(2/3, u) )/(2 + 3*chi*u)**4), 0, np.inf)[0]
gaunt = np.vectorize(gaunt)

def d2Pdtdw_q(gg, ge, a0):
    """
        quantum synchrotron spectrum
        
        gg - gamma of photon
        gm - gamma of electron
        a0 - local a0
    """
    w_p = 2*pi*c/(0.8e-6) # laser frequency
    ES = m_e**2*c**3/(e*hbar) # Schwinger field
    Eref = m_e*c*w_p/e # normalizing field
    aS = ES/Eref # normalized Schwinger field
    
    eta = 2*ge*a0/aS
    chi = 2*gg*a0/aS
    xi = chi/eta;

    if (xi>1e-4) and (xi<1):
        nu = 2*xi/(3*eta*(1-xi)); #chitil
        int53 = quad(lambda y: kv(5/3,y), nu, np.inf)[0]
        Gtil = sqrt(3)/(2*pi) * xi * (3/2*chi*nu*kv(2/3,nu)+int53);
        return Gtil/ge/gg;
    else:
        return 0
d2Pdtdw_q = np.vectorize(d2Pdtdw_q)


def a_of_phi(a0, phi, n):
    """
        laser field value as function of phase
    """
    return np.abs( a0 * sin(phi/n)**2 * sin(phi) ) * np.heaviside(n*pi-phi,0)

def dgdt(g, t, a0, T, n):
    """
        computes semi-classical Larmor Power energy loss (including gaunt factor)
    """

    phi = t*2/pi

    a = a_of_phi(a0, phi, n) 
    eta = 2*g*a/aS
    
    dpdt = -2/3 * alpha * 1/lbdc * m_e*c**2 * eta**2

    return gaunt(eta) * dpdt/(m_e*c) * T /4.9 # 
    
def goft(a0, g0, tau_osiris):
    """
        solves ODE for average electron energy as a function of laser phase
    """
    
    t_osiris = np.linspace(0, tau_osiris, 201) # 70
    
    tau = tau_osiris/w_p; #[fs]
    T = lbd/c; #[s]
    n = tau/T; #[]
    
    sol = odeint(dgdt, g0, t_osiris, args=(a0, T, n) )

    return t_osiris, sol

def photon_spectrum(gglst, g0, a0, tau_fs):
    """
        computes integrated photon spectrum accounting for semi-classical evolution of the average electron energy along the laser pulse
    """
    
    lbd = 0.8e-6; # laser wavelength
    T = lbd/c;
    n = tau_fs*1e-15/T; # number of laser cycles
    tau_osiris = tau_fs * (w_p*1e-15) # laser pulse duration in osiris units

    # solve ODE for average electron energy
    t_osiris, g_phase = goft(a0, g0, tau_osiris)
    g_phase = g_phase.flatten()

    alst = a_of_phi(a0, t_osiris, n) # get laser field value in phase
    
    dNdgg = np.zeros_like(gglst)
    
    # integrate photon spectrum
    for i in range(len(t_osiris)):
        if alst[i]>1e-3:
            dNdgg = dNdgg + d2Pdtdw_q(gglst, g_phase[i], alst[i]) # g0/(1+g0*9.8492e-05*(a0/12)**2 / 2)
    return dNdgg
    
