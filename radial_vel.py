#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np 
from astropy import constants
pi = np.pi
Ms = constants.M_sun.value
c = constants.c.value
G = constants.G.value
Rs = constants.R_sun.value
Mj = constants.M_jup.value
AU = constants.au.value
Me = constants.M_earth.value
angle_conv = np.pi/180/3600/365.25 #arcsecs/yr -> rad/day

def RV_Model(par,times,fixpar,plus=False): 

    '''
    K - semi amplitude of the planet
    Ks- semi amplitude of the star
    omega - argument of peristron in radian
    T_0 - Time of periastron
    plus: False, if True is uses RV_model_plus function to include precession: fixpar needs to include: omegadot M0, & M1 values.

    '''
# Using flux-weighted=False so exact value radius_1, radius_2, sbratio not important
    Kp, Ks, theta, offset = par
    if plus:
        try:
            period,T0p,ecc,omega,omegadot,M0,M1 = fixpar
        except:
            print('Please provide primary mass estimate in fixpar [period,T_0,ecc,omega,omegadot,M0,M1].')
    else:
        period,T0p,ecc,omega = fixpar
    
    #add phase shift theta
    T0p = T0p + theta*period
    
    if plus:
        model = keplerian_plusplus(times, period, Kp, ecc, omega-pi,omegadot, T0p, offset,M0=M0,R0=0,M1=M1,relativistic_correction=True,tidal_correction=False)
        star  = keplerian_plusplus(times, period, Ks, ecc, omega, omegadot, T0p, offset,M0=M0,R0=0,M1=M1,relativistic_correction=True,tidal_correction=False)

    else:
        model = keplerian(times, period, Kp, ecc, omega-pi, T0p, offset)
        star  = keplerian(times, period, Ks, ecc, omega, T0p, offset)

    return model,star

def RV_Model_plusplus(par,times,fixpar): 

    '''
    Modified version that uses keplerian_plusplus
    K - semiamplitude of planet to be explored
    Ks- known semiamplitude of the star
    omega - argument of peristron in radian
    omdot - change of argument of periastron with with
    T_0 - Time of periastron

    '''
# Using flux-weighted=False so exact value radius_1, radius_2, sbratio not important
    K, Ks, theta, offset = par
    try:
        period,T_0,ecc,omega,omegadot,M0,M1 = fixpar
    except:
        print('Please provide primary mass estimate in fixpar [period,T_0,ecc,omega,omegadot,M0,M1].')

    #add phase shift theta
    T_0 = T_0 + theta*period
    
    model = keplerian_plusplus(times, period, K, ecc, omega-np.pi,omegadot, T_0, offset,M0=M0,R0=0,M1=M1,relativistic_correction=True,tidal_correction=False)
    star  = keplerian_plusplus(times, period, Ks, ecc, omega, omegadot, T_0, offset,M0=M0,R0=0,M1=M1,relativistic_correction=True,tidal_correction=False)

    return model,star

"""
Radial velocity functions

Created on Wed Nov 16 15:23:29 2022

author: Thomas Baycroft
"""

def keplerian(time, p, k, ecc, omega, t0, vsys):
    '''
    keplerian function given:
        time: array of times
        p: Period (float or array of floats)
        k:semi-amplitude (float or array of floats)
        ecc:eccentricity (float or array of floats)
        omega: argument of pericentre (float or array of floats)
        t0: time of pericentre passage (float or array of floats)
        vsys: systemic velocity (float)
    '''
    vel = np.zeros_like(time)
    p, k, ecc, omega, t0 = np.atleast_1d(p, k, ecc, omega, t0)

    with np.errstate(divide='raise'):
        for i in range(p.size):
            M = 2.*np.pi * (time-t0[i]) / p[i]
            E = ecc_anomaly(M, ecc[i])
            nu = true_anomaly(E, ecc[i])
            vel += k[i] * (np.cos(omega[i]+nu)+ ecc[i]*np.cos(omega[i]))
        vel += vsys
    return vel

def true_anomaly(E, e):
    '''
    calculate true anomaly from the eccentric anomaly
    '''
    return 2. * np.arctan( np.sqrt((1.+e)/(1.-e)) * np.tan(E/2.))

def ecc_anomaly(M, e):
    M = np.atleast_1d(M)
    E=M			#startvalues
    E_n=M		#startvalues
    i = 0
    while 1==1:
        i+=1 
        E_n = E
        E = E_n - (E_n-e*np.sin(E_n)-M)/(1.-e*np.cos(E_n))
        if max(abs(E*180/np.pi-E_n*180/np.pi))<=1E-7:
            break
        if i ==200:
            # no convergence, return the best estimate
            break
    return E

def keplerian_plus(time, p, k, ecc, omega, omdot, t0, vsys):
    '''
    keplerian function with a linear apsidal precession term added given:
        time: array of times
        p: Period (float or array of floats)
        k:semi-amplitude (float or array of floats)
        ecc:eccentricity (float or array of floats)
        omega: argument of pericentre (float or array of floats)
        omdot: rate of change of omega (float or array of floats)
        t0: time of pericentre passage (float or array of floats)
        vsys: systemic velocity (float)
    '''
    vel = np.zeros_like(time)
    p, k, ecc, omega, omdot, t0 = np.atleast_1d(p, k, ecc, omega, omdot, t0)

    with np.errstate(divide='raise'):
        for i in range(p.size):
            p[i] = Period_change(p[i], ecc[i], omdot[i])
            M = 2.*pi * (time-t0[i]) / p[i]
            E = ecc_anomaly(M, ecc[i])
            nu = true_anomaly(E, ecc[i])
            wdot = omdot[i]*angle_conv #convert from arcsec per year to radians per day
            w = omega[i]+wdot*(time-t0[i])
            vel += k[i] * (np.cos(nu+w) + ecc[i]*np.cos(w))
        vel += vsys 
        
def keplerian_plusplus(time, p, k, ecc, omega, omdot, t0, vsys, M0, R0, M1=None, relativistic_correction=True, tidal_correction=True):
    '''
    RV keplerian function with a linear apsidal precession term added and the option to include relativistic and tidal corrections, given:
        time: array of times
        p: Period (float or array of floats)
        k:semi-amplitude (float or array of floats)
        ecc:eccentricity (float or array of floats)
        omega: argument of pericentre (float or array of floats)
        omdot: rate of change of omega (float or array of floats)
        t0: time of pericentre passage (float or array of floats)
        vsys: systemic velocity (float)
        M0, R0: Mass and Radius of primary body (on which RV data is obtained) (floats)
        M1: Mass of secondary body (float)
        relativistic_correction: flag to turn on relativistic correction (bool)
        tidal_correction: flag to turn on Tidal correction (bool)
    '''
    vel = np.zeros_like(time)
    p, k, ecc, omega, omdot, t0 = np.atleast_1d(p, k, ecc, omega, omdot, t0)

    with np.errstate(divide='raise'):
        for i in range(p.size):
            p[i] = Period_change(p[i], ecc[i], omdot[i])
            M = 2.*pi * (time-t0[i]) / p[i]
            E = ecc_anomaly(M, ecc[i])
            nu = true_anomaly(E, ecc[i])
            wdot = omdot[i]*angle_conv #convert from arcsec per year to radians per day
            w = omega[i]+wdot*(time-t0[i])
            vel += k[i] * (np.cos(nu+w) + ecc[i]*np.cos(w))
            extra_vel = post_newtonian(k[i], nu, ecc[i], w, p[i], M0, R0, M1, relativistic_correction, tidal_correction)

        vel += vsys + extra_vel
    return vel

def post_newtonian(K0, nu, ecc, w, P, M0, R0, M1, relativistic_correction, tidal_correction):
    '''
    Calculates the post-newtonian contributions as in Baycroft et. al 2023
    '''
    sini = 1
    v = 0.0
    if relativistic_correction or tidal_correction:
        if M1 == None:
            K1 = get_K2(K0,M0,P,ecc)
            M1 = M0*K0/K1
        else:
            K1 = K0*M0/M1
    if tidal_correction:
        if R0 == None:
            R0 = M0**0.8
        delta_v_tide = v_tide(R0, M0, P, nu, w, M1, "M_s")
        v += delta_v_tide
    if relativistic_correction:
        cl = c
        delta_LT = K0**2*np.sin(nu+w)**2*(1+ecc*np.cos(nu))/cl
        delta_TD = K0**2*(1 + ecc*np.cos(nu) - (1-ecc**2)/2)/(cl*sini**2)
        delta_GR = K0*(K0+K1)*(1+ecc*np.cos(nu))/(cl*sini**2)
        v += delta_LT + delta_TD + delta_GR
    
    return v

def get_K2(K1, M, P, ecc):
    '''
    Calculate the semi-amplitude that the primary body would induce ont he secondary body, using Newton-raphson
    '''
    M_est = mass(K1,P,ecc,M*Ms)
    k = semiamp(M,M_est,P,ecc)
    while abs(k-K1)>50:
        M_est -= f_M(K1,M,M_est,P,ecc)/f_dash_M(K1,M,M_est,P,ecc)
        k = semiamp(M,M_est,P,ecc)

    K2 = K1*M/M_est
    
    return K2

def f_M(K1, M0, M1, P, ecc):
    '''
    funtion for Newton-Raphson
    '''
    return semiamp(M0, M1, P, ecc) - K1
    
def f_dash_M(K, M0, M1, P, ecc):
    '''
    funtional derivative for Newton-Raphson (M0 and M1 in solar mass, P in days returns in m/s)
    '''

    MjMs = 1047.5655
    m1 = M1*MjMs
    m01 = M0 + M1
    
    f_dash_1 = 28.4329*(1-ecc**2)**(-1/2)*MjMs*m01**(-2/3)*(P/365)**(-1/3)
    f_dash_2 = -2/3*28.4329*(1-ecc**2)**(-1/2)*m1*m01**(-5/3)*(P/365)**(-1/3)
    
    return f_dash_1 - f_dash_2


def mass(K,P,e=0,M0=Ms):
    '''
    Mass from semi-amplitude period and eccentricity (Lovis & Fischer)
    '''

    return ((K/28.4329)*(1-e**2)**(1/2)*(M0/Ms)**(2/3)*(P/365)**(1/3))*Mj/Ms

def v_tide(R_star, M_star, P, f, w, M_p, M_p_unit):
    '''
    https://arxiv.org/pdf/1107.6005.pdf Equation 17

    Takes:
        R_star = Radius of star (solar radius) float
        M_star = Mass of star (solar masses) float
        P = Period (days) float
        f = Mean motion
        phi_0 = 
        M_p = Mass of planet (or orbiting body) (M_j) float
        M_p_unit = Unit "M_j"/"M_s" str

    returns:
        v_tide = Velocity of tide (m/s) float
        
    '''
    
    phi_0 = np.pi/2 - w #phi_0 defined relative to argument of pericenter
    
    M_ratio = Ms/Mj
    if M_p_unit == "M_s":
        M_p *= M_ratio
        
    v_tide = ((((1.13 * M_p)/(M_star * (M_star + (M_p/M_ratio)))) * (R_star**4)) / P**3)*np.sin(2*(f-phi_0))

    return v_tide


def semiamp(M0,M1,P,e):
    '''
    
    from Lovis and Fischer 
    Parameters
    ----------
    M0 : float
        Body 0 mass (Ms)
    M1 : float
        Body 1 mass (Ms)
    P : float
        Orbital Period (days)
    e : float
        eccentricity

    Returns
    -------
    float
        semiamplitude of RV signal on body 0 due to body 1 (m/s)

    '''
    m1 = M1*1047.5655
    m01 = M0 + M1
    return 28.4329*(1-e**2)**(-1/2)*(m1)*(m01)**(-2/3)*(P/365)**(-1/3)
        
def Period_change(P2,e,wdot):
    '''
    Gives the anomalistic period from the observed period

    Assume spread evenly/circular orbit
    '''
    return P2*(1+wdot/(3600*180*365*2)*P2)