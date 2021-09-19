import numpy as np
import sys, os

import random

class Bunch:
    def __init__(self, **kwds):
        self.__dict__.update(kwds)


#===========initializing mapping variables==========
def initMap(param):
    """
    initialize Mapping variables q and p
    dimensionality q[nstate,nb] so do p
    """
    q = np.zeros((param.nstate,param.nb))
    p = np.zeros((param.nstate,param.nb))
    i0 = param.initState
    for i in range(param.nstate):
       for ib in range(param.nb):
            η = np.sqrt(1 + 2*(i==i0))
            θ = random() * 2 * np.pi
            q[i,ib] = η * np.cos(θ) 
            p[i,ib] = η * np.sin(θ) 
    return p,q
#============================================================

#===============intializing nuclear positions with Monte-carlo sampling===========
def monte_carlo(param, steps = 3000, dR = 0.5):
    R = np.zeros((param.ndof,param.nb))
    ndof, nb = R.shape
    βn = param.beta/nb 
    Hel0 = param.initHel0
    #Monte carlo loop
    for i in range(steps):
        rDof  = np.random.choice(range(ndof))
        rBead = np.random.choice(range(nb))

        # Energy before Move
        En0 = ringPolymer(R[rDof,:], param) + Hel0(R[rDof,rBead]) 

         # update a bead -------------
        dR0 = dR * (random() - 0.5)
        R[rDof, rBead] += dR0
        #----------------------------
        # Energy after Move
        En1 = ringPolymer(R[rDof,:], param) + Hel0(R[rDof,rBead]) 

        # Probality of MC
        Pr = np.min([1.0,np.exp(-βn * (En1-En0))])
        # a Random number
        r = random()
        # Accepted
        if r < Pr:
            pass
        # Rejected
        else:
            R[rDof, rBead] -= dR0
    return R


def ringPolymer(R,param):
    """
    Compute Ringpolymer Energy
    E = ∑ 0.5  (m nb^2/β^2)  (Ri-Ri+1)^2
    """
    nb = param.nb
    βn = (param.beta)/nb 
    Ω  = (1 / βn)  
    M  = param.M
    E = 0
    for k in range(-1,nb-1):
        E+= 0.5 * M * Ω**2 * (R[k] - R[k+1])**2
    return E
#==========================================================

#=============initializing nuclear momentum================        
def initP(param):
    nb, ndof = param.nb , param.ndof
    sigp = (param.M * param.nb/param.beta)**0.5
    return np.random.normal(size = (ndof, nb )) * sigp
#==========================================================

#========Calculation of non-adiabatic force term=======
def Force(R,q,p,dHij,dH0):
    """
    Nuclear Force
    dH => grad of H matrix element
          must NOT include state independent
          part as well
    - 0.5 ∑ dHij (qi * qj + pi * pj - dij) 
    """      

    F = np.zeros((R.shape)) # ndof nbead

    #----- state independent part-----------
    F[:] = -dH0 
    
    #------- state dependent part------------
    qiqj = np.outer(q,q)
    pipj = np.outer(p,p)
    γ = np.identity(len(q))
    rhoij = 0.5 * ( qiqj + pipj - γ) 
    #------ total force term--------------- 
    for i in range(len(F)):
        F[i] -= np.sum(rhoij * dHij[:,:,i])
    return F
#======================================================
#============ normal mode transformation==============
def nm_t(P,R,param):

    """
    normal mode transformation = fourier transform from bead
    representation to mode representation

    see Eq(36) and (37) at J. Chem. Phys. 154, 124124 (2021)
    """  

    nb = param.nb
    ndof = param.ndof
    lb_n = param.lb_n
    ub_n = param.ub_n
    
    cmat = np.zeros((nb,nb))    # normal mode transformation matrix
    pibyn = math.acos(-1.0)/nb


    P_norm = np.zeros((ndof,nb)) #normal modes for momenta
    Q_norm = np.zeros((ndof,nb)) #normal modes for position

    for j in range(nb):
        for i in range(nb):
            l=(i-int(nb/2))
            if l==0:
                cmat[j,l] = 1.0
            elif l >= lb_n and l<0:
                cmat[j,l] = np.sqrt(2.0)*np.sin(2.0*pibyn*(j+1)*l)
            elif l > 0 and l <= ub_n:
                cmat[j,l] = np.sqrt(2.0)*np.cos(2.0*pibyn*(j+1)*l)



    pnew = np.zeros((ndof,nb))
    qnew = np.zeros((ndof,nb))

    for j in range(nb):
        for i in range(nb):
            l=(i-int(nb/2))
            for m in range(ndof):
                pnew[m,l]+= P[m,j]*cmat[j,l]
                qnew[m,l]+= R[m,j]*cmat[j,l]


    P_norm = pnew/nb
    Q_norm = qnew/nb

    return P_norm, Q_norm
#==========================================================

#========== back normal mode transformation================
def back_nm_t(P_norm,Q_norm,param):

    """
    back normal mode transformation = fourier transform from mode
    representation to bead representation

    see Eq(36) and (37) at J. Chem. Phys. 154, 124124 (2021)
    """  

    nb = param.nb
    ndof = param.ndof
    lb_n = param.lb_n
    ub_n = param.ub_n

    cmat = np.zeros((nb,nb)) # transformation matrices
    pibyn = math.acos(-1.0)/nb

    P = np.zeros((ndof,nb)) # bead representation momenta
    R = np.zeros((ndof,nb)) # bead representation position


    for j in range(nb):
        for i in range(nb):
            l=(i-int(nb/2))
            if l==0:
                cmat[j,l] = 1.0
            elif l >= lb_n and l<0:
                cmat[j,l] = np.sqrt(2.0)*np.sin(2.0*pibyn*(j+1)*l)
            elif l > 0 and l <= ub_n:
                cmat[j,l] = np.sqrt(2.0)*np.cos(2.0*pibyn*(j+1)*l)

    pnew = np.zeros((ndof,nb))
    qnew = np.zeros((ndof,nb))    

    for j in range(nb):
        for i in range(nb):
            l=(i-int(nb/2))
            for m in range(ndof):
                pnew[m,j]+= P_norm[m,l]*cmat[j,l]
                qnew[m,j]+= Q_norm[m,l]*cmat[j,l]

    P = pnew
    R = qnew

    return P,R
#==================================================================
#======== polynomials for free ring-polymer propagation===============
def ring(param):
    
    """
    see Ceriotti, Parinello JCP 2010 
    """

    nb = param.nb
    ndof = param.ndof
    dt = param.dtN
    M = param.M
    beta = param.beta
    lb_n = param.lb_n
    ub_n = param.ub_n

    poly = np.zeros((4,nb))
    
    #Monodromy matrix for free ring-polymer update

    betan = beta/nb
    twown = 2.0/(betan)
    pibyn = math.acos(-1.0)/nb

    for i in range(nb):
        l=(i-int(nb/2))

        if l==0:
            poly[0,0] = 1.0
            poly[1,0] = 0.0
            poly[2,0] = dt/M
            poly[3,0] = 1.0
            
        elif l >= lb_n and l<0:
            poly[0,l]=np.cos(twown*np.sin(l*pibyn)*dt)
            poly[1,l]=-twown*np.sin(l*pibyn)*M*np.sin(twown*np.sin(l*pibyn)*dt)
            poly[2,l]=np.sin(twown*np.sin(l*pibyn)*dt)/(twown*np.sin(l*pibyn)*M)
            poly[3,l]=np.cos(twown*np.sin(l*pibyn)*dt)
            
        elif l > 0 and l <= ub_n:
            poly[0,l]=np.cos(twown*np.sin(l*pibyn)*dt)
            poly[1,l]=-twown*np.sin(l*pibyn)*M*np.sin(twown*np.sin(l*pibyn)*dt)
            poly[2,l]=np.sin(twown*np.sin(l*pibyn)*dt)/(twown*np.sin(l*pibyn)*M)
            poly[3,l]=np.cos(twown*np.sin(l*pibyn)*dt)

    return poly

#==============================================================================
#=========== free ring-polymer propagation ============================
def freerp(P,R,param):

    """
    see Eq. 22 to 24 of Ceriotti, Parinello JCP 2010 

    """
    nb = param.nb
    ndof = param.ndof

    P_norm = np.zeros((ndof,nb)) #normal modes for momenta
    Q_norm = np.zeros((ndof,nb)) #normal modes for position
    poly = np.zeros((4,nb))

    poly = ring(param)  # calling ring() function to obtain the polynomials

    P_norm, Q_norm = nm_t(P,R,param) #normal mode obtained

    for k in range(nb):
        for j in range(ndof):
            l=(k-int(nb/2))
            
            pjknew = P_norm[j,l]*poly[0,l] + Q_norm[j,l]*poly[1,l]
            Q_norm[j,l] = P_norm[j,l]*poly[2,l] + Q_norm[j,l]*poly[3,l]
            P_norm[j,l] = pjknew


    P,R = back_nm_t(P_norm,Q_norm,param) # from normal mode to bead

    return P,R
#=======================================================================
