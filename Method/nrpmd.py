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