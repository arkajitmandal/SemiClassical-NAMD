import numpy as np
import sys, os
from numpy import pi as π
from numpy.random import random as ℜ

class Bunch:
    def __init__(self, **kwds):
        self.__dict__.update(kwds)

# Initialization of the mapping Variables
def initMapping(Nstates, initState = 0, stype = "square"):
    qF = np.zeros((Nstates))
    pF = np.zeros((Nstates))
    if (stype == "square" or stype == "□"):
        γ = (np.sqrt(3.0) -1)/2
        η = 2 * γ * ℜ(Nstates)  
        θ = 2 * π * ℜ(Nstates)

    if (stype == "triangle" or stype == "Δ"):
        η = np.zeros(Nstates)
        θ = 2 * π * ℜ(Nstates)
        # For initial State
        while (True):
            η[initState] = ℜ()
            if ( 1 - η[initState] >= ℜ() ):
                break
        
        # For other States
        for i in range(Nstates):
            if (i != initState):
                η[i] = ℜ() * ( 1 - η[initState] )
    
    η[initState] += 1.0
    qF =  np.sqrt( 2 * η ) * np.cos(θ)
    pF = -np.sqrt( 2 * η ) * np.sin(θ)
    return qF, pF 

def Uqm(dat, dt):
    qFin, pFin = dat.qF * 1.0, dat.pF * 1.0  # Store input position and momentum for verlet propogation
    # Store initial array containing sums to use at second derivative step
    qF, pF  = dat.qF * 1.0, dat.pF * 1.0
    VMatxqF =  dat.Hij @ qFin #np.array([np.sum(VMat[k,:] * qFin[:]) for k in range(NStates)])

    # Update momenta using input positions (first-order in dt)
    pF -= 0.5 * dt * VMatxqF  # VMat @ qFin  
    # Now update positions with input momenta (first-order in dt)
    qF += dt * dat.Hij @ pFin  
    # Update positions to second order in dt
    qF -=  (dt**2/2.0) * dat.Hij @ VMatxqF
    #-----------------------------------------------------------------------------
    # Update momenta using output positions (first-order in dt)
    pF -= 0.5 * dt * dat.Hij @ qF  
    dat.qF, dat.pF = qF * 1.0, pF * 1.0 
    return dat

def Force(dat):
    γ = dat.γ
    dH = dat.dHij #dHel(R) Nxnxn Matrix, N = Nuclear DOF, n = NStates 
    dH0 = dat.dH0
    qF, pF =  dat.qF * 1.0, dat.pF * 1.0
    # F = np.zeros((len(dat.R)))
    F = -dH0
    for i in range(len(qF)):
        F -= 0.5 * dH[i,i,:] * ( qF[i] ** 2 + pF[i] ** 2 - 2 * γ)
        for j in range(i+1, len(qF)):
            F -= dH[i,j,:] * ( qF[i] * qF[j] + pF[i] * pF[j])
    return F


def popSquare(dat):
    
    qF, pF = dat.qF * 1.0, dat.pF * 1.0 
    N = len(qF)
    η = 0.5 * ( qF**2 + pF**2 )
    γ = dat.γ
    ρij = np.outer(qF + 1j * pF, qF - 1j * pF) * 0 # have to recheck coherences
    ρij[np.diag_indices(N)] = np.ones((N))

    # Inspired from Braden's (Braden Weight) Implementation
    # Check his Package : https://github.com/bradenmweight/QuantumDynamicsMethodsSuite
    for i in range(N):
        for j in range(N):
            if ( η[j] - (i == j) < 0.0 or η[j] - (i == j) > 2 * γ ):
                ρij[i,i] = 0

    return ρij

def popTriangle(dat):
    qF, pF = dat.qF * 1.0, dat.pF * 1.0 
    N = len(qF)
    η = 0.5 * ( qF**2 + pF**2 )
    γ = dat.γ
    ρij = np.outer(qF + 1j * pF, qF - 1j * pF) * 0 # have to recheck coherences
    ρij[np.diag_indices(N)] = np.ones((N))
    # Inspired from Braden's (Braden Weight) Implementation
    # Check his Package : https://github.com/bradenmweight/QuantumDynamicsMethodsSuite
    for i in range(N):
        for j in range(N):
                if ( (i == j and η[j] < 1.0) or (i != j and η[j] >= 1.0) ):
                    ρij[i,i] = 0
    return ρij

def runTraj(parameters):
    #------- Seed --------------------
    try:
        np.random.seed(parameters.SEED)
    except:
        pass
    #------------------------------------
    ## Parameters -------------
    NSteps = parameters.NSteps
    NTraj = parameters.NTraj
    NStates = parameters.NStates
    initState = parameters.initState # intial state
    stype = parameters.stype
    nskip = parameters.nskip
    #---------------------------
    if NSteps%nskip == 0:
        pl = 0
    else :
        pl = 1
    rho_ensemble = np.zeros((NStates,NStates,NSteps//nskip + pl), dtype=complex)
    # Ensemble
    for itraj in range(NTraj): 
        # Trajectory data
        dat = Bunch(param =  parameters )
        dat.R, dat.P = parameters.initR()

        # set propagator
        dat.force = Force
        dat.Uqm = Uqm
        vv  = parameters.vv

        # Call function to initialize mapping variables
        dat.qF, dat.pF = initMapping(NStates, initState, stype) 
        if stype == "square" or stype == "□":
            dat.γ = (np.sqrt(3.0) - 1.0)/2.0
            pop = popSquare
        if stype == "triangle" or stype == "Δ":
            dat.γ = 1/3.0 
            pop = popTriangle
        #----- Initial QM --------
        dat.Hij  = parameters.Hel(dat.R)
        dat.dHij = parameters.dHel(dat.R)
        dat.dH0  = parameters.dHel0(dat.R)
        #----------------------------
        iskip = 0 # please modify
        for i in range(NSteps): # One trajectory
            #------- ESTIMATORS-------------------------------------
            if (i % nskip == 0):
                rho_ensemble[:,:,iskip] += pop(dat)
                iskip += 1
            #-------------------------------------------------------
            dat = vv(dat)

    return rho_ensemble
