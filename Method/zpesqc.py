import numpy as np
import sys, os
from numpy import pi as π
from numpy.random import random as ℜ

class Bunch:
    def __init__(self, **kwds):
        self.__dict__.update(kwds)

# Initialization of the mapping Variables
def initMapping(Nstates, initState = 0, stype = "square"):
    qF  = np.zeros((Nstates))
    pF  = np.zeros((Nstates))
    γ0  = np.zeros((Nstates)) # Adjusted ZPE

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

    for i in range(Nstates):
        γ0[i] = η[i] - 1 * (i == initState)

    return qF, pF, γ0

def Umap(qF, pF, dt, VMat):
    qFin, pFin = qF * 1.0, pF * 1.0  # Store input position and momentum for verlet propogation
    # Store initial array containing sums to use at second derivative step

    VMatxqF =  VMat @ qFin #np.array([np.sum(VMat[k,:] * qFin[:]) for k in range(NStates)])

    # Update momenta using input positions (first-order in dt)
    pF -= 0.5 * dt * VMatxqF  # VMat @ qFin  
    # Now update positions with input momenta (first-order in dt)
    qF += dt * VMat @ pFin  
    # Update positions to second order in dt
    qF -=  (dt**2/2.0) * VMat @ VMatxqF
       #-----------------------------------------------------------------------------
    # Update momenta using output positions (first-order in dt)
    pF -= 0.5 * dt * VMat @ qF  

    return qF, pF 

def Force(dat):
    γ0 = dat.γ0
    dH = dat.dHij #dHel(R) Nxnxn Matrix, N = Nuclear DOF, n = NStates 
    dH0 = dat.dH0
    qF, pF =  dat.qF * 1.0, dat.pF * 1.0
    # F = np.zeros((len(dat.R)))
    F = -dH0
    for i in range(len(qF)):
        F -= 0.5 * dH[i,i,:] * ( qF[i] ** 2 + pF[i] ** 2 - 2 * γ0[i])
        for j in range(i+1, len(qF)):
            F -= dH[i,j,:] * ( qF[i] * qF[j] + pF[i] * pF[j])
    return F

def VelVer(dat) : # R, P, qF, qB, pF, pB, dtI, dtE, F1, Hij,M=1): # Ionic position, ionic velocity, etc.
 
    # data 
    qF, pF = dat.qF * 1.0, dat.pF * 1.0 
    par =  dat.param
    v = dat.P/par.M
    EStep = int(par.dtN/par.dtE)
    dtE = par.dtN/EStep
    # half-step mapping
    for t in range(int(np.floor(EStep/2))):
        qF, pF = Umap(qF, pF, dtE, dat.Hij)
    dat.qF, dat.pF = qF * 1, pF * 1

    # ======= Nuclear Block ==================================
    F1    =  Force(dat) # force with {qF(t+dt/2)} * dH(R(t))
    dat.R += v * par.dtN + 0.5 * F1 * par.dtN ** 2 / par.M
    
    #------ Do QM ----------------
    dat.Hij  = par.Hel(dat.R)
    dat.dHij = par.dHel(dat.R)
    dat.dH0  = par.dHel0(dat.R)
    #-----------------------------
    F2 = Force(dat) # force with {qF(t+dt/2)} * dH(R(t+ dt))
    v += 0.5 * (F1 + F2) * par.dtN / par.M

    dat.P = v * par.M
    # =======================================================
    
    # half-step mapping
    dat.Hij = par.Hel(dat.R) # do QM
    for t in range(int(np.ceil(EStep/2))):
        qF, pF = Umap(qF, pF, dtE, dat.Hij)
    dat.qF, dat.pF = qF * 1, pF * 1
    
    return dat



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
        vv  = VelVer

        # Call function to initialize mapping variables
        dat.qF, dat.pF, dat.γ0 = initMapping(NStates, initState, stype) 
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

if __name__ == "__main__": 
    import spinBoson as model
    par =  model.parameters
    
    par.dHel = model.dHel
    par.dHel0 = model.dHel0
    par.initR = model.initR
    par.Hel   = model.Hel

    rho_ensemble = runTraj(par)
    
    NSteps = model.parameters.NSteps
    NTraj = model.parameters.NTraj
    NStates = model.parameters.NStates

    PiiFile = open("Pii.txt","w") 
    for t in range(NSteps):
        PiiFile.write(f"{t * model.parameters.nskip} \t")
        for i in range(NStates):
            PiiFile.write(str(rho_ensemble[i,i,t].real / NTraj) + "\t")
        PiiFile.write("\n")
    PiiFile.close()

