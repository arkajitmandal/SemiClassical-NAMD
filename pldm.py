import numpy as np
import sys, os
sys.path.append(os.popen("pwd").read().replace("\n","")+"/Model")
import spinBoson as model
from spinBoson import Hel, dHel, initR, dHel0
import random

# Initialization of the mapping Variables
def initMapping(Nstates, initState = 0, stype = "focused"):
    #global qF, qB, pF, pB, qF0, qB0, pF0, pB0
    qF = np.zeros((Nstates))
    qB = np.zeros((Nstates))
    pF = np.zeros((Nstates))
    pB = np.zeros((Nstates))
    if (stype == "focused"):
        qF[initState] = 1.0
        qB[initState] = 1.0
        pF[initState] = 1.0
        pB[initState] = -1.0 # This minus sign allows for backward motion of fictitious oscillator
    elif (stype == "sampled"):
       qF = np.array([ np.random.normal() for i in range(Nstates)]) 
       qB = np.array([ np.random.normal() for i in range(Nstates)]) 
       pF = np.array([ np.random.normal() for i in range(Nstates)]) 
       pB = np.array([ np.random.normal() for i in range(Nstates)]) 
    return qF, qB, pF, pB 

def Umap(qF, qB, pF, pB, dt, VMat):
    qFin, qBin, pFin, pBin = qF, qB, pF, pB # Store input position and momentum for verlet propogation
    # Store initial array containing sums to use at second derivative step

    VMatxqB =  VMat @ qBin #np.array([np.sum(VMat[k,:] * qBin[:]) for k in range(NStates)])
    VMatxqF =  VMat @ qFin #np.array([np.sum(VMat[k,:] * qFin[:]) for k in range(NStates)])

    # Update momenta using input positions (first-order in dt)
    pB -= 0.5 * dt * VMatxqB  # VMat @ qBin  
    pF -= 0.5 * dt * VMatxqF  # VMat @ qFin  
    # Now update positions with input momenta (first-order in dt)
    qB += dt * VMat @ pBin  
    qF += dt * VMat @ pFin  
    # Update positions to second order in dt
    qB -=  (dt**2/2.0) * VMat @ VMatxqB 
    qF -=  (dt**2/2.0) * VMat @ VMatxqF
       #-----------------------------------------------------------------------------
    # Update momenta using output positions (first-order in dt)
    pB -= 0.5 * dt * VMat @ qB  
    pF -= 0.5 * dt * VMat @ qF  

    return qF, qB, pF, pB

def Force(R, qF, qB, pF, pB):
    dH = dHel(R) # Nxnxn Matrix, N = Nuclear DOF, n = NStates 
    dH0 = dHel0(R)
    F = np.zeros((len(R)))
    F -= dH0
    for i in range(len(qF)):
        F -= 0.25 * dH[i,i,:] * ( qF[i] ** 2 + pF[i] ** 2 + qB[i] ** 2 + pB[i] ** 2)
        for j in range(i+1, len(qF)):
            F -= 0.5 * dH[i,j,:] * ( qF[i] * qF[j] + pF[i] * pF[j] + qB[i] * qB[j] + pB[i] * pB[j])
    return F

def VelVer(R, P, qF, qB, pF, pB, dtI, dtE, F1, Hij,M=1): # Ionic position, ionic velocity, etc.
    v = P/M
    EStep = int(dtI/dtE)
    # Hij = Hel(R)
    for t in range(EStep):
        qF, qB, pF, pB = Umap(qF, qB, pF, pB, dtE/2, Hij)
    #F1 = Force(R, qF, qB, pF, pB)
    R += v * dtI + 0.5 * F1 * dtI ** 2 / M
    Hij = Hel(R)
    for t in range(EStep):
        qF, qB, pF, pB = Umap(qF, qB, pF, pB, dtE/2, Hij)
    F2 = Force(R, qF, qB, pF, pB)
    v += 0.5 * (F1 + F2) * dtI / M
    return R, v*M, qF, qB, pF, pB, F2, Hij

def pop(qF, qB, pF, pB, rho0):
    return np.outer(qF + 1j*pF, qB-1j*pB) * rho0

def runTraj(parameters = model.parameters):
    #------- Seed --------------------
    try:
        np.random.seed(parameters.SEED)
    except:
        pass
    #------------------------------------
    ## Parameters -------------
    dtE = parameters.dtE
    dtN = parameters.dtN
    NSteps = parameters.NSteps
    NTraj = parameters.NTraj
    NStates = parameters.NStates
    M = parameters.M #mass
    initState = parameters.initState # intial state
    stype = parameters.stype
    nskip = parameters.nskip
    #---------------------------

    rho_ensemble = np.zeros((NStates,NStates,NSteps//nskip), dtype=complex)
    # Ensemble
    for itraj in range(NTraj): 
        R,P = initR()
        vv  = VelVer

        # Call function to initialize fictitious oscillators 
        # according to focused ("Default") or according 
        # to gaussian random distribution
        qF, qB, pF, pB = initMapping(NStates, initState, stype) 

        # Set initial values of fictitious oscillator variables for future use
        qF0, qB0, pF0, pB0 = qF[initState], qB[initState], pF[initState], pB[initState] 
        rho0 = 0.25 * (qF0 - 1j*pF0) * (qB0 + 1j*pB0)

        #----- Initial Force --------
        F1 = Force(R, qF, qB, pF, pB)
        Hij = Hel(R)
        iskip = 0
        for i in range(NSteps): # One trajectory
            #------- ESTIMATORS-------------------------------------
            if (i % nskip == 0):
                rho_ensemble[:,:,iskip] += pop(qF, qB, pF, pB, rho0)
                iskip += 1
            #-------------------------------------------------------
            R, P, qF, qB, pF, pB, F1, Hij = vv(R, P, qF, qB, pF, pB, dtN, dtE, F1, Hij, M)

    return rho_ensemble

if __name__ == "__main__": 
    rho_ensemble = runTraj(model.parameters)
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

