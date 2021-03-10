import numpy as np
import model
from model import Hel, dHel, initR


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
    else:
       qF = np.array([ np.random.normal() for i in range(Nstates)]) 
       qB = np.array([ np.random.normal() for i in range(Nstates)]) 
       pF = np.array([ np.random.normal() for i in range(Nstates)]) 
       pB = np.array([ np.random.normal() for i in range(Nstates)]) 
    return qF, qB, pF, pB 

def Umap(qF, qB, pF, pB, dt, R):
    NStates = len(qF)
    VMat = Hel(R)
    qFin, qBin, pFin, pBin = qF, qB, pF, pB # Store input position and momentum for verlet propogation
    # Store initial array containing sums to use at second derivative step

    VMatxqB =  np.array([np.sum(VMat[k,:] * qBin[:]) for k in range(NStates)])
    VMatxqF =  np.array([np.sum(VMat[k,:] * qFin[:]) for k in range(NStates)])
    """
    for i in range(NStates): # Loop over q's and p's for initial update of positions
       # Update momenta using input positions (first-order in dt)
       pB[i] -= 0.5 * dt * np.sum(VMat[i,:] * qBin[:]) ## First Derivatives ##
       pF[i] -= 0.5 * dt * np.sum(VMat[i,:] * qFin[:])
       
       # Now update positions with input momenta (first-order in dt)
       qB[i] += dt * np.sum(VMat[i,:] * pBin[:])
       qF[i] += dt * np.sum(VMat[i,:] * pFin[:])
       #----------------------------------------------------------------------------
       for k in range(NStates):
           # Update positions to second order in dt
           qB[i] -= (dt**2/2.0) * (VMat[i,k])* VMatxqB[k] ## Second Derivatives ##
           qF[i] -= (dt**2/2.0) * (VMat[i,k])* VMatxqF[k]
    """
    # Update momenta using input positions (first-order in dt)
    pB -= 0.5 * dt * VMat @ qBin #np.sum(VMat[i,:] * qBin[:])
    pF -= 0.5 * dt * VMat @ qFin # np.sum(VMat[i,:] * qFin[:])
    # Now update positions with input momenta (first-order in dt)
    qB += dt * VMat @ pBin # np.sum(VMat[i,:] * pBin[:])
    qF += dt * VMat @ pFin # np.sum(VMat[i,:] * pFin[:])
    # Update positions to second order in dt
    qB -=  (dt**2/2.0) * VMat @ VMatxqB 
    qF -=  (dt**2/2.0) * VMat @ VMatxqF
       #-----------------------------------------------------------------------------
    # Update momenta using output positions (first-order in dt)
    pB -= 0.5 * dt * VMat @ qB # np.sum(VMat[i,:] * qB[:])
    pF -= 0.5 * dt * VMat @ qF # np.sum(VMat[i,:] * qF[:])
    """
    for i in range(NStates): # Loop over q's and p's for final update of fictitious momentum
       pB[i] -= 0.5 * dt * np.sum(VMat[i,:] * qB[:])
       pF[i] -= 0.5 * dt * np.sum(VMat[i,:] * qF[:])
    """
    return qF, qB, pF, pB

def Force(R, qF, qB, pF, pB):
    dH = dHel(R) # Nxnxn Matrix, N = Nuclear DOF, n = NStates
    F = np.zeros((len(R)))
    for i in range(len(qF)):
        for j in range(len(qF)):
            F -= 0.25 * dH[i,j,:] * ( qF[i] * qF[j] + pF[i] * pF[j] + qB[i] * qB[j] + pB[i] * pB[j])
    return F

def VelVerF(R, P, qF, qB, pF, pB, dtI, dtE, F1,  M=1): # Ionic position, ionic velocity, etc.
    v = P/M
    #F1 = Force(R, qF, qB, pF, pB)
    R += v * dtI + 0.5 * F1 * dtI ** 2 / M
    EStep = int(dtI/dtE)
    for t in range(EStep):
        qF, qB, pF, pB = Umap(qF, qB, pF, pB, dtE, R)
    F2 = Force(R, qF, qB, pF, pB)
    v += 0.5 * (F1 + F2) * dtI / M
    return R, v*M, qF, qB, pF, pB, F2

def getPopulation(qF, qB, pF, pB, qF0, qB0, pF0, pB0, step):
    #print (step/NSteps * 100, "%")
    rho = np.zeros(( len(qF), len(qF) ), dtype=complex) # Define density matrix
    rho0 = (qF0 - 1j*pF0) * (qB0 + 1j*pB0)
    for i in range(len(qF)):
       for j in range(j,len(qF)):
          rho[i,j] = 0.25 * (qF[i] + 1j*pF[i]) * (qB[j] - 1j*pB[j]) * rho0
          rho[j,i] = rho[i,j] 
    return rho


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
    Ntraj = parameters.NTraj
    stype = parameters.stype
    #---------------------------

    rho_ensemble = np.zeros((NStates,NStates,NSteps), dtype=complex)
    for itraj in range(NTraj): # Ensemble
        R,P = initR()

        # Call function to initialize fictitious oscillators 
        # according to focused ("Default") or according 
        # to gaussian random distribution
        qF, qB, pF, pB = initMapping(NStates, initState, stype) 

        # Set initial values of fictitious oscillator variables for future use
        qF0, qB0, pF0, pB0 = qF[initState], qB[initState], pF[initState], pB[initState] 
        F1 = Force(R, qF, qB, pF, pB)

        for i in range(NSteps): # One trajectory

            if (i % 1 == 0):
                rho_current = getPopulation(qF, qB, pF, pB, qF0, qB0, pF0, pB0, i)
                rho_ensemble[:,:,i] += rho_current
            R, P, qF, qB, pF, pB, F1 = VelVerF(R, P, qF, qB, pF, pB, dtN, dtE, F1, M)

    return rho_ensemble

if __name__ == "__main__": 
    rho_ensemble = runTraj(model.parameters)
    NSteps = model.parameters.NSteps
    NTraj = model.parameters.NTraj
    NStates = model.parameters.NStates
    PiiFile = open("Pii.txt","w") 
    for t in range(NSteps):
        PiiFile.write(str(t) + "\t")
        for i in range(NStates):
            PiiFile.write(str(rho_ensemble[i,i,t].real / NTraj) + "\t")
        PiiFile.write("\n")
    PiiFile.close()

