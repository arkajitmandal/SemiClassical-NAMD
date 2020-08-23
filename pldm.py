import numpy as np
import model

dt = model.parameters.dt
NSteps = model.parameters.NSteps

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

def propagateMapVars(qF, qB, pF, pB, dt): # Note that I removed "R" from this list temporarily
    VMat = model.Hel() # Get interaction Hamiltonian from model file
    qFin, qBin, pFin, pBin = qF, qB, pF, pB # Store input position and momentum for verlet propogation
    # Store initial array containing sums to use at second derivative step
    VMatxqB =  np.array([np.sum(VMat[k,:] * qBin[:]) for k in range(N)])
    VMatxqF =  np.array([np.sum(VMat[k,:] * qFin[:]) for k in range(N)])
    for i in range(NStates): # Loop over q's and p's for initial update of positions
       pB[i] -= 0.5 * dt * np.sum(VMat[i,:] * qBin[:]) ## First Derivatives ##
       pF[i] -= 0.5 * dt * np.sum(VMat[i,:] * qFin[:])
       qB[i] += dt * np.sum(VMat[i,:] * pBin[:])
       qF[i] += dt * np.sum(VMat[i,:] * pFin[:])
       for k in range(NStates):
           qB[i] -= (dt**2/2.0) * (VMat[i,k])* VMatxqB[k] ## Second Derivatives ##
           qF[i] -= (dt**2/2.0) * (VMat[i,k])* VMatxqF[k]

    for i in range(N): # Loop over q's and p's for final update of fictitious momentum
       pB[i] -= 0.5 * dt * np.sum(Vij[i,:] * qB[:])
       pF[i] -= 0.5 * dt * np.sum(Vij[i,:] * qF[:])
    return qF, qB, pF, pB

def getPopulation(qF, qB, pF, pB, qF0, qB0, pF0, pB0):
    rho = np.zeros(( len(qF), len(qF) ), dtype=complex) # Define density matrix
    rho0 = (qF0 - 1j*pF0) * (qB0 + 1j*pB0)
    for i in range(len(qF)):
       for j in range(len(qF)):
          if (j >= i):
              rho[i,j] = 0.25 * (qF[i] + 1j*pF[i]) * (qB[j] - 1j*pB[j]) * rho0
    return rho


## Start Main Program

initState = 0
qF, qB, pF, pB = initMapping(2,initState)
qF0, qB0, pF0, pB0 = qF[initState], qB[initState], pF[initState], pB[initState] # Set initial values of fictitious oscillator variables

file01 = open("output_rho.txt","w") # Density natrix: Step, 11, 12, 13, 14, 22, 23, 24, 33, 34, 44, or similar for uuper triangle of NxN matrix
#file02 = open("output_map.txt","w") # Fictitious Oscillator Motion (Use for sanity check): Step, qF[0], PF

for i in range(NSteps):
   rho_current = getPopulation(qF, qB, pF, pB, qF0, qB0, pF0, pB0)
   file01.write( str(i) + "\t" + "\t".join(rho_current.flatten().real.astype("str"))+ "\n")
   #file02.write( str(i) + "\t" + str(qF[0]) + "\t" + str(pF[0]) + + "\t" + str(qB[0]) + "\t" + str(pB[0]) + "\n")
   qF, qB, pF, pB = propagateMapVars(qF, qB, pF, pB, dt)
file01.close()
#file02.close()
   





