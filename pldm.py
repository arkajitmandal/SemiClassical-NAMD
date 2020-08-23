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
        pB[initState] = 1.0
    else:
       qF = np.array([ np.random.normal() for i in range(Nstates)]) 
       qB = np.array([ np.random.normal() for i in range(Nstates)]) 
       pF = np.array([ np.random.normal() for i in range(Nstates)]) 
       pB = np.array([ np.random.normal() for i in range(Nstates)]) 
    return qF, qB, pF, pB 

def propagateMapVars(qF, qB, pF, pB, dt): # Note that I removed "R" from this list temporarily
    Vij = model.Hel() 
    N = len(Vij)
    qF1, qB1, pF1, pB1 = qF, qB, pF, pB
    VijxqB =  np.array([np.sum(Vij[k,:] * qB1[:]) for k in range(N)])
    VijxqF =  np.array([np.sum(Vij[k,:] * qF1[:]) for k in range(N)])
    for i in range(N): # Loop over q's and p's
       pB[i] -= 0.5 * dt * np.sum(Vij[i,:] * qB1[:])
       pF[i] -= 0.5 * dt * np.sum(Vij[i,:] * qF1[:])
       qB[i] += dt * np.sum(Vij[i,:] * pB1[:])
       qF[i] += dt * np.sum(Vij[i,:] * pF1[:])
       for k in range(N):
           qB[i] -= (dt**2/2.0) * (Vij[i,k])* VijxqB[k] 	 # Second loop over index of Vij
           qF[i] -= (dt**2/2.0) * (Vij[i,k])* VijxqF[k]	 # Second loop over index of Vij

    for i in range(N): # Loop over q's and p's
       pB[i] -= 0.5 * dt * np.sum(Vij[i,:] * qB[:])
       pF[i] -= 0.5 * dt * np.sum(Vij[i,:] * qF[:])
    return qF, qB, pF, pB

def getPopulation(qF, qB, pF, pB, qF0, qB0, pF0, pB0):
    rho = np.zeros(( len(qF), len(qF) ), dtype=complex) # Define density matrix
    rho0 = (qF0 - 1j*pF0) * (qB0 + 1j*pB0)
    for i in range(len(qF)):
       for j in range(len(qF)):
          rho[i,j] = 0.25 * (qF[i] + 1j*pF[i]) * (qB[j] - 1j*pB[j]) * rho0
    return rho


## Start Main Program

initState = 0
qF, qB, pF, pB = initMapping(2,initState)
qF0, qB0, pF0, pB0 = qF[initState], qB[initState], pF[initState], pB[initState] # Set initial values

file01 = open("output_rho.txt","w")
file02 = open("output_map.txt","w")

for i in range(NSteps):
   rho_current = getPopulation(qF, qB, pF, pB, qF0, qB0, pF0, pB0)
   file01.write( str(i) + "\t" + "\t".join(rho_current.flatten().real.astype("str"))+ "\n")
   file02.write( str(i) + "\t" + str(qF[0]) + "\t" + str(pF[0]) + "\n")
   qF, qB, pF, pB = propagateMapVars(qF, qB, pF, pB, dt)
   #print (rho_current[0,0] + rho_current[1,1])
file01.close()
file02.close()
   

# q(t+dt) = q(t) + 0.25 * SUM_J(vij*Pj)*dt + (1/8) SUM_j(-0.25 SUM_(jk) vjk*qk)*dt^2




