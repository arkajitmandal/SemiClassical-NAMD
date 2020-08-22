import numpy as np
import model

# Initialization of the mapping Variables
def initMapping(Nstates, initState = 0, stype = "focused"):
    global qF, qB, pF, pB
    qF = np.zeros((Nstates))
    qB = np.zeros((Nstates))
    pF = np.zeros((Nstates))
    pB = np.zeros((Nstates))
    if (stype == "focused"):
        qF[initState] = 1.0
        qB[initState] = 1.0
        pF[initState] = 1.0
        pB[initState] = -1.0
    else:
       qF = np.array([ np.random.normal() for i in range(Nstates)]) 
       qB = np.array([ np.random.normal() for i in range(Nstates)]) 
       pF = np.array([ np.random.normal() for i in range(Nstates)]) 
       pB = np.array([ np.random.normal() for i in range(Nstates)]) 
    return qF, qB, pF, pB 

def propagateMapVars(qF, qB, pF, pB, dt): # Note that I removed "R" from this list temporarily
    Vij = model.Hel() 
    for i in range(len(Vij)): # Loop over q's and p's
       for j in range(np.shape(Vij)[0]): # Loop over index of Vij
          pB[i] -= 0.5 * dt * np.sum(Vij[i,:] * qB[:])
          pF[i] -= 0.5 * dt * np.sum(Vij[i,:] * qF[:])
          qB[i] += dt * np.sum(Vij[i,:] * pB[:])
          qF[i] += dt * np.sum(Vij[i,:] * pF[:])
          for k in range(np.shape(Vij)[0]): # Second loop over index of Vij
             qB[i] -= (dt**2/2.0) * np.sum(Vij[i,:]*np.sum(Vij[:,] * qB[:]))
             qF[i] -= (dt**2/2.0) * np.sum(Vij[i,:]*np.sum(Vij[:,] * qF[:]))
          pB[i] -= 0.5 * dt * np.sum(Vij[i,:] * qB[:])
          pF[i] -= 0.5 * dt * np.sum(Vij[i,:] * qF[:])             
    return qF, qB, pF, pB

initMapping(2)
propagateMapVars(qF, qB, pF, pB, 1)


# q(t+dt) = q(t) + 0.25 * SUM_J(vij*Pj)*dt + (1/8) SUM_j(-0.25 SUM_(jk) vjk*qk)*dt^2




