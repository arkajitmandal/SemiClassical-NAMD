import numpy as np
from scipy.linalg import toeplitz

class parameters():
   NSteps = 10**2 #int(2*10**6)
   dt = 2*10**(-4)
   NGrid = 100

def HelTwoLevel():
    VMat = np.zeros((2,2))
    VMat[0,0] = 0.1 # Hartrees (27.2114 eV/Hartree)
    VMat[1,0] = 0.01
    VMat[0,1] = 0.01
    VMat[1,1] = 0.0
    return VMat

def HelEqualEnergyManifold(N = 2):
    zeros = np.zeros((N-2))
    matrixvec = [0,0.01]
    matrixvec.extend(zeros)
    print ([0,0.01].extend(zeros))
    VMat = toeplitz(matrixvec,matrixvec)
    for n in range(N):
        for m in range(N):
            if (m == n):
                VMat[n,m] += 0.2 * (1 - n/N) - 0.1
    print (VMat)
    return VMat

def HelMarcusTheory(NGrid = 100):
    R = np.arange(NGrid)
    print (R**2)
    VMat = np.zeros((2,2,NGrid))
    VMat[0,0] = 0.1 * R**2
    VMat[1,0] = 0.01
    VMat[0,1] = 0.01
    VMat[1,1] = 0.1 * (R-1)**2
    return VMat

def getForce(VMat, qF, qB, pF, pB): # Take derivatives w.r.t. some value R along RxN coordinate
    # R finds its way into VMat somehow: e.g., VMat[1,1] = (R-1)^2, VMat[2,2] = (R-2)^2 + 1    (???)
    Hmap = np.zeros((NGrid)) # Mapping Ham
    force = np.zeros((NGrid)) # Deriv. of Mapping Ham
    dx = 1 # Arbitrary grid spacing
    for i in range(NGrid):
        for n in range(len(VMat)):
            for m in range(len(VMat)):
                Hmap[i] += 0.5 * ( 0.5 * VMat[n,m,i] * ( qF[n] * qF[m] + pF[n] * pF[m] ) + 0.5 * VMat[n,m,i] * ( qB[n] * qB[m] + pB[n] * pB[m] ) )
    # Derivative Operator (gradient) in 1D w.r.t. "R"
    # dfdR(i) = [ f(i+1) - f(i-1) ] / (2*dx)
    for i in range(len(NGrid)):
        if (i == 0):
            force[i] = ( Hmap[i+1] - HMap[i] ) / dx
        if (i == NGrid):
            force[i] = ( Hmap[i] - HMap[i-1] ) / dx
        else:
            force[i] = ( Hmap[i+1] - HMap[i-1] ) / (2*dx)
    return force


