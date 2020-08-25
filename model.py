import numpy as np
from scipy.linalg import toeplitz

class parameters():
   NSteps = 10**2 #int(2*10**6)
   dtI = 1
   dtE = dtI/20
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

def initR():
    R0 = -9.0
    P0 = 30
    alpha = 1.0
    sigR = 1.0/np.sqrt(2.0*alpha)
    sigP = np.sqrt(alpha/2.0)
    R = np.random.normal()*sigR + R0
    P = np.random.normal()*sigP + P0
    return R, P