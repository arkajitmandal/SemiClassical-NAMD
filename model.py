#def param():

import numpy as np
from scipy.linalg import toeplitz

class parameters():
   NSteps = int(2*10**6)
   dt = 2*10**(-4)

def HelTwoLevel():
    VMat = np.zeros((2,2))
    VMat[0,0] = 0.1 # Hartrees (27.2114 eV/Hartree)
    VMat[1,0] = 0.01
    VMat[0,1] = 0.01
    VMat[1,1] = 0.0
    return VMat

def HelEqualManifold(N = 3):
    zeros = np.zeros((N-2))
    matrixvec = [0,0.01]
    matrixvec.extend(zeros)
    print ([0,0.01].extend(zeros))
    VMat = toeplitz(matrixvec,matrixvec)
    for n in range(N):
        for m in range(N):
            if (m == n):
                VMat[n,m] += 0.2 * (1 - n/N)
    print (VMat)
    return VMat
