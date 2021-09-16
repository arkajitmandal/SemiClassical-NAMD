import numpy as np

class parameters():
   NSteps = 600 #int(2*10**6)
   NTraj = 100
   dtN = 2
   dtE = dtN/20
   NStates = 2
   M = 2000
   initState = 0
   nskip = 5

def Hel(R):
    V = np.zeros((2,2))
    A = 0.01
    B = 1.6
    C = 0.005
    D = 1.0
    V[0,0] = A*np.sign(R)*(1-np.exp(-B*np.abs(R)))
    V[1,1] = -V[0,0]
    V[0,1] = C*np.exp(-D*R**2)
    V[1,0] = V[0,1]

    return Vij

def dHel0(R):
    return 0

def dHel(R):
    A = 0.01
    B = 1.6
    C = 0.005
    D = 1.0
    dVMat = np.zeros((2,2,1))
    dVMat[0,0,0] = A*B*np.exp(-B*np.abs(x))
    dVMat[1,0,0] = C * np.exp(-D * R**2) * (-2 * D * R)
    dVMat[0,1,0] = dVMat[1,0,0]
    dVMat[1,1,0] = -dV[0,0,0]

    return dVMat

def initR():
    R0 = -9.0
    P0 = 30
    α = 1.0
    sigR = 1.0/np.sqrt(2.0*α)
    sigP = np.sqrt(α/2.0)
    R = np.random.normal()*sigR + R0
    P = np.random.normal()*sigP + P0
    return np.array([R]), np.array([P])