import numpy as np

class parameters():
   NSteps = 1200 #int(2*10**6)
   NTraj = 400
   dtN = 1
   dtE = dtN/20
   NStates = 2
   M = 2000
   initState = 0
   stype = "multiple focused"
   nskip = 5

def Hel(R):
    A = 0.1
    B = 0.28
    C = 0.015
    D = 0.06
    E0 = 0.05
    VMat = np.zeros((2,2))
    VMat[0,0] = 0.0
    VMat[1,0] = C * np.exp(-D * R**2)
    VMat[0,1] = VMat[1,0]
    VMat[1,1] = -A * np.exp(-B * R**2) + E0
    return VMat
    
def dHel0(R):
    return 0

def dHel(R):
    A = 0.1
    B = 0.28
    C = 0.015
    D = 0.06
    dVMat = np.zeros((2,2,1))
    dVMat[0,0,0] = 0.0
    dVMat[1,0,0] = C * np.exp(-D * R**2) * (-2 * D * R)
    dVMat[0,1,0] = dVMat[1,0]
    dVMat[1,1,0] = -A * np.exp(-B * R**2) * (-2 * B * R)
    return dVMat

def initR():
    R0 = -9.0
    P0 = 30
    alpha = 1.0
    sigR = 1.0/np.sqrt(2.0*alpha)
    sigP = np.sqrt(alpha/2.0)
    R = np.random.normal()*sigR + R0
    P = np.random.normal()*sigP + P0
    return np.array([R]), np.array([P])