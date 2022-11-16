import numpy as np

class parameters():
   NSteps = 4300 * 5 #int(2*10**6)
   NTraj = 20
   dtN = 5
   dtE = dtN/20
   NStates = 2
   M = 550 * 1836
   initState = 1
   nskip = 5
   ndof = 1

def Hel(R):
    V = np.zeros((2,2))
    
    gc = 0.136/27.2114
    wc = 0.06
    v1 = v(R, 0.049244, 0.183747, -0.75)
    v2 = v(R, 0.010657, 0.183747,  0.85)
    v3 = v(R, 0.428129, 0.183747, -1.15)
    v4 = v(R, 0.373005, 0.146997,   1.25)

    D1 = 0.073499
    D2 = 0.514490

    Eg = 0.5 * (v1 + v2) - (D1**2 + (v1-v2)**2/4)**0.5  #+ wc
    Ee = 0.5 * (v3 + v4) - (D2**2 + (v3-v4)**2/4)**0.5 
    V[0,0] = Eg + wc
    V[1,1] = Ee  
    V[0,1] = gc
    V[1,0] = V[0,1]

    return V


def v(R, A1, B1, R1):
    return A1 +  B1 * (R- R1)**2

def dHel0(R):
    return 0


def dHel(R):
    h = 0.0000001
    R1 = R + h
    R2 = R - h
    
    H1 = Hel(R1)
    H2 = Hel(R2)

    dH = (H1 - H2)/(2 * h)
    
    dVij = np.zeros((2,2,1))
    dVij[:,:,0] = dH
    return dVij

def initR():
    R0 = -0.7
    P0 = 0.0
    α = 609
    sigR = 1.0/np.sqrt(2.0*α)
    sigP = np.sqrt(α/2.0)
    R = np.random.normal()*sigR + R0
    P = np.random.normal()*sigP + P0
    return np.array([R]), np.array([P])