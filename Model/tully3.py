import numpy as np

class parameters():
   NSteps = 600 #int(2*10**6)
   NTraj = 50
   dtN = 2
   dtE = dtN/40
   NStates = 2
   M = 2000
   initState = 0
   nskip = 10
   ndof = 1

def Hel(R):

    Vij = np.zeros((2,2))
    A = 6*10**-4
    B = 0.1 
    C = 0.9 
    Vij[0,0] = A

    if ( R < 0 ):
        Vij[1,0] = B * np.exp( C*R )
    else:
        Vij[1,0] = B * ( 2 - np.exp( -C*R ) )

    Vij[0,1] = Vij[1,0]
    Vij[1,1] = -A

    return Vij
    
def dHel0(R):
    return np.zeros((len(R)))


def dHel(R):
    dVij = np.zeros((2,2,1))

    B = 0.1
    C = 0.9

    if ( R < 0 ):
        dVij[1,0,0] = - B * C * np.exp( C*R )
    else:
        dVij[1,0,0] = B * C * np.exp( -C*R )
    dVij[0,1,0] = dVij[1,0,0]
    return dVij

def initR():
    R0 = -9.0
    P0 = 30.0
    alpha = 1.0
    sigR = 1.0/np.sqrt(2.0*alpha)
    sigP = np.sqrt(alpha/2.0)
    R = np.random.normal()*sigR + R0
    P = np.random.normal()*sigP + P0
    return np.array([R]), np.array([P])